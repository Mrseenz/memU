from __future__ import annotations

import base64
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - optional dependency in constrained envs
    httpx = None  # type: ignore[assignment]

from memu.llm.backends.base import LLMBackend
from memu.llm.backends.doubao import DoubaoLLMBackend
from memu.llm.backends.gguf import GGUFLLMBackend
from memu.llm.backends.grok import GrokBackend
from memu.llm.backends.openai import OpenAILLMBackend
from memu.llm.backends.openrouter import OpenRouterLLMBackend


# Minimal embedding backend support (moved from embedding module)
class _EmbeddingBackend:
    name: str
    embedding_endpoint: str

    def build_embedding_payload(self, *, inputs: list[str], embed_model: str) -> dict[str, Any]:
        raise NotImplementedError

    def parse_embedding_response(self, data: dict[str, Any]) -> list[list[float]]:
        raise NotImplementedError


class _OpenAIEmbeddingBackend(_EmbeddingBackend):
    name = "openai"
    embedding_endpoint = "/embeddings"

    def build_embedding_payload(self, *, inputs: list[str], embed_model: str) -> dict[str, Any]:
        return {"model": embed_model, "input": inputs}

    def parse_embedding_response(self, data: dict[str, Any]) -> list[list[float]]:
        return [cast(list[float], d["embedding"]) for d in data["data"]]


class _DoubaoEmbeddingBackend(_EmbeddingBackend):
    name = "doubao"
    embedding_endpoint = "/api/v3/embeddings"

    def build_embedding_payload(self, *, inputs: list[str], embed_model: str) -> dict[str, Any]:
        return {"model": embed_model, "input": inputs, "encoding_format": "float"}

    def parse_embedding_response(self, data: dict[str, Any]) -> list[list[float]]:
        return [cast(list[float], d["embedding"]) for d in data["data"]]


class _OpenRouterEmbeddingBackend(_EmbeddingBackend):
    """OpenRouter uses OpenAI-compatible embedding API."""

    name = "openrouter"
    embedding_endpoint = "/api/v1/embeddings"

    def build_embedding_payload(self, *, inputs: list[str], embed_model: str) -> dict[str, Any]:
        return {"model": embed_model, "input": inputs}

    def parse_embedding_response(self, data: dict[str, Any]) -> list[list[float]]:
        return [cast(list[float], d["embedding"]) for d in data["data"]]


logger = logging.getLogger(__name__)


def _require_httpx() -> None:
    if httpx is None:
        msg = "httpx is required for HTTPLLMClient. Install with `pip install httpx`."
        raise ModuleNotFoundError(msg)

LLM_BACKENDS: dict[str, Callable[[], LLMBackend]] = {
    OpenAILLMBackend.name: OpenAILLMBackend,
    DoubaoLLMBackend.name: DoubaoLLMBackend,
    GGUFLLMBackend.name: GGUFLLMBackend,
    GrokBackend.name: GrokBackend,
    OpenRouterLLMBackend.name: OpenRouterLLMBackend,
}


class HTTPLLMClient:
    """HTTP client for LLM APIs (chat, vision, transcription) and embeddings."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        chat_model: str,
        provider: str = "openai",
        endpoint_overrides: dict[str, str] | None = None,
        capability_autodetect: bool = True,
        capability_models_endpoint: str = "/models",
        extra_headers: dict[str, str] | None = None,
        timeout: int = 60,
        embed_model: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.chat_model = chat_model
        self.provider = provider.lower()
        self.backend = self._load_backend(self.provider)
        self.embedding_backend = self._load_embedding_backend(self.provider)
        overrides = endpoint_overrides or {}
        self.summary_endpoint = overrides.get("chat") or overrides.get("summary") or self.backend.summary_endpoint
        self.embedding_endpoint = (
            overrides.get("embeddings")
            or overrides.get("embedding")
            or overrides.get("embed")
            or self.embedding_backend.embedding_endpoint
        )
        self.timeout = timeout
        self.embed_model = embed_model or chat_model
        self.extra_headers = dict(extra_headers or {})
        self.capability_autodetect = capability_autodetect
        self.capability_models_endpoint = capability_models_endpoint
        self._capabilities: dict[str, bool | None] = {"chat": None, "vision": None, "embeddings": None}
        self._capabilities_probed = False

    @staticmethod
    def _infer_capabilities_from_models(
        models: list[dict[str, Any]], *, chat_model: str, embed_model: str
    ) -> dict[str, bool | None]:
        def _get_modalities(model: dict[str, Any]) -> set[str]:
            values: set[str] = set()
            for key in ("modalities", "input_modalities", "output_modalities"):
                raw = model.get(key)
                if isinstance(raw, list):
                    for item in raw:
                        if isinstance(item, str):
                            values.add(item.lower())
            return values

        def _find_model(model_id: str) -> dict[str, Any] | None:
            needle = model_id.strip().lower()
            for model in models:
                if str(model.get("id", "")).strip().lower() == needle:
                    return model
            return None

        def _is_embedding_model(model: dict[str, Any]) -> bool:
            mid = str(model.get("id", "")).lower()
            mods = _get_modalities(model)
            return bool({"embedding", "embeddings"} & mods) or "embedding" in mid

        def _is_vision_model(model: dict[str, Any]) -> bool:
            mid = str(model.get("id", "")).lower()
            mods = _get_modalities(model)
            return bool({"image", "vision"} & mods) or any(t in mid for t in ("vision", "vl", "llava", "gpt-4o"))

        chat_supported = bool(models)
        embed_supported = any(_is_embedding_model(model) for model in models)
        vision_supported = any(_is_vision_model(model) for model in models)

        chat_target = _find_model(chat_model)
        if chat_target is not None:
            chat_supported = True
            vision_supported = _is_vision_model(chat_target)

        embed_target = _find_model(embed_model)
        if embed_target is not None:
            embed_supported = _is_embedding_model(embed_target)

        return {"chat": chat_supported, "vision": vision_supported, "embeddings": embed_supported}

    async def _ensure_capabilities(self) -> None:
        if not self.capability_autodetect or self._capabilities_probed:
            return
        self._capabilities_probed = True
        _require_httpx()
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
                resp = await client.get(self.capability_models_endpoint, headers=self._headers())
                resp.raise_for_status()
                payload = resp.json()
        except Exception:
            logger.info(
                "Capability auto-detection skipped for provider=%s base_url=%s endpoint=%s",
                self.provider,
                self.base_url,
                self.capability_models_endpoint,
                exc_info=True,
            )
            return

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return
        models = [m for m in data if isinstance(m, dict)]
        self._capabilities = self._infer_capabilities_from_models(
            models, chat_model=self.chat_model, embed_model=self.embed_model
        )

    async def _supports(self, capability: Literal["chat", "vision", "embeddings"]) -> bool | None:
        await self._ensure_capabilities()
        return self._capabilities.get(capability)

    @staticmethod
    def _looks_unsupported_error(exc: Exception, operation: str) -> bool:
        if httpx is None or not isinstance(exc, httpx.HTTPStatusError):
            return False
        status = exc.response.status_code
        if status not in {400, 404, 405, 422, 501}:
            return False
        text = exc.response.text.lower()
        markers = {
            "chat": ("chat", "completions", "unsupported"),
            "vision": ("vision", "image", "multimodal", "unsupported"),
            "embeddings": ("embedding", "embeddings", "unsupported"),
        }
        return any(marker in text for marker in markers[operation])

    async def chat(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> tuple[str, dict[str, Any]]:
        """Generic chat completion."""
        if (await self._supports("chat")) is False:
            msg = f"Provider '{self.provider}' does not advertise chat capability via {self.capability_models_endpoint}."
            raise RuntimeError(msg)
        messages: list[dict[str, Any]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        _require_httpx()
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
                resp = await client.post(self.summary_endpoint, json=payload, headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            if self._looks_unsupported_error(exc, "chat"):
                self._capabilities["chat"] = False
            raise
        logger.debug("HTTP LLM chat response: %s", data)
        return self.backend.parse_summary_response(data), data

    async def summarize(
        self, text: str, max_tokens: int | None = None, system_prompt: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        payload = self.backend.build_summary_payload(
            text=text, system_prompt=system_prompt, chat_model=self.chat_model, max_tokens=max_tokens
        )
        _require_httpx()
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            resp = await client.post(self.summary_endpoint, json=payload, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
        logger.debug("HTTP LLM summarize response: %s", data)
        return self.backend.parse_summary_response(data), data

    async def vision(
        self,
        prompt: str,
        image_path: str,
        *,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Call Vision API with an image.

        Args:
            prompt: Text prompt to send with the image
            image_path: Path to the image file
            max_tokens: Maximum tokens in response
            system_prompt: Optional system prompt

        Returns:
            Tuple of (LLM response text, raw response dict)
        """
        if (await self._supports("vision")) is False:
            fallback_prompt = (
                "Vision capability is not available on this provider. "
                "Respond with a concise best-effort answer based on the text prompt only.\n\n"
                f"Original prompt:\n{prompt}"
            )
            return await self.chat(fallback_prompt, max_tokens=max_tokens, system_prompt=system_prompt)

        # Read and encode image as base64
        image_data = Path(image_path).read_bytes()
        base64_image = base64.b64encode(image_data).decode("utf-8")

        # Detect image format
        suffix = Path(image_path).suffix.lower()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(suffix, "image/jpeg")

        payload = self.backend.build_vision_payload(
            prompt=prompt,
            base64_image=base64_image,
            mime_type=mime_type,
            system_prompt=system_prompt,
            chat_model=self.chat_model,
            max_tokens=max_tokens,
        )

        _require_httpx()
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
                resp = await client.post(self.summary_endpoint, json=payload, headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            if self._looks_unsupported_error(exc, "vision"):
                self._capabilities["vision"] = False
                fallback_prompt = (
                    "Vision request failed because image understanding is unsupported. "
                    "Provide a response from text only.\n\n"
                    f"Original prompt:\n{prompt}"
                )
                return await self.chat(fallback_prompt, max_tokens=max_tokens, system_prompt=system_prompt)
            raise
        logger.debug("HTTP LLM vision response: %s", data)
        return self.backend.parse_summary_response(data), data

    async def embed(self, inputs: list[str]) -> tuple[list[list[float]], dict[str, Any]]:
        """Create text embeddings using the provider-specific embedding API."""
        if (await self._supports("embeddings")) is False:
            msg = (
                f"Provider '{self.provider}' does not advertise embedding support via {self.capability_models_endpoint}. "
                "Configure an embedding-capable profile or use retrieve.method='llm'."
            )
            raise RuntimeError(msg)
        payload = self.embedding_backend.build_embedding_payload(inputs=inputs, embed_model=self.embed_model)
        _require_httpx()
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
                resp = await client.post(self.embedding_endpoint, json=payload, headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            if self._looks_unsupported_error(exc, "embeddings"):
                self._capabilities["embeddings"] = False
                msg = (
                    "Embedding request failed because this provider likely does not support embeddings. "
                    "Switch to an embedding-capable profile or use retrieve.method='llm'."
                )
                raise RuntimeError(msg) from exc
            raise
        logger.debug("HTTP embedding response: %s", data)
        return self.embedding_backend.parse_embedding_response(data), data

    async def transcribe(
        self,
        audio_path: str,
        *,
        prompt: str | None = None,
        language: str | None = None,
        response_format: str = "text",
    ) -> tuple[str, dict[str, Any] | None]:
        """
        Transcribe audio file using OpenAI Audio API.

        Args:
            audio_path: Path to the audio file
            prompt: Optional prompt to guide the transcription
            language: Optional language code (e.g., 'en', 'zh')
            response_format: Response format ('text', 'json', 'verbose_json')

        Returns:
            Tuple of (transcribed text, raw response dict or None for text format)
        """
        try:
            raw_response: dict[str, Any] | None = None
            # Prepare multipart form data
            with open(audio_path, "rb") as audio_file:
                files = {"file": (Path(audio_path).name, audio_file, "application/octet-stream")}
                data = {
                    "model": "gpt-4o-mini-transcribe",
                    "response_format": response_format,
                }
                if prompt:
                    data["prompt"] = prompt
                if language:
                    data["language"] = language

                _require_httpx()
                async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout * 3) as client:
                    resp = await client.post(
                        "/v1/audio/transcriptions",
                        files=files,
                        data=data,
                        headers=self._headers(),
                    )
                    resp.raise_for_status()

                    if response_format == "text":
                        result = resp.text
                    else:
                        raw_response = resp.json()
                        result = raw_response.get("text", "")

            logger.debug("HTTP audio transcribe response for %s: %s chars", audio_path, len(result))
        except Exception:
            logger.exception("Audio transcription failed for %s", audio_path)
            raise
        else:
            return result or "", raw_response

    def _headers(self) -> dict[str, str]:
        headers = dict(self.extra_headers)
        if self.api_key.strip():
            headers.setdefault("Authorization", f"Bearer {self.api_key}")
        return headers

    def _load_backend(self, provider: str) -> LLMBackend:
        factory = LLM_BACKENDS.get(provider)
        if not factory:
            msg = f"Unsupported LLM provider '{provider}'. Available: {', '.join(LLM_BACKENDS.keys())}"
            raise ValueError(msg)
        return factory()

    def _load_embedding_backend(self, provider: str) -> _EmbeddingBackend:
        backends: dict[str, type[_EmbeddingBackend]] = {
            _OpenAIEmbeddingBackend.name: _OpenAIEmbeddingBackend,
            _DoubaoEmbeddingBackend.name: _DoubaoEmbeddingBackend,
            "gguf": _OpenAIEmbeddingBackend,
            "grok": _OpenAIEmbeddingBackend,
            _OpenRouterEmbeddingBackend.name: _OpenRouterEmbeddingBackend,
        }
        factory = backends.get(provider)
        if not factory:
            msg = f"Unsupported embedding provider '{provider}'. Available: {', '.join(backends.keys())}"
            raise ValueError(msg)
        return factory()

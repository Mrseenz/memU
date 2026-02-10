from __future__ import annotations

from memu.llm.backends.openai import OpenAILLMBackend


class GGUFLLMBackend(OpenAILLMBackend):
    """Backend for local GGUF models served via an OpenAI-compatible API."""

    name = "gguf"


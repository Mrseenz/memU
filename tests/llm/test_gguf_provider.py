from memu.app.settings import LLMConfig
from memu.llm.http_client import HTTPLLMClient


def test_gguf_settings_defaults():
    config = LLMConfig(provider="gguf")

    assert config.base_url == "http://127.0.0.1:8080/v1"
    assert config.api_key == ""
    assert config.chat_model == "local-gguf"
    assert config.capability_autodetect is True
    assert config.capability_models_endpoint == "/models"


def test_gguf_http_client_supports_provider_and_no_auth_header():
    client = HTTPLLMClient(
        base_url="http://127.0.0.1:8080/v1",
        api_key="",
        chat_model="local-gguf",
        provider="gguf",
    )

    assert client.provider == "gguf"
    assert client._headers() == {}
    assert client.backend.name == "gguf"


def test_gguf_http_client_supports_custom_headers_without_api_key():
    client = HTTPLLMClient(
        base_url="http://127.0.0.1:8080/v1",
        api_key="",
        chat_model="local-gguf",
        provider="gguf",
        extra_headers={"X-Api-Key": "local-secret", "X-Client": "memu"},
    )

    assert client._headers() == {"X-Api-Key": "local-secret", "X-Client": "memu"}


def test_gguf_http_client_preserves_explicit_authorization_header():
    client = HTTPLLMClient(
        base_url="http://127.0.0.1:8080/v1",
        api_key="token-from-config",
        chat_model="local-gguf",
        provider="gguf",
        extra_headers={"Authorization": "Bearer explicit-token"},
    )

    assert client._headers()["Authorization"] == "Bearer explicit-token"


def test_infer_capabilities_from_models_prefers_target_models():
    models = [
        {"id": "text-embedding-3-small", "modalities": ["embedding"]},
        {"id": "llava-1.6", "input_modalities": ["text", "image"]},
        {"id": "local-gguf", "modalities": ["text"]},
    ]

    result = HTTPLLMClient._infer_capabilities_from_models(
        models,
        chat_model="local-gguf",
        embed_model="text-embedding-3-small",
    )

    assert result["chat"] is True
    assert result["vision"] is False
    assert result["embeddings"] is True


def test_infer_capabilities_detects_vision_on_chat_model():
    models = [
        {"id": "local-gguf", "input_modalities": ["text", "image"]},
    ]

    result = HTTPLLMClient._infer_capabilities_from_models(
        models,
        chat_model="local-gguf",
        embed_model="local-gguf",
    )

    assert result["chat"] is True
    assert result["vision"] is True
    assert result["embeddings"] is False

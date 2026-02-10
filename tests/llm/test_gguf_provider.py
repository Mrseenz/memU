from memu.app.settings import LLMConfig
from memu.llm.http_client import HTTPLLMClient


def test_gguf_settings_defaults():
    config = LLMConfig(provider="gguf")

    assert config.base_url == "http://127.0.0.1:8080/v1"
    assert config.api_key == ""
    assert config.chat_model == "local-gguf"


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

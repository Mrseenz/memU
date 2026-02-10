"""
Example 6: Local GGUF model via OpenAI-compatible endpoint (llama.cpp, etc.)

Usage:
    # Start a local OpenAI-compatible server first, e.g.:
    # llama.cpp server --model ./your-model.gguf --port 8080
    python examples/example_6_local_gguf.py
"""

import asyncio

from memu.app import MemoryService


async def main() -> None:
    service = MemoryService(
        llm_profiles={
            "default": {
                "provider": "gguf",
                "client_backend": "httpx",
                "base_url": "http://127.0.0.1:8080/v1",
                "api_key": "",  # no auth for most local setups
                # Optional: pass local gateway specific headers
                # "http_headers": {"X-Api-Key": "local-secret"},
                "chat_model": "local-gguf",
                # Set embed_model only if your local endpoint supports /embeddings
                "embed_model": "local-gguf",
            },
        },
    )

    result = await service.memorize(
        resource_url="examples/resources/conversations/conv1.json",
        modality="conversation",
        user={"user_id": "local_gguf_demo"},
    )

    print(f"memorize: items={len(result.get('items', []))}, categories={len(result.get('categories', []))}")

    retrieval = await service.retrieve(
        queries=[{"role": "user", "content": {"text": "What are this user's preferences?"}}],
        where={"user_id": "local_gguf_demo"},
    )
    print(f"retrieve: items={len(retrieval.get('items', []))}, categories={len(retrieval.get('categories', []))}")


if __name__ == "__main__":
    asyncio.run(main())

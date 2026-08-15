"""
Dense embeddings via SiliconFlow's OpenAI-compatible ``/embeddings`` endpoint
(bge-m3, 1024-dim). Texts are sent in batches to bound request size.
"""
import logging

import httpx

from app.agent.rag.config import rag_settings

logger = logging.getLogger(__name__)


async def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, preserving order. Returns one vector per input."""
    if not texts:
        return []

    vectors: list[list[float]] = []
    batch = rag_settings.rag_embed_batch_size
    headers = {"Authorization": f"Bearer {rag_settings.siliconflow_api_key}"}

    async with httpx.AsyncClient(base_url=rag_settings.siliconflow_base_url, timeout=60) as client:
        for start in range(0, len(texts), batch):
            chunk = texts[start : start + batch]
            resp = await client.post(
                "/embeddings",
                headers=headers,
                json={"model": rag_settings.embedding_model, "input": chunk},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            vectors.extend(item["embedding"] for item in sorted(data, key=lambda d: d["index"]))
    return vectors


async def embed_one(text: str) -> list[float]:
    return (await embed([text]))[0]

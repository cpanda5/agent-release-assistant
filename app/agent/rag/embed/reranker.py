"""
Cross-encoder reranking via SiliconFlow's ``/rerank`` endpoint (bge-reranker-v2-m3).
Takes the fused candidate set and reorders it by true query–document relevance,
returning the indices + scores of the top results.
"""
import logging

import httpx

from app.agent.rag.config import rag_settings

logger = logging.getLogger(__name__)


async def rerank(query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
    """Return ``[(original_index, score), ...]`` ordered best-first, capped at top_n."""
    if not documents:
        return []

    headers = {"Authorization": f"Bearer {rag_settings.siliconflow_api_key}"}
    async with httpx.AsyncClient(base_url=rag_settings.siliconflow_base_url, timeout=60) as client:
        resp = await client.post(
            "/rerank",
            headers=headers,
            json={
                "model": rag_settings.rerank_model,
                "query": query,
                "documents": documents,
                "top_n": min(top_n, len(documents)),
            },
        )
        resp.raise_for_status()
        results = resp.json()["results"]
    return [(r["index"], r["relevance_score"]) for r in results]

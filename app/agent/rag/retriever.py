"""
Release-aware hybrid retriever.

    query → metadata filter (scope→release_id / project_id)
          → Qdrant hybrid (dense bge-m3 + sparse BM25, RRF fusion)
          → bge-reranker-v2-m3 → top_k
          → Documents(page_content, metadata={document_id, chunk_id, source, score, scope})

Scoping is enforced by the Qdrant payload filter so a release-scoped query never
recalls another release's documents. Every hop degrades gracefully: if Qdrant or
SiliconFlow is unreachable the retriever logs and returns ``[]`` (matching the old
stub), so the agent keeps running; if only rerank fails, results fall back to the
RRF-fused order.
"""
import logging

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.agent.rag.config import rag_settings
from app.agent.rag.embed import embedder, reranker, sparse
from app.agent.rag.storage import qdrant_store

logger = logging.getLogger(__name__)


def _point_to_document(point) -> Document:
    payload = point.payload or {}
    return Document(
        page_content=payload.get("content", ""),
        metadata={
            "document_id": payload.get("document_id", ""),
            "chunk_id": payload.get("chunk_id", ""),
            "source": payload.get("source", ""),
            "scope": payload.get("scope", ""),
            "seq": payload.get("seq", 0),
            "score": float(point.score),
        },
    )


async def _hybrid_search(
    query: str, *, project_id: int | None, release_id: int | None, scope: str, limit: int
) -> list[Document]:
    dense = await embedder.embed_one(query)
    sp = await sparse.embed_query(query)
    query_filter = qdrant_store.build_filter(
        project_id=project_id, release_id=release_id, scope=scope
    )
    points = await qdrant_store.hybrid_query(
        dense=dense, sparse=sp, query_filter=query_filter, limit=limit
    )
    return [_point_to_document(p) for p in points]


async def _rerank(query: str, docs: list[Document], top_k: int) -> list[Document]:
    if not docs:
        return []
    try:
        ranked = await reranker.rerank(query, [d.page_content for d in docs], top_n=top_k)
    except Exception:
        logger.warning("Rerank unavailable — falling back to fusion order")
        return docs[:top_k]
    out: list[Document] = []
    for idx, score in ranked:
        doc = docs[idx]
        doc.metadata["score"] = float(score)
        out.append(doc)
    return out


class ReleaseAwareRetriever(BaseRetriever):
    """A ``BaseRetriever`` bound to a single (release_id, project_id, scope) filter."""

    release_id: int | None = None
    project_id: int | None = None
    scope: str = "release"
    top_k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        raise NotImplementedError("Use the async interface (ainvoke).")

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        try:
            candidates = await _hybrid_search(
                query,
                project_id=self.project_id,
                release_id=self.release_id,
                scope=self.scope,
                limit=rag_settings.rag_rerank_top_n,
            )
        except Exception:
            logger.exception("Hybrid retrieval failed (scope=%s) — returning []", self.scope)
            return []
        return await _rerank(query, candidates, self.top_k)


async def retrieve(
    query: str,
    *,
    release_id: int | None = None,
    project_id: int | None = None,
    top_k: int | None = None,
) -> list[Document]:
    """
    Union release-scoped + project-scoped hybrid hits, rerank the merged set,
    return top_k. Used by the graph ``rag`` node and ``project_memory.search``.
    """
    top_k = top_k or rag_settings.rag_top_k
    candidates: list[Document] = []
    try:
        if release_id is not None:
            candidates += await _hybrid_search(
                query, project_id=project_id, release_id=release_id,
                scope="release", limit=rag_settings.rag_rerank_top_n,
            )
        if project_id is not None:
            candidates += await _hybrid_search(
                query, project_id=project_id, release_id=None,
                scope="project", limit=rag_settings.rag_rerank_top_n,
            )
    except Exception:
        logger.exception("Hybrid retrieval failed — returning []")
        return []
    return await _rerank(query, candidates, top_k)

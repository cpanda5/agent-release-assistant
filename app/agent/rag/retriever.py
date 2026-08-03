"""
Release-aware RAG retriever — **interface placeholder**.

The concrete vector store (Qdrant) + embedding + ingestion pipeline is deferred
to a dedicated later design. This module defines the *contract* the rest of the
system codes against so nothing else has to change when the real implementation
lands.

Contract / future design
-------------------------
Documents (colleague-uploaded Markdown, PRDs, design docs) are ingested with
metadata::

    {"scope": "release" | "project", "release_id": int, "project_id": int, "source": str}

Retrieval is **metadata-filtered** so:

  * ``scope="release"`` recalls only documents tagged with the *current*
    ``release_id`` — no cross-release contamination.
  * ``scope="project"`` recalls project-wide shared knowledge by ``project_id``.

Planned implementation (later):
  * Embedding: DashScope ``text-embedding-v*`` (same API key).
  * Vector store: Qdrant, one collection with a payload index on
    ``release_id`` / ``project_id`` / ``scope`` for cheap filtering.
  * Ingestion: MD → split (recursive/markdown splitter) → embed → upsert with the
    metadata above.

Until then every method returns empty results, and the ``rag`` graph node is a
no-op that simply passes the (empty) chunk list through.
"""
import logging

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


class ReleaseAwareRetriever(BaseRetriever):
    """
    A ``BaseRetriever`` bound to a single (release_id, project_id, scope) filter.

    Stub: returns ``[]``. Wire up Qdrant here later — apply the metadata filter
    described in the module docstring.
    """

    release_id: int | None = None
    project_id: int | None = None
    scope: str = "release"  # "release" | "project"
    top_k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return []

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        # TODO(rag): embed `query`, query Qdrant with the metadata filter for this
        # release/project scope, return the top_k matches as Documents.
        return []


async def retrieve(
    query: str,
    *,
    release_id: int | None = None,
    project_id: int | None = None,
    top_k: int = 5,
) -> list[Document]:
    """
    Convenience helper used by the graph ``rag`` node.

    Recalls both release-scoped and project-scoped documents for the query,
    metadata-filtered. Currently returns ``[]`` (see module docstring).
    """
    docs: list[Document] = []
    if release_id is not None:
        docs += await ReleaseAwareRetriever(
            release_id=release_id, project_id=project_id, scope="release", top_k=top_k
        ).ainvoke(query)
    if project_id is not None:
        docs += await ReleaseAwareRetriever(
            project_id=project_id, scope="project", top_k=top_k
        ).ainvoke(query)
    return docs

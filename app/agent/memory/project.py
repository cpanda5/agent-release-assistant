"""
Layer 3: Project Memory — delegates to the RAG retriever.

The actual vector search lives in ``app/agent/rag`` (Qdrant-backed hybrid
retrieval). AgentContext carries the results in its ``rag_chunks`` field.
"""
from app.agent.memory.schemas import KnowledgeChunk


class ProjectMemory:
    async def search(self, query: str, project_id: str) -> list[KnowledgeChunk]:
        raise NotImplementedError

    async def ingest(self, content: str, metadata: dict) -> str:
        raise NotImplementedError


class _RagProjectMemory(ProjectMemory):
    """Bridges the legacy KnowledgeChunk API onto the LangChain retriever."""

    async def search(self, query: str, project_id: str) -> list[KnowledgeChunk]:
        from app.agent.rag import retrieve

        pid = _maybe_int(project_id)
        docs = await retrieve(query, project_id=pid)
        return [
            KnowledgeChunk(
                content=d.page_content,
                source=str(d.metadata.get("source", "")),
                score=float(d.metadata.get("score", 0.0)),
            )
            for d in docs
        ]

    async def ingest(self, content: str, metadata: dict) -> str:
        from app.agent.rag.ingest.pipeline import ingest_document

        project_id = _maybe_int(metadata.get("project_id"))
        if project_id is None:
            return ""
        doc = await ingest_document(
            file_bytes=content.encode("utf-8"),
            filename=metadata.get("source", "inline.md"),
            project_id=project_id,
            release_id=_maybe_int(metadata.get("release_id")),
            scope=metadata.get("scope", "project"),
            mime_type="text/markdown",
        )
        return doc.uuid


def _maybe_int(value) -> int | None:
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


project_memory = _RagProjectMemory()

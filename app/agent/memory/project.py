"""
Layer 3: Project Memory — delegates to the RAG retriever.

The actual vector search lives in ``app/agent/rag`` (currently a placeholder).
AgentContext already has a ``rag_chunks`` field so callers don't change when the
real Qdrant-backed retriever is implemented.
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
        # TODO(rag): chunk + embed + upsert into Qdrant with release/project metadata.
        return ""


def _maybe_int(value) -> int | None:
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


project_memory = _RagProjectMemory()

"""
Layer 3: Project Memory — interface stub.

Module 4 implements this with vector search (RAG).
AgentContext already has a rag_chunks field so the caller doesn't change.
"""
from app.agent.memory.schemas import KnowledgeChunk


class ProjectMemory:
    async def search(self, query: str, project_id: str) -> list[KnowledgeChunk]:
        raise NotImplementedError

    async def ingest(self, content: str, metadata: dict) -> str:
        raise NotImplementedError


class _NoopProjectMemory(ProjectMemory):
    async def search(self, query: str, project_id: str) -> list[KnowledgeChunk]:
        return []

    async def ingest(self, content: str, metadata: dict) -> str:
        return ""


project_memory = _NoopProjectMemory()

# Phase 2: RAG retrieval service
# Will connect to a vector DB (e.g. Chroma / Weaviate / pgvector)
# and retrieve knowledge filtered by project_id / release_id / scope.


async def retrieve_knowledge(
    query: str,
    project_id: str | None = None,
    release_id: str | None = None,
) -> str:
    """Return relevant context for the query. Empty string in Phase 1."""
    return ""

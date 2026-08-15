from app.agent.rag.retriever import ReleaseAwareRetriever, retrieve

__all__ = ["ReleaseAwareRetriever", "retrieve"]


async def init_stores() -> None:
    """Best-effort store setup at startup: ensure the Qdrant collection + MinIO bucket."""
    import logging

    logger = logging.getLogger(__name__)
    try:
        from app.agent.rag.storage import qdrant_store

        await qdrant_store.ensure_collection()
    except Exception:
        logger.warning("Qdrant init skipped (unreachable) — retrieval will degrade to []")
    try:
        from app.agent.rag.storage import minio_store

        await minio_store.ensure_bucket()
    except Exception:
        logger.warning("MinIO init skipped (unreachable) — originals won't be stored")

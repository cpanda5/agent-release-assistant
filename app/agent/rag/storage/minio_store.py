"""
MinIO object storage for original uploaded files — kept so every chunk stays
traceable back to the exact bytes it came from. The MinIO SDK is synchronous, so
each call is offloaded to a thread to stay non-blocking.
"""
import asyncio
import io
import logging

from minio import Minio

from app.agent.rag.config import rag_settings

logger = logging.getLogger(__name__)

_client: Minio | None = None


def get_client() -> Minio:
    global _client
    if _client is None:
        _client = Minio(
            rag_settings.minio_endpoint,
            access_key=rag_settings.minio_access_key,
            secret_key=rag_settings.minio_secret_key,
            secure=rag_settings.minio_secure,
        )
    return _client


async def ensure_bucket() -> None:
    def _ensure() -> None:
        client = get_client()
        if not client.bucket_exists(rag_settings.minio_bucket):
            client.make_bucket(rag_settings.minio_bucket)
            logger.info("MinIO bucket '%s' created", rag_settings.minio_bucket)

    await asyncio.to_thread(_ensure)


async def put(object_key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    def _put() -> str:
        get_client().put_object(
            rag_settings.minio_bucket,
            object_key,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return object_key

    return await asyncio.to_thread(_put)


async def get(object_key: str) -> bytes:
    def _get() -> bytes:
        resp = get_client().get_object(rag_settings.minio_bucket, object_key)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    return await asyncio.to_thread(_get)


async def delete(object_key: str) -> None:
    await asyncio.to_thread(
        get_client().remove_object, rag_settings.minio_bucket, object_key
    )

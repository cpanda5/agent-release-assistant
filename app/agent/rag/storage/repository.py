"""
aiomysql CRUD for knowledge_base / document / chunk.

Follows the project's ``id BIGINT PK + uuid VARCHAR(36)`` convention: rows carry
both, the app works in uuids over the API and resolves to the BIGINT id at write
time (mirrors ``identity.resolve_session_pk``). Unlike tracing, ingestion *needs*
a database — every function raises ``NoDatabaseError`` when the pool is absent so
the router can answer 503 instead of silently succeeding.
"""
import json
import logging
import uuid as _uuid

import aiomysql

from app.agent.rag.models import Chunk, Document
from app.db import get_db_pool

logger = logging.getLogger(__name__)


class NoDatabaseError(RuntimeError):
    """Raised when a RAG DB operation runs without a MySQL pool."""


# project_id → knowledge_base.id
_kb_cache: dict[int, int] = {}


def _pool():
    pool = get_db_pool()
    if pool is None:
        raise NoDatabaseError("MySQL is not configured; knowledge base requires a database.")
    return pool


async def ensure_knowledge_base(project_id: int, *, name: str | None = None) -> int:
    """Return the knowledge_base.id for a project, creating the row on first use."""
    cached = _kb_cache.get(project_id)
    if cached is not None:
        return cached

    pool = _pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT id FROM knowledge_base WHERE project_id = %s AND deleted_at IS NULL",
                (project_id,),
            )
            row = await cur.fetchone()
            if row:
                _kb_cache[project_id] = row[0]
                return row[0]

            await cur.execute(
                """
                INSERT INTO knowledge_base (uuid, project_id, name)
                VALUES (%s, %s, %s)
                """,
                (str(_uuid.uuid4()), project_id, name or f"project-{project_id}-kb"),
            )
            await conn.commit()
            kb_id = cur.lastrowid
    _kb_cache[project_id] = kb_id
    return kb_id


async def find_document_by_hash(knowledge_base_id: int, file_hash: str) -> dict | None:
    pool = _pool()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                """
                SELECT uuid, filename, parse_status, chunk_count
                FROM document
                WHERE knowledge_base_id = %s AND file_hash = %s AND deleted_at IS NULL
                LIMIT 1
                """,
                (knowledge_base_id, file_hash),
            )
            return await cur.fetchone()


async def insert_document(doc: Document) -> Document:
    pool = _pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO document
                    (uuid, knowledge_base_id, project_id, release_id, scope, filename,
                     mime_type, doc_type, minio_bucket, minio_object_key, file_size,
                     file_hash, parse_status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    doc.uuid, doc.knowledge_base_id, doc.project_id, doc.release_id,
                    doc.scope, doc.filename, doc.mime_type, doc.doc_type,
                    doc.minio_bucket, doc.minio_object_key, doc.file_size,
                    doc.file_hash, doc.parse_status,
                ),
            )
            await conn.commit()
            doc.id = cur.lastrowid
    return doc


async def update_document_status(
    document_id: int,
    *,
    parse_status: str,
    chunk_count: int = 0,
    char_count: int = 0,
    parse_error: str | None = None,
) -> None:
    pool = _pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE document
                SET parse_status = %s, chunk_count = %s, char_count = %s, parse_error = %s
                WHERE id = %s
                """,
                (parse_status, chunk_count, char_count, parse_error, document_id),
            )
            await conn.commit()


async def insert_chunks(chunks: list[Chunk]) -> None:
    if not chunks:
        return
    pool = _pool()
    rows = [
        (
            c.uuid, c.document_id, c.knowledge_base_id, c.project_id, c.release_id,
            c.scope, c.seq, c.content, c.token_count, c.qdrant_point_id,
            json.dumps(c.metadata, ensure_ascii=False),
        )
        for c in chunks
    ]
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.executemany(
                """
                INSERT INTO chunk
                    (uuid, document_id, knowledge_base_id, project_id, release_id,
                     scope, seq, content, token_count, qdrant_point_id, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )
            await conn.commit()


async def get_document_by_uuid(document_uuid: str) -> dict | None:
    pool = _pool()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT * FROM document WHERE uuid = %s AND deleted_at IS NULL",
                (document_uuid,),
            )
            return await cur.fetchone()


async def list_documents(project_id: int) -> list[dict]:
    pool = _pool()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                """
                SELECT uuid, filename, doc_type, scope, release_id, parse_status,
                       chunk_count, char_count, file_size, created_at
                FROM document
                WHERE project_id = %s AND deleted_at IS NULL
                ORDER BY created_at DESC
                """,
                (project_id,),
            )
            return list(await cur.fetchall())


async def list_chunks(document_uuid: str) -> list[dict]:
    pool = _pool()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                """
                SELECT c.uuid, c.seq, c.content, c.token_count, c.qdrant_point_id, c.metadata
                FROM chunk c
                JOIN document d ON d.id = c.document_id
                WHERE d.uuid = %s
                ORDER BY c.seq
                """,
                (document_uuid,),
            )
            return list(await cur.fetchall())


async def delete_document(document_uuid: str) -> dict | None:
    """Soft-delete the document, hard-delete its chunks. Returns the document row."""
    doc = await get_document_by_uuid(document_uuid)
    if doc is None:
        return None
    pool = _pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM chunk WHERE document_id = %s", (doc["id"],))
            await cur.execute(
                "UPDATE document SET deleted_at = CURRENT_TIMESTAMP(3) WHERE id = %s",
                (doc["id"],),
            )
            await conn.commit()
    return doc

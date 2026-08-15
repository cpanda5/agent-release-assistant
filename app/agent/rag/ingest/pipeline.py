"""
End-to-end ingestion:

    minio (original) → parse → table-to-text → split → embed (dense + sparse)
                     → qdrant upsert → persist document + chunk rows

Structured as one async function so a later redis/worker can call it unchanged.
Storing the original in MinIO is best-effort (a warning, not a failure); the parse
→ embed → persist core is transactional in spirit: any failure flips the document
to ``parse_status='failed'`` and re-raises so the caller sees the real error.
"""
import hashlib
import logging
import uuid as _uuid

from app.agent.rag.config import rag_settings
from app.agent.rag.embed import embedder, sparse
from app.agent.rag.ingest import parsers, splitter
from app.agent.rag.models import Chunk, Document
from app.agent.rag.storage import minio_store, qdrant_store, repository

logger = logging.getLogger(__name__)


async def ingest_document(
    *,
    file_bytes: bytes,
    filename: str,
    project_id: int,
    release_id: int | None = None,
    scope: str = "project",
    mime_type: str | None = None,
) -> Document:
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    kb_id = await repository.ensure_knowledge_base(project_id)

    existing = await repository.find_document_by_hash(kb_id, file_hash)
    if existing and existing["parse_status"] == "parsed":
        logger.info("Duplicate upload '%s' (hash match) — returning existing document", filename)
        return Document(
            uuid=existing["uuid"], knowledge_base_id=kb_id, project_id=project_id,
            filename=existing["filename"], scope=scope, release_id=release_id,
            file_hash=file_hash, parse_status="parsed", chunk_count=existing["chunk_count"],
        )

    doc_type = parsers.doc_type_for(filename)
    object_key = f"{project_id}/{file_hash}/{filename}"
    try:
        await minio_store.ensure_bucket()
        await minio_store.put(object_key, file_bytes, mime_type or "application/octet-stream")
        stored_key = object_key
    except Exception:
        logger.warning("MinIO store failed for '%s' — continuing without original", filename)
        stored_key = None

    doc = Document(
        uuid=str(_uuid.uuid4()), knowledge_base_id=kb_id, project_id=project_id,
        release_id=release_id, scope=scope, filename=filename, mime_type=mime_type,
        doc_type=doc_type, minio_bucket=rag_settings.minio_bucket if stored_key else None,
        minio_object_key=stored_key, file_size=len(file_bytes), file_hash=file_hash,
        parse_status="parsing",
    )
    await repository.insert_document(doc)

    try:
        elements = parsers.parse(file_bytes, filename)
        full_text = parsers.elements_to_text(elements)
        texts = splitter.split(full_text)
        if not texts:
            raise parsers.ParserError("No extractable text in document")

        dense_vectors = await embedder.embed(texts)
        sparse_vectors = await sparse.embed_documents(texts)

        chunks, points = _build(doc, texts, dense_vectors, sparse_vectors)
        await qdrant_store.ensure_collection()
        await qdrant_store.upsert_points(points)
        await repository.insert_chunks(chunks)

        await repository.update_document_status(
            doc.id, parse_status="parsed", chunk_count=len(chunks), char_count=len(full_text)
        )
        doc.parse_status = "parsed"
        doc.chunk_count = len(chunks)
        doc.char_count = len(full_text)
        logger.info("Ingested '%s' → %d chunks", filename, len(chunks))
        return doc
    except Exception as exc:
        await repository.update_document_status(
            doc.id, parse_status="failed", parse_error=str(exc)[:1000]
        )
        doc.parse_status = "failed"
        doc.parse_error = str(exc)
        logger.exception("Ingestion failed for '%s'", filename)
        raise


def _build(doc, texts, dense_vectors, sparse_vectors):
    chunks: list[Chunk] = []
    points: list[dict] = []
    for seq, (text, dense, sp) in enumerate(zip(texts, dense_vectors, sparse_vectors)):
        point_id = str(_uuid.uuid4())
        chunks.append(
            Chunk(
                uuid=str(_uuid.uuid4()), document_id=doc.id, knowledge_base_id=doc.knowledge_base_id,
                project_id=doc.project_id, release_id=doc.release_id, scope=doc.scope,
                seq=seq, content=text, qdrant_point_id=point_id,
                metadata={"filename": doc.filename, "doc_type": doc.doc_type},
            )
        )
        points.append(
            {
                "id": point_id,
                "dense": dense,
                "sparse": sp,
                "payload": {
                    "chunk_id": chunks[-1].uuid,
                    "document_id": doc.uuid,
                    "knowledge_base_id": doc.knowledge_base_id,
                    "project_id": doc.project_id,
                    "release_id": doc.release_id,
                    "scope": doc.scope,
                    "source": doc.filename,
                    "seq": seq,
                    "content": text,
                },
            }
        )
    return chunks, points

"""FastAPI endpoints for the knowledge base: upload, list, inspect, delete, debug-search.

Public API accepts project / release **UUIDs** (resolved to internal BIGINT PKs
here at the boundary; storage and Qdrant metadata stay numeric). Legacy callers
passing numeric id strings keep working via the passthrough in the resolvers.
"""
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.agent.identity import resolve_project_pk, resolve_release_pk
from app.agent.rag import retriever
from app.agent.rag.ingest import parsers
from app.agent.rag.ingest.pipeline import ingest_document
from app.agent.rag.storage import minio_store, qdrant_store, repository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


async def _project_pk_or_404(ref: str) -> int:
    pk = await resolve_project_pk(ref)
    if pk is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return pk


async def _release_pk_or_404(ref: str) -> int:
    pk = await resolve_release_pk(ref)
    if pk is None:
        raise HTTPException(status_code=404, detail="Release not found")
    return pk


class SearchRequest(BaseModel):
    query: str
    project_id: str
    release_id: str | None = None
    top_k: int = 5


@router.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    project_id: str = Form(...),
    release_id: str | None = Form(None),
    scope: str = Form("project"),
):
    project_pk = await _project_pk_or_404(project_id)
    release_pk = await _release_pk_or_404(release_id) if release_id else None
    file_bytes = await file.read()
    try:
        doc = await ingest_document(
            file_bytes=file_bytes,
            filename=file.filename or "upload.bin",
            project_id=project_pk,
            release_id=release_pk,
            scope=scope,
            mime_type=file.content_type,
        )
    except repository.NoDatabaseError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except parsers.ParserError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return {
        "document_uuid": doc.uuid,
        "filename": doc.filename,
        "doc_type": doc.doc_type,
        "chunk_count": doc.chunk_count,
        "parse_status": doc.parse_status,
    }


@router.get("/documents")
async def list_documents(project_id: str):
    project_pk = await _project_pk_or_404(project_id)
    try:
        return {"documents": await repository.list_documents(project_pk)}
    except repository.NoDatabaseError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/documents/{document_uuid}/chunks")
async def list_chunks(document_uuid: str):
    try:
        doc = await repository.get_document_by_uuid(document_uuid)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"document_uuid": document_uuid, "chunks": await repository.list_chunks(document_uuid)}
    except repository.NoDatabaseError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.delete("/documents/{document_uuid}")
async def delete_document(document_uuid: str):
    try:
        doc = await repository.delete_document(document_uuid)
    except repository.NoDatabaseError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        await qdrant_store.delete_by_document(document_uuid)
    except Exception:
        logger.warning("Failed to delete Qdrant points for %s", document_uuid)
    if doc.get("minio_object_key"):
        try:
            await minio_store.delete(doc["minio_object_key"])
        except Exception:
            logger.warning("Failed to delete MinIO object for %s", document_uuid)

    return {"document_uuid": document_uuid, "deleted": True}


@router.post("/search")
async def debug_search(req: SearchRequest):
    project_pk = await _project_pk_or_404(req.project_id)
    release_pk = await _release_pk_or_404(req.release_id) if req.release_id else None
    docs = await retriever.retrieve(
        req.query, release_id=release_pk, project_id=project_pk, top_k=req.top_k
    )
    return {
        "query": req.query,
        "results": [
            {"content": d.page_content, **d.metadata} for d in docs
        ],
    }

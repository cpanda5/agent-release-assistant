"""FastAPI endpoints for the knowledge base: upload, list, inspect, delete, debug-search."""
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.agent.rag import retriever
from app.agent.rag.ingest import parsers
from app.agent.rag.ingest.pipeline import ingest_document
from app.agent.rag.storage import minio_store, qdrant_store, repository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


class SearchRequest(BaseModel):
    query: str
    project_id: int
    release_id: int | None = None
    top_k: int = 5


@router.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    project_id: int = Form(...),
    release_id: int | None = Form(None),
    scope: str = Form("project"),
):
    file_bytes = await file.read()
    try:
        doc = await ingest_document(
            file_bytes=file_bytes,
            filename=file.filename or "upload.bin",
            project_id=project_id,
            release_id=release_id,
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
async def list_documents(project_id: int):
    try:
        return {"documents": await repository.list_documents(project_id)}
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
    docs = await retriever.retrieve(
        req.query, release_id=req.release_id, project_id=req.project_id, top_k=req.top_k
    )
    return {
        "query": req.query,
        "results": [
            {"content": d.page_content, **d.metadata} for d in docs
        ],
    }

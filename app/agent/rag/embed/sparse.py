"""
Sparse (BM25) vectors via FastEmbed's ``Qdrant/bm25`` model. The model emits raw
term weights; Qdrant applies IDF at query time (collection ``Modifier.IDF``), so
documents and queries stay symmetric. FastEmbed is synchronous and loads its
vocabulary lazily on first use, so the model is built once and calls run in a
thread.
"""
import asyncio
import logging

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        from fastembed import SparseTextEmbedding

        _model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _model


def _to_dict(embedding) -> dict:
    return {"indices": embedding.indices.tolist(), "values": embedding.values.tolist()}


async def embed_documents(texts: list[str]) -> list[dict]:
    def _run() -> list[dict]:
        return [_to_dict(e) for e in _get_model().embed(texts)]

    return await asyncio.to_thread(_run)


async def embed_query(text: str) -> dict:
    def _run() -> dict:
        return _to_dict(next(iter(_get_model().query_embed(text))))

    return await asyncio.to_thread(_run)

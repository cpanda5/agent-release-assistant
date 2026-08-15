"""
Qdrant hybrid store: one collection holding a dense vector (bge-m3) and a sparse
vector (BM25) per chunk. Retrieval prefetches both and fuses with RRF via the
Query API, so lexical and semantic matches are combined in a single round-trip.

The client connects lazily; callers are expected to catch failures and degrade
(retrieval returns ``[]``, ingestion surfaces the error).
"""
import logging

from qdrant_client import AsyncQdrantClient, models

from app.agent.rag.config import rag_settings

logger = logging.getLogger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"

_client: AsyncQdrantClient | None = None


def get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(
            url=rag_settings.qdrant_url,
            api_key=rag_settings.qdrant_api_key or None,
        )
    return _client


async def ensure_collection() -> None:
    """Create the collection (dense + sparse) and payload indexes if missing."""
    client = get_client()
    if await client.collection_exists(rag_settings.qdrant_collection):
        return

    await client.create_collection(
        collection_name=rag_settings.qdrant_collection,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(
                size=rag_settings.embedding_dim, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR: models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        },
    )
    for field in ("project_id", "release_id", "scope", "document_id"):
        await client.create_payload_index(
            collection_name=rag_settings.qdrant_collection,
            field_name=field,
            field_schema=(
                models.PayloadSchemaType.KEYWORD
                if field in ("scope", "document_id")
                else models.PayloadSchemaType.INTEGER
            ),
        )
    logger.info("Qdrant collection '%s' created", rag_settings.qdrant_collection)


def build_filter(
    *, project_id: int | None, release_id: int | None, scope: str
) -> models.Filter:
    """Scope filter: release scope pins release_id, project scope matches the project."""
    must: list = [models.FieldCondition(key="scope", match=models.MatchValue(value=scope))]
    if scope == "release" and release_id is not None:
        must.append(
            models.FieldCondition(key="release_id", match=models.MatchValue(value=release_id))
        )
    elif project_id is not None:
        must.append(
            models.FieldCondition(key="project_id", match=models.MatchValue(value=project_id))
        )
    return models.Filter(must=must)


async def upsert_points(points: list[dict]) -> None:
    """Upsert chunks. Each point: {id, dense: list[float], sparse: {indices, values}, payload}."""
    client = get_client()
    qdrant_points = [
        models.PointStruct(
            id=p["id"],
            vector={
                DENSE_VECTOR: p["dense"],
                SPARSE_VECTOR: models.SparseVector(
                    indices=p["sparse"]["indices"], values=p["sparse"]["values"]
                ),
            },
            payload=p["payload"],
        )
        for p in points
    ]
    await client.upsert(collection_name=rag_settings.qdrant_collection, points=qdrant_points)


async def hybrid_query(
    *,
    dense: list[float],
    sparse: dict,
    query_filter: models.Filter,
    limit: int,
) -> list[models.ScoredPoint]:
    """Prefetch dense + sparse candidates, fuse with RRF, return top `limit`."""
    client = get_client()
    result = await client.query_points(
        collection_name=rag_settings.qdrant_collection,
        prefetch=[
            models.Prefetch(query=dense, using=DENSE_VECTOR, filter=query_filter, limit=limit),
            models.Prefetch(
                query=models.SparseVector(indices=sparse["indices"], values=sparse["values"]),
                using=SPARSE_VECTOR,
                filter=query_filter,
                limit=limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return result.points


async def delete_by_document(document_uuid: str) -> None:
    client = get_client()
    await client.delete(
        collection_name=rag_settings.qdrant_collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_uuid)
                    )
                ]
            )
        ),
    )

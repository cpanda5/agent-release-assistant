"""
RAG subsystem settings.

Kept in a dedicated ``RagSettings`` (instead of bloating ``app.config.Settings``)
so the knowledge-base subsystem stays self-contained: it reads the *same* ``.env``
file but owns its own namespace of keys (SiliconFlow / Qdrant / MinIO / chunking).

Nothing here performs I/O or validates connectivity — construction is cheap and
side-effect free, so importing the RAG package never touches the network. The
stores connect lazily on first use and degrade gracefully when unreachable.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class RagSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # SiliconFlow — OpenAI-compatible embeddings + rerank
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    embedding_dim: int = 1024

    # Qdrant — dense + sparse hybrid store
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "agent_knowledge"

    # MinIO — original-file object storage (endpoint is host:port, no scheme)
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False
    minio_bucket: str = "knowledge-docs"

    # Chunking / retrieval
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 120
    rag_top_k: int = 5                # final chunks returned to the agent
    rag_rerank_top_n: int = 20        # candidates pulled from Qdrant before rerank
    rag_embed_batch_size: int = 16


rag_settings = RagSettings()

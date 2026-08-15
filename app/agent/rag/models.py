"""Plain dataclasses shared across the RAG subsystem."""
from dataclasses import dataclass, field


@dataclass
class KnowledgeBase:
    id: int
    uuid: str
    project_id: int
    name: str
    embedding_model: str = "BAAI/bge-m3"


@dataclass
class Document:
    uuid: str
    knowledge_base_id: int
    project_id: int
    filename: str
    scope: str = "project"                 # project | release
    release_id: int | None = None
    mime_type: str | None = None
    doc_type: str = "text"                 # text | image | audio
    minio_bucket: str | None = None
    minio_object_key: str | None = None
    file_size: int = 0
    file_hash: str | None = None
    parse_status: str = "pending"          # pending | parsing | parsed | failed
    parse_error: str | None = None
    chunk_count: int = 0
    char_count: int = 0
    id: int | None = None


@dataclass
class Chunk:
    uuid: str
    document_id: int
    knowledge_base_id: int
    project_id: int
    scope: str
    seq: int
    content: str
    qdrant_point_id: str
    release_id: int | None = None
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A hybrid-search hit, traceable back to its document and chunk."""
    content: str
    score: float
    document_id: str                       # document uuid
    chunk_id: str                          # chunk uuid
    source: str                            # filename
    scope: str
    seq: int = 0

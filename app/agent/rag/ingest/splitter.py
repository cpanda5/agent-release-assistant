"""Text splitting via LangChain's RecursiveCharacterTextSplitter."""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.agent.rag.config import rag_settings


def split(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=rag_settings.rag_chunk_size,
        chunk_overlap=rag_settings.rag_chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
    )
    return [c for c in splitter.split_text(text) if c.strip()]

"""
Parser dispatch. Each file type maps to a parser that lazy-imports its heavy ML
dependency, so a missing dep only breaks *that* file type — core install and the
other parsers keep working. Every parser returns a list of ``Element``; the
pipeline flattens them to text, routing ``Table`` elements through
``table.to_text``.
"""
from dataclasses import dataclass
from pathlib import Path

TEXT_EXTS = {".md", ".txt", ".pdf", ".doc", ".docx", ".xlsx", ".xls", ".csv", ".html", ".htm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}


class ParserError(Exception):
    """Unsupported type, or a parser's dependency is not installed."""


@dataclass
class Element:
    text: str
    category: str = "Text"          # Text | Title | Table | ...
    table_html: str | None = None


def doc_type_for(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    return "text"


def parse(file_bytes: bytes, filename: str) -> list[Element]:
    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXTS:
        from app.agent.rag.ingest.parsers import image

        return image.parse(file_bytes, filename)
    if ext in AUDIO_EXTS:
        from app.agent.rag.ingest.parsers import audio

        return audio.parse(file_bytes, filename)
    if ext in TEXT_EXTS or ext == "":
        from app.agent.rag.ingest.parsers import text

        return text.parse(file_bytes, filename)
    raise ParserError(f"Unsupported file type: {ext or filename}")


def elements_to_text(elements: list[Element]) -> str:
    from app.agent.rag.ingest.parsers import table

    parts: list[str] = []
    for el in elements:
        if el.category == "Table":
            parts.append(table.to_text(el))
        elif el.text and el.text.strip():
            parts.append(el.text.strip())
    return "\n\n".join(parts)

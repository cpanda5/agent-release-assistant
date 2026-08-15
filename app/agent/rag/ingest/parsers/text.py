"""Text/document parsing via Unstructured (md/txt/pdf/doc/docx/xlsx/csv/html)."""
import io

from app.agent.rag.ingest.parsers import Element, ParserError


def parse(file_bytes: bytes, filename: str) -> list[Element]:
    try:
        from unstructured.partition.auto import partition
    except ImportError as exc:
        raise ParserError(
            "Text parsing needs the 'rag' extra (unstructured). Install: uv sync --extra rag"
        ) from exc

    els = partition(file=io.BytesIO(file_bytes), metadata_filename=filename)
    out: list[Element] = []
    for el in els:
        category = getattr(el, "category", type(el).__name__)
        table_html = getattr(getattr(el, "metadata", None), "text_as_html", None)
        out.append(
            Element(text=str(el), category="Table" if category == "Table" else category, table_html=table_html)
        )
    return out

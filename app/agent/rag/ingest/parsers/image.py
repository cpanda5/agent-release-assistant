"""
Image parsing via PaddleOCR PP-Structure: OCR for plain text plus table-structure
recovery (regions returned as HTML, handed to table.to_text downstream).
"""
import tempfile

from app.agent.rag.ingest.parsers import Element, ParserError


def parse(file_bytes: bytes, filename: str) -> list[Element]:
    try:
        from paddleocr import PPStructure
    except ImportError as exc:
        raise ParserError(
            "Image parsing needs the 'rag' extra (paddleocr/paddlepaddle). "
            "Install: uv sync --extra rag"
        ) from exc

    suffix = "." + filename.rsplit(".", 1)[-1] if "." in filename else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    engine = PPStructure(table=True, ocr=True, show_log=False)
    regions = engine(tmp_path)

    out: list[Element] = []
    for region in regions:
        rtype = region.get("type")
        res = region.get("res")
        if rtype == "table" and isinstance(res, dict) and res.get("html"):
            out.append(Element(text="", category="Table", table_html=res["html"]))
        elif isinstance(res, list):
            lines = [line.get("text", "") for line in res if isinstance(line, dict)]
            text = "\n".join(t for t in lines if t)
            if text:
                out.append(Element(text=text, category="Text"))
    return out

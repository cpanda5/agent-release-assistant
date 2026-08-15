"""Audio transcription via FunASR (paraformer-zh)."""
import tempfile

from app.agent.rag.ingest.parsers import Element, ParserError

_model = None


def _get_model():
    global _model
    if _model is None:
        from funasr import AutoModel

        _model = AutoModel(model="paraformer-zh", disable_update=True)
    return _model


def parse(file_bytes: bytes, filename: str) -> list[Element]:
    try:
        model = _get_model()
    except ImportError as exc:
        raise ParserError(
            "Audio parsing needs the 'rag' extra (funasr). Install: uv sync --extra rag"
        ) from exc

    suffix = "." + filename.rsplit(".", 1)[-1] if "." in filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    result = model.generate(input=tmp_path)
    text = " ".join(seg.get("text", "") for seg in result) if result else ""
    return [Element(text=text, category="Text")] if text.strip() else []

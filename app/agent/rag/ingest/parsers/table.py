"""
Rule-based Table-to-text: flatten an HTML table into pipe-delimited rows so the
splitter and embedder see readable text instead of markup. Falls back to the raw
element text when no HTML is available.
"""
from html.parser import HTMLParser

from app.agent.rag.ingest.parsers import Element


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def to_text(element: Element) -> str:
    if not element.table_html:
        return element.text.strip()
    parser = _TableParser()
    parser.feed(element.table_html)
    lines = [" | ".join(row) for row in parser.rows if any(c for c in row)]
    return "\n".join(lines) if lines else element.text.strip()

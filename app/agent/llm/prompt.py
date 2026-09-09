from functools import lru_cache
from pathlib import Path

import yaml

# templates/ is at the project root, four levels up from this file
TEMPLATES_DIR = Path(__file__).parent.parent.parent.parent / "templates"


class _SafeDict(dict):
    """format_map helper: leaves unknown placeholders intact rather than raising."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class PromptTemplate:
    def __init__(self, data: dict):
        self.name: str = data["name"]
        self.version: str = data.get("version", "v1")
        self.model: str | None = data.get("model")
        self.max_tokens: int | None = data.get("max_tokens")
        self.temperature: float | None = data.get("temperature")
        self._system: str = data.get("system", "")
        self._user_template: str = data.get("user_template", "{user_input}")

    def render(self, **kwargs) -> list[dict]:
        """
        Render template variables into a messages list.
        Unknown placeholders are left as-is so missing context doesn't crash.
        """
        safe = _SafeDict(**kwargs)
        messages: list[dict] = []
        if self._system:
            messages.append({"role": "system", "content": self._system.format_map(safe)})
        messages.append({"role": "user", "content": self._user_template.format_map(safe)})
        return messages

    def render_system(self, **kwargs) -> str:
        """Render just the system portion. Unknown placeholders left intact."""
        return self._system.format_map(_SafeDict(**kwargs)) if self._system else ""

    def render_user(self, **kwargs) -> str:
        """Render just the user portion. Unknown placeholders left intact."""
        return self._user_template.format_map(_SafeDict(**kwargs))


@lru_cache(maxsize=64)
def load_template(name: str, version: str = "v1") -> PromptTemplate:
    path = TEMPLATES_DIR / name / f"{version}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return PromptTemplate(data)

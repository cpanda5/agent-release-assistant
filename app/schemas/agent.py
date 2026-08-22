from pydantic import BaseModel, field_validator

from app.config import ALLOWED_MODELS


def _validate_model(value: str | None) -> str | None:
    """Accept only the three supported models; None means 'use server default'."""
    if value is None:
        return None
    if value not in ALLOWED_MODELS:
        raise ValueError(
            f"Unsupported model '{value}'. Choose one of: {sorted(ALLOWED_MODELS)}"
        )
    return value


# ── Shared ─────────────────────────────────────────────────────────────────── #

class UsageInfo(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


# ── /agent/chat (sync + stream) ────────────────────────────────────────────── #

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    release_id: str | None = None
    project_id: str | None = None
    user_id: str | None = None
    model: str | None = None
    enable_thinking: bool | None = None
    # optional template override
    prompt_template: str | None = None
    prompt_template_version: str = "v1"

    _check_model = field_validator("model")(_validate_model)


class ChatResponse(BaseModel):
    response: str
    session_id: str
    trace_id: str
    usage: UsageInfo


# ── /agent/session/close ───────────────────────────────────────────────────── #

class CloseSessionRequest(BaseModel):
    session_id: str
    release_id: int | None = None

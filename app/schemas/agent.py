from pydantic import BaseModel


# ── Legacy /agent/run ──────────────────────────────────────────────────────── #

class AgentRunRequest(BaseModel):
    message: str
    release_id: str | None = None
    project_id: str | None = None
    session_id: str | None = None


class UsageInfo(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class AgentRunResponse(BaseModel):
    response: str
    session_id: str
    usage: UsageInfo


# ── /agent/chat (sync + stream) ────────────────────────────────────────────── #

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    release_id: str | None = None
    project_id: str | None = None
    # optional template override
    prompt_template: str | None = None
    prompt_template_version: str = "v1"


class ChatResponse(BaseModel):
    response: str
    session_id: str
    trace_id: str
    usage: UsageInfo


# ── /agent/session/close ───────────────────────────────────────────────────── #

class CloseSessionRequest(BaseModel):
    session_id: str
    release_id: int | None = None

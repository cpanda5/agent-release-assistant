from pydantic import BaseModel


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

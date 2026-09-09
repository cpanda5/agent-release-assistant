from datetime import datetime

from pydantic import BaseModel


class SessionOut(BaseModel):
    uuid: str
    release_uuid: str
    title: str | None = None
    status: str
    message_count: int
    total_tokens: int
    created_at: datetime | None = None
    updated_at: datetime | None = None


class MessageOut(BaseModel):
    uuid: str
    role: str
    content: str
    token_count: int | None = None
    created_at: datetime | None = None

from datetime import datetime

from pydantic import BaseModel, Field


# ── Requests ─────────────────────────────────────────────────────────────────── #

class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")
    password: str = Field(min_length=6, max_length=128)
    display_name: str | None = Field(default=None, max_length=64)
    email: str | None = Field(default=None, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


# ── Responses ────────────────────────────────────────────────────────────────── #

class UserOut(BaseModel):
    uuid: str
    username: str
    display_name: str | None = None
    email: str | None = None
    created_at: datetime | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: float  # unix timestamp
    user: UserOut

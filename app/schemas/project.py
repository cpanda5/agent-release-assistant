from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ReleaseStatus = Literal["draft", "in_progress", "review", "done", "archived"]


# ── Project ──────────────────────────────────────────────────────────────────── #

class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str | None = None
    context_summary: str | None = None
    tech_constraints: Any | None = None


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = None
    context_summary: str | None = None
    tech_constraints: Any | None = None
    status: int | None = Field(default=None, ge=0, le=1)


class ProjectOut(BaseModel):
    uuid: str
    name: str
    description: str | None = None
    context_summary: str | None = None
    tech_constraints: Any | None = None
    status: int
    created_at: datetime | None = None
    updated_at: datetime | None = None


# ── Release ──────────────────────────────────────────────────────────────────── #

class ReleaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str | None = None
    status: ReleaseStatus = "draft"


class ReleaseUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = None
    status: ReleaseStatus | None = None


class ReleaseOut(BaseModel):
    uuid: str
    project_uuid: str
    name: str
    description: str | None = None
    status: str
    created_at: datetime | None = None
    updated_at: datetime | None = None

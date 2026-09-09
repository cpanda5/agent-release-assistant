"""Project / Release CRUD — every operation is scoped to the current user's projects."""
import json
import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth.deps import get_current_user
from app.db import get_db_pool
from app.schemas.project import (
    ProjectCreate,
    ProjectOut,
    ProjectUpdate,
    ReleaseCreate,
    ReleaseOut,
    ReleaseUpdate,
)
from app.schemas.session import MessageOut, SessionOut

logger = logging.getLogger(__name__)

router = APIRouter(tags=["project"])

_PROJECT_COLS = "uuid, name, description, context_summary, tech_constraints, status, created_at, updated_at"
_RELEASE_COLS = "r.uuid, p.uuid, r.name, r.description, r.status, r.created_at, r.updated_at"


def _require_pool():
    pool = get_db_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return pool


def _project_out(row) -> ProjectOut:
    tc = row[4]
    if isinstance(tc, str):
        tc = json.loads(tc)
    return ProjectOut(
        uuid=row[0], name=row[1], description=row[2], context_summary=row[3],
        tech_constraints=tc, status=row[5], created_at=row[6], updated_at=row[7],
    )


def _release_out(row) -> ReleaseOut:
    return ReleaseOut(
        uuid=row[0], project_uuid=row[1], name=row[2], description=row[3],
        status=row[4], created_at=row[5], updated_at=row[6],
    )


async def _user_pk(cur, user_uuid: str) -> int:
    """Internal BIGINT id of the current user (token subject must exist)."""
    await cur.execute(
        "SELECT id FROM `user` WHERE uuid = %s AND deleted_at IS NULL", (user_uuid,)
    )
    row = await cur.fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return row[0]


async def _owned_project_row(cur, project_uuid: str, owner_pk: int, row_cols: str = _PROJECT_COLS):
    await cur.execute(
        f"SELECT {row_cols} FROM project WHERE uuid = %s AND owner_id = %s AND deleted_at IS NULL",
        (project_uuid, owner_pk),
    )
    row = await cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return row


# ── Projects ─────────────────────────────────────────────────────────────────── #

@router.post("/projects", response_model=ProjectOut)
async def create_project(request: ProjectCreate, user: dict = Depends(get_current_user)) -> ProjectOut:
    pool = _require_pool()
    project_uuid = str(uuid4())
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            owner_pk = await _user_pk(cur, user["user_uuid"])
            await cur.execute(
                f"INSERT INTO project (uuid, name, description, context_summary, tech_constraints, owner_id) "
                f"VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    project_uuid,
                    request.name,
                    request.description,
                    request.context_summary,
                    json.dumps(request.tech_constraints, ensure_ascii=False) if request.tech_constraints is not None else None,
                    owner_pk,
                ),
            )
            await conn.commit()
            row = await _owned_project_row(cur, project_uuid, owner_pk)
    return _project_out(row)


@router.get("/projects", response_model=list[ProjectOut])
async def list_projects(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user: dict = Depends(get_current_user),
) -> list[ProjectOut]:
    pool = _require_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            owner_pk = await _user_pk(cur, user["user_uuid"])
            await cur.execute(
                f"SELECT {_PROJECT_COLS} FROM project "
                f"WHERE owner_id = %s AND deleted_at IS NULL "
                f"ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (owner_pk, limit, offset),
            )
            rows = await cur.fetchall()
    return [_project_out(r) for r in rows]


@router.get("/projects/{project_uuid}", response_model=ProjectOut)
async def get_project(project_uuid: str, user: dict = Depends(get_current_user)) -> ProjectOut:
    pool = _require_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            owner_pk = await _user_pk(cur, user["user_uuid"])
            row = await _owned_project_row(cur, project_uuid, owner_pk)
    return _project_out(row)


@router.put("/projects/{project_uuid}", response_model=ProjectOut)
async def update_project(
    project_uuid: str, request: ProjectUpdate, user: dict = Depends(get_current_user)
) -> ProjectOut:
    fields = request.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    if "tech_constraints" in fields and fields["tech_constraints"] is not None:
        fields["tech_constraints"] = json.dumps(fields["tech_constraints"], ensure_ascii=False)

    set_clause = ", ".join(f"{k} = %s" for k in fields)
    pool = _require_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            owner_pk = await _user_pk(cur, user["user_uuid"])
            await _owned_project_row(cur, project_uuid, owner_pk)  # 404 guard
            await cur.execute(
                f"UPDATE project SET {set_clause} WHERE uuid = %s",
                (*fields.values(), project_uuid),
            )
            await conn.commit()
            row = await _owned_project_row(cur, project_uuid, owner_pk)
    return _project_out(row)


@router.delete("/projects/{project_uuid}")
async def delete_project(project_uuid: str, user: dict = Depends(get_current_user)) -> dict:
    pool = _require_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            owner_pk = await _user_pk(cur, user["user_uuid"])
            await _owned_project_row(cur, project_uuid, owner_pk)  # 404 guard
            # soft delete: project + its releases
            await cur.execute(
                "UPDATE project SET deleted_at = NOW(3) WHERE uuid = %s", (project_uuid,)
            )
            await cur.execute(
                "UPDATE releases r JOIN project p ON r.project_id = p.id "
                "SET r.deleted_at = NOW(3) WHERE p.uuid = %s AND r.deleted_at IS NULL",
                (project_uuid,),
            )
            await conn.commit()
    return {"status": "deleted", "uuid": project_uuid}


# ── Releases ─────────────────────────────────────────────────────────────────── #

async def _owned_release_row(cur, release_uuid: str, owner_pk: int, row_cols: str = _RELEASE_COLS):
    await cur.execute(
        f"SELECT {row_cols} FROM releases r "
        f"JOIN project p ON r.project_id = p.id "
        f"WHERE r.uuid = %s AND p.owner_id = %s "
        f"AND r.deleted_at IS NULL AND p.deleted_at IS NULL",
        (release_uuid, owner_pk),
    )
    row = await cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Release not found")
    return row


@router.post("/projects/{project_uuid}/releases", response_model=ReleaseOut)
async def create_release(
    project_uuid: str, request: ReleaseCreate, user: dict = Depends(get_current_user)
) -> ReleaseOut:
    pool = _require_pool()
    release_uuid = str(uuid4())
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            owner_pk = await _user_pk(cur, user["user_uuid"])
            proj = await _owned_project_row(cur, project_uuid, owner_pk, row_cols="id")
            await cur.execute(
                "INSERT INTO releases (uuid, project_id, name, description, status) "
                "VALUES (%s, %s, %s, %s, %s)",
                (release_uuid, proj[0], request.name, request.description, request.status),
            )
            await conn.commit()
            row = await _owned_release_row(cur, release_uuid, owner_pk)
    return _release_out(row)


@router.get("/projects/{project_uuid}/releases", response_model=list[ReleaseOut])
async def list_releases(
    project_uuid: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user: dict = Depends(get_current_user),
) -> list[ReleaseOut]:
    pool = _require_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            owner_pk = await _user_pk(cur, user["user_uuid"])
            await _owned_project_row(cur, project_uuid, owner_pk, row_cols="uuid")  # 404 guard
            await cur.execute(
                f"SELECT {_RELEASE_COLS} FROM releases r "
                f"JOIN project p ON r.project_id = p.id "
                f"WHERE p.uuid = %s AND r.deleted_at IS NULL "
                f"ORDER BY r.created_at DESC LIMIT %s OFFSET %s",
                (project_uuid, limit, offset),
            )
            rows = await cur.fetchall()
    return [_release_out(r) for r in rows]


@router.get("/releases/{release_uuid}", response_model=ReleaseOut)
async def get_release(release_uuid: str, user: dict = Depends(get_current_user)) -> ReleaseOut:
    pool = _require_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            owner_pk = await _user_pk(cur, user["user_uuid"])
            row = await _owned_release_row(cur, release_uuid, owner_pk)
    return _release_out(row)


@router.put("/releases/{release_uuid}", response_model=ReleaseOut)
async def update_release(
    release_uuid: str, request: ReleaseUpdate, user: dict = Depends(get_current_user)
) -> ReleaseOut:
    fields = request.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")

    set_clause = ", ".join(f"{k} = %s" for k in fields)
    pool = _require_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            owner_pk = await _user_pk(cur, user["user_uuid"])
            await _owned_release_row(cur, release_uuid, owner_pk)  # 404 guard
            await cur.execute(
                f"UPDATE releases SET {set_clause} WHERE uuid = %s",
                (*fields.values(), release_uuid),
            )
            await conn.commit()
            row = await _owned_release_row(cur, release_uuid, owner_pk)
    return _release_out(row)


@router.delete("/releases/{release_uuid}")
async def delete_release(release_uuid: str, user: dict = Depends(get_current_user)) -> dict:
    pool = _require_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            owner_pk = await _user_pk(cur, user["user_uuid"])
            await _owned_release_row(cur, release_uuid, owner_pk)  # 404 guard
            await cur.execute(
                "UPDATE releases SET deleted_at = NOW(3) WHERE uuid = %s", (release_uuid,)
            )
            await conn.commit()
    return {"status": "deleted", "uuid": release_uuid}


# ── Sessions & history (release-scoped) ────────────────────────────────────── #

@router.get("/releases/{release_uuid}/sessions", response_model=list[SessionOut])
async def list_release_sessions(
    release_uuid: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user: dict = Depends(get_current_user),
) -> list[SessionOut]:
    """List chat sessions belonging to a release (newest first)."""
    pool = _require_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            owner_pk = await _user_pk(cur, user["user_uuid"])
            rel = await _owned_release_row(cur, release_uuid, owner_pk, row_cols="r.id")
            await cur.execute(
                "SELECT s.uuid, %s, s.title, s.status, "
                "  (SELECT COUNT(*) FROM message m "
                "     WHERE m.session_id = s.id AND m.content_type = 'text' "
                "       AND m.role IN ('user','assistant')), "
                "  s.total_tokens, s.created_at, s.updated_at "
                "FROM sessions s "
                "WHERE s.release_id = %s AND s.deleted_at IS NULL "
                "ORDER BY s.created_at DESC LIMIT %s OFFSET %s",
                (release_uuid, rel[0], limit, offset),
            )
            rows = await cur.fetchall()
    return [
        SessionOut(
            uuid=r[0], release_uuid=r[1], title=r[2], status=r[3],
            message_count=r[4], total_tokens=r[5], created_at=r[6], updated_at=r[7],
        )
        for r in rows
    ]


@router.get("/sessions/{session_uuid}/messages", response_model=list[MessageOut])
async def get_session_messages(
    session_uuid: str,
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    user: dict = Depends(get_current_user),
) -> list[MessageOut]:
    """
    Full text history of one session (oldest → newest). Accessible when the
    session's release is owned by the caller, or the session itself was created
    by them (``sessions.user_id``).
    """
    pool = _require_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            owner_pk = await _user_pk(cur, user["user_uuid"])
            await cur.execute(
                "SELECT s.id FROM sessions s "
                "LEFT JOIN releases r ON s.release_id = r.id AND r.deleted_at IS NULL "
                "LEFT JOIN project p ON r.project_id = p.id AND p.deleted_at IS NULL "
                "WHERE s.uuid = %s AND s.deleted_at IS NULL "
                "  AND (p.owner_id = %s OR s.user_id = %s)",
                (session_uuid, owner_pk, user["user_uuid"]),
            )
            session = await cur.fetchone()
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")
            await cur.execute(
                "SELECT uuid, role, content, token_count, created_at FROM message "
                "WHERE session_id = %s AND content_type = 'text' "
                "  AND role IN ('user','assistant') "
                "ORDER BY id ASC LIMIT %s OFFSET %s",
                (session[0], limit, offset),
            )
            rows = await cur.fetchall()
    return [
        MessageOut(uuid=r[0], role=r[1], content=r[2], token_count=r[3], created_at=r[4])
        for r in rows
    ]

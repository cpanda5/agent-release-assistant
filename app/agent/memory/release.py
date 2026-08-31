"""
Release memory layer — MySQL access for sessions / message / session_summary /
release_context, plus prompt-block rendering.

Summary watermark: a session_summary row covers all messages created before its
`created_at`. Writers store an explicit cutoff (DB NOW(3) captured before
reading the messages); replay loads messages created after the latest cutoff.
"""
import json
import logging
import uuid as _uuid
from datetime import datetime

from app.agent.identity import cache_session_pk, resolve_session_pk
from app.agent.memory.schemas import Message
from app.db import get_db_pool

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# sessions                                                             #
# ------------------------------------------------------------------ #

async def ensure_session(
    *,
    session_id: str,
    release_id: int | None,
    user_id: str | None,
) -> int | None:
    """Idempotently register a session by UUID; returns its BIGINT id."""
    pool = get_db_pool()
    if pool is None:
        return None
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO sessions (uuid, release_id, user_id)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        id         = LAST_INSERT_ID(id),
                        release_id = COALESCE(VALUES(release_id), release_id),
                        user_id    = COALESCE(VALUES(user_id), user_id)
                    """,
                    (session_id, release_id, user_id),
                )
                await conn.commit()
                pk = cur.lastrowid
                if pk:
                    cache_session_pk(session_id, pk)
                return pk
    except Exception:
        logger.exception("Failed to ensure session=%s", session_id)
        return None


async def db_now() -> datetime | None:
    """Current DB time (DATETIME(3)) — used as the summary cutoff watermark."""
    pool = get_db_pool()
    if pool is None:
        return None
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT NOW(3)")
                row = await cur.fetchone()
                return row[0] if row else None
    except Exception:
        logger.exception("Failed to read DB time")
        return None


# ------------------------------------------------------------------ #
# message (Session History)                                            #
# ------------------------------------------------------------------ #

async def insert_message(
    *,
    session_id: str,
    role: str,
    content: str,
    token_count: int = 0,
    trace_id: str | None = None,
) -> int | None:
    """Insert one text message; returns message.id (None when MySQL is off)."""
    pool = get_db_pool()
    if pool is None:
        return None
    session_pk = await resolve_session_pk(session_id)
    if session_pk is None:
        return None
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO message
                        (uuid, session_id, role, content, content_type, llm_trace_id, token_count)
                    VALUES (%s, %s, %s, %s, 'text', %s, %s)
                    """,
                    (str(_uuid.uuid4()), session_pk, role, content, trace_id, token_count),
                )
                await conn.commit()
                return cur.lastrowid
    except Exception:
        logger.exception("Failed to persist message for session=%s", session_id)
        return None


async def load_history(
    session_id: str,
    *,
    after: datetime | None = None,
    limit: int = 50,
) -> list[Message]:
    """Replayable history (user/assistant text), oldest → newest, after cutoff."""
    pool = get_db_pool()
    if pool is None:
        return []
    session_pk = await resolve_session_pk(session_id)
    if session_pk is None:
        return []
    try:
        sql = (
            "SELECT role, content, token_count FROM message "
            "WHERE session_id = %s AND content_type = 'text' "
            "AND role IN ('user', 'assistant') "
            "AND content IS NOT NULL AND content <> '' "
        )
        params: list = [session_pk]
        if after is not None:
            sql += "AND created_at > %s "
        sql += "ORDER BY id DESC LIMIT %s"
        params.append(limit)
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()
        rows.reverse()
        return [
            Message(role=r[0], content=r[1], token_count=r[2] or 0)
            for r in rows
        ]
    except Exception:
        logger.exception("Failed to load history for session=%s", session_id)
        return []


# ------------------------------------------------------------------ #
# session_summary                                                      #
# ------------------------------------------------------------------ #

async def load_latest_summary(session_id: str) -> dict | None:
    """Latest summary row: {summary, key_decisions, version, cutoff}."""
    pool = get_db_pool()
    if pool is None:
        return None
    session_pk = await resolve_session_pk(session_id)
    if session_pk is None:
        return None
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT summary, key_decisions, version, created_at "
                    "FROM session_summary WHERE session_id = %s "
                    "ORDER BY version DESC LIMIT 1",
                    (session_pk,),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                return {
                    "summary": row[0] or "",
                    "key_decisions": _json_load(row[1]),
                    "version": row[2],
                    "cutoff": row[3],
                }
    except Exception:
        logger.exception("Failed to load latest summary for session=%s", session_id)
        return None


async def write_session_summary(
    *,
    session_id: str,
    release_id: int,
    summary: str,
    key_decisions: list[str] | None = None,
    token_count: int = 0,
    version: int = 1,
    cutoff: datetime | None = None,
) -> None:
    """Append-only insert; `cutoff` becomes created_at (the replay watermark)."""
    pool = get_db_pool()
    if pool is None:
        return
    session_pk = await resolve_session_pk(session_id)
    if session_pk is None:
        return
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO session_summary
                        (uuid, session_id, release_id, summary, key_decisions,
                         token_count, version, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, COALESCE(%s, NOW(3)))
                    """,
                    (
                        str(_uuid.uuid4()), session_pk, release_id, summary,
                        json.dumps(key_decisions or [], ensure_ascii=False),
                        token_count, version, cutoff,
                    ),
                )
                await conn.commit()
    except Exception:
        logger.exception("Failed to write session_summary for session=%s", session_id)


def build_summary_prompt_block(summary: dict | None) -> str:
    if not summary or not summary.get("summary"):
        return ""
    lines = ["--- 本 session 早期对话摘要 ---", summary["summary"]]
    decisions = summary.get("key_decisions") or []
    if decisions:
        lines.append("\n关键决策：")
        lines += [f"  • {d}" for d in decisions]
    lines.append("--- End ---")
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# release_context                                                      #
# ------------------------------------------------------------------ #

async def load_release_context(release_id: int) -> dict | None:
    """Return the current release_context row, or None if not yet created."""
    pool = get_db_pool()
    if pool is None:
        return None
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT progress_summary, pending_tasks, tech_constraints "
                    "FROM release_context WHERE release_id = %s",
                    (release_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                return {
                    "progress_summary": row[0] or "",
                    "pending_tasks": _json_load(row[1]),
                    "tech_constraints": _json_load(row[2]),
                }
    except Exception:
        logger.exception("Failed to load release_context for release_id=%s", release_id)
        return None


async def upsert_release_context(
    *,
    release_id: int,
    progress_summary: str,
    pending_tasks: list | None = None,
    tech_constraints: list | None = None,
    updated_by: str | None = None,
) -> None:
    pool = get_db_pool()
    if pool is None:
        return
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO release_context
                        (release_id, progress_summary, pending_tasks, tech_constraints, updated_by)
                    VALUES (%s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        progress_summary = VALUES(progress_summary),
                        pending_tasks    = VALUES(pending_tasks),
                        tech_constraints = VALUES(tech_constraints),
                        updated_by       = VALUES(updated_by),
                        version          = version + 1
                    """,
                    (
                        release_id,
                        progress_summary,
                        json.dumps(pending_tasks or [], ensure_ascii=False),
                        json.dumps(tech_constraints or [], ensure_ascii=False),
                        updated_by,
                    ),
                )
                await conn.commit()
    except Exception:
        logger.exception("Failed to upsert release_context for release_id=%s", release_id)


def build_release_prompt_block(ctx: dict | None) -> str:
    """Render a release_context row as the system-prompt injection block."""
    if not ctx or not (
        ctx.get("progress_summary") or ctx.get("pending_tasks") or ctx.get("tech_constraints")
    ):
        return ""

    lines = ["--- Release 上下文 ---"]
    if ctx.get("progress_summary"):
        lines.append(ctx["progress_summary"])
    pending = ctx.get("pending_tasks") or []
    if pending:
        lines.append("\n待完成：")
        lines += [f"  • {t}" for t in pending]
    constraints = ctx.get("tech_constraints") or []
    if constraints:
        lines.append("\n技术约束：")
        lines += [f"  • {c}" for c in constraints]
    lines.append("--- End ---")
    return "\n".join(lines)


def _json_load(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []

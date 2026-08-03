"""
Layer 2: Release Memory

Reads/writes session_summary and release_context tables.
Provides the release-progress snapshot injected into each system prompt.
"""
import json
import logging
import uuid as _uuid

from app.agent.identity import cache_session_pk, resolve_session_pk
from app.db import get_db_pool

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# sessions table                                                       #
# ------------------------------------------------------------------ #

async def ensure_session(
    *,
    session_id: str,
    release_id: int | None,
    user_id: str | None,
) -> int | None:
    """
    Idempotently register a session by its UUID and return its BIGINT id.

    Uses the ``LAST_INSERT_ID(id)`` upsert trick so ``cur.lastrowid`` yields the
    id on both insert and duplicate. Caches uuid → id for later writers.
    """
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


# ------------------------------------------------------------------ #
# messages table                                                       #
# ------------------------------------------------------------------ #

async def persist_message(
    *,
    session_id: str,
    role: str,
    content: str,
    token_count: int,
    trace_id: str | None,
    content_type: str = "text",
    tool_call_id: str | None = None,
) -> None:
    """Insert one row into `message`, resolving the session UUID to its BIGINT id."""
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
                    INSERT INTO message
                        (uuid, session_id, role, content, content_type,
                         tool_call_id, llm_trace_id, token_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(_uuid.uuid4()), session_pk,
                        role, content, content_type,
                        tool_call_id, trace_id, token_count,
                    ),
                )
                await conn.commit()
    except Exception:
        logger.exception("Failed to persist message for session=%s", session_id)


# ------------------------------------------------------------------ #
# session_summary table                                                #
# ------------------------------------------------------------------ #

async def write_session_summary(
    *,
    session_id: str,
    release_id: int,
    summary: str,
    key_decisions: list[str] | None = None,
    artifacts_generated: list[dict] | None = None,
    token_count: int = 0,
) -> None:
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
                        (uuid, session_id, release_id, summary, key_decisions, artifacts_generated, token_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        summary              = VALUES(summary),
                        key_decisions        = VALUES(key_decisions),
                        artifacts_generated  = VALUES(artifacts_generated),
                        token_count          = VALUES(token_count)
                    """,
                    (
                        str(_uuid.uuid4()), session_pk, release_id,
                        summary,
                        json.dumps(key_decisions or [], ensure_ascii=False),
                        json.dumps(artifacts_generated or [], ensure_ascii=False),
                        token_count,
                    ),
                )
                await conn.commit()
    except Exception:
        logger.exception("Failed to write session_summary for session=%s", session_id)


# ------------------------------------------------------------------ #
# release_context table                                                #
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
                    "SELECT progress_summary, key_decisions, pending_tasks, tech_constraints "
                    "FROM release_context WHERE release_id = %s",
                    (release_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                return {
                    "progress_summary": row[0] or "",
                    "key_decisions": _json_load(row[1]),
                    "pending_tasks": _json_load(row[2]),
                    "tech_constraints": _json_load(row[3]),
                }
    except Exception:
        logger.exception("Failed to load release_context for release_id=%s", release_id)
        return None


async def upsert_release_context(
    *,
    release_id: int,
    progress_summary: str,
    key_decisions: list | None = None,
    pending_tasks: list | None = None,
    tech_constraints: list | None = None,
    last_updated_by: str | None = None,
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
                        (release_id, progress_summary, key_decisions, pending_tasks, tech_constraints, last_updated_by)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        progress_summary  = VALUES(progress_summary),
                        key_decisions     = VALUES(key_decisions),
                        pending_tasks     = VALUES(pending_tasks),
                        tech_constraints  = VALUES(tech_constraints),
                        last_updated_by   = VALUES(last_updated_by),
                        version           = version + 1,
                        updated_at        = CURRENT_TIMESTAMP(3)
                    """,
                    (
                        release_id,
                        progress_summary,
                        json.dumps(key_decisions or [], ensure_ascii=False),
                        json.dumps(pending_tasks or [], ensure_ascii=False),
                        json.dumps(tech_constraints or [], ensure_ascii=False),
                        last_updated_by,
                    ),
                )
                await conn.commit()
    except Exception:
        logger.exception("Failed to upsert release_context for release_id=%s", release_id)


def build_release_prompt_block(ctx: dict | None) -> str:
    """Render a release_context row as the system-prompt injection block."""
    if not ctx or not ctx.get("progress_summary"):
        return ""

    lines = ["--- Release 上下文 ---", ctx["progress_summary"]]

    decisions = ctx.get("key_decisions") or []
    if decisions:
        lines.append("\n关键决策：")
        lines += [f"  • {d}" for d in decisions]

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
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return []

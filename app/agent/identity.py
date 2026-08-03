"""
Session identity resolution: external UUID ↔ internal BIGINT id.

The app generates a session UUID client-side (LangGraph style) and passes that
string around everywhere. The database, however, uses the ``id + uuid`` pattern:
``sessions.id`` (BIGINT PK) is the foreign key that child tables (message,
llm_trace, tool_call_trace, session_summary) reference.

This module is the single place that translates a session UUID to its BIGINT
``sessions.id``. It caches the mapping per process (populated when the session is
ensured at the start of each turn), and falls back to a lookup by UUID.

It imports only ``app.db`` (the lowest layer), so every writer — memory/release,
llm/tracer, tools/tracer — can use it without creating an import cycle.
"""
import logging

from app.db import get_db_pool

logger = logging.getLogger(__name__)

# session UUID → sessions.id (BIGINT)
_pk_cache: dict[str, int] = {}


def cache_session_pk(session_uuid: str, pk: int) -> None:
    _pk_cache[session_uuid] = pk


async def resolve_session_pk(session_uuid: str | None) -> int | None:
    """Return sessions.id for a session UUID, or None if unknown / no DB."""
    if not session_uuid:
        return None
    cached = _pk_cache.get(session_uuid)
    if cached is not None:
        return cached

    pool = get_db_pool()
    if pool is None:
        return None
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id FROM sessions WHERE uuid = %s", (session_uuid,)
                )
                row = await cur.fetchone()
                if row:
                    _pk_cache[session_uuid] = row[0]
                    return row[0]
    except Exception:
        logger.exception("Failed to resolve session pk for uuid=%s", session_uuid)
    return None

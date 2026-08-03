import json
import logging

from app.db import get_db_pool

logger = logging.getLogger(__name__)

_OUTPUT_MAX_CHARS = 2000


async def write_tool_trace(
    *,
    trace_id: str,
    session_id: str,
    release_id: int | None,
    tool_name: str,
    tool_call_id: str,
    input_params: dict,
    output_content: str | None,
    status: str,
    error_type: str | None = None,
    error_message: str | None = None,
    retry_count: int = 0,
    latency_ms: int = 0,
    iteration: int = 1,
) -> None:
    """Write one row to tool_call_trace. Failures are logged but never re-raised."""
    pool = get_db_pool()
    if pool is None:
        logger.debug("DB pool unavailable, skipping tool trace %s", tool_call_id)
        return

    from app.agent.identity import resolve_session_pk

    session_pk = await resolve_session_pk(session_id)

    output_truncated = None
    if output_content is not None:
        output_truncated = output_content[:_OUTPUT_MAX_CHARS]

    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO tool_call_trace (
                        trace_id, session_id, release_id,
                        tool_name, tool_call_id,
                        input_params, output_content,
                        status, error_type, error_message,
                        retry_count, latency_ms, iteration
                    ) VALUES (
                        %s, %s, %s,
                        %s, %s,
                        %s, %s,
                        %s, %s, %s,
                        %s, %s, %s
                    )
                    ON DUPLICATE KEY UPDATE
                        status        = VALUES(status),
                        error_type    = VALUES(error_type),
                        error_message = VALUES(error_message),
                        retry_count   = VALUES(retry_count),
                        latency_ms    = VALUES(latency_ms)
                    """,
                    (
                        trace_id, session_pk, release_id,
                        tool_name, tool_call_id,
                        json.dumps(input_params, ensure_ascii=False), output_truncated,
                        status, error_type, error_message[:512] if error_message else None,
                        retry_count, latency_ms, iteration,
                    ),
                )
                await conn.commit()
    except Exception:
        logger.exception("Failed to write tool_call_trace for tool_call_id=%s", tool_call_id)

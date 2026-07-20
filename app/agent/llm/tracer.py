import json
import logging
import time
import uuid
from dataclasses import dataclass, field

from app.db import get_db_pool

logger = logging.getLogger(__name__)


@dataclass
class TraceContext:
    """Immutable request-side context captured before the LLM call."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: int | None = None
    release_id: int | None = None
    project_id: int | None = None
    requested_model: str = ""
    prompt_template: str | None = None
    prompt_template_version: str | None = None
    system_prompt_hash: str | None = None
    is_streaming: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    _start: float = field(default_factory=time.monotonic, repr=False)

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._start) * 1000)


@dataclass
class TraceResult:
    """Mutable result side, populated after the LLM call completes."""
    model: str = ""
    fallback_model: str | None = None
    fallback_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    ttfb_ms: int | None = None
    retry_count: int = 0
    retry_reasons: list = field(default_factory=list)
    finish_reason: str | None = None
    status: str = "success"
    error_type: str | None = None
    error_message: str | None = None


async def write_trace(ctx: TraceContext, result: TraceResult) -> None:
    """Write one row to llm_trace. Failures are logged but never re-raised."""
    pool = get_db_pool()
    if pool is None:
        logger.debug("DB pool unavailable, skipping trace %s", ctx.trace_id)
        return
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO llm_trace (
                        trace_id, session_id, release_id, project_id,
                        model, requested_model, fallback_model, fallback_reason,
                        prompt_template, prompt_template_version, system_prompt_hash,
                        prompt_tokens, completion_tokens, total_tokens, cost_usd,
                        latency_ms, ttfb_ms,
                        retry_count, retry_reasons,
                        finish_reason, status, error_type, error_message,
                        is_streaming, temperature, max_tokens
                    ) VALUES (
                        %s,%s,%s,%s,
                        %s,%s,%s,%s,
                        %s,%s,%s,
                        %s,%s,%s,%s,
                        %s,%s,
                        %s,%s,
                        %s,%s,%s,%s,
                        %s,%s,%s
                    )
                    """,
                    (
                        ctx.trace_id, ctx.session_id, ctx.release_id, ctx.project_id,
                        result.model, ctx.requested_model, result.fallback_model, result.fallback_reason,
                        ctx.prompt_template, ctx.prompt_template_version, ctx.system_prompt_hash,
                        result.prompt_tokens, result.completion_tokens, result.total_tokens, result.cost_usd,
                        result.latency_ms, result.ttfb_ms,
                        result.retry_count, json.dumps(result.retry_reasons),
                        result.finish_reason, result.status, result.error_type, result.error_message,
                        int(ctx.is_streaming), ctx.temperature, ctx.max_tokens,
                    ),
                )
                await conn.commit()
    except Exception:
        logger.exception("Failed to write llm_trace %s", ctx.trace_id)

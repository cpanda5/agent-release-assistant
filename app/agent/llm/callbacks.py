"""
LangChain callback handler that persists LLM call traces to MySQL ``llm_trace``.

Adapts the LangChain callback lifecycle to the ``tracer.write_trace`` sink. One
row per underlying model call (retries and model fallbacks each get a row).

Per-request context (session / release / project) is threaded in via the run's
``metadata`` (set by the gateway); LangChain also injects ``ls_*`` metadata such
as ``ls_model_name``.

Tool traces are **not** handled here — they are written by ``tools/executor.py``
which owns retry/confirmation semantics, so routing them through callbacks too
would double-write ``tool_call_trace``.
"""
import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from app.agent.llm.tracer import TraceContext, TraceResult, write_trace

logger = logging.getLogger(__name__)


class MySQLTraceCallbackHandler(AsyncCallbackHandler):
    def __init__(self) -> None:
        # run_id → captured request-side context
        self._runs: dict[UUID, dict[str, Any]] = {}

    async def on_chat_model_start(
        self,
        serialized: dict,
        messages: list,
        *,
        run_id: UUID,
        metadata: dict | None = None,
        **kwargs: Any,
    ) -> None:
        md = metadata or {}
        self._runs[run_id] = {
            "start": time.monotonic(),
            "model": md.get("ls_model_name") or _model_from_serialized(serialized),
            "session_id": md.get("session_id"),
            "release_id": _maybe_int(md.get("release_id")),
            "project_id": _maybe_int(md.get("project_id")),
            "temperature": md.get("ls_temperature"),
            "max_tokens": md.get("ls_max_tokens"),
            "is_streaming": bool(md.get("is_streaming", False)),
        }

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        info = self._runs.pop(run_id, None)
        if info is None:
            return
        usage, finish_reason = _extract_usage(response)
        ctx, result = _make_trace(info)
        result.prompt_tokens = usage.get("input_tokens", 0)
        result.completion_tokens = usage.get("output_tokens", 0)
        result.total_tokens = result.prompt_tokens + result.completion_tokens
        result.finish_reason = finish_reason or "stop"
        result.status = "success"
        result.latency_ms = int((time.monotonic() - info["start"]) * 1000)
        await write_trace(ctx, result)

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        info = self._runs.pop(run_id, None)
        if info is None:
            return
        ctx, result = _make_trace(info)
        result.status = "failed"
        result.error_type = type(error).__name__
        result.error_message = str(error)[:512]
        result.latency_ms = int((time.monotonic() - info["start"]) * 1000)
        await write_trace(ctx, result)


# ── Helpers ───────────────────────────────────────────────────────────── #

def _make_trace(info: dict) -> tuple[TraceContext, TraceResult]:
    ctx = TraceContext(
        requested_model=info["model"] or "",
        session_id=info["session_id"],
        release_id=info["release_id"],
        project_id=info["project_id"],
        is_streaming=info["is_streaming"],
        temperature=info["temperature"],
        max_tokens=info["max_tokens"],
    )
    return ctx, TraceResult(model=info["model"] or "")


def _extract_usage(response: LLMResult) -> tuple[dict, str | None]:
    finish_reason: str | None = None
    usage: dict = {}
    try:
        gen = response.generations[0][0]
        finish_reason = (gen.generation_info or {}).get("finish_reason")
        msg = getattr(gen, "message", None)
        if msg is not None and getattr(msg, "usage_metadata", None):
            um = msg.usage_metadata
            usage = {
                "input_tokens": um.get("input_tokens", 0),
                "output_tokens": um.get("output_tokens", 0),
            }
    except (IndexError, AttributeError):
        pass
    if not usage and response.llm_output:
        tu = response.llm_output.get("token_usage") or {}
        usage = {
            "input_tokens": tu.get("prompt_tokens", 0),
            "output_tokens": tu.get("completion_tokens", 0),
        }
    return usage, finish_reason


def _model_from_serialized(serialized: dict | None) -> str:
    if not serialized:
        return ""
    kwargs = serialized.get("kwargs") or {}
    return kwargs.get("model") or kwargs.get("model_name") or ""


def _maybe_int(value: Any) -> int | None:
    """llm_trace stores integer ids; UUID session strings map to None."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


# Module-level singleton — safe to share; per-run state is keyed by run_id.
trace_callback = MySQLTraceCallbackHandler()

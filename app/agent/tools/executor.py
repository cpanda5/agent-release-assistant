import asyncio
import json
import logging
import time

from app.agent.tools.base import BaseTool, ToolContext, ToolResult
from app.agent.tools.errors import (
    ToolErrorCategory,
    ToolErrorType,
    classify_tool_error,
)
from app.agent.tools.registry import tool_registry
from app.agent.tools.tracer import write_tool_trace

logger = logging.getLogger(__name__)

_MAX_TOOL_RETRIES = 2


class ToolExecutor:
    """
    Dispatches one or many tool calls, handles retries, writes traces, and
    produces tool_result messages ready to be appended to the conversation.
    """

    async def dispatch(self, tool_call: dict, context: ToolContext) -> dict:
        """
        Execute a single tool_call dict (from LLM response) and return the
        tool_result message dict to append to the next LLM request.
        """
        tool_call_id: str = tool_call["id"]
        fn = tool_call["function"]
        tool_name: str = fn["name"]
        start_ms = _now_ms()

        # Parse arguments
        try:
            params = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError as e:
            await write_tool_trace(
                trace_id=context.trace_id,
                session_id=context.session_id,
                release_id=context.release_id,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                input_params={},
                output_content=None,
                status="failed",
                error_type=ToolErrorType.PARAM_INVALID,
                error_message=f"Invalid JSON arguments: {e}",
                latency_ms=_elapsed_ms(start_ms),
                iteration=context.iteration,
            )
            return _error_msg(tool_call_id, f"Invalid tool arguments JSON: {e}")

        tool = tool_registry.get(tool_name)
        if tool is None:
            await write_tool_trace(
                trace_id=context.trace_id,
                session_id=context.session_id,
                release_id=context.release_id,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                input_params=params,
                output_content=None,
                status="failed",
                error_type=ToolErrorType.UNKNOWN,
                error_message=f"Unknown tool: {tool_name}",
                latency_ms=_elapsed_ms(start_ms),
                iteration=context.iteration,
            )
            return _error_msg(tool_call_id, f"Unknown tool: {tool_name}")

        # Validate parameters against JSON schema
        validation = tool.validate_params(params)
        if not validation.valid:
            await write_tool_trace(
                trace_id=context.trace_id,
                session_id=context.session_id,
                release_id=context.release_id,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                input_params=params,
                output_content=None,
                status="failed",
                error_type=ToolErrorType.PARAM_INVALID,
                error_message=validation.error,
                latency_ms=_elapsed_ms(start_ms),
                iteration=context.iteration,
            )
            return _error_msg(tool_call_id, f"Parameter validation failed: {validation.error}")

        # Confirmation gate for dangerous operations
        if tool.require_confirmation:
            await write_tool_trace(
                trace_id=context.trace_id,
                session_id=context.session_id,
                release_id=context.release_id,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                input_params=params,
                output_content=None,
                status="rejected",
                error_type=ToolErrorType.USER_REJECTED,
                error_message="Tool requires user confirmation (not yet supported in this version)",
                latency_ms=_elapsed_ms(start_ms),
                iteration=context.iteration,
            )
            return _error_msg(
                tool_call_id,
                f"Tool '{tool_name}' requires user confirmation before execution. "
                "Please confirm via the management interface and retry.",
            )

        # Execute with retry for transient failures
        result, retry_count = await self._execute_with_retry(tool, params, context)
        latency = _elapsed_ms(start_ms)

        status = "success" if result.success else (
            "timeout" if result.error_type == ToolErrorType.TIMEOUT else "failed"
        )
        await write_tool_trace(
            trace_id=context.trace_id,
            session_id=context.session_id,
            release_id=context.release_id,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            input_params=params,
            output_content=result.content,
            status=status,
            error_type=result.error_type,
            error_message=result.error_message,
            retry_count=retry_count,
            latency_ms=latency,
            iteration=context.iteration,
        )

        if result.success:
            return {"role": "tool", "tool_call_id": tool_call_id, "content": result.content}
        return _error_msg(tool_call_id, result.error_message or "Tool execution failed")

    async def dispatch_parallel(
        self,
        tool_calls: list[dict],
        context: ToolContext,
    ) -> list[dict]:
        """Dispatch multiple tool calls concurrently; preserve input order in results."""
        return list(
            await asyncio.gather(*[self.dispatch(tc, context) for tc in tool_calls])
        )

    # ------------------------------------------------------------------ #

    async def _execute_with_retry(
        self,
        tool: BaseTool,
        params: dict,
        context: ToolContext,
    ) -> tuple[ToolResult, int]:
        last_result: ToolResult | None = None
        for attempt in range(_MAX_TOOL_RETRIES + 1):
            try:
                result = await asyncio.wait_for(
                    tool.execute(params, context),
                    timeout=tool.timeout_seconds,
                )
                if result.success or not result.should_retry:
                    return result, attempt
                last_result = result
            except Exception as exc:
                err = classify_tool_error(exc)
                if err.category != ToolErrorCategory.RETRYABLE or attempt >= _MAX_TOOL_RETRIES:
                    return ToolResult(
                        success=False,
                        content="",
                        error_type=err.error_type,
                        error_message=str(err),
                    ), attempt
                last_result = ToolResult(
                    success=False,
                    content="",
                    error_type=err.error_type,
                    error_message=str(err),
                    should_retry=True,
                )
                await asyncio.sleep(0.5 * (attempt + 1))

        return last_result or ToolResult(success=False, content="", error_message="Max retries exceeded"), _MAX_TOOL_RETRIES


def _error_msg(tool_call_id: str, message: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": f"Error: {message}"}


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


def _elapsed_ms(start_ms: int) -> int:
    return _now_ms() - start_ms


tool_executor = ToolExecutor()

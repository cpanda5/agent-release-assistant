import logging
from uuid import uuid4

from app.agent.memory import memory_manager
from app.schemas.agent import AgentRunRequest, AgentRunResponse, UsageInfo

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 10


class AgentRunner:
    """
    Agent execution loop.

    Phase 1: observe → LLM → respond (stateless)
    Phase 2: observe → RAG retrieve → LLM → respond
    Phase 4: observe → RAG → LLM ↔ tool calling → respond  ← current
    Phase 5: full MemoryManager integration (active)
    """

    async def run(self, request: AgentRunRequest) -> AgentRunResponse:
        from app.agent.llm import llm_client
        from app.agent.tools import tool_registry
        from app.agent.tools.executor import tool_executor
        from app.agent.tools.base import ToolContext

        session_id = request.session_id or str(uuid4())
        release_id = _to_int(request.release_id)
        project_id = _to_int(request.project_id)

        # ── Assemble memory context (Layers 1, 2, 3) ────────────────── #
        ctx = await memory_manager.build_context(
            session_id=session_id,
            release_id=release_id,
            project_id=project_id,
            user_query=request.message,
        )

        await memory_manager.append_message(
            session_id=session_id,
            role="user",
            content=request.message,
            release_id=release_id,
            project_id=project_id,
        )

        # Build the initial messages list for the LLM
        messages = ctx.to_llm_messages()
        messages.append({"role": "user", "content": request.message})

        tools = tool_registry.to_llm_tools_spec()
        tool_context = ToolContext(
            session_id=session_id,
            trace_id="",  # updated after first LLM call
            release_id=release_id,
            project_id=project_id,
        )

        final_text = ""
        final_usage: dict = {}
        final_trace_id = ""

        # ── Tool calling loop ────────────────────────────────────────── #
        for iteration in range(1, MAX_TOOL_ITERATIONS + 1):
            tool_context.iteration = iteration

            text, tool_calls, usage, trace_id = await llm_client.complete_with_tools(
                messages,
                tools=tools,
                release_id=release_id,
                project_id=project_id,
            )
            tool_context.trace_id = trace_id
            final_usage = usage
            final_trace_id = trace_id

            if not tool_calls:
                # LLM finished — no more tool calls
                final_text = text or ""
                await memory_manager.append_message(
                    session_id=session_id,
                    role="assistant",
                    content=final_text,
                    token_count=usage.get("output_tokens", 0),
                    trace_id=trace_id,
                    release_id=release_id,
                    project_id=project_id,
                )
                break

            # LLM wants to call tools
            # 1. Record assistant turn (with tool_calls) in working memory
            await memory_manager.append_message(
                session_id=session_id,
                role="assistant",
                content=text,
                tool_calls=tool_calls,
                token_count=usage.get("output_tokens", 0),
                trace_id=trace_id,
                release_id=release_id,
                project_id=project_id,
            )

            # 2. Append assistant message to the conversation
            assistant_msg: dict = {"role": "assistant", "content": text, "tool_calls": tool_calls}
            messages.append(assistant_msg)

            # 3. Execute all tool calls in parallel
            tool_result_msgs = await tool_executor.dispatch_parallel(tool_calls, tool_context)

            # 4. Record each tool result in working memory and extend the conversation
            for tr_msg in tool_result_msgs:
                await memory_manager.append_message(
                    session_id=session_id,
                    role="tool",
                    content=tr_msg["content"],
                    tool_call_id=tr_msg.get("tool_call_id"),
                    release_id=release_id,
                    project_id=project_id,
                )
                messages.append(tr_msg)

            logger.info(
                "session=%s iteration=%d/%d tool_calls=%d",
                session_id, iteration, MAX_TOOL_ITERATIONS, len(tool_calls),
            )

        else:
            # Exhausted iterations without LLM finishing
            final_text = "Agent 已达到最大工具调用迭代次数，无法继续处理请求。"
            logger.warning("session=%s reached max tool iterations (%d)", session_id, MAX_TOOL_ITERATIONS)

        return AgentRunResponse(
            response=final_text,
            session_id=session_id,
            usage=UsageInfo(
                input_tokens=final_usage.get("input_tokens", 0),
                output_tokens=final_usage.get("output_tokens", 0),
            ),
        )


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

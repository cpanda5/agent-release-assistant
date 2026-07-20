from uuid import uuid4

from app.agent.memory import memory_manager
from app.schemas.agent import AgentRunRequest, AgentRunResponse, UsageInfo


class AgentRunner:
    """
    Agent execution loop.

    Phase 1: observe → LLM → respond (stateless)
    Phase 2: observe → RAG retrieve → LLM → respond
    Phase 4: observe → RAG → LLM ↔ tool calling → respond
    Phase 5: full MemoryManager integration (current)
    """

    async def run(self, request: AgentRunRequest) -> AgentRunResponse:
        from app.agent.llm import llm_client

        session_id = request.session_id or str(uuid4())
        release_id = _to_int(request.release_id)
        project_id = _to_int(request.project_id)

        # Layer 1+2+3: assemble context
        ctx = await memory_manager.build_context(
            session_id=session_id,
            release_id=release_id,
            project_id=project_id,
            user_query=request.message,
        )

        # Record user turn
        await memory_manager.append_message(
            session_id=session_id,
            role="user",
            content=request.message,
            release_id=release_id,
            project_id=project_id,
        )

        # Call LLM with assembled context
        messages = ctx.to_llm_messages()
        messages.append({"role": "user", "content": request.message})

        text, usage, trace_id = await llm_client.complete(
            messages,
            release_id=release_id,
            project_id=project_id,
        )

        # Record assistant turn
        await memory_manager.append_message(
            session_id=session_id,
            role="assistant",
            content=text,
            token_count=usage.get("output_tokens", 0),
            trace_id=trace_id,
            release_id=release_id,
            project_id=project_id,
        )

        return AgentRunResponse(
            response=text,
            session_id=session_id,
            usage=UsageInfo(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            ),
        )


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

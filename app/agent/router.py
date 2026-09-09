import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.agent.core import build_initial_state, invoke_agent
from app.agent.graph import agent_graph
from app.agent.llm.errors import LLMError
from app.agent.memory import memory_manager
from app.schemas.agent import (
    ChatRequest,
    ChatResponse,
    CloseSessionRequest,
    UsageInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


# ── /agent/chat — sync (full LangGraph workflow) ─────────────────────────── #

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = await invoke_agent(request)
    except LLMError as e:
        _raise_http(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ChatResponse(
        response=result.final_text,
        session_id=result.session_id,
        trace_id=result.trace_id,
        usage=UsageInfo(
            input_tokens=result.usage.get("input_tokens", 0),
            output_tokens=result.usage.get("output_tokens", 0),
        ),
    )


# ── /agent/chat/stream — SSE (full LangGraph workflow, streamed) ─────────── #

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Stream the full LangGraph workflow as SSE events.

    Event types emitted on the wire::

        {"type": "text",      "content": "..."}           # LLM answer delta
        {"type": "reasoning", "content": "..."}           # qwen3.7 thinking delta
        {"type": "tool_call", "name": "...", "input": {...}}
        {"type": "tool_result","name": "...", "output": "..."}
        {"type": "plan",      "plan": "..."}              # planner output
        {"type": "done",      "final_text": "...", "session_id": "...",
         "trace_id": "...", "usage": {...}}
        {"type": "error",     "error": "...", "detail": "..."}

    The client still terminates the stream by reading ``type: done`` (or
    ``type: error``); the literal ``data: [DONE]`` sentinel is emitted as a
    trailing frame for backward compatibility with the previous endpoint.
    """
    initial_state = await build_initial_state(request)
    session_id = initial_state["session_id"]
    trace_id = initial_state["trace_id"]
    accumulator = _StreamAccumulator()

    async def sse_generator():
        try:
            async for event in agent_graph.astream_events(
                initial_state, version="v2",
            ):
                for frame in accumulator.process(event):
                    yield frame
        except LLMError as e:
            yield _sse({"type": "error", "error": e.error_type, "detail": str(e)})
        except Exception as e:  # noqa: BLE001
            logger.exception("SSE stream failed")
            yield _sse({"type": "error", "error": "internal_error", "detail": str(e)})

        yield _sse({
            "type": "done",
            "session_id": session_id,
            "trace_id": trace_id,
            "final_text": accumulator.final_text,
            "usage": accumulator.usage,
        })
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


# ── Session lifecycle ────────────────────────────────────────────────────── #

@router.post("/session/close")
async def close_session(request: CloseSessionRequest) -> dict:
    from app.agent.identity import resolve_release_pk

    await memory_manager.close_session(
        session_id=request.session_id,
        release_id=await resolve_release_pk(request.release_id),
    )
    return {"status": "closed", "session_id": request.session_id}


# ── Health ────────────────────────────────────────────────────────────────── #

@router.get("/health")
async def health():
    return {"status": "ok"}


# ── Helpers ───────────────────────────────────────────────────────────────── #

class _StreamAccumulator:
    """
    Converts astream_events() output into SSE frames, while accumulating
    the final answer text and token usage for the trailing ``done`` event.
    """

    def __init__(self) -> None:
        self.final_text = ""
        self.usage: dict = {}

    def process(self, event: dict) -> list[str]:
        """Return zero or more SSE-formatted ``data: ...\\n\\n`` frames."""
        kind = event.get("event")
        frames: list[str] = []

        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk is None:
                return frames
            # AIMessageChunk: .content may be str or list (tool-call partials)
            content = chunk.content if isinstance(chunk.content, str) else ""
            reasoning = (chunk.additional_kwargs or {}).get("reasoning_content")
            # tool_calls arrive as partial chunks with content="" — we surface
            # them via on_tool_start instead, which carries full args.
            if reasoning:
                frames.append(_sse({"type": "reasoning", "content": reasoning}))
            if content:
                self.final_text += content
                frames.append(_sse({"type": "text", "content": content}))
            # usage is only populated on the final chunk; capture if present.
            usage = getattr(chunk, "usage_metadata", None)
            if usage:
                self.usage = {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                }

        elif kind == "on_tool_start":
            # event["data"]["input"] carries the tool arguments; name is on the event root.
            frames.append(_sse({
                "type": "tool_call",
                "name": event.get("name", ""),
                "input": event.get("data", {}).get("input"),
            }))

        elif kind == "on_tool_end":
            # output may be a str or a LangChain ToolOutput — stringify defensively.
            output = event.get("data", {}).get("output", "")
            if not isinstance(output, str):
                output = str(output)
            frames.append(_sse({
                "type": "tool_result",
                "name": event.get("name", ""),
                "output": output,
            }))

        elif kind == "on_chain_end" and event.get("name") == "planner":
            data = event.get("data", {}).get("output") or {}
            if isinstance(data, dict) and data.get("plan"):
                frames.append(_sse({"type": "plan", "plan": data["plan"]}))

        return frames


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _raise_http(e: LLMError) -> None:
    from app.agent.llm.errors import LLMErrorType

    code_map = {
        LLMErrorType.AUTH_ERROR: 401,
        LLMErrorType.RATE_LIMIT: 429,
        LLMErrorType.INVALID_REQUEST: 400,
        LLMErrorType.CONTENT_FILTERED: 422,
    }
    status = code_map.get(e.error_type, 502)
    raise HTTPException(status_code=status, detail=f"{e.error_type}: {e}")

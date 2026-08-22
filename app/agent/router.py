import json
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.agent.core import invoke_agent
from app.agent.llm import SYSTEM_PROMPT, llm_gateway
from app.agent.llm.callbacks import trace_callback
from app.agent.llm.errors import LLMError
from app.agent.memory import memory_manager
from app.schemas.agent import (
    ChatRequest,
    ChatResponse,
    CloseSessionRequest,
    UsageInfo,
)

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


# ── /agent/chat/stream — SSE (direct model stream) ───────────────────────── #

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Lightweight streaming endpoint: streams a single model turn (no tool loop).
    Emits reasoning deltas separately from answer deltas so a client can render
    qwen3.7 deep-thinking when ``enable_thinking`` is on.
    """
    session_id = request.session_id or str(uuid4())
    messages = _build_chat_messages(request)
    run_metadata = {
        "session_id": session_id,
        "release_id": request.release_id,
        "project_id": request.project_id,
        "trace_id": str(uuid4()),
    }

    async def sse_generator():
        try:
            async for delta in llm_gateway.astream(
                messages,
                model=request.model,
                thinking=request.enable_thinking,
                callbacks=[trace_callback],
                run_metadata=run_metadata,
            ):
                payload = {"session_id": session_id}
                if delta.get("reasoning"):
                    payload["reasoning"] = delta["reasoning"]
                if delta.get("content"):
                    payload["text"] = delta["content"]
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except LLMError as e:
            yield f"data: {json.dumps({'error': e.error_type, 'detail': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


# ── Session lifecycle ────────────────────────────────────────────────────── #

@router.post("/session/close")
async def close_session(request: CloseSessionRequest) -> dict:
    await memory_manager.close_session(
        session_id=request.session_id,
        release_id=request.release_id,
    )
    return {"status": "closed", "session_id": request.session_id}


# ── Health ────────────────────────────────────────────────────────────────── #

@router.get("/health")
async def health():
    return {"status": "ok"}


# ── Helpers ───────────────────────────────────────────────────────────────── #

def _build_chat_messages(request: ChatRequest) -> list[dict]:
    content = request.message
    if request.release_id:
        content = f"[Release: {request.release_id}]\n\n{content}"
    elif request.project_id:
        content = f"[Project: {request.project_id}]\n\n{content}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


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

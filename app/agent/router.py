import json
from uuid import uuid4

import openai
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.agent.core import AgentRunner
from app.agent.llm import SYSTEM_PROMPT, llm_client
from app.agent.llm.errors import LLMError
from app.agent.memory import memory_manager
from app.schemas.agent import (
    AgentRunRequest,
    AgentRunResponse,
    ChatRequest,
    ChatResponse,
    CloseSessionRequest,
    UsageInfo,
)

router = APIRouter(prefix="/agent", tags=["agent"])
_runner = AgentRunner()


# ── Legacy endpoint (Phase 1 compat) ─────────────────────────────────────── #

@router.post("/run", response_model=AgentRunResponse)
async def run_agent(request: AgentRunRequest) -> AgentRunResponse:
    try:
        return await _runner.run(request)
    except openai.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid DASHSCOPE_API_KEY.")
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit reached. Retry later.")
    except openai.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /agent/chat — sync ────────────────────────────────────────────────────── #

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or str(uuid4())
    messages = _build_chat_messages(request)
    try:
        text, usage, trace_id = await llm_client.complete(
            messages,
            prompt_template=request.prompt_template,
            prompt_template_version=request.prompt_template_version,
        )
    except LLMError as e:
        _raise_http(e)
    return ChatResponse(
        response=text,
        session_id=session_id,
        trace_id=trace_id,
        usage=UsageInfo(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        ),
    )


# ── /agent/chat/stream — SSE ─────────────────────────────────────────────── #

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    session_id = request.session_id or str(uuid4())
    messages = _build_chat_messages(request)

    async def sse_generator():
        try:
            async for chunk in llm_client.stream(
                messages,
                prompt_template=request.prompt_template,
                prompt_template_version=request.prompt_template_version,
            ):
                yield f"data: {json.dumps({'text': chunk, 'session_id': session_id})}\n\n"
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
    from app.agent.llm.errors import LLMErrorCategory, LLMErrorType

    code_map = {
        LLMErrorType.AUTH_ERROR: 401,
        LLMErrorType.RATE_LIMIT: 429,
        LLMErrorType.INVALID_REQUEST: 400,
        LLMErrorType.CONTENT_FILTERED: 422,
    }
    status = code_map.get(e.error_type, 502)
    raise HTTPException(status_code=status, detail=f"{e.error_type}: {e}")

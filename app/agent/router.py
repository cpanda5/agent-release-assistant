import anthropic
from fastapi import APIRouter, HTTPException

from app.agent.core import AgentRunner
from app.schemas.agent import AgentRunRequest, AgentRunResponse

router = APIRouter(prefix="/agent", tags=["agent"])
_runner = AgentRunner()


@router.post("/run", response_model=AgentRunResponse)
async def run_agent(request: AgentRunRequest) -> AgentRunResponse:
    try:
        return await _runner.run(request)
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid ANTHROPIC_API_KEY.")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit reached. Retry later.")
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    return {"status": "ok"}

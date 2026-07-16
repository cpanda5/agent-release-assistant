from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent.router import router as agent_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.config import settings
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill in your key.")
    yield


app = FastAPI(
    title="Agent Release Assistant",
    description="AI SDLC Agent — 研发交付智能助手",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(agent_router)


@app.get("/")
async def root():
    return {
        "service": "Agent Release Assistant",
        "version": "0.1.0",
        "docs": "/docs",
        "phase": "Phase 1 — LLM Core",
    }

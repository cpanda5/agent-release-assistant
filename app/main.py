import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent.router import router as agent_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.config import settings

    if not settings.dashscope_api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set.")

    if settings.mysql_host:
        from app.db import init_db_pool, close_db_pool

        try:
            await init_db_pool()
        except Exception:
            logger.exception("MySQL pool init failed — traces will be skipped")

    yield

    if settings.mysql_host:
        from app.db import close_db_pool

        await close_db_pool()


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

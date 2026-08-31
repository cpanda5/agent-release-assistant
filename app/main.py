import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent.rag.router import router as knowledge_router
from app.agent.router import router as agent_router
from app.auth import auth_router
from app.project import project_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.config import settings

    if not settings.dashscope_api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set.")

    if settings.mysql_host:
        from app.db import init_db_pool

        try:
            await init_db_pool()
        except Exception:
            logger.exception("MySQL pool init failed — traces will be skipped")

    from app.agent.tools import register_all_tools
    register_all_tools()
    logger.info("Tool registry initialised")

    from app.agent.rag import init_stores
    await init_stores()

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

# Dev-friendly CORS so the Vite frontend can call the API directly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(project_router)
app.include_router(agent_router)
app.include_router(knowledge_router)


@app.get("/")
async def root():
    return {
        "service": "Agent Release Assistant",
        "version": "0.1.0",
        "docs": "/docs",
        "engine": "LangGraph workflow (planner → rag → agent ⇄ tools → memory)",
    }

"""
Project memory layer — stable background from the `project` table
(context_summary / tech_constraints). Read-only for the agent; maintained via
admin channels. RAG retrieval lives in the graph's `rag` node, not here.
"""
import json
import logging

from app.db import get_db_pool

logger = logging.getLogger(__name__)


async def load_project_context(project_id: int) -> dict | None:
    pool = get_db_pool()
    if pool is None:
        return None
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT context_summary, tech_constraints FROM project "
                    "WHERE id = %s AND status = 1 AND deleted_at IS NULL",
                    (project_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                return {
                    "context_summary": row[0] or "",
                    "tech_constraints": _json_load(row[1]),
                }
    except Exception:
        logger.exception("Failed to load project context for project_id=%s", project_id)
        return None


def build_project_prompt_block(ctx: dict | None) -> str:
    if not ctx or not (ctx.get("context_summary") or ctx.get("tech_constraints")):
        return ""
    lines = ["--- Project 上下文 ---"]
    if ctx.get("context_summary"):
        lines.append(ctx["context_summary"])
    constraints = ctx.get("tech_constraints") or []
    if constraints:
        lines.append("\n技术约束：")
        lines += [f"  • {c}" for c in constraints]
    lines.append("--- End ---")
    return "\n".join(lines)


def _json_load(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []

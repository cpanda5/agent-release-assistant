"""
invoke_agent — thin adapter over the LangGraph workflow
(``app/agent/graph``: load_memory → planner → rag → agent ⇄ tools → persist_memory).
Builds the initial state, invokes the compiled graph, and maps the result onto
the response schemas.
"""
import logging
from dataclasses import dataclass
from uuid import uuid4

from app.agent.graph import agent_graph
from app.agent.identity import resolve_project_pk, resolve_release_pk

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    final_text: str
    session_id: str
    trace_id: str
    usage: dict


async def invoke_agent(request) -> AgentResult:
    """
    Run the agent graph for any request exposing the standard fields
    (message / session_id / release_id / project_id / user_id / model /
    enable_thinking / prompt_template / prompt_template_version).
    """
    initial_state = await build_initial_state(request)
    final_state = await agent_graph.ainvoke(initial_state)
    return AgentResult(
        final_text=final_state.get("final_text", ""),
        session_id=initial_state["session_id"],
        trace_id=initial_state["trace_id"],
        usage=final_state.get("usage", {}) or {},
    )


async def build_initial_state(request) -> dict:
    """
    Build the initial AgentState from any request-like object exposing the
    standard ChatRequest fields. Reused by both the sync ``invoke_agent``
    path and the SSE streaming path so the graph always sees the same inputs.

    ``release_id`` / ``project_id`` arrive as public UUIDs (or legacy numeric
    ids) and are resolved to the internal BIGINT PKs the memory / trace layers
    key on. Unresolvable references resolve to None (the graph degrades to
    session-only scoping) rather than erroring, matching the no-DB behaviour.
    """
    session_id = getattr(request, "session_id", None) or str(uuid4())
    trace_id = str(uuid4())

    return {
        "session_id": session_id,
        "release_id": await resolve_release_pk(getattr(request, "release_id", None)),
        "project_id": await resolve_project_pk(getattr(request, "project_id", None)),
        "user_id": getattr(request, "user_id", None),
        "model": getattr(request, "model", None),
        "enable_thinking": getattr(request, "enable_thinking", None),
        "user_query": request.message,
        "trace_id": trace_id,
        "iteration": 0,
        "prompt_template": getattr(request, "prompt_template", None),
        "prompt_template_version": getattr(request, "prompt_template_version", None),
    }

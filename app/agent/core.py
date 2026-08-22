"""
invoke_agent — thin adapter over the LangGraph workflow.

The former hand-rolled ReAct loop now lives in ``app/agent/graph`` as a
``StateGraph`` (load_memory → planner → rag → agent ⇄ tools → persist_memory).
This module just builds the initial state, invokes the compiled graph, and maps
the result onto the response schemas — keeping the router interface stable.
"""
import logging
from dataclasses import dataclass
from uuid import uuid4

from app.agent.graph import agent_graph

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
    enable_thinking).
    """
    session_id = request.session_id or str(uuid4())
    trace_id = str(uuid4())

    initial_state = {
        "session_id": session_id,
        "release_id": _to_int(request.release_id),
        "project_id": _to_int(request.project_id),
        "user_id": getattr(request, "user_id", None),
        "model": getattr(request, "model", None),
        "enable_thinking": getattr(request, "enable_thinking", None),
        "user_query": request.message,
        "trace_id": trace_id,
        "iteration": 0,
    }

    final_state = await agent_graph.ainvoke(initial_state)
    return AgentResult(
        final_text=final_state.get("final_text", ""),
        session_id=session_id,
        trace_id=trace_id,
        usage=final_state.get("usage", {}) or {},
    )


def _to_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

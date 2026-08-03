from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """
    State threaded through the LangGraph workflow:

        load_memory → planner → rag → agent ⇄ tools → persist_memory → END

    ``messages`` uses the ``add_messages`` reducer so each node returns only the
    messages it wants to append. Memory persistence (MySQL + working memory) is
    handled out-of-band by ``memory_manager`` inside the nodes.
    """
    messages: Annotated[list[AnyMessage], add_messages]

    # Request context
    session_id: str
    release_id: int | None
    project_id: int | None
    user_id: str | None
    model: str | None
    enable_thinking: bool | None
    user_query: str
    trace_id: str

    # Intermediate / output
    plan: str
    rag_chunks: list[str]
    iteration: int
    usage: dict[str, Any]
    final_text: str

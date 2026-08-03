"""
Assembles the LangGraph workflow:

    START → load_memory → planner → rag → agent ⇄ tools → persist_memory → END
"""
from langgraph.graph import END, START, StateGraph

from app.agent.graph.nodes import (
    agent,
    load_memory,
    persist_memory,
    planner,
    rag,
    route_after_agent,
    tools,
)
from app.agent.graph.state import AgentState


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("load_memory", load_memory)
    g.add_node("planner", planner)
    g.add_node("rag", rag)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("persist_memory", persist_memory)

    g.add_edge(START, "load_memory")
    g.add_edge("load_memory", "planner")
    g.add_edge("planner", "rag")
    g.add_edge("rag", "agent")
    g.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "persist_memory": "persist_memory"},
    )
    g.add_edge("tools", "agent")
    g.add_edge("persist_memory", END)

    return g.compile()


# Module-level compiled graph — reused across requests.
agent_graph = build_graph()

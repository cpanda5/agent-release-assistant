"""
LangGraph node functions for the agent workflow.

Each node returns a partial state update; ``messages`` uses the
``add_messages`` reducer, so nodes append by returning a list of new messages.

Memory reads happen only in ``load_memory``; writes only in ``persist_memory``
(user turn + final answer → ``message`` table, plus a background summary check).
Intermediate tool turns are not replayed — ``tool_call_trace`` records them.
"""
import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from app.agent.graph.state import AgentState
from app.agent.llm import llm_gateway
from app.agent.llm.callbacks import trace_callback
from app.agent.llm.prompt import load_template
from app.agent.memory import memory_manager
from app.agent.memory.schemas import AgentContext
from app.agent.rag import retrieve as rag_retrieve
from app.agent.tools import tool_registry
from app.agent.tools.base import ToolContext
from app.agent.tools.executor import tool_executor

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 10


def _run_metadata(state: AgentState) -> dict:
    return {
        "session_id": state["session_id"],
        "release_id": state.get("release_id"),
        "project_id": state.get("project_id"),
        "trace_id": state.get("trace_id", ""),
    }


class Plan(BaseModel):
    """Structured planner output — the SDLC stage + short execution plan."""

    class Stage(str):
        pass

    stage: str = Field(
        description="SDLC 阶段：需求分析 / 技术设计 / 代码生成 / 代码Review / GitHub操作 / 其他"
    )
    description: str = Field(description="一到两句话的简短执行计划")


# ── load_memory ───────────────────────────────────────────────────────── #

async def load_memory(state: AgentState) -> dict:
    # Register the session, then read all four context layers (project /
    # release / session summary / history) into the prompt + message list.
    await memory_manager.ensure_session(
        session_id=state["session_id"],
        release_id=state.get("release_id"),
        user_id=state.get("user_id"),
    )

    # Render the template's system portion now and pass it as an override.
    # Placeholders that depend on RAG (rag_context) are filled later, in the rag node.
    system_override: str | None = None
    if state.get("prompt_template"):
        try:
            tpl = load_template(
                state["prompt_template"],
                state.get("prompt_template_version") or "v1",
            )
            system_override = tpl.render_system(
                user_input=state.get("user_query", ""),
                project_description="",
                release_name="",
                tech_stack="",
            )
        except FileNotFoundError:
            logger.warning(
                "prompt_template '%s' not found; falling back to default",
                state.get("prompt_template"),
            )

    ctx = await memory_manager.build_context(
        session_id=state["session_id"],
        release_id=state.get("release_id"),
        project_id=state.get("project_id"),
        system_prompt_override=system_override,
    )
    return {"messages": _ctx_to_messages(ctx), "iteration": 0, "usage": {}}


# ── planner (structured output) ───────────────────────────────────────── #

async def planner(state: AgentState) -> dict:
    """
    Run the planner via LLM structured output. The result is written directly
    to ``state["plan"]`` for downstream observability; it is not injected into
    the conversation as a SystemMessage.
    """
    prompt = [
        SystemMessage(
            content=(
                "你是研发交付流程的规划器。判断用户请求属于哪个 SDLC 阶段"
                "（需求分析 / 技术设计 / 代码生成 / 代码Review / GitHub操作 / 其他），"
                "并用一到两句话给出简短执行计划。"
            )
        ),
        HumanMessage(content=state["user_query"]),
    ]
    try:
        plan_obj = await llm_gateway.ainvoke_with_schema(
            prompt,
            Plan,
            model=state.get("model"),
            callbacks=[trace_callback],
            run_metadata=_run_metadata(state),
        )
        plan_text = f"[{plan_obj.stage}] {plan_obj.description}"
    except Exception:
        logger.exception("planner failed; continuing without a plan")
        plan_text = ""

    return {"plan": plan_text}


# ── rag ─────────────────────────────────────────────────────────────────── #

async def rag(state: AgentState) -> dict:
    docs = await rag_retrieve(
        state["user_query"],
        release_id=state.get("release_id"),
        project_id=state.get("project_id"),
    )
    new_messages: list = []
    chunks: list[str] = []
    context = ""
    if docs:
        chunks = [d.page_content for d in docs]
        context = "\n\n".join(chunks)
        new_messages.append(SystemMessage(content=f"参考资料（本 release / project 知识库）：\n{context}"))

    # A requested template's rendered user portion (may weave in RAG context)
    # replaces the raw user_query.
    user_content = state["user_query"]
    if state.get("prompt_template"):
        try:
            tpl = load_template(
                state["prompt_template"],
                state.get("prompt_template_version") or "v1",
            )
            user_content = tpl.render_user(
                user_input=state["user_query"],
                rag_context=context,
                project_description="",
                release_name="",
                tech_stack="",
            )
        except FileNotFoundError:
            pass  # already warned in load_memory

    new_messages.append(HumanMessage(content=user_content))
    return {"rag_chunks": chunks, "messages": new_messages}


# ── agent (LLM ⇄ tools) ───────────────────────────────────────────────── #

async def agent(state: AgentState) -> dict:
    tools = tool_registry.to_langchain_tools()
    ai = await llm_gateway.ainvoke(
        state["messages"],
        model=state.get("model"),
        thinking=state.get("enable_thinking"),
        tools=tools,
        callbacks=[trace_callback],
        run_metadata=_run_metadata(state),
    )

    usage = ai.usage_metadata or {}
    return {
        "messages": [ai],
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        },
    }


# ── tools ─────────────────────────────────────────────────────────────── #

async def tools(state: AgentState) -> dict:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    iteration = state.get("iteration", 0) + 1

    context = ToolContext(
        session_id=state["session_id"],
        trace_id=state.get("trace_id", ""),
        release_id=state.get("release_id"),
        project_id=state.get("project_id"),
        iteration=iteration,
    )
    oai_calls = [_lc_tool_call_to_oai(tc) for tc in tool_calls]
    results = await tool_executor.dispatch_parallel(oai_calls, context)

    tool_msgs: list = []
    for r in results:
        tool_msgs.append(ToolMessage(content=r["content"], tool_call_id=r["tool_call_id"]))

    logger.info(
        "session=%s iteration=%d/%d tool_calls=%d",
        state["session_id"], iteration, MAX_TOOL_ITERATIONS, len(oai_calls),
    )
    return {"messages": tool_msgs, "iteration": iteration}


# ── persist_memory (finalize) ─────────────────────────────────────────── #

async def persist_memory(state: AgentState) -> dict:
    final_text = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.text.strip():
            final_text = msg.text
            break
    if not final_text and state.get("iteration", 0) >= MAX_TOOL_ITERATIONS:
        final_text = "Agent 已达到最大工具调用迭代次数，无法继续处理请求。"

    # Single write point: user turn + final answer → `message`, then a
    # background check whether the session needs a new summary version.
    await memory_manager.persist_turn(
        session_id=state["session_id"],
        release_id=state.get("release_id"),
        user_query=state["user_query"],
        final_text=final_text,
        trace_id=state.get("trace_id"),
    )
    return {"final_text": final_text}


# ── Routing ───────────────────────────────────────────────────────────── #

def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if (
        isinstance(last, AIMessage)
        and last.tool_calls
        and state.get("iteration", 0) < MAX_TOOL_ITERATIONS
    ):
        return "tools"
    return "persist_memory"


# ── Helpers ───────────────────────────────────────────────────────────── #

def _ctx_to_messages(ctx: AgentContext) -> list:
    """Convert assembled memory into LangChain messages (system prompt + replay)."""
    out: list = [SystemMessage(content=ctx.system_prompt)]
    for m in ctx.messages:
        if m.role == "user":
            out.append(HumanMessage(content=m.content or ""))
        elif m.role == "assistant":
            out.append(AIMessage(content=m.content or ""))
    return out


def _lc_tool_call_to_oai(tc: dict) -> dict:
    """LangChain tool_call ({name, args, id}) → OpenAI-format dict for the executor."""
    return {
        "id": tc["id"],
        "type": "function",
        "function": {
            "name": tc["name"],
            "arguments": json.dumps(tc.get("args", {}), ensure_ascii=False),
        },
    }

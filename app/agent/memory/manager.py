"""
MemoryManager — the only class AgentCore imports from this package.

Three public methods:
  build_context()   — assemble AgentContext before each LLM call
  append_message()  — record one turn (user or assistant)
  close_session()   — summarise and persist when session ends
"""
import asyncio
import logging

from app.agent.llm.client import SYSTEM_PROMPT, llm_client
from app.agent.memory import project as _proj_layer
from app.agent.memory import release as _rel_layer
from app.agent.memory import working as _work_layer
from app.agent.memory.schemas import AgentContext, Message
from app.agent.memory.tokens import estimate_tokens

logger = logging.getLogger(__name__)

# Token budget slices (see Module 2 spec §7)
_TOTAL_BUDGET = 100_000
_RELEASE_CTX_MAX = 2_000
_PROJECT_RAG_MAX = 3_000
_COMPLETION_RESERVE = 5_000


class MemoryManager:

    # ------------------------------------------------------------------ #
    # build_context                                                        #
    # ------------------------------------------------------------------ #

    async def build_context(
        self,
        *,
        session_id: str,
        release_id: int | None,
        project_id: int | None,
        user_query: str,
    ) -> AgentContext:
        """
        Concurrently reads all three memory layers and assembles AgentContext.
        Token budget is enforced in priority order (spec §7).
        """
        wm = _work_layer.get_or_create(session_id)

        # Parallel I/O: release context + project RAG
        release_ctx_task = asyncio.create_task(
            _rel_layer.load_release_context(release_id) if release_id else _noop_task(None)
        )
        rag_task = asyncio.create_task(
            _proj_layer.project_memory.search(user_query, str(project_id))
            if project_id else _noop_task([])
        )
        release_ctx, rag_chunks = await asyncio.gather(release_ctx_task, rag_task)

        # ── Build system prompt ──────────────────────────────────────── #
        system_parts = [SYSTEM_PROMPT]

        summary_preamble = wm.combined_summary()
        if summary_preamble:
            system_parts.append(f"\n\n--- 本 session 早期对话摘要 ---\n{summary_preamble}\n---")

        release_block = _rel_layer.build_release_prompt_block(release_ctx)
        if release_block:
            release_block = _truncate_to_tokens(release_block, _RELEASE_CTX_MAX)
            system_parts.append(f"\n\n{release_block}")

        system_prompt = "".join(system_parts)

        # ── Compute remaining budget ─────────────────────────────────── #
        budget = _TOTAL_BUDGET
        budget -= estimate_tokens(system_prompt)
        budget -= estimate_tokens(user_query)
        budget -= _COMPLETION_RESERVE
        for chunk in rag_chunks:
            budget -= estimate_tokens(chunk.content)
        budget = max(budget, 0)

        # ── Select working-memory messages within budget ─────────────── #
        live_msgs = _select_messages_within_budget(wm, budget)

        ctx = AgentContext(
            system_prompt=system_prompt,
            messages=live_msgs,
            rag_chunks=rag_chunks,
            estimated_tokens=_TOTAL_BUDGET - budget,
        )

        # Trigger compression if working memory is growing large
        if wm.needs_compression():
            asyncio.create_task(self._compress(wm))

        return ctx

    # ------------------------------------------------------------------ #
    # append_message                                                       #
    # ------------------------------------------------------------------ #

    async def append_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str | None = None,
        token_count: int = 0,
        trace_id: str | None = None,
        release_id: int | None = None,
        project_id: int | None = None,
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        """
        1. Append to in-process WorkingMemory.
        2. Persist to MySQL message table (fire-and-forget).
        """
        import json as _json
        wm = _work_layer.get_or_create(session_id)
        persist_content = content or ""
        if tool_calls and not content:
            persist_content = _json.dumps(tool_calls, ensure_ascii=False)
        if token_count == 0:
            token_count = estimate_tokens(persist_content)
        wm.append(role, content, trace_id, tool_calls=tool_calls, tool_call_id=tool_call_id)

        # Persist without blocking the response path
        asyncio.create_task(
            _rel_layer.persist_message(
                session_id=_int_or_zero(session_id),
                release_id=release_id,
                project_id=project_id,
                role=role,
                content=persist_content,
                token_count=token_count,
                trace_id=trace_id,
            )
        )

    # ------------------------------------------------------------------ #
    # update_release_context_with_artifact                                 #
    # ------------------------------------------------------------------ #

    async def update_release_context_with_artifact(
        self,
        *,
        release_id: int,
        session_id: str,
        artifact: dict,
    ) -> None:
        """
        Called by GenerateArtifactTool after a successful artifact creation.
        Appends an artifact mention to the release progress summary so future
        sessions know what has already been produced.
        """
        existing = await _rel_layer.load_release_context(release_id) or {}
        artifact_note = f"[交付物] {artifact.get('type', '')}：{artifact.get('title', '')} (id={artifact.get('artifact_id', '')})"
        prev_summary = existing.get("progress_summary", "")
        new_summary = (prev_summary + "\n" + artifact_note).strip() if prev_summary else artifact_note
        new_summary = _truncate_to_tokens(new_summary, _RELEASE_CTX_MAX)
        await _rel_layer.upsert_release_context(
            release_id=release_id,
            progress_summary=new_summary,
            key_decisions=existing.get("key_decisions"),
            pending_tasks=existing.get("pending_tasks"),
            tech_constraints=existing.get("tech_constraints"),
            last_updated_by=session_id,
        )

    # ------------------------------------------------------------------ #
    # close_session                                                        #
    # ------------------------------------------------------------------ #

    async def close_session(
        self,
        *,
        session_id: str,
        release_id: int | None = None,
    ) -> None:
        """
        Generate a final session summary → write session_summary →
        update release_context.
        """
        wm = _work_layer.get(session_id)
        if wm is None:
            return

        full_text = wm.full_text_for_final_summary()
        if not full_text.strip():
            _work_layer.evict(session_id)
            return

        summary, key_decisions = await _generate_session_summary(full_text)
        token_count = estimate_tokens(summary)

        if release_id:
            await _rel_layer.write_session_summary(
                session_id=_int_or_zero(session_id),
                release_id=release_id,
                summary=summary,
                key_decisions=key_decisions,
                token_count=token_count,
            )
            await _update_release_context(release_id, summary, key_decisions, session_id)

        _work_layer.evict(session_id)
        logger.info("session=%s closed, summary written (%d tokens)", session_id, token_count)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    async def _compress(self, wm: _work_layer.WorkingMemory) -> None:
        """Background task: summarise overflow and apply compression."""
        overflow = wm.get_overflow_messages()
        if not overflow:
            return
        text = "\n".join(f"[{m.role}]: {m.content}" for m in overflow)
        summary, _ = await _generate_session_summary(text)
        wm.apply_compression(summary)


# ── Module-level singleton ─────────────────────────────────────────── #

memory_manager = MemoryManager()


# ── Private helpers ───────────────────────────────────────────────── #

async def _noop_task(value):
    return value


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    # Rough truncation: shave characters until within budget
    while estimate_tokens(text) > max_tokens and len(text) > 0:
        text = text[: int(len(text) * 0.9)]
    return text + "\n[截断]"


def _select_messages_within_budget(
    wm: _work_layer.WorkingMemory, token_budget: int
) -> list[Message]:
    """
    Priority: newest messages first, include as many as budget allows.
    Always include at least the most recent message.
    """
    msgs = wm.messages
    if not msgs:
        return []

    selected: list[Message] = []
    used = 0
    for msg in reversed(msgs):
        cost = msg.token_count or estimate_tokens(msg.content)
        if used + cost > token_budget and selected:
            break
        selected.insert(0, msg)
        used += cost

    return selected


async def _generate_session_summary(text: str) -> tuple[str, list[str]]:
    """Call LLM to produce a brief summary + key decisions list."""
    prompt = (
        "请对以下对话历史进行简洁总结（不超过300字），并以JSON格式返回：\n"
        '{"summary": "...", "key_decisions": ["决策1", "决策2"]}\n\n'
        f"对话历史：\n{text[:8000]}"  # hard cap to avoid runaway costs
    )
    try:
        import json as _json
        text_out, _, _ = await llm_client.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.1,
        )
        # Try to parse structured output; fall back to raw text
        start = text_out.find("{")
        end = text_out.rfind("}") + 1
        if start != -1 and end > start:
            data = _json.loads(text_out[start:end])
            return data.get("summary", text_out), data.get("key_decisions", [])
    except Exception:
        logger.exception("Session summary generation failed")
    return text[:500], []


async def _update_release_context(
    release_id: int,
    new_summary: str,
    new_decisions: list[str],
    session_id: str,
) -> None:
    existing = await _rel_layer.load_release_context(release_id) or {}
    prev_summary = existing.get("progress_summary", "")
    all_decisions = list(dict.fromkeys(
        (existing.get("key_decisions") or []) + new_decisions
    ))
    combined_summary = (prev_summary + "\n\n" + new_summary).strip() if prev_summary else new_summary
    combined_summary = _truncate_to_tokens(combined_summary, _RELEASE_CTX_MAX)
    await _rel_layer.upsert_release_context(
        release_id=release_id,
        progress_summary=combined_summary,
        key_decisions=all_decisions,
        pending_tasks=existing.get("pending_tasks"),
        tech_constraints=existing.get("tech_constraints"),
        last_updated_by=session_id,
    )


def _int_or_zero(value: str) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0

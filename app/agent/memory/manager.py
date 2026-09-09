"""
MemoryManager — the only class the graph imports from this package.

Four MySQL-backed context layers: Project Context (`project`, read-only),
Release Context (`release_context`), Session Summary (`session_summary`,
versioned) and Session History (`message`).

Reads happen once per turn in `build_context`, writes once in `persist_turn`.
Summaries are produced by a background task past a size threshold and
finalized by `close_session`.
"""
import asyncio
import logging

from app.agent.llm.gateway import SYSTEM_PROMPT, llm_gateway
from app.agent.memory import project as _proj_layer
from app.agent.memory import release as _rel_layer
from app.agent.memory.schemas import AgentContext, Message
from app.agent.memory.tokens import estimate_tokens

logger = logging.getLogger(__name__)

# Token caps for the injected prompt blocks
_PROJECT_CTX_MAX = 1_000
_RELEASE_CTX_MAX = 2_000
_SUMMARY_MAX = 2_000
# History replay window
_HISTORY_LIMIT = 50
_HISTORY_TOKEN_MAX = 8_000
# Summarize when unsummarized history exceeds either threshold
_SUMMARY_TRIGGER_MESSAGES = 40
_SUMMARY_TRIGGER_TOKENS = 8_000
# Hard cap on LLM input when generating a summary
_SUMMARY_INPUT_CHARS = 12_000

# Strong refs to background tasks (asyncio only keeps weak refs)
_bg_tasks: set[asyncio.Task] = set()


def _spawn_bg(coro) -> None:
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


class MemoryManager:

    # ------------------------------------------------------------------ #
    # session                                                              #
    # ------------------------------------------------------------------ #

    async def ensure_session(
        self,
        *,
        session_id: str,
        release_id: int | None = None,
        user_id: str | None = None,
    ) -> int | None:
        """Register the session row and return its BIGINT id."""
        return await _rel_layer.ensure_session(
            session_id=session_id, release_id=release_id, user_id=user_id
        )

    # ------------------------------------------------------------------ #
    # read path — load_memory node                                         #
    # ------------------------------------------------------------------ #

    async def build_context(
        self,
        *,
        session_id: str,
        release_id: int | None,
        project_id: int | None,
        system_prompt_override: str | None = None,
    ) -> AgentContext:
        """Read all four layers and assemble system prompt + replay history.

        ``system_prompt_override`` replaces the default ``SYSTEM_PROMPT`` as the
        first system block when a request-level prompt template is active.
        Project / release / summary blocks are still appended on top, so the
        agent keeps its workspace context even with a custom template.
        """
        project_ctx, release_ctx, summary = await asyncio.gather(
            _proj_layer.load_project_context(project_id) if project_id else _noop(None),
            _rel_layer.load_release_context(release_id) if release_id else _noop(None),
            _rel_layer.load_latest_summary(session_id),
        )

        cutoff = summary["cutoff"] if summary else None
        history = await _rel_layer.load_history(session_id, after=cutoff, limit=_HISTORY_LIMIT)

        system_parts = [system_prompt_override or SYSTEM_PROMPT]
        for block, cap in (
            (_proj_layer.build_project_prompt_block(project_ctx), _PROJECT_CTX_MAX),
            (_rel_layer.build_release_prompt_block(release_ctx), _RELEASE_CTX_MAX),
            (_rel_layer.build_summary_prompt_block(summary), _SUMMARY_MAX),
        ):
            if block:
                system_parts.append("\n\n" + _truncate_tail(block, cap))

        system_prompt = "".join(system_parts)
        messages = _select_messages_within_budget(history, _HISTORY_TOKEN_MAX)
        used = sum(m.token_count or estimate_tokens(m.content) for m in messages)
        return AgentContext(
            system_prompt=system_prompt,
            messages=messages,
            estimated_tokens=estimate_tokens(system_prompt) + used,
        )

    # ------------------------------------------------------------------ #
    # write path — persist_memory node                                     #
    # ------------------------------------------------------------------ #

    async def persist_turn(
        self,
        *,
        session_id: str,
        release_id: int | None,
        user_query: str,
        final_text: str,
        trace_id: str | None = None,
    ) -> None:
        """Persist this turn (user + final assistant), then check the summary threshold."""
        await _rel_layer.insert_message(
            session_id=session_id, role="user", content=user_query,
            token_count=estimate_tokens(user_query),
        )
        if final_text:
            await _rel_layer.insert_message(
                session_id=session_id, role="assistant", content=final_text,
                token_count=estimate_tokens(final_text), trace_id=trace_id,
            )
        if release_id is not None:  # session_summary.release_id is NOT NULL
            _spawn_bg(self._maybe_summarize(session_id, release_id))

    async def update_release_context_with_artifact(
        self,
        *,
        release_id: int,
        session_id: str,
        artifact: dict,
    ) -> None:
        """Called by GenerateArtifactTool; appends an artifact note to the progress summary."""
        existing = await _rel_layer.load_release_context(release_id) or {}
        note = (
            f"[交付物] {artifact.get('type', '')}：{artifact.get('title', '')} "
            f"(id={artifact.get('artifact_id', '')})"
        )
        prev = existing.get("progress_summary", "")
        progress = (prev + "\n" + note).strip() if prev else note
        await _rel_layer.upsert_release_context(
            release_id=release_id,
            progress_summary=_truncate_head(progress, _RELEASE_CTX_MAX),
            pending_tasks=existing.get("pending_tasks"),
            tech_constraints=existing.get("tech_constraints"),
            updated_by=session_id,
        )

    # ------------------------------------------------------------------ #
    # session close                                                        #
    # ------------------------------------------------------------------ #

    async def close_session(
        self,
        *,
        session_id: str,
        release_id: int | None = None,
    ) -> None:
        """Final summary → session_summary, then merge progress into release_context."""
        if release_id is None:
            logger.warning("close_session without release_id: summary skipped (session=%s)", session_id)
            return
        prev = await _rel_layer.load_latest_summary(session_id)
        cutoff, pending = await self._read_unsummarized(session_id, prev)
        if not pending:
            return
        summary, decisions = await self._write_summary(session_id, release_id, prev, cutoff, pending)
        if summary is None:
            return
        await self._merge_into_release_context(release_id, summary, decisions, session_id)
        logger.info("session=%s closed, final summary written", session_id)

    # ------------------------------------------------------------------ #
    # summarization internals                                              #
    # ------------------------------------------------------------------ #

    async def _maybe_summarize(self, session_id: str, release_id: int) -> None:
        """Background: summarize once unsummarized history exceeds a threshold."""
        try:
            prev = await _rel_layer.load_latest_summary(session_id)
            cutoff, pending = await self._read_unsummarized(session_id, prev)
            tokens = sum(m.token_count or estimate_tokens(m.content) for m in pending)
            if len(pending) < _SUMMARY_TRIGGER_MESSAGES and tokens < _SUMMARY_TRIGGER_TOKENS:
                return
            await self._write_summary(session_id, release_id, prev, cutoff, pending)
        except Exception:
            logger.exception("Background summarization failed for session=%s", session_id)

    async def _read_unsummarized(self, session_id: str, prev: dict | None):
        """
        Return (cutoff, messages since the previous summary).

        The cutoff is DB NOW(3) captured *before* the read: messages written
        while the summary LLM call runs land after the cutoff and are replayed
        verbatim next turn. Worst case a message is both summarized and
        replayed — harmless duplication, never a gap.
        """
        cutoff = await _rel_layer.db_now()
        after = prev["cutoff"] if prev else None
        pending = await _rel_layer.load_history(session_id, after=after, limit=500)
        return cutoff, pending

    async def _write_summary(self, session_id, release_id, prev, cutoff, pending):
        """Incremental summary: LLM(prev summary + new messages) → new version row."""
        if not pending:
            return None, []
        summary, decisions = await _generate_summary(
            _serialize_history(pending),
            prev["summary"] if prev else "",
        )
        version = (prev["version"] + 1) if prev else 1
        await _rel_layer.write_session_summary(
            session_id=session_id,
            release_id=release_id,
            summary=summary,
            key_decisions=decisions,
            token_count=estimate_tokens(summary),
            version=version,
            cutoff=cutoff,
        )
        return summary, decisions

    async def _merge_into_release_context(
        self, release_id: int, summary: str, decisions: list[str], session_id: str
    ) -> None:
        """Fold the session's outcome into the release-level progress summary."""
        existing = await _rel_layer.load_release_context(release_id) or {}
        merged = summary
        if decisions:
            merged += "\n关键决策：" + "；".join(decisions)
        prev = existing.get("progress_summary", "")
        combined = (prev + "\n\n" + merged).strip() if prev else merged
        await _rel_layer.upsert_release_context(
            release_id=release_id,
            progress_summary=_truncate_head(combined, _RELEASE_CTX_MAX),
            pending_tasks=existing.get("pending_tasks"),
            tech_constraints=existing.get("tech_constraints"),
            updated_by=session_id,
        )


# ── Module-level singleton ─────────────────────────────────────────── #

memory_manager = MemoryManager()


# ── Helpers ───────────────────────────────────────────────────────── #

async def _noop(value):
    return value


def _serialize_history(messages: list[Message]) -> str:
    lines = [f"[{m.role}]: {m.content}" for m in messages]
    return "\n".join(lines)[:_SUMMARY_INPUT_CHARS]


def _truncate_tail(text: str, max_tokens: int) -> str:
    """Keep the head, shave the tail — for prompt blocks."""
    if estimate_tokens(text) <= max_tokens:
        return text
    while estimate_tokens(text) > max_tokens and text:
        text = text[: int(len(text) * 0.9)]
    return text + "\n[截断]"


def _truncate_head(text: str, max_tokens: int) -> str:
    """Keep the recent tail, shave the head — for append-only progress summaries."""
    if estimate_tokens(text) <= max_tokens:
        return text
    while estimate_tokens(text) > max_tokens and text:
        text = text[int(len(text) * 0.1):]
    return "[早期内容已截断]\n" + text


def _select_messages_within_budget(
    messages: list[Message], token_budget: int
) -> list[Message]:
    """Newest first; always keep the most recent message."""
    selected: list[Message] = []
    used = 0
    for msg in reversed(messages):
        cost = msg.token_count or estimate_tokens(msg.content)
        if used + cost > token_budget and selected:
            break
        selected.insert(0, msg)
        used += cost
    return selected


async def _generate_summary(text: str, prev_summary: str = "") -> tuple[str, list[str]]:
    """LLM-compress a conversation slice; merges with the previous summary if given."""
    prompt = (
        "请对以下对话历史进行简洁总结（不超过300字），并以JSON格式返回：\n"
        '{"summary": "...", "key_decisions": ["决策1", "决策2"]}\n'
    )
    if prev_summary:
        prompt += f"\n已有摘要（请增量合并，保留仍然有效的信息）：\n{prev_summary}\n"
    prompt += f"\n对话历史：\n{text}"
    try:
        import json as _json
        ai = await llm_gateway.ainvoke([{"role": "user", "content": prompt}])
        out = ai.text or ""
        start, end = out.find("{"), out.rfind("}") + 1
        if start != -1 and end > start:
            data = _json.loads(out[start:end])
            return data.get("summary", out), data.get("key_decisions", [])
    except Exception:
        logger.exception("Session summary generation failed")
    return text[:500], []

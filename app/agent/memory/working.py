"""
Layer 1: Working Memory

In-process per-session message store with sliding-window compression.
Lives only as long as the process; persisted to MySQL by MemoryManager.
"""
import logging
from dataclasses import dataclass, field

from app.agent.memory.schemas import Message
from app.agent.memory.tokens import estimate_tokens

logger = logging.getLogger(__name__)

# Token budget before compression is triggered (leaves 20% completion buffer)
DEFAULT_COMPRESSION_THRESHOLD = 80_000
# How many recent messages to always preserve verbatim
KEEP_RECENT_N = 10


@dataclass
class WorkingMemory:
    session_id: str
    messages: list[Message] = field(default_factory=list)
    summary_history: list[str] = field(default_factory=list)
    total_estimated_tokens: int = 0
    compression_threshold: int = DEFAULT_COMPRESSION_THRESHOLD

    # ------------------------------------------------------------------ #
    # Append                                                               #
    # ------------------------------------------------------------------ #

    def append(self, role: str, content: str, trace_id: str | None = None) -> Message:
        tokens = estimate_tokens(content)
        msg = Message(role=role, content=content, token_count=tokens, trace_id=trace_id)
        self.messages.append(msg)
        self.total_estimated_tokens += tokens
        return msg

    # ------------------------------------------------------------------ #
    # Context window budget                                                #
    # ------------------------------------------------------------------ #

    def needs_compression(self) -> bool:
        return self.total_estimated_tokens > self.compression_threshold

    def get_messages_for_llm(self) -> list[Message]:
        """Return the verbatim tail of the history that fits within budget."""
        return self.messages[-KEEP_RECENT_N:]

    def get_overflow_messages(self) -> list[Message]:
        """Messages that have been pushed out of the live window."""
        if len(self.messages) <= KEEP_RECENT_N:
            return []
        return self.messages[:-KEEP_RECENT_N]

    def apply_compression(self, summary: str) -> None:
        """
        Replace overflow messages with a summary entry.
        Called by MemoryManager after it receives the LLM-generated summary.
        """
        overflow = self.get_overflow_messages()
        if not overflow:
            return

        self.summary_history.append(summary)
        # Recalculate token budget: keep only recent N + new summary
        kept = self.messages[-KEEP_RECENT_N:]
        self.messages = kept
        self.total_estimated_tokens = sum(m.token_count for m in kept)
        logger.debug(
            "session=%s compressed %d messages into summary (%d tokens freed)",
            self.session_id, len(overflow), sum(m.token_count for m in overflow),
        )

    # ------------------------------------------------------------------ #
    # Summaries                                                            #
    # ------------------------------------------------------------------ #

    def combined_summary(self) -> str:
        """All accumulated summaries joined — used as preamble in system prompt."""
        return "\n\n".join(self.summary_history)

    def full_text_for_final_summary(self) -> str:
        """All messages serialised for session-close summarisation."""
        lines = []
        for m in self.messages:
            lines.append(f"[{m.role}]: {m.content}")
        return "\n".join(lines)


# ── Per-process registry (session_id → WorkingMemory) ─────────────────── #

_registry: dict[str, WorkingMemory] = {}


def get_or_create(session_id: str) -> WorkingMemory:
    if session_id not in _registry:
        _registry[session_id] = WorkingMemory(session_id=session_id)
    return _registry[session_id]


def get(session_id: str) -> WorkingMemory | None:
    return _registry.get(session_id)


def evict(session_id: str) -> None:
    _registry.pop(session_id, None)

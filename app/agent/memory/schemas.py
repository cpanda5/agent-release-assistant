from dataclasses import dataclass, field


@dataclass
class Message:
    """One replayable history message (user / assistant text only)."""
    role: str
    content: str
    token_count: int = 0


@dataclass
class AgentContext:
    """Assembled per-turn context produced by MemoryManager.build_context()."""
    system_prompt: str
    messages: list[Message] = field(default_factory=list)
    estimated_tokens: int = 0

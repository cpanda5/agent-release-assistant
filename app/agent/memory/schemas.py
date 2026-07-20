from dataclasses import dataclass, field


@dataclass
class Message:
    role: str           # "system" | "user" | "assistant"
    content: str
    token_count: int = 0
    trace_id: str | None = None


@dataclass
class KnowledgeChunk:
    content: str
    source: str
    score: float = 0.0


@dataclass
class AgentContext:
    """
    The assembled context passed to LLMClient.complete().

    system_prompt  — base prompt + release progress injection
    messages       — compressed working memory (ready for LLM)
    rag_chunks     — project knowledge (empty until Module 4)
    estimated_tokens — pre-flight budget check
    """
    system_prompt: str
    messages: list[Message]
    rag_chunks: list[KnowledgeChunk] = field(default_factory=list)
    estimated_tokens: int = 0

    def to_llm_messages(self) -> list[dict]:
        result = [{"role": "system", "content": self.system_prompt}]
        result += [{"role": m.role, "content": m.content} for m in self.messages]
        return result

from dataclasses import dataclass, field


@dataclass
class Message:
    role: str                         # "system" | "user" | "assistant" | "tool"
    content: str | None               # None is valid for assistant turn that only has tool_calls
    token_count: int = 0
    trace_id: str | None = None
    tool_calls: list[dict] | None = None   # set on assistant messages that trigger tool use
    tool_call_id: str | None = None        # set on tool-result messages


@dataclass
class KnowledgeChunk:
    content: str
    source: str
    score: float = 0.0


@dataclass
class AgentContext:
    """
    The assembled context consumed by the graph's load_memory node.

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
        for m in self.messages:
            if m.role == "tool":
                result.append({
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": m.content or "",
                })
            elif m.role == "assistant" and m.tool_calls:
                result.append({
                    "role": "assistant",
                    "content": m.content,  # may be None
                    "tool_calls": m.tool_calls,
                })
            else:
                result.append({"role": m.role, "content": m.content or ""})
        return result

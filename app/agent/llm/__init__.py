from app.agent.llm.client import SYSTEM_PROMPT, LLMClient, llm_client
from app.agent.llm.errors import LLMError, LLMErrorCategory, LLMErrorType
from app.agent.llm.prompt import PromptTemplate, load_template
from app.agent.llm.tracer import TraceContext, TraceResult


async def call_llm(messages: list[dict]) -> tuple[str, dict]:
    """Backward-compatible wrapper used by AgentRunner. Prepends SYSTEM_PROMPT."""
    full = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    text, usage, _ = await llm_client.complete(full)
    return text, usage


__all__ = [
    "LLMClient",
    "LLMError",
    "LLMErrorCategory",
    "LLMErrorType",
    "PromptTemplate",
    "SYSTEM_PROMPT",
    "TraceContext",
    "TraceResult",
    "call_llm",
    "llm_client",
    "load_template",
]

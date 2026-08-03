from app.agent.llm.errors import LLMError, LLMErrorCategory, LLMErrorType
from app.agent.llm.gateway import SYSTEM_PROMPT, LLMGateway, llm_gateway
from app.agent.llm.prompt import PromptTemplate, load_template
from app.agent.llm.tracer import TraceContext, TraceResult


async def call_llm(messages: list[dict]) -> tuple[str, dict]:
    """Backward-compatible helper: prepends SYSTEM_PROMPT and returns (text, usage)."""
    full: list = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    ai = await llm_gateway.ainvoke(full)
    usage = ai.usage_metadata or {}
    return ai.text, {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }


__all__ = [
    "LLMGateway",
    "LLMError",
    "LLMErrorCategory",
    "LLMErrorType",
    "PromptTemplate",
    "SYSTEM_PROMPT",
    "TraceContext",
    "TraceResult",
    "call_llm",
    "llm_gateway",
    "load_template",
]

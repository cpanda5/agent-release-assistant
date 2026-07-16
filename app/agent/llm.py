import anthropic

from app.config import settings

SYSTEM_PROMPT = """You are an AI SDLC Agent (AI研发交付助手), an intelligent assistant for enterprise software development lifecycle management.

You help engineering teams deliver software efficiently across the full SDLC:
- Requirements analysis (需求分析) and user story decomposition
- Technical design (技术设计) and architecture decisions
- Code generation (代码生成) following enterprise best practices
- Documentation (文档沉淀) and knowledge management
- API design (API设计) and database schema design
- Task decomposition (任务拆解) and sprint planning
- GitHub PR preparation and code review

You operate within a "Release" workspace — each release is a self-contained AI work context for one deliverable version. You may also reference project-level shared knowledge that spans multiple releases.

Be concise, technically precise, and actionable. Respond in the same language as the user."""

_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


async def call_llm(messages: list[dict]) -> tuple[str, dict]:
    """Call LLM with prompt caching on the system prompt."""
    async with _client.messages.stream(
        model=settings.model,
        max_tokens=settings.max_tokens,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=messages,
    ) as stream:
        final = await stream.get_final_message()

    text = next((b.text for b in final.content if b.type == "text"), "")
    usage = {
        "input_tokens": final.usage.input_tokens,
        "output_tokens": final.usage.output_tokens,
        "cache_read_input_tokens": getattr(final.usage, "cache_read_input_tokens", 0),
        "cache_creation_input_tokens": getattr(final.usage, "cache_creation_input_tokens", 0),
    }
    return text, usage

import hashlib
import logging

from openai import AsyncOpenAI

from app.agent.llm.errors import LLMError, LLMErrorCategory, classify_error
from app.agent.llm.retry import with_retry
from app.agent.llm.streaming import StreamingHandler
from app.agent.llm.tracer import TraceContext, TraceResult, write_trace
from app.config import settings

logger = logging.getLogger(__name__)

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


class LLMClient:
    def __init__(self):
        self._http = AsyncOpenAI(
            api_key=settings.dashscope_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    # ------------------------------------------------------------------ #
    # Public: sync complete                                                #
    # ------------------------------------------------------------------ #

    async def complete(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        session_id: int | None = None,
        release_id: int | None = None,
        project_id: int | None = None,
        prompt_template: str | None = None,
        prompt_template_version: str | None = None,
    ) -> tuple[str, dict, str]:
        """Call LLM synchronously. Returns (text, usage, trace_id)."""
        req_model = model or settings.model
        req_max_tokens = max_tokens or settings.max_tokens

        ctx = TraceContext(
            requested_model=req_model,
            session_id=session_id,
            release_id=release_id,
            project_id=project_id,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            system_prompt_hash=_hash_system(messages),
            is_streaming=False,
            temperature=temperature,
            max_tokens=req_max_tokens,
        )
        result = TraceResult(model=req_model)

        try:
            text, usage, retry_log, actual_model = await self._call_with_fallback(
                messages, req_model, req_max_tokens, temperature
            )
            result.model = actual_model
            if actual_model != req_model:
                result.fallback_model = req_model
                result.fallback_reason = "model_overloaded_or_context_too_long"
                result.status = "fallback"
            result.prompt_tokens = usage.get("input_tokens", 0)
            result.completion_tokens = usage.get("output_tokens", 0)
            result.total_tokens = result.prompt_tokens + result.completion_tokens
            result.retry_count = len(retry_log)
            result.retry_reasons = retry_log
            result.finish_reason = usage.get("finish_reason", "stop")
            result.latency_ms = ctx.elapsed_ms()
            return text, usage, ctx.trace_id
        except LLMError as e:
            result.status = "failed"
            result.error_type = e.error_type
            result.error_message = str(e)[:512]
            result.latency_ms = ctx.elapsed_ms()
            raise
        finally:
            await write_trace(ctx, result)

    # ------------------------------------------------------------------ #
    # Public: tool-calling complete                                        #
    # ------------------------------------------------------------------ #

    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        session_id: int | None = None,
        release_id: int | None = None,
        project_id: int | None = None,
    ) -> tuple[str | None, list[dict] | None, dict, str]:
        """
        Call LLM with tool definitions.

        Returns (text, tool_calls, usage, trace_id).
        Exactly one of text / tool_calls is non-None:
        - text is set when the LLM produces a final answer (finish_reason=stop)
        - tool_calls is set when the LLM wants to invoke tools (finish_reason=tool_calls)
        """
        req_model = model or settings.model
        req_max_tokens = max_tokens or settings.max_tokens

        ctx = TraceContext(
            requested_model=req_model,
            session_id=session_id,
            release_id=release_id,
            project_id=project_id,
            is_streaming=False,
            temperature=temperature,
            max_tokens=req_max_tokens,
        )
        result = TraceResult(model=req_model)

        try:
            (text, tool_calls, usage), retry_log, actual_model = await self._call_with_tools_fallback(
                messages, tools, req_model, req_max_tokens, temperature
            )
            result.model = actual_model
            if actual_model != req_model:
                result.fallback_model = req_model
                result.fallback_reason = "model_overloaded_or_context_too_long"
                result.status = "fallback"
            result.prompt_tokens = usage.get("input_tokens", 0)
            result.completion_tokens = usage.get("output_tokens", 0)
            result.total_tokens = result.prompt_tokens + result.completion_tokens
            result.retry_count = len(retry_log)
            result.retry_reasons = retry_log
            result.finish_reason = usage.get("finish_reason", "stop")
            result.latency_ms = ctx.elapsed_ms()
            return text, tool_calls, usage, ctx.trace_id
        except LLMError as e:
            result.status = "failed"
            result.error_type = e.error_type
            result.error_message = str(e)[:512]
            result.latency_ms = ctx.elapsed_ms()
            raise
        finally:
            await write_trace(ctx, result)

    # ------------------------------------------------------------------ #
    # Public: streaming                                                    #
    # ------------------------------------------------------------------ #

    async def stream(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        session_id: int | None = None,
        release_id: int | None = None,
        project_id: int | None = None,
        prompt_template: str | None = None,
        prompt_template_version: str | None = None,
    ):
        """
        Async generator: yields text chunks.
        Writes llm_trace after the stream is fully consumed or on error.

        Usage:
            async for chunk in client.stream(messages, ...):
                yield chunk
        """
        req_model = model or settings.model
        req_max_tokens = max_tokens or settings.max_tokens

        ctx = TraceContext(
            requested_model=req_model,
            session_id=session_id,
            release_id=release_id,
            project_id=project_id,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            system_prompt_hash=_hash_system(messages),
            is_streaming=True,
            temperature=temperature,
            max_tokens=req_max_tokens,
        )
        result = TraceResult(model=req_model)

        try:
            raw_stream, retry_log = await with_retry(
                lambda: self._open_stream(req_model, messages, req_max_tokens, temperature)
            )
            result.retry_count = len(retry_log)
            result.retry_reasons = retry_log

            handler = StreamingHandler(raw_stream)
            async for chunk in handler:
                yield chunk

            result.model = req_model
            result.ttfb_ms = handler.ttfb_ms
            result.finish_reason = handler.finish_reason
            result.prompt_tokens = handler.usage.get("input_tokens", 0)
            result.completion_tokens = handler.usage.get("output_tokens", 0)
            result.total_tokens = result.prompt_tokens + result.completion_tokens
            result.latency_ms = ctx.elapsed_ms()
        except LLMError as e:
            result.status = "failed"
            result.error_type = e.error_type
            result.error_message = str(e)[:512]
            result.latency_ms = ctx.elapsed_ms()
            raise
        finally:
            await write_trace(ctx, result)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    async def _call_with_fallback(
        self,
        messages: list[dict],
        req_model: str,
        max_tokens: int,
        temperature: float | None,
    ) -> tuple[str, dict, list[dict], str]:
        """Try primary model, fall back to fallback_model on FALLBACK errors."""
        chain = [req_model]
        if settings.fallback_model and settings.fallback_model != req_model:
            chain.append(settings.fallback_model)

        last_err: LLMError | None = None
        for model in chain:
            try:
                (text, usage), retry_log = await with_retry(
                    lambda m=model: self._single_call(m, messages, max_tokens, temperature)
                )
                return text, usage, retry_log, model
            except LLMError as e:
                last_err = e
                if e.category != LLMErrorCategory.FALLBACK:
                    raise
        raise last_err  # type: ignore[misc]

    async def _single_call(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float | None,
    ) -> tuple[str, dict]:
        try:
            kwargs: dict = {"model": model, "messages": messages, "max_tokens": max_tokens}
            if temperature is not None:
                kwargs["temperature"] = temperature
            completion = await self._http.chat.completions.create(**kwargs)
            text = completion.choices[0].message.content or ""
            usage = {
                "input_tokens": completion.usage.prompt_tokens if completion.usage else 0,
                "output_tokens": completion.usage.completion_tokens if completion.usage else 0,
                "finish_reason": completion.choices[0].finish_reason or "stop",
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            }
            return text, usage
        except Exception as exc:
            raise classify_error(exc) from exc

    async def _open_stream(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float | None,
    ):
        try:
            kwargs: dict = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": True}
            if temperature is not None:
                kwargs["temperature"] = temperature
            return await self._http.chat.completions.create(**kwargs)
        except Exception as exc:
            raise classify_error(exc) from exc


    async def _call_with_tools_fallback(
        self,
        messages: list[dict],
        tools: list[dict],
        req_model: str,
        max_tokens: int,
        temperature: float | None,
    ) -> tuple[tuple[str | None, list[dict] | None, dict], list[dict], str]:
        chain = [req_model]
        if settings.fallback_model and settings.fallback_model != req_model:
            chain.append(settings.fallback_model)

        last_err: LLMError | None = None
        for model in chain:
            try:
                (text, tool_calls, usage), retry_log = await with_retry(
                    lambda m=model: self._single_call_with_tools(m, messages, max_tokens, temperature, tools)
                )
                return (text, tool_calls, usage), retry_log, model
            except LLMError as e:
                last_err = e
                if e.category != LLMErrorCategory.FALLBACK:
                    raise
        raise last_err  # type: ignore[misc]

    async def _single_call_with_tools(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float | None,
        tools: list[dict],
    ) -> tuple[str | None, list[dict] | None, dict]:
        try:
            kwargs: dict = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "tools": tools,
                "tool_choice": "auto",
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            completion = await self._http.chat.completions.create(**kwargs)
            choice = completion.choices[0]
            usage = {
                "input_tokens": completion.usage.prompt_tokens if completion.usage else 0,
                "output_tokens": completion.usage.completion_tokens if completion.usage else 0,
                "finish_reason": choice.finish_reason or "stop",
            }
            raw_tool_calls = getattr(choice.message, "tool_calls", None)
            if raw_tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in raw_tool_calls
                ]
                return None, tool_calls, usage
            return choice.message.content or "", None, usage
        except Exception as exc:
            raise classify_error(exc) from exc


def _hash_system(messages: list[dict]) -> str | None:
    sys_content = next((m["content"] for m in messages if m.get("role") == "system"), None)
    if sys_content:
        return hashlib.sha256(sys_content.encode()).hexdigest()[:16]
    return None


# Module-level singleton — all callers share one HTTP connection pool
llm_client = LLMClient()

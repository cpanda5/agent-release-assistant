"""
LLM Gateway — the single entry point for all LLM I/O.

LangChain ``ChatOpenAI`` (DashScope-compatible endpoint) plus the enterprise
orchestration:

  * model selection / validation      → ``settings.resolve_model``
  * retry with backoff                 → ``llm/retry.with_retry``
  * error classification + fallback    → ``llm/errors.classify_error``
  * observability                      → LangChain callbacks (see ``callbacks.py``)

LangGraph nodes call ``llm_gateway.ainvoke(...)`` / ``llm_gateway.astream(...)``;
they never construct ChatModels themselves. Tracing is attached by passing a
callback handler — the gateway does not write traces directly.
"""
import logging

from typing import Type

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.agent.llm.errors import LLMError, LLMErrorCategory, classify_error
from app.agent.llm.retry import with_retry
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


class LLMGateway:
    def __init__(self) -> None:
        # Cache base ChatModel instances keyed by (model, thinking) so we reuse
        # one HTTP connection pool per configuration instead of building a fresh
        # AsyncOpenAI client on every call.
        self._models: dict[tuple[str, bool], ChatOpenAI] = {}

    # ------------------------------------------------------------------ #
    # Model construction                                                   #
    # ------------------------------------------------------------------ #

    def resolve_model(self, name: str | None) -> str:
        return settings.resolve_model(name)

    def _base_model(self, model: str, thinking: bool) -> ChatOpenAI:
        key = (model, thinking)
        cached = self._models.get(key)
        if cached is None:
            cached = ChatOpenAI(
                model=model,
                api_key=settings.dashscope_api_key,
                base_url=settings.llm_base_url,
                max_tokens=settings.max_tokens,
                # qwen3.7 deep-thinking toggle (default off). Note: some Qwen
                # models only honour enable_thinking on the streaming path.
                extra_body={"enable_thinking": thinking},
            )
            self._models[key] = cached
        return cached

    # ------------------------------------------------------------------ #
    # Public: invoke (with retry + model fallback)                         #
    # ------------------------------------------------------------------ #

    async def ainvoke(
        self,
        messages: list[BaseMessage] | list[dict],
        *,
        model: str | None = None,
        thinking: bool | None = None,
        tools: list[BaseTool] | None = None,
        callbacks: list | None = None,
        run_metadata: dict | None = None,
    ) -> AIMessage:
        """
        Invoke the LLM once (optionally with tools bound).

        Retries RETRYABLE errors, and on FALLBACK-category errors (overload /
        context-too-long) transparently downgrades to ``settings.fallback_model``.
        Tracing is emitted by the provided callbacks; ``run_metadata`` (session /
        release / project ids) is forwarded to them.
        """
        req_model = self.resolve_model(model)
        use_thinking = settings.enable_thinking if thinking is None else thinking

        chain = [req_model]
        if settings.fallback_model and settings.fallback_model != req_model:
            chain.append(settings.fallback_model)

        last_err: LLMError | None = None
        for m in chain:
            try:
                result, _retry_log = await with_retry(
                    lambda mm=m: self._single_ainvoke(mm, messages, use_thinking, tools, callbacks, run_metadata)
                )
                return result
            except LLMError as e:
                last_err = e
                if e.category != LLMErrorCategory.FALLBACK:
                    raise
        raise last_err  # type: ignore[misc]

    async def _single_ainvoke(
        self,
        model: str,
        messages: list,
        thinking: bool,
        tools: list[BaseTool] | None,
        callbacks: list | None,
        run_metadata: dict | None,
    ) -> AIMessage:
        runnable = self._base_model(model, thinking)
        if tools:
            runnable = runnable.bind_tools(tools)
        config = _build_config(callbacks, run_metadata)
        try:
            return await runnable.ainvoke(messages, config=config)
        except Exception as exc:  # noqa: BLE001 — re-classified into LLMError
            raise classify_error(_unwrap(exc)) from exc

    # ------------------------------------------------------------------ #
    # Public: structured output                                            #
    # ------------------------------------------------------------------ #

    async def ainvoke_with_schema(
        self,
        messages: list[BaseMessage] | list[dict],
        schema: Type[BaseModel],
        *,
        model: str | None = None,
        callbacks: list | None = None,
        run_metadata: dict | None = None,
    ) -> BaseModel:
        """
        Invoke the LLM once and return a validated Pydantic instance.

        Uses LangChain's ``with_structured_output`` (JSON-schema function calling
        on the ChatModel). Retries RETRYABLE errors; on FALLBACK, downgrades to
        ``settings.fallback_model``. Thinking is forced off — structured output
        doesn't mix with qwen3.7 deep-thinking.
        """
        req_model = self.resolve_model(model)

        chain = [req_model]
        if settings.fallback_model and settings.fallback_model != req_model:
            chain.append(settings.fallback_model)

        last_err: LLMError | None = None
        for m in chain:
            try:
                result, _retry_log = await with_retry(
                    lambda mm=m: self._single_ainvoke_schema(
                        mm, messages, schema, callbacks, run_metadata
                    )
                )
                return result
            except LLMError as e:
                last_err = e
                if e.category != LLMErrorCategory.FALLBACK:
                    raise
        raise last_err  # type: ignore[misc]

    async def _single_ainvoke_schema(
        self,
        model: str,
        messages: list,
        schema: Type[BaseModel],
        callbacks: list | None,
        run_metadata: dict | None,
    ) -> BaseModel:
        runnable = self._base_model(model, thinking=False).with_structured_output(schema)
        config = _build_config(callbacks, run_metadata)
        try:
            return await runnable.ainvoke(messages, config=config)
        except Exception as exc:  # noqa: BLE001
            raise classify_error(_unwrap(exc)) from exc

    # ------------------------------------------------------------------ #
    # Public: streaming                                                    #
    # ------------------------------------------------------------------ #

    async def astream(
        self,
        messages: list[BaseMessage] | list[dict],
        *,
        model: str | None = None,
        thinking: bool | None = None,
        callbacks: list | None = None,
        run_metadata: dict | None = None,
    ):
        """
        Async generator yielding ``{"reasoning": str|None, "content": str}`` per
        chunk. ``reasoning`` carries qwen3.7 deep-thinking deltas (only present
        when thinking is enabled); ``content`` carries the final answer deltas.

        Streaming does not attempt model fallback — a stream cannot be resumed on
        a different model mid-flight.
        """
        req_model = self.resolve_model(model)
        use_thinking = settings.enable_thinking if thinking is None else thinking
        llm = self._base_model(req_model, use_thinking)
        config = _build_config(callbacks, {**(run_metadata or {}), "is_streaming": True})
        try:
            async for chunk in llm.astream(messages, config=config):
                reasoning = chunk.additional_kwargs.get("reasoning_content")
                content = chunk.content if isinstance(chunk.content, str) else ""
                if reasoning or content:
                    yield {"reasoning": reasoning, "content": content}
        except Exception as exc:  # noqa: BLE001
            raise classify_error(_unwrap(exc)) from exc


def _build_config(callbacks: list | None, run_metadata: dict | None) -> dict | None:
    config: dict = {}
    if callbacks:
        config["callbacks"] = callbacks
    if run_metadata:
        # Drop None values so LangChain metadata stays clean.
        config["metadata"] = {k: v for k, v in run_metadata.items() if v is not None}
    return config or None


def _unwrap(exc: Exception) -> Exception:
    """LangChain may wrap the underlying openai exception; classify the cause."""
    import openai

    if isinstance(exc, openai.OpenAIError):
        return exc
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, openai.OpenAIError):
        return cause
    return exc


# Module-level singleton — shared connection pools across all callers.
llm_gateway = LLMGateway()

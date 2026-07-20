import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.agent.llm.errors import LLMError, LLMErrorCategory, LLMErrorType

T = TypeVar("T")

# 不同错误类型的最大重试次数
_MAX_RETRIES: dict[str, int] = {
    LLMErrorType.RATE_LIMIT: 3,
    LLMErrorType.SERVICE_UNAVAILABLE: 3,
    LLMErrorType.TIMEOUT: 2,
}

_BASE_DELAY_S = 1.0
_JITTER_S = 0.5


async def with_retry(fn: Callable[[], Awaitable[T]]) -> tuple[T, list[dict]]:
    """
    Retry fn with exponential backoff + jitter for RETRYABLE errors.
    Returns (result, retry_log) where retry_log records each retry attempt.

    Jitter prevents thundering herd: 100 simultaneous rate-limit hits won't
    all retry at the same second.
    """
    retry_log: list[dict] = []
    attempt = 0

    while True:
        try:
            result = await fn()
            return result, retry_log
        except LLMError as e:
            if e.category != LLMErrorCategory.RETRYABLE:
                raise
            max_retries = _MAX_RETRIES.get(e.error_type, 3)
            if attempt >= max_retries:
                raise

            # Prefer retry-after header (RateLimitError); otherwise exponential backoff
            if e.retry_after_ms is not None:
                wait_ms = e.retry_after_ms
            else:
                wait_ms = int((_BASE_DELAY_S * 2**attempt + random.uniform(0, _JITTER_S)) * 1000)

            retry_log.append({"attempt": attempt + 1, "error": e.error_type, "wait_ms": wait_ms})
            await asyncio.sleep(wait_ms / 1000)
            attempt += 1

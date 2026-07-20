import openai


class LLMErrorType:
    RATE_LIMIT = "RateLimitError"
    SERVICE_UNAVAILABLE = "ServiceUnavailable"
    TIMEOUT = "Timeout"
    MODEL_OVERLOADED = "ModelOverloaded"
    CONTEXT_TOO_LONG = "ContextTooLong"
    AUTH_ERROR = "AuthError"
    INVALID_REQUEST = "InvalidRequest"
    CONTENT_FILTERED = "ContentFiltered"
    UNKNOWN = "Unknown"


class LLMErrorCategory:
    RETRYABLE = "retryable"
    FALLBACK = "fallback"
    NOT_RETRYABLE = "not_retryable"
    UNKNOWN = "unknown"


class LLMError(Exception):
    def __init__(
        self,
        error_type: str,
        category: str,
        message: str,
        original: Exception | None = None,
        retry_after_ms: int | None = None,
    ):
        self.error_type = error_type
        self.category = category
        self.original = original
        self.retry_after_ms = retry_after_ms
        super().__init__(message)


def classify_error(exc: Exception) -> LLMError:
    if isinstance(exc, openai.AuthenticationError):
        return LLMError(LLMErrorType.AUTH_ERROR, LLMErrorCategory.NOT_RETRYABLE, str(exc), exc)

    if isinstance(exc, openai.RateLimitError):
        retry_after_ms = None
        try:
            if hasattr(exc, "response") and exc.response is not None:
                ra = exc.response.headers.get("retry-after")
                if ra:
                    retry_after_ms = int(float(ra) * 1000)
        except Exception:
            pass
        return LLMError(LLMErrorType.RATE_LIMIT, LLMErrorCategory.RETRYABLE, str(exc), exc, retry_after_ms)

    if isinstance(exc, openai.APITimeoutError):
        return LLMError(LLMErrorType.TIMEOUT, LLMErrorCategory.RETRYABLE, str(exc), exc)

    if isinstance(exc, openai.BadRequestError):
        msg = str(exc).lower()
        if any(k in msg for k in ("context_length_exceeded", "context too long", "maximum context")):
            return LLMError(LLMErrorType.CONTEXT_TOO_LONG, LLMErrorCategory.FALLBACK, str(exc), exc)
        if any(k in msg for k in ("content_filter", "content policy", "violat")):
            return LLMError(LLMErrorType.CONTENT_FILTERED, LLMErrorCategory.NOT_RETRYABLE, str(exc), exc)
        return LLMError(LLMErrorType.INVALID_REQUEST, LLMErrorCategory.NOT_RETRYABLE, str(exc), exc)

    if isinstance(exc, openai.APIStatusError):
        code = exc.status_code
        if code == 429:
            return LLMError(LLMErrorType.RATE_LIMIT, LLMErrorCategory.RETRYABLE, str(exc), exc)
        if code in (503, 529):
            return LLMError(LLMErrorType.MODEL_OVERLOADED, LLMErrorCategory.FALLBACK, str(exc), exc)
        if code >= 500:
            return LLMError(LLMErrorType.SERVICE_UNAVAILABLE, LLMErrorCategory.RETRYABLE, str(exc), exc)
        return LLMError(LLMErrorType.UNKNOWN, LLMErrorCategory.UNKNOWN, str(exc), exc)

    if isinstance(exc, openai.APIConnectionError):
        return LLMError(LLMErrorType.SERVICE_UNAVAILABLE, LLMErrorCategory.RETRYABLE, str(exc), exc)

    return LLMError(LLMErrorType.UNKNOWN, LLMErrorCategory.UNKNOWN, str(exc), exc)

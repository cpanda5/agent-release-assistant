import asyncio


class ToolErrorType:
    TIMEOUT = "ToolTimeout"
    NETWORK = "ToolNetwork"
    PARAM_INVALID = "ParamInvalid"
    RESOURCE_NOT_FOUND = "ResourceNotFound"
    PERMISSION_DENIED = "PermissionDenied"
    USER_REJECTED = "UserRejected"
    NOT_IMPLEMENTED = "NotImplemented"
    INTERNAL = "ToolInternalError"
    UNKNOWN = "Unknown"


class ToolErrorCategory:
    RETRYABLE = "retryable"       # framework retries silently; LLM never sees the failure
    PASS_TO_LLM = "pass_to_llm"  # send error as tool_result; LLM decides next step
    FATAL = "fatal"               # terminate the agent loop immediately


class ToolError(Exception):
    def __init__(self, error_type: str, category: str, message: str):
        self.error_type = error_type
        self.category = category
        super().__init__(message)


def classify_tool_error(exc: Exception) -> ToolError:
    if isinstance(exc, ToolError):
        return exc
    if isinstance(exc, asyncio.TimeoutError):
        return ToolError(ToolErrorType.TIMEOUT, ToolErrorCategory.RETRYABLE, str(exc) or "Tool execution timed out")
    if isinstance(exc, ConnectionError):
        return ToolError(ToolErrorType.NETWORK, ToolErrorCategory.RETRYABLE, str(exc))
    return ToolError(ToolErrorType.UNKNOWN, ToolErrorCategory.PASS_TO_LLM, str(exc))

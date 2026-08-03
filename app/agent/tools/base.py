from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolContext:
    """Execution context injected into every tool call — avoids threading IDs through params."""
    session_id: str
    trace_id: str
    release_id: int | None = None
    project_id: int | None = None
    iteration: int = 1


@dataclass
class ValidationResult:
    valid: bool
    error: str | None = None


@dataclass
class ToolResult:
    success: bool
    content: str          # returned to LLM verbatim (error message when success=False)
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict = field(default_factory=dict)  # tracing only, never sent to LLM
    should_retry: bool = False


class BaseTool(ABC):
    """
    All tools must inherit BaseTool and define the three class-level fields.
    The framework calls validate_params() then execute(); tools never need to
    know about retry logic or tracing.
    """
    name: str
    description: str
    parameters_schema: dict   # JSON Schema fed directly to the LLM
    timeout_seconds: int = 30
    require_confirmation: bool = False  # dangerous ops (e.g. git push) set this True

    @abstractmethod
    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        raise NotImplementedError

    def validate_params(self, params: dict) -> ValidationResult:
        try:
            import jsonschema
            jsonschema.validate(params, self.parameters_schema)
            return ValidationResult(valid=True)
        except ImportError:
            return ValidationResult(valid=True)
        except jsonschema.ValidationError as e:
            return ValidationResult(valid=False, error=e.message)
        except Exception as e:
            return ValidationResult(valid=False, error=str(e))

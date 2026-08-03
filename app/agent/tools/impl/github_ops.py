from app.agent.tools.base import BaseTool, ToolContext, ToolResult
from app.agent.tools.errors import ToolError, ToolErrorCategory, ToolErrorType


class GitHubOpsTool(BaseTool):
    """
    GitHub 操作：创建分支、提交代码、开 PR 等。
    Stub — will be implemented in Module 5.
    require_confirmation=True because push operations are irreversible.
    """
    name = "github_ops"
    description = (
        "执行 GitHub 操作：创建分支、提交代码、开 Pull Request、查询 PR 状态。"
        "涉及代码推送的操作需要用户确认后才会执行。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["create_branch", "create_pr", "get_pr_status", "list_branches"],
                "description": "要执行的 GitHub 操作",
            },
            "repo": {
                "type": "string",
                "description": "仓库名称，格式为 owner/repo",
            },
            "branch": {
                "type": "string",
                "description": "分支名（create_branch 时必填）",
            },
            "pr_title": {
                "type": "string",
                "description": "PR 标题（create_pr 时必填）",
            },
            "pr_body": {
                "type": "string",
                "description": "PR 描述正文（create_pr 时可选）",
            },
        },
        "required": ["operation", "repo"],
    }
    timeout_seconds = 30
    require_confirmation = True

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        # Executor never reaches here for require_confirmation=True tools
        # (executor short-circuits before calling execute).
        # This path is only hit in unit tests that bypass the executor.
        raise ToolError(
            ToolErrorType.NOT_IMPLEMENTED,
            ToolErrorCategory.PASS_TO_LLM,
            f"github_ops.{params.get('operation')} is not yet implemented. Coming in Module 5.",
        )

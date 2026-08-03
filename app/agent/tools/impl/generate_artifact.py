import json
import logging
import time

from app.agent.tools.base import BaseTool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class GenerateArtifactTool(BaseTool):
    """
    Saves a structured deliverable (requirement doc, tech design, API spec, etc.)
    and records it in the release context so subsequent LLM calls know it exists.
    """
    name = "generate_artifact"
    description = (
        "生成并保存一个交付物（需求文档、技术方案、API 设计、任务列表、代码片段等）。"
        "调用此工具将交付物持久化到 release 上下文，后续会话中 Agent 仍能感知到它的存在。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "artifact_type": {
                "type": "string",
                "enum": ["requirement_doc", "tech_design", "api_spec", "task_list", "code_snippet"],
                "description": "交付物类型",
            },
            "title": {
                "type": "string",
                "description": "交付物标题，简洁描述其内容",
            },
            "content": {
                "type": "string",
                "description": "交付物正文内容（Markdown 格式）",
            },
        },
        "required": ["artifact_type", "title", "content"],
    }
    timeout_seconds = 15

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        artifact_type: str = params["artifact_type"]
        title: str = params["title"]
        content: str = params["content"]

        artifact_id = f"{artifact_type}_{int(time.time())}"

        if context.release_id:
            try:
                from app.agent.memory.manager import memory_manager
                await memory_manager.update_release_context_with_artifact(
                    release_id=context.release_id,
                    session_id=context.session_id,
                    artifact={"type": artifact_type, "title": title, "artifact_id": artifact_id},
                )
            except Exception:
                logger.exception("Failed to update release context after artifact generation")

        return ToolResult(
            success=True,
            content=json.dumps(
                {
                    "artifact_id": artifact_id,
                    "artifact_type": artifact_type,
                    "title": title,
                    "status": "created",
                    "content_length": len(content),
                },
                ensure_ascii=False,
            ),
            metadata={
                "artifact_id": artifact_id,
                "artifact_type": artifact_type,
                "content_length": len(content),
            },
        )

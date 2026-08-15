import json
import logging
import time

from app.agent.tools.base import BaseTool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class GenerateArtifactTool(BaseTool):
    """
    Persists a structured deliverable (requirement doc, tech design, API spec,
    task list, code snippet) to the `artifact` table, then records a mention in
    the release context so subsequent sessions know it exists.
    """
    name = "generate_artifact"
    description = (
        "生成并保存一个交付物（需求文档、技术方案、API 设计、任务列表、代码片段等）。"
        "调用此工具会把交付物正文持久化到 artifact 表，并登记进 release 上下文，"
        "后续会话中 Agent 仍能感知到它的存在。修订已有交付物时可传 parent_artifact_id。"
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
            "parent_artifact_id": {
                "type": "string",
                "description": "可选：被修订的上一版交付物 id（uuid），用于串联版本链",
            },
        },
        "required": ["artifact_type", "title", "content"],
    }
    timeout_seconds = 15

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        artifact_type: str = params["artifact_type"]
        title: str = params["title"]
        content: str = params["content"]
        parent_uuid: str | None = params.get("parent_artifact_id")

        artifact_id: str | None = None
        version = 1
        persisted = False

        # ── Persist to the artifact table (needs a release_id + DB) ──────── #
        if context.release_id:
            try:
                from app.agent.artifacts import create_artifact

                source_trace = {
                    "session_uuid": context.session_id,
                    "trace_id": context.trace_id,
                    "iteration": context.iteration,
                }
                saved = await create_artifact(
                    release_id=context.release_id,
                    session_uuid=context.session_id,
                    artifact_type=artifact_type,
                    title=title,
                    content=content,
                    source_trace=source_trace,
                    parent_uuid=parent_uuid,
                )
                if saved:
                    artifact_id = saved["artifact_id"]
                    version = saved["version"]
                    persisted = True
            except Exception:
                logger.exception("Failed to persist artifact to DB")

            # Record a mention in release context (progress awareness).
            try:
                from app.agent.memory.manager import memory_manager

                await memory_manager.update_release_context_with_artifact(
                    release_id=context.release_id,
                    session_id=context.session_id,
                    artifact={"type": artifact_type, "title": title, "artifact_id": artifact_id or ""},
                )
            except Exception:
                logger.exception("Failed to update release context after artifact generation")

        # Fallback id when there is no DB / no release (keeps the tool usable).
        if artifact_id is None:
            artifact_id = f"{artifact_type}_{int(time.time())}"

        return ToolResult(
            success=True,
            content=json.dumps(
                {
                    "artifact_id": artifact_id,
                    "artifact_type": artifact_type,
                    "title": title,
                    "version": version,
                    "status": "created",
                    "content_length": len(content),
                },
                ensure_ascii=False,
            ),
            metadata={
                "artifact_id": artifact_id,
                "artifact_type": artifact_type,
                "version": version,
                "content_length": len(content),
                "persisted": persisted,
            },
        )

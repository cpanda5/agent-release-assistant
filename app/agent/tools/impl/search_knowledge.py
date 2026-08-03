from app.agent.tools.base import BaseTool, ToolContext, ToolResult


class SearchKnowledgeTool(BaseTool):
    """
    Searches the project knowledge base.
    Stub implementation — returns a placeholder until Module 4 adds real RAG.
    """
    name = "search_knowledge"
    description = (
        "搜索项目知识库，获取相关文档、架构决策、技术规范和历史决策。"
        "当需要了解项目背景、已有设计或技术约束时调用此工具。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索查询，描述你想了解的内容",
            },
            "top_k": {
                "type": "integer",
                "description": "返回的最相关结果数量",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["query"],
    }
    timeout_seconds = 10

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        # Module 4: replace with real vector search via project_memory.search()
        return ToolResult(
            success=True,
            content=(
                f"[知识库搜索 — 占位结果]\n"
                f"查询：'{query}'（top_k={top_k}）\n"
                "Module 4 接入 RAG 后此工具将返回真实的项目文档内容。"
            ),
            metadata={"query": query, "top_k": top_k, "stub": True},
        )

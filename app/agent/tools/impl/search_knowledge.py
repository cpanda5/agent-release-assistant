from app.agent.tools.base import BaseTool, ToolContext, ToolResult


class SearchKnowledgeTool(BaseTool):
    """
    Searches the project knowledge base via release/project-scoped hybrid retrieval
    (dense + BM25, RRF fusion, reranked). Results carry a `[filename#chunk]` citation
    so the agent can attribute claims back to a specific document chunk.
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
        from app.agent.rag import retrieve

        query = params.get("query", "")
        top_k = params.get("top_k", 5)

        docs = await retrieve(
            query,
            release_id=context.release_id,
            project_id=context.project_id,
            top_k=top_k,
        )
        if not docs:
            return ToolResult(
                success=True,
                content=f"知识库中未找到与“{query}”相关的内容。",
                metadata={"query": query, "top_k": top_k, "hits": 0},
            )

        blocks: list[str] = []
        citations: list[str] = []
        for d in docs:
            source = d.metadata.get("source", "unknown")
            chunk_id = str(d.metadata.get("chunk_id", ""))[:8]
            citation = f"{source}#{chunk_id}"
            citations.append(citation)
            blocks.append(f"[{citation}] (score={d.metadata.get('score', 0):.3f})\n{d.page_content}")

        return ToolResult(
            success=True,
            content="\n\n---\n\n".join(blocks),
            metadata={"query": query, "top_k": top_k, "hits": len(docs), "citations": citations},
        )

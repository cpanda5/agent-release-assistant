import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool

from app.agent.tools.base import BaseTool, ToolContext

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Maps tool names to tool instances. Singleton used by executor and LLM client.

    Adding a new tool: instantiate it, call registry.register(MyTool()). Zero
    changes needed to framework code.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._lc_tools: list[StructuredTool] | None = None

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        self._lc_tools = None  # invalidate cached LangChain adapters
        logger.info("Registered tool: %s", tool.name)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def to_llm_tools_spec(self) -> list[dict]:
        """Generate the `tools` list accepted by the OpenAI-compatible chat API."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters_schema,
                },
            }
            for t in self._tools.values()
        ]

    def to_langchain_tools(self) -> list[StructuredTool]:
        """
        Expose every registered tool as a LangChain ``StructuredTool``.

        These are what we ``bind_tools`` onto the ChatModel — the JSON schema is
        identical to ``to_llm_tools_spec``. Each adapter is also independently
        invocable: its coroutine reconstructs a ``ToolContext`` from the run's
        config metadata and delegates to the underlying tool. In the normal graph
        flow, however, execution goes through ``tool_executor`` (retry /
        confirmation / tracing), not through these coroutines.
        """
        if self._lc_tools is None:
            self._lc_tools = [self._to_langchain_tool(t) for t in self._tools.values()]
        return self._lc_tools

    @staticmethod
    def _to_langchain_tool(tool: BaseTool) -> StructuredTool:
        async def _coroutine(config: RunnableConfig = None, **kwargs) -> str:
            md = (config or {}).get("metadata", {}) or {}
            ctx = ToolContext(
                session_id=str(md.get("session_id", "")),
                trace_id=str(md.get("trace_id", "")),
                release_id=md.get("release_id"),
                project_id=md.get("project_id"),
            )
            result = await tool.execute(kwargs, ctx)
            return result.content

        return StructuredTool(
            name=tool.name,
            description=tool.description,
            args_schema=tool.parameters_schema,
            coroutine=_coroutine,
        )


tool_registry = ToolRegistry()

from app.agent.tools.registry import tool_registry


def register_all_tools() -> None:
    """Called once at application startup to populate the tool registry."""
    from app.agent.tools.impl.search_knowledge import SearchKnowledgeTool
    from app.agent.tools.impl.generate_artifact import GenerateArtifactTool
    from app.agent.tools.impl.github_ops import GitHubOpsTool

    tool_registry.register(SearchKnowledgeTool())
    tool_registry.register(GenerateArtifactTool())
    tool_registry.register(GitHubOpsTool())


__all__ = ["tool_registry", "register_all_tools"]

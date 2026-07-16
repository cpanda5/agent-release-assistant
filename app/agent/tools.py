# Phase 4: Tool calling layer
# Planned tools: GitHub (create branch / commit / PR), DB query, code exec.
# Each tool will be registered here and dispatched by core.py.

TOOL_REGISTRY: dict[str, callable] = {}


def register_tool(name: str):
    def decorator(fn):
        TOOL_REGISTRY[name] = fn
        return fn
    return decorator


async def execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name not in TOOL_REGISTRY:
        raise NotImplementedError(f"Tool '{tool_name}' is not registered.")
    return await TOOL_REGISTRY[tool_name](**tool_input)

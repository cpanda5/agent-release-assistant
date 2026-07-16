# Phase 5: Memory system
# Will store conversation summaries, agent-generated artifacts,
# and cross-session context per release / project.


async def update_memory(session_id: str, message: str, response: str) -> None:
    """Persist turn to memory store. No-op in Phase 1."""
    pass


async def load_memory(session_id: str) -> list[dict]:
    """Load previous turns for a session. Returns empty list in Phase 1."""
    return []

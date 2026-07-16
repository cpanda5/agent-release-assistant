from uuid import uuid4

from app.agent.llm import call_llm
from app.schemas.agent import AgentRunRequest, AgentRunResponse, UsageInfo


class AgentRunner:
    """
    Agent execution loop. Phase evolution:
      Phase 1: observe → LLM → respond
      Phase 2: observe → RAG retrieve → LLM → respond
      Phase 4: observe → RAG → LLM ↔ tool calling → respond
      Phase 5: + memory update after each turn
    """

    async def run(self, request: AgentRunRequest) -> AgentRunResponse:
        session_id = request.session_id or str(uuid4())

        # Phase 1: direct LLM call
        messages = self._build_messages(request)
        response_text, usage_dict = await call_llm(messages)

        return AgentRunResponse(
            response=response_text,
            session_id=session_id,
            usage=UsageInfo(**usage_dict),
        )

    def _build_messages(self, request: AgentRunRequest) -> list[dict]:
        content = request.message
        if request.release_id:
            content = f"[Release: {request.release_id}]\n\n{content}"
        if request.project_id and not request.release_id:
            content = f"[Project: {request.project_id}]\n\n{content}"
        return [{"role": "user", "content": content}]

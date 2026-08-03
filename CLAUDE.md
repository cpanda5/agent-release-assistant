# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Setup (uv required):**
```bash
uv sync
cp .env.example .env   # then fill in DASHSCOPE_API_KEY and MySQL creds
```

**Run dev server:**
```bash
uv run uvicorn app.main:app --reload --port 8000
```

**API docs:** `http://localhost:8000/docs`

There are no tests yet. When adding tests, use `uv run pytest`.

## Architecture

An AI SDLC Agent (研发交付智能助手) built on **LangChain + LangGraph**. LangGraph
orchestrates the workflow; LangChain provides the LLM / tool / retriever
abstractions. The former hand-rolled infra (retry, fallback, tracing, memory
compression) is preserved and wired into the LangChain layer. The full vision is
a FastAPI service consumed by a SpringBoot business layer via MQ; currently it
runs standalone.

### Request flow

```
POST /agent/chat  (or /agent/run legacy)
  → router.py → core.invoke_agent() → agent_graph.ainvoke()   (LangGraph)
      load_memory → planner → rag → agent ⇄ tools → persist_memory → END
  → each LLM call goes through llm/gateway.py (LLMGateway)
      → with_retry()        — backoff for RETRYABLE errors (reused)
      → model fallback      — FALLBACK errors downgrade to fallback_model (reused)
      → ChatOpenAI.ainvoke  — LangChain model, DashScope-compatible endpoint
      → trace_callback      — MySQLTraceCallbackHandler writes llm_trace

POST /agent/chat/stream (SSE) → llm_gateway.astream()  — single turn, separates
      qwen3.7 reasoning_content from answer content (no tool loop)
```

### Key modules

| Path | Role |
|---|---|
| `app/config.py` | `Settings` + `ALLOWED_MODELS` + `resolve_model()`. Reads `.env`. |
| `app/main.py` | FastAPI app + lifespan: validates API key, optional MySQL pool, registers tools. |
| `app/agent/router.py` | Endpoints: `/agent/run` (legacy), `/agent/chat`, `/agent/chat/stream`, `/agent/session/close`, `/agent/health`. |
| `app/agent/core.py` | `invoke_agent()` / `AgentRunner` — thin adapter that invokes the compiled graph. |
| `app/agent/graph/` | LangGraph workflow: `state.py` (`AgentState`), `nodes.py` (load_memory/planner/rag/agent/tools/persist_memory), `builder.py` (`agent_graph`). |
| `app/agent/llm/gateway.py` | `LLMGateway` — LangChain `ChatOpenAI` + reused retry/fallback + model selection. `SYSTEM_PROMPT` lives here. |
| `app/agent/llm/callbacks.py` | `MySQLTraceCallbackHandler` — adapts LangChain callback lifecycle → `llm_trace`. |
| `app/agent/llm/errors.py` | `classify_error()` maps OpenAI exceptions → `LLMError` (RETRYABLE / FALLBACK / NOT_RETRYABLE). |
| `app/agent/llm/retry.py` | `with_retry()` — exponential backoff + jitter, respects `retry-after`. |
| `app/agent/llm/tracer.py` | `write_trace()` — async MySQL insert into `llm_trace`; never raises. |
| `app/agent/tools/` | `BaseTool` + `registry` (`to_langchain_tools()`) + `executor` (retry/confirmation/tool_trace). Impls in `tools/impl/`. |
| `app/agent/memory/` | 3-layer memory: `working` (in-proc), `release` (MySQL sessions/messages/summary/context), `project` (RAG bridge). `manager.py` orchestrates + token budget + compression. |
| `app/agent/rag/` | `ReleaseAwareRetriever` (LangChain `BaseRetriever`) — **stub**; Qdrant + metadata filter deferred. |
| `app/db.py` | `aiomysql` pool. MySQL is optional — if unset/unreachable, tracing + persistence silently skip. |
| `app/schemas/agent.py` | Pydantic request/response models; requests carry `model` (validated), `user_id`, `enable_thinking`. |
| `schema.sql` | DDL for sessions/messages/session_summary/release_context/llm_trace/tool_call_trace. |
| `templates/` | YAML prompt templates, loaded via `llm/prompt.load_template()` (`@lru_cache`). |

### LLM provider

Uses **Alibaba DashScope / 百炼** via the OpenAI-compatible endpoint in
`settings.llm_base_url` (a dedicated `*.maas.aliyuncs.com` deployment). Three
user-selectable models: `qwen3.7-flash` (default), `qwen3.7-plus` (fallback),
`qwen3.7-max`. Deep-thinking (`enable_thinking` → `reasoning_content`) is
configurable and off by default. LangChain `ChatOpenAI` is the client; the API
key is read from `.env`.

### Error handling pattern

`classify_error()` converts OpenAI SDK exceptions to `LLMError(error_type, category, ...)`. The gateway retries RETRYABLE errors and downgrades to `fallback_model` on FALLBACK-category errors (503, 529, context-too-long). The router maps `LLMErrorType` → HTTP status.

### Prompt templates

Templates live in `templates/<name>/v1.yaml`. Use `load_template(name, version)` to load; call `.render(**kwargs)` to get a messages list. Unknown `{placeholders}` are left as-is (safe formatting). Pass `prompt_template` and `prompt_template_version` in chat requests to override the default system prompt.

### Memory & scoping

Sessions carry a `release_id` (and `user_id`): `sessions(session_id, release_id, user_id, created_at)` and `messages(message_id, session_id, role, content, created_at, ...)`. Memory is three-tiered — current session (working memory), current release (session summaries + release_context + release-scoped RAG docs), and project (project-scoped RAG). RAG documents are ingested with `release_id`/`project_id` metadata so retrieval is filtered per release and never leaks across releases.

### Next steps

RAG is a stub (`ReleaseAwareRetriever` returns `[]`): the Qdrant vector store, DashScope embeddings, and the MD ingestion pipeline (chunk → tag with `release_id` → upsert) are a separate follow-up design.

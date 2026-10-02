# AI SDLC Agent - 研发交付智能助手

面向软件开发生命周期（SDLC）的 AI Agent 系统，基于 LangChain + LangGraph 构建，支持在 Release 工作空间中完成需求分析、技术设计、代码生成、文档沉淀和 GitHub PR 提交等研发交付流程。

## 项目背景

传统研发流程中，需求分析、技术设计、代码编写、文档沉淀等环节割裂，知识难以复用。本项目构建了一个 AI Agent 系统，让 AI 深度参与软件开发生命周期，实现：

- **Release 级工作空间隔离**：每个版本独立上下文，知识不泄漏
- **四层记忆体系**：项目、发布、会话摘要、会话历史，支撑长短期记忆
- **RAG 知识检索**：混合检索（dense + sparse）+ 重排序，知识精准召回
- **工具调用系统**：GitHub 操作、知识搜索、文档生成等工具编排
- **LLM 网关**：重试、降级、追踪、多模型切换

## 核心功能

### 1. LangGraph 工作流编排

```
用户输入 → 加载记忆 → 知识检索 → 规划 → Agent ⇄ 工具 → 持久化记忆
```

通过 LangGraph 状态机编排完整工作流，每个节点可独立测试和替换。

### 2. 四层记忆系统

| 层级 | 存储 | 作用 |
|------|------|------|
| Project Context | `project` 表 | 项目级架构知识、技术约束 |
| Release Context | `release_context` 表 | 版本进度、待办事项 |
| Session Summary | `session_summary` 表 | 历史会话压缩，带时间水印 |
| Session History | `message` 表 | 原始对话记录 |

会话关闭时自动触发摘要生成，合并到 Release Context，实现知识沉淀。

### 3. RAG 知识检索

- **混合检索**：bge-m3 向量 + FastEmbed BM25 稀疏，RRF 融合
- **重排序**：bge-reranker-v2-m3 精排 Top-K
- **多格式解析**：文本、表格（PaddleOCR）、图片、音频（FunASR）
- **元数据过滤**：按 `release_id` / `project_id` 隔离，防止知识泄漏
- **存储**：Qdrant（向量）+ MinIO（原文）+ MySQL（元数据）

### 4. LLM 网关

- **多模型支持**：qwen3.7-flash（默认）、qwen3.7-plus（降级）、qwen3.7-max
- **错误分类**：RETRYABLE（重试）/ FALLBACK（降级）/ NOT_RETRYABLE
- **重试策略**：指数退避 + 抖动，尊重 `retry-after`
- **可观测性**：每次调用写入 `llm_trace`，支持成本追踪

### 5. 工具系统

- **GitHub 操作**：创建分支、提交代码、开启 PR
- **知识搜索**：RAG 检索封装
- **文档生成**：需求分析、技术设计、代码生成等模板
- **工具执行**：重试、确认、追踪（`tool_call_trace`）

### 6. 认证与项目 API

- **用户认证**：PBKDF2 哈希 + HMAC 签名 Token（无 JWT 依赖）
- **项目/发布管理**：CRUD + 所有权隔离 + 软删除
- **会话管理**：Session 关联 Release，支持多会话并行

## 技术栈

| 类别 | 技术 |
|------|------|
| **后端框架** | FastAPI + Uvicorn |
| **Agent 编排** | LangGraph（状态机）+ LangChain（LLM/工具抽象）|
| **LLM 服务** | 阿里云百炼（DashScope），OpenAI 兼容接口 |
| **向量数据库** | Qdrant（dense + sparse，RRF 融合）|
| **对象存储** | MinIO（原文存储）|
| **关系数据库** | MySQL（aiomysql 异步连接池）|
| **Embedding** | bge-m3（dense）+ FastEmbed（BM25 sparse）|
| **重排序** | bge-reranker-v2-m3（SiliconFlow）|
| **文档解析** | Unstructured / PaddleOCR PP-Structure / FunASR |
| **依赖管理** | uv + pyproject.toml |

## 系统架构

```mermaid
graph TB
    subgraph Client
        FE[前端 / SpringBoot]
    end

    subgraph FastAPI Service
        Router[路由层<br/>/agent/chat, /knowledge/*]
        Core[Agent Core<br/>invoke_agent]
        
        subgraph LangGraph Workflow
            LoadMem[加载记忆]
            Planner[规划节点]
            RAG[RAG 检索]
            Agent[Agent 节点]
            Tools[工具执行]
            Persist[持久化记忆]
        end
        
        subgraph LLM Gateway
            Retry[重试 + 降级]
            Fallback[模型切换]
            Tracer[调用追踪]
        end
        
        subgraph Memory System
            Project[Project Context]
            Release[Release Context]
            Summary[Session Summary]
            History[Session History]
        end
        
        subgraph RAG System
            Ingest[文档解析<br/>文本/表格/图片/音频]
            Embed[Embedding<br/>bge-m3 + BM25]
            Retrieve[混合检索<br/>RRF + 重排序]
        end
        
        subgraph Tool Layer
            GitHub[GitHub 操作]
            Search[知识搜索]
            Artifact[文档生成]
        end
    end

    subgraph Storage
        MySQL[(MySQL<br/>sessions/messages/trace)]
        Qdrant[(Qdrant<br/>向量 + 稀疏)]
        MinIO[(MinIO<br/>原文存储)]
    end

    subgraph External
        LLM[阿里云百炼<br/>qwen3.7]
        GH[GitHub API]
        SF[SiliconFlow<br/>Embedding/Rerank]
    end

    FE --> Router
    Router --> Core
    Core --> LoadMem --> Planner --> RAG --> Agent
    Agent --> Tools --> Persist
    Agent -.需要工具.-> Tools
    Tools -.执行完成.-> Agent
    
    Agent --> LLM Gateway
    LLM Gateway --> LLM
    
    LoadMem --> Memory System
    Persist --> Memory System
    Memory System --> MySQL
    
    RAG --> Retrieve
    Retrieve --> Qdrant
    Retrieve --> MySQL
    
    Ingest --> Embed --> Qdrant
    Ingest --> MinIO
    
    Tools --> GitHub
    Tools --> Search
    Tools --> Artifact
    Search --> Retrieve
    
    Embed -.调用.-> SF
    Retrieve -.重排序.-> SF
    GitHub --> GH
```

## 关键设计

### 1. LangGraph 状态机

使用 LangGraph 编排工作流，状态在节点间流转：

```python
State: AgentState
  - messages: 对话历史
  - memory_context: 记忆上下文
  - rag_context: 检索到的知识
  - tool_calls: 工具调用记录

Nodes:
  - load_memory: 从 MySQL 加载四层记忆
  - planner: 规划下一步动作
  - rag: 检索相关知识
  - agent: LLM 决策（是否调用工具）
  - tools: 执行工具调用
  - persist_memory: 持久化记忆 + 触发摘要
```

### 2. 错误处理与降级

```python
classify_error() 将 OpenAI 异常映射为:
  - RETRYABLE: 429, 超时, 网络错误 → 指数退避重试
  - FALLBACK: 503, 529, 上下文过长 → 降级到 fallback_model
  - NOT_RETRYABLE: 401, 403, 400 → 直接返回错误
```

### 3. RAG 混合检索

```
查询 → bge-m3 向量检索 + BM25 稀疏检索
     → RRF (Reciprocal Rank Fusion) 融合
     → bge-reranker 重排序
     → Top-K 返回
```

每个 chunk 可追溯到原始文档，支持元数据过滤（`release_id` / `project_id`）。

### 4. 记忆压缩与沉淀

```
Session 对话 > 阈值 → 后台任务生成摘要
                   → 写入 session_summary (带版本号)
Session 关闭 → 合并摘要到 release_context
             → 作为下次会话的背景知识
```

### 5. 知识隔离

RAG 文档入库时携带 `release_id` / `project_id` 元数据，检索时强制过滤，确保：
- 同一 Release 内的知识可检索
- 跨 Release 的知识不泄漏
- Project 级知识全局可复用

## 快速启动

### 1. 环境准备

```bash
# 安装依赖（需要 uv）
uv sync

# 可选：安装 RAG 重依赖（文档解析）
uv sync --extra rag

# 配置环境变量
cp .env.example .env
# 填写 DASHSCOPE_API_KEY, SILICONFLOW_API_KEY, MySQL 配置

# 启动 RAG 基础设施
docker compose up -d qdrant minio
```

### 2. 启动服务

```bash
uv run uvicorn app.main:app --reload --port 8000
```

访问 `http://localhost:8000/docs` 查看 API 文档。

### 3. 主要接口

| 接口 | 说明 |
|------|------|
| `POST /agent/chat` | Agent 对话（完整工作流）|
| `POST /agent/chat/stream` | Agent 对话（SSE 流式）|
| `POST /knowledge/upload` | 上传知识文档 |
| `POST /knowledge/search` | 知识检索 |
| `POST /auth/register` | 用户注册 |
| `POST /auth/login` | 用户登录 |
| `GET /projects` | 项目列表 |
| `POST /projects` | 创建项目 |

## 项目结构

```
app/
├── agent/
│   ├── graph/          # LangGraph 工作流
│   ├── llm/            # LLM 网关（重试/降级/追踪）
│   ├── memory/         # 四层记忆系统
│   ├── rag/            # RAG 检索与入库
│   ├── tools/          # 工具系统
│   ├── router.py       # Agent 路由
│   └── core.py         # Agent 入口
├── auth/               # 用户认证
├── project/            # 项目/发布管理
├── schemas/            # Pydantic 模型
├── config.py           # 配置管理
├── db.py               # MySQL 连接池
└── main.py             # FastAPI 应用入口
```

## 设计亮点

1. **Release 级上下文隔离**：每个版本独立工作空间，知识不污染
2. **四层记忆体系**：项目、发布、会话、历史，支撑长短期记忆
3. **混合 RAG**：dense + sparse + reranker，检索精准度高
4. **LLM 网关抽象**：重试、降级、追踪、多模型切换，生产级健壮性
5. **工具编排**：LangGraph 状态机驱动，工具调用可追溯
6. **知识沉淀**：会话摘要自动合并到 Release Context，持续积累
7. **可观测性**：每次 LLM 调用、工具执行写入 trace，支持成本分析

## License

MIT

# 🚀 AI Agent Release Assistant（研发交付智能助手）设计文档

---

# 1. 项目概述

本项目旨在构建一个面向研发交付流程的 AI Agent 系统，模拟“AI 参与软件开发生命周期（SDLC）”的工作方式。

系统支持用户在“Release 工作空间”中完成：

* 需求分析
* 技术设计
* 代码生成
* 文档沉淀
* GitHub PR 提交
* 知识库构建与复用

---

# 2. 核心目标

## 2.1 产品目标

构建一个类似：

* Claude Code
* Cursor Agent
* Copilot Workspace

但更偏向：

> **企业研发交付流程自动化 Agent**

---

## 2.2 技术目标

覆盖以下能力：

* LLM Agent（Planner + Executor）
* RAG 知识库系统
* Tool Calling / MCP
* 多层知识管理（Project / Release / Artifact）
* 异步任务系统（MQ）
* GitHub 自动化
* 可观测性（Trace / Log）

---

# 3. 总体架构设计

```
Frontend（Vue）
   ↓
SpringBoot（业务层 / Release管理 / API Gateway）
   ↓
MQ（RabbitMQ）
   ↓
FastAPI（Agent Core）
   ├── LLM Orchestrator
   ├── RAG Service
   ├── Tool Calling Layer
   ├── Memory System
   └── Artifact Generator
   ↓
外部系统：
- GitHub
- Vector DB
- OSS / MinIO
- MySQL / MongoDB
```

---

# 4. 核心业务模型设计

## 4.1 Project（项目）

> 顶层业务容器

```
Project
 ├── name
 ├── description
 ├── global_knowledge_base
 └── releases[]
```

作用：

* 存储长期架构知识
* 通用设计规范
* 跨 release 复用信息

---

## 4.2 Release（核心工作单元）

> 一个可交付版本的 AI 工作空间

```
Release
 ├── id
 ├── project_id
 ├── name
 ├── status
 ├── documents[]
 ├── sessions[]
 ├── artifacts[]
 └── dependencies[]
```

特点：

* 一个 release = 一个 AI 工作上下文
* 支持多个 release 并行

---

## 4.3 Artifact（AI产物）

```
Artifact
 ├── type (design / code / api / test / pr)
 ├── content
 ├── source_trace
 ├── created_by_agent
```

---

## 4.4 Knowledge Unit（知识单元）

```
Knowledge
 ├── content
 ├── embedding
 ├── scope (global / release)
 ├── metadata
```

---

# 5. 知识库设计（重点）

## 5.1 统一向量库设计

所有知识进入统一 vector store：

```
vector_db
```

通过 metadata 区分：

```
project_id
release_id
artifact_type
scope
```

---

## 5.2 检索策略

### Release级：

```
filter:
project_id + release_id
```

### Project级：

```
filter:
project_id
```

### 跨Release：

```
filter:
project_id
AND scope != current_release
```

---

## 5.3 知识来源

* PRD
* API文档
* Agent生成内容
* GitHub代码
* 用户对话总结

---

# 6. Agent 系统设计（核心）

## 6.1 Agent定义

Agent =

> LLM + 状态机 + 工具调用 + 记忆系统 + 执行控制器

---

## 6.2 Agent执行循环

```
while not done:

    1. observe context
    2. retrieve knowledge (RAG)
    3. plan next step
    4. tool calling (if needed)
    5. update memory
```

---

## 6.3 Agent能力

* 需求分析
* 任务拆解
* 代码生成
* 文档生成
* Git操作
* API设计
* 数据库设计

---

# 7. MQ异步任务系统设计

## 7.1 MQ用途

### 1️⃣ Agent任务执行

```
agent_task_queue
```

---

### 2️⃣ 文档解析

```
document_parse_queue
```

---

### 3️⃣ 工具执行

```
tool_execution_queue
```

---

## 7.2 任务流

```
SpringBoot
   ↓
MQ
   ↓
FastAPI Worker
   ↓
LLM / RAG / Tools
   ↓
GitHub / DB
```

---

# 8. FastAPI 设计（AI核心服务）

## 8.1 模块结构

```
agent/
 ├── router.py
 ├── core.py
 ├── llm.py
 ├── rag.py
 ├── tools.py
 ├── memory.py
 └── executor.py
```

---

## 8.2 核心接口

### Agent入口

```
POST /agent/run
```

---

## 8.3 最小Agent实现（MVP）

```
用户输入 → LLM → 返回结果
```

逐步升级：

* tool calling
* memory
* RAG
* workflow

---

# 9. SpringBoot设计（业务层）

## 9.1 职责

SpringBoot 不负责 AI，只负责：

* Project / Release 管理
* 用户系统
* Session管理
* API Gateway
* MQ投递

---

## 9.2 核心接口

```
POST /release/create
POST /release/{id}/chat
POST /release/{id}/task
```

---

## 9.3 请求流

```
Frontend → SpringBoot → FastAPI
```

---

# 10. GitHub 集成（Tool Layer）

Agent可调用：

* create branch
* commit code
* open PR

通过：

```
tool_execution_queue
```

---

# 11. 系统演进路线（非常重要）

## Phase 1（MVP）

* FastAPI 单 Agent
* SpringBoot 转发
* 无 MQ

---

## Phase 2

* 加 RAG
* 加知识库

---

## Phase 3

* 引入 MQ（异步Agent）

---

## Phase 4

* Tool Calling（GitHub / DB）

---

## Phase 5

* Memory + Artifact System

---

## Phase 6

* Model Routing（多模型调度）
* Observability（trace + cost）

---

# 12. 技术亮点总结（面试核心）

* Agent Workflow System
* 多层知识体系（Project / Release / Artifact）
* RAG + 向量检索
* Tool Calling / MCP思想
* MQ驱动异步Agent执行
* GitHub自动化交付
* 可观测AI系统

---

# 13. 项目本质定位

本项目不是：

> ❌ AI聊天系统

也不是：

> ❌ 代码生成工具

而是：

> ✅ 面向研发交付流程的 AI SDLC Agent 系统

---

# 14. 最小启动建议

第一步只做：

```
FastAPI /agent/run
→ 调用LLM
→ 返回结果
```

再逐步扩展系统。

---

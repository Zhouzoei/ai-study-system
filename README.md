# AI 学习系统

一个基于 **RAG（检索增强生成）** 的个性化学习助手。支持多轮对话、知识图谱增强检索、SM-2 间隔重复复习与自适应出题。

采用分层架构：**Facade（Pipeline）→ Service（4 个领域服务）→ Engine（30+ 引擎模块）**，包含 3 个确定性 Bug 修复（已修复）、~8000 行 Python 代码。

---

## 目录

- [快速开始](#快速开始)
- [系统架构](#系统架构)
- [核心特性](#核心特性)
- [API 密钥](#api-密钥)
- [项目结构](#项目结构)
- [技术栈](#技术栈)
- [运行测试](#运行测试)
- [Docker 部署](#docker-部署)
- [RAG 评估结果](#rag-评估结果)

---

## 快速开始

```bash
git clone <repo-url>
cd QAsystem

# 创建虚拟环境
python -m venv .venv

# 激活
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 配置 API Key
cp .env.example .env
# 编辑 .env，填入密钥后：
python app.py
```

浏览器访问 `http://127.0.0.1:7861`

---

## 系统架构

```
用户 → Streamlit UI
         │
         ▼
  EnhancedRAGPipeline (Facade)
         │
         ▼
  ┌──────────────────────────────────────────────┐
  │           LearningLoop (调度层)               │
  │  Phase 1:  知识状态预检 (薄弱点/到期复习)      │
  │  Phase 1.5: Graph RAG 实体提取 + 查询扩展     │
  │  Phase 2:  路由 → Orchestrator / Agents      │
  │  Phase 2.5: Self-RAG 质量检查 + 可选重生成    │
  │  Phase 3:  LearnerModel 事件记录 + 掌握度更新 │
  └──────────────────────────────────────────────┘
         │
  ┌──────┬──────┬──────┬──────┐
  │      │      │      │      │
  ▼      ▼      ▼      ▼      ▼
  检索    知识   进度    Agent  Learner
  Service Graph Service Service Model
```

### 领域服务

| 服务 | 职责 | 核心模块 |
|------|------|---------|
| **RetrievalService** | 文档分块、混合检索、重排序、评估 | `HybridRetriever`, `CrossEncoderReranker`, `MMRReranker`, `RAGASEvaluator` |
| **KnowledgeService** | 知识图谱构建与查询、知识蒸馏、学习上下文 | `KnowledgeGraphBuilder`, `KnowledgeDistiller`, `LearningContextBuilder` |
| **ProgressService** | SM-2 间隔重复、学习进度、复习提醒 | `ProgressTracker`, `LearningPlanner`, `LearningReminder`, `Analytics` |
| **AgentService** | 意图路由、Agent 生成、工具注册、事件总线 | `IntentRouter`, `OrchestratorAgent`, `QuizAgent`, `SummaryAgent`, `ReviewAgent` |

### 学习闭环

```
                  ┌──────────────────────────────┐
                  │  用户提问 / 交互              │
                  └──────────────┬───────────────┘
                                 ▼
    ┌─────────────────────────────────────────────┐
    │  LearningLoop.Process()                      │
    │                                              │
    │  ① 知识状态预检                               │
    │     ├─ LearnerModel 查薄弱点 → 针对性出题      │
    │     ├─ SM-2 到期 → 间隔复习                   │
    │     └─ 正常 → 进入检索                        │
    │                                              │
    │  ② Graph RAG 增强                            │
    │     ├─ LLM 提取关键实体                       │
    │     ├─ KG 查实体描述 + 关联关系               │
    │     └─ 扩展查询词                             │
    │                                              │
    │  ③ 路由                                      │
    │     ├─ QA → Orchestrator (ReAct 循环)        │
    │     ├─ Quiz → QuizAgent                       │
    │     ├─ Summary → SummaryAgent                 │
    │     └─ Review → ReviewAgent                   │
    │                                              │
    │  ④ Self-RAG 质量检查                          │
    │     ├─ 完整性 / 依据性 / 有用性打分            │
    │     └─ 低于阈值 → 带改进提示重生成             │
    │                                              │
    │  ⑤ 后处理                                    │
    │     ├─ LearnerModel.emit() 记录事件            │
    │     └─ ProgressTracker 更新掌握度              │
    └─────────────────────────────────────────────┘
```

---

## 核心特性

### 💬 智能问答

- **ReAct 循环**：Orchestrator 通过 ReAct（推理+行动）循环调用多个工具（知识检索、网络搜索、进度查询、笔记记录）逐步回答问题
- **检索增强**：BM25 + 向量混合检索 → RRF 融合 → MMR 多样性重排
- **查询改写**：自动扩展同义查询、生成 HyDE 伪文档
- **流式输出**：LLM 流式逐 token 生成回答

### 🧪 Graph RAG

- 对用户问题自动提取关键实体
- 从知识图谱查询实体描述和关联关系（邻居节点）
- 将 KG 上下文注入生成 prompt，提升回答的准确性
- 实体模糊匹配（向量 + LLM 双重判断同义概念）

### 🔍 Self-RAG

- **生成前**：对每个检索段落进行相关性打分（LLM 二分类），低分段落自动丢弃
- **生成后**：对 AI 回答进行完整性/依据性/有用性三维度评估
- 低于质量阈值时自动带改进提示重新生成

### 🔗 知识图谱

- 文档上传后自动抽取实体和关系（LLM 驱动）
- 存储实体描述、类型、关联关系、属性
- 支持向量搜索、名称精确查询、模糊 alias 匹配、多跳路径查询
- 关联复习（基于 KG 邻居生成复习内容）

### 📈 SM-2 间隔重复

- 5 级掌握度：未知 → 接触 → 熟悉 → 熟练 → 掌握
- 自动计算下次复习时间（ease factor 动态调整）
- 到期自动提醒 + 个性化复习内容生成
- 薄弱点自动标记 + 针对性出题

### 🎯 自适应 Agent

| 意图 | 处理器 | 说明 |
|------|--------|------|
| QA | OrchestratorAgent | ReAct 循环，多工具协作回答 |
| Quiz | QuizAgent | 选择题/判断/填空/简答，支持自适应出题和风格化出题 |
| Summary | SummaryAgent | 按主题/章节/文档多粒度总结 |
| Review | ReviewAgent | 间隔复习/薄弱点复习/关联复习 |
| Explain | OrchestratorAgent | 通俗解释复杂概念 |
| Compare | OrchestratorAgent | 对比两个或多个概念 |
| Tutor | TutorAgent | 生成学习路径建议 |

### 📊 学习仪表盘

- 侧边栏实时显示掌握度进度条、薄弱点数量、待复习数量
- Plotly 掌握度分布图和复习活动热力图
- 知识图谱可视化
- 后台 Agent 自动检测到期复习和长时间未学习

### 🔐 安全设计

- `.env` 被 `.gitignore` 和 `.dockerignore` 双重阻止进入版本控制和 Docker 镜像
- Dockerfile 采用逐目录精确 COPY，不复制整个构建上下文
- Circuit Breaker 熔断机制：LLM 连续失败自动降级
- 优雅降级：无 LLM 时降级为纯检索模式

---

## API 密钥

| 服务 | 用途 | 必填 | 获取地址 |
|------|------|------|---------|
| DeepSeek | LLM 推理 | ✅ | [platform.deepseek.com](https://platform.deepseek.com/) |
| DashScope | 文本嵌入 | ✅ | [dashscope.aliyun.com](https://dashscope.aliyun.com/) |
| Qdrant Cloud | 向量数据库 | ✅ | [cloud.qdrant.io](https://cloud.qdrant.io/) |
| Tavily | 网络搜索（可选） | ❌ | [tavily.com](https://tavily.com/) |

### 最小配置 `.env`

```ini
LLM_API_KEY=sk-your-deepseek-key
EMBED_API_KEY=sk-your-dashscope-key
QDRANT_URL=https://your-instance.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-key
```

---

## 项目结构

```
├── app.py                     # 入口：初始化 Pipeline + 启动 Streamlit
├── config.py                  # 统一配置（环境变量 → 类属性）
├── requirements.txt           # Python 依赖
├── Dockerfile                 # Docker 镜像构建（逐目录 COPY）
├── docker-compose.yml         # Docker Compose 编排
├── .env.example               # 环境变量模板（不含真实密钥）
├── .dockerignore              # Docker 构建上下文过滤
│
├── ui/
│   └── app.py                 # Streamlit 界面（对话/文档/出题/进度/图谱）
│
├── engines/                   # 引擎层（30+ 模块）
│   ├── pipeline.py            # Facade: 注入 4 个领域服务 + GraphRAG + SelfRAG
│   ├── learning_loop.py       # 学习闭环调度器（预检→增强→路由→质检→更新）
│   ├── graph_rag.py           # 知识图谱增强检索（实体提取+KG上下文注入）
│   ├── self_rag.py            # 自反检索（相关性过滤+生成后质量自查）
│   ├── orchestrator.py        # ReAct 循环 Agent
│   ├── agents.py              # QuizAgent / SummaryAgent / ReviewAgent
│   ├── agent_service.py       # Agent 领域服务（意图路由+工具注册+事件总线）
│   ├── retrieval_service.py   # 检索领域服务（分块+混合检索+重排序+评估）
│   ├── knowledge_service.py   # 知识领域服务（图谱+蒸馏+上下文）
│   ├── progress_service.py    # 进度领域服务（追踪+计划+提醒+分析）
│   ├── hybrid_retriever.py    # BM25 + 向量混合检索（PMI 自适应分词典）
│   ├── hierarchical_retriever.py  # 层级上下文组装（4 种策略）
│   ├── adaptive_retriever.py  # 自适应检索路由
│   ├── reranker.py            # CrossEncoder 重排序
│   ├── mmr_reranker.py        # MMR 多样性重排
│   ├── qa_engine.py           # QA 引擎
│   ├── evaluator.py           # RAGAS 评估
│   ├── intent_router.py       # LLM 意图分类
│   ├── resilience.py          # Circuit Breaker + 降级 + 错误码枚举
│   ├── tool_registry.py       # 工具注册表 + Prompt 模板
│   ├── learner_model.py       # 学习者模型（事件记录+笔记+范围文档）
│   ├── tutor_agent.py         # 学习路径生成
│   ├── message_bus.py         # 事件总线
│   ├── guided_learning.py     # 引导式学习
│   └── learning_analytics.py  # 学习分析统计
│
├── core/                      # 核心数据层
│   ├── hierarchical_chunker.py    # 三层分块（L1/L2/L3）
│   ├── tree_storage.py            # Qdrant 树形存储
│   ├── knowledge_graph.py         # 知识图谱（实体+关系+别名）
│   ├── progress_tracker.py        # SM-2 间隔重复追踪
│   ├── conversation_memory.py     # 分层记忆（短/中/长期）
│   ├── document_manager.py        # 文档管理（版本控制）
│   ├── knowledge_distiller.py     # 知识蒸馏
│   ├── learning_context.py        # 学习上下文构建
│   ├── learning_planner.py        # 学习计划
│   ├── learning_reminder.py       # 复习提醒
│   ├── background_agent.py        # 后台检测线程
│   ├── course_manager.py          # 课程管理
│   └── database.py                # SQLite 数据库管理
│
├── utils/                     # 基础设施
│   ├── llm_service.py         # LLM 封装（流式/非流式，多provider）
│   ├── embedding_service.py   # Embedding 封装
│   ├── retrieval_utils.py     # 检索工具函数
│   └── web_search.py          # 网络搜索（Tavily + 备选）
│
├── tests/                     # 测试（阶段 1-4）
├── docs/
│   └── performance_report.md  # RAG 评估报告
└── user_data/                 # 用户数据（L1事件/笔记/范围）
```

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 前端 | Streamlit |
| RAG 管线 | 三层分块 + BM25 + 向量检索 + RRF 融合 + MMR 去重 |
| 向量数据库 | Qdrant Cloud |
| Embedding | DashScope text-embedding-v3 |
| LLM | DeepSeek Chat / OpenAI / OpenRouter / SiliconFlow |
| 知识图谱 | LLM 实体抽取 + SQLite + 向量模糊匹配 |
| 间隔重复 | SM-2 算法 |
| 检索增强 | Graph RAG + Self-RAG |
| Agent 机制 | ReAct 循环 + 工具注册 + 事件总线 |
| 可视化 | Plotly |
| 部署 | Docker + docker-compose |
| 评估 | RAGAS（Faithfulness / Relevancy / Precision / Recall） |
| CI | GitHub Actions（pytest） |
| 安全 | `.gitignore` + `.dockerignore` + 逐目录 COPY |

---

## Docker 部署

```bash
# 1. 构建镜像
docker compose build --no-cache

# 2. 确保 .env 文件存在且已配置密钥
#    （Docker 运行时用 --env-file 加载，不进入镜像）

# 3. 启动
docker compose up -d

# 访问 http://localhost:7861
```

**安全说明**：`.env` 文件被 `.dockerignore` 明确排除，不会进入镜像层。Docker Compose 通过 `env_file` 指令在运行时注入环境变量。

---

## RAG 评估结果

基于 20 个深度学习问答样本的 RAGAS 评估：

### 最优策略

| 策略 | Context Recall | Context Precision | Faithfulness | Answer Relevancy |
|:----|:--------------:|:-----------------:|:------------:|:----------------:|
| **hybrid** 🏆 | **0.874** | **0.923** | **1.0** | **1.0** |
| vector_only | 0.851 | 0.940 | 0.96 | 1.0 |
| hybrid + reranker | 0.854 | 0.928 | 1.0 | 0.95 |

### 关键提升

```
语义 Chunking:            Recall 0.07 → 0.584   (+734%)
Balanced 上下文策略:      Recall 0.584 → 0.893  (+53%)
PMI 自适应分词典:         Recall 0.858 → 0.874  (+1.6%)
```

完整报告见 [docs/performance_report.md](docs/performance_report.md)

---

## 运行测试

```bash
pytest tests/ -v
```

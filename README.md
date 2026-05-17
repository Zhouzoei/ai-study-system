# AI 学习系统

一个基于 **RAG（检索增强生成）** 的个性化学习助手，支持智能问答、知识图谱可视化、间隔重复复习与学习进度追踪。

---

## 目录

- [快速开始](#快速开始)
- [本地环境搭建](#本地环境搭建)
- [系统架构](#系统架构)
- [核心特性](#核心特性)
- [RAG 评估结果](#rag-评估结果)
- [项目结构](#项目结构)
- [技术栈](#技术栈)
- [运行测试](#运行测试)

---

## 快速开始

```bash
git clone https://github.com/your-username/QAsystem.git
cd QAsystem

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
# .venv\Scripts\activate.bat
# macOS / Linux:
# source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 配置 API Key
cp .env.example .env
```

编辑 `.env`，填入以下密钥：

| 配置项 | 获取方式 |
|--------|---------|
| `LLM_API_KEY` | [DeepSeek 平台](https://platform.deepseek.com/) |
| `EMBED_API_KEY` | [阿里云 DashScope](https://dashscope.aliyun.com/) |
| `QDRANT_URL` + `QDRANT_API_KEY` | [Qdrant Cloud](https://cloud.qdrant.io/) |

然后启动：

```bash
python app.py
```

浏览器访问 `http://127.0.0.1:7861`

---

## 本地环境搭建

### 前置要求

| 软件 | 版本要求 | 验证 |
|------|---------|------|
| Python | 3.11+ | `python --version` |
| pip | (随 Python 安装) | `pip --version` |

### 详细步骤

**1. 克隆仓库**

```bash
git clone https://github.com/your-username/QAsystem.git
cd QAsystem
```

**2. 创建虚拟环境（推荐）**

```bash
# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

看到终端前面出现 `(.venv)` 即表示激活成功。

**3. 安装依赖**

```bash
pip install -r requirements.txt
```

如果安装速度慢，可以换国内镜像源：

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

**4. 配置环境变量**

```bash
cp .env.example .env
```

打开 `.env`，填入以下 API 密钥：

```ini
# LLM（使用 DeepSeek）
LLM_MODEL_ID=deepseek-chat
LLM_API_KEY=sk-your-deepseek-api-key
LLM_BASE_URL=https://api.deepseek.com

# Embedding（使用阿里云 DashScope）
EMBED_MODEL_TYPE=dashscope
EMBED_MODEL_NAME=text-embedding-v3
EMBED_API_KEY=sk-your-dashscope-api-key
EMBED_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Qdrant 向量数据库
QDRANT_URL=https://your-qdrant-instance.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
```

**5. 启动系统**

```bash
python app.py
```

首次启动会自动初始化 Pipeline 并连接 Qdrant 云服务。终端输出类似：

```
[TreeStorage] Qdrant cloud connected: hierarchical_chunks
[App] 系统初始化完成，启动 Web 界面...
* Running on local URL:  http://127.0.0.1:7861
```

### 使用 Docker

```bash
docker compose up
```

---

## 系统架构

```
用户 → Gradio UI
         ↓
  EnhancedRAGPipeline
         ↓
  ┌─────────────┬──────────────┬─────────────┐
  │ 分层分块器   │ 混合检索器    │ 知识图谱     │
  │ (Markdown   │ (BM25+向量   │ (LLM 实体    │
  │  章节解析)  │  + RRF融合)  │  抽取)       │
  ├─────────────┼──────────────┼─────────────┤
  │ 重排序器     │ MMR 去重     │ 间隔重复     │
  │ (CrossEncoder│             │ (SM-2算法)  │
  └─────────────┴──────────────┴─────────────┘
         ↓
   LLM (DeepSeek) → 流式回答
```

---

## 核心特性

### 💬 双循环智能问答

采用 **Dual-Loop Agent 机制**：

1. **分析循环** — QueryClassifier 分析问题类型（事实/推理/比较/探索/步骤），提取关键词和实体
2. **求解循环** — 根据分析结果动态调整检索策略，RAG 检索后流式生成回答

支持 3 种检索模式：

| 模式 | 策略 | 适用场景 |
|------|------|---------|
| ⚡ naive | 纯向量检索 | 快速事实查找 |
| 🎯 hybrid | 向量 + BM25 混合 | 日常问答（默认） |
| 🔬 deep | 混合 + 重排序 + 改写 | 复杂推理问题 |

回答自动标注来源 `[来源 1]`，底部展示原文片段。

### 🔗 知识图谱

- 文档上传后自动抽取实体和关系
- 支持模糊匹配查询（输入"反传"也能找到"反向传播"）
- 多跳路径查询

### 📈 学习进度

- SM-2 间隔重复算法追踪知识点掌握度
- 5 级掌握度：未知 → 接触 → 熟悉 → 熟练 → 掌握
- 自动生成待复习列表
- 学习仪表盘 + 掌握度分布图（Plotly）

### 🎯 智能多模式

同一对话框中自动识别用户意图：
- **出题** — 根据知识点自动生成选择题
- **总结** — 智能摘要生成
- **问答** — RAG 检索回答

---

## RAG 评估结果

基于 20 个深度学习问答样本的 RAGAS 评估，使用 47 节点文档（L1=2, L2=5, L3=40）。

### 最优策略（5 策略对比）

| 策略 | Context Recall | Context Precision | Faithfulness | Answer Relevancy |
|:----|:--------------:|:-----------------:|:------------:|:----------------:|
| **hybrid** 🏆 | **0.874** | **0.923** | **1.0** | **1.0** |
| vector_only | 0.851 | 0.940 | 0.96 | 1.0 |
| hybrid + reranker | 0.854 | 0.928 | 1.0 | 0.95 |
| hybrid + rewrite | 0.851 | 0.923 | 1.0 | 1.0 |
| hybrid + rewrite + rank | 0.844 | 0.945 | 1.0 | 1.0 |

**结论**：hybrid（BM25 + 向量混合检索）在 Recall 和 Precision 之间取得最佳平衡，无需 reranker 和查询改写。

### 关键发现

```
语义 Chunking:            Recall 0.07 → 0.584   (🔺+734%)
Balanced 上下文策略修复:   Recall 0.584 → 0.893  (🔺+53%)
分词优化（PMI 领域自适应）: Recall 0.858 → 0.874  (🔺+1.6%)
```

完整评估报告见 [docs/performance_report.md](docs/performance_report.md)

---

## 项目结构

```
├── app.py                    # 入口（仅 25 行，启动入口）
├── config.py                 # 统一配置（环境变量映射）
├── requirements.txt          # Python 依赖
├── Dockerfile                # Docker 镜像构建
├── docker-compose.yml        # Docker 编排
├── .env.example              # 环境变量模板
│
├── ui/
│   ├── __init__.py
│   └── app.py                # Gradio 界面（3 导航模块）
│
├── engines/                  # 引擎层
│   ├── pipeline.py           # RAG 主管线（ingest / query / generate）
│   ├── adaptive_retriever.py # 自适应检索 + 查询分类器
│   ├── hybrid_retriever.py   # BM25 + 向量混合检索
│   ├── hierarchical_retriever.py  # 层级上下文组装
│   ├── reranker.py           # CrossEncoder 重排序
│   ├── mmr_reranker.py       # MMR 多样性重排
│   ├── qa_engine.py          # QA 引擎
│   ├── evaluator.py          # RAGAS 评估
│   ├── learning_analytics.py # 学习分析
│   ├── query_rewriter.py     # 查询改写
│   └── query_expander.py     # 查询扩展
│
├── core/                     # 核心模块
│   ├── hierarchical_chunker.py   # 分层分块（Markdown 解析）
│   ├── tree_storage.py           # 树形存储 + Qdrant 向量
│   ├── knowledge_graph.py        # 知识图谱构建与查询
│   ├── progress_tracker.py       # SM-2 间隔重复
│   ├── learning_planner.py       # 学习计划
│   ├── conversation_memory.py    # 对话记忆
│   ├── document_manager.py       # 文档管理
│   └── learning_reminder.py      # 复习提醒
│
├── utils/                    # 服务封装
│   ├── llm_service.py        # LLM（流式/非流式）
│   └── embedding_service.py  # Embedding 服务
│
├── docs/
│   └── performance_report.md # RAG 评估报告
│
└── tests/                    # 测试（4 个阶段）
    ├── test_phase1.py        # 核心管线测试
    ├── test_phase2.py        # 对话 + 学习计划 + 进度
    ├── test_phase3.py        # 知识图谱 + 评估
    ├── test_phase4.py        # 提醒 + 文档管理 + 分析
    └── ragas_baseline_report.json  # 评估原始数据
```

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 前端 | Gradio |
| RAG 管线 | 分层分块 + BM25 + 向量检索 + RRF 融合 + MMR 去重 |
| 向量数据库 | Qdrant Cloud |
| Embedding | DashScope text-embedding-v3 |
| LLM | DeepSeek Chat |
| 知识图谱 | LLM 自动抽取 + SQLite + 模糊匹配 |
| 间隔重复 | SM-2 算法 |
| 可视化 | Plotly |
| 部署 | Docker + docker-compose |
| 评估 | RAGAS（Faithfulness / Relevancy / Precision / Recall） |
| CI | GitHub Actions（pytest） |

---

## 运行测试

```bash
# 安装测试依赖
pip install pytest

# 运行全部测试
pytest tests/ -v

# 运行单个阶段测试
pytest tests/test_phase1.py -v
```

---

## 许可证

MIT

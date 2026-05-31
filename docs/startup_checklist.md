# 本地运行环境启动检查清单

使用前逐项核对，在完成项前打 `[x]`。

---

## 1. 前置环境检查

- [ ] **Python 版本 ≥ 3.11**
  ```bash
  python --version
  ```
- [ ] **pip 可用**
  ```bash
  pip --version
  ```
- [ ] **Git 已安装**（首次拉取代码时需要）
  ```bash
  git --version
  ```

---

## 2. API 密钥准备

| 服务 | 用途 | 获取地址 | 必填 |
|------|------|---------|------|
| DeepSeek | LLM 推理 (问答/出题/总结) | [platform.deepseek.com](https://platform.deepseek.com/) | ✅ |
| 阿里云 DashScope | 文本嵌入 (向量化) | [dashscope.aliyun.com](https://dashscope.aliyun.com/) | ✅ |
| Qdrant Cloud | 向量数据库 | [cloud.qdrant.io](https://cloud.qdrant.io/) | ✅ |
| Neo4j Aura | 图数据库 (可选) | [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura/) | ❌ |
| Tavily | 网页搜索工具 (可选) | [tavily.com](https://tavily.com/) | ❌ |
| SerpApi | 网页搜索工具 (可选) | [serpapi.com](https://serpapi.com/) | ❌ |

提前打开以上页面，获取密钥后再开始配置。

---

## 3. 环境变量配置

- [ ] **复制配置文件**
  ```bash
  cp .env.example .env
  ```
- [ ] **填写 LLM 配置**（DeepSeek）
  ```ini
  LLM_MODEL_ID=deepseek-chat
  LLM_API_KEY=sk-你的DeepSeek密钥
  LLM_BASE_URL=https://api.deepseek.com
  ```
- [ ] **填写 Embedding 配置**（阿里云 DashScope）
  ```ini
  EMBED_MODEL_TYPE=dashscope
  EMBED_MODEL_NAME=text-embedding-v3
  EMBED_API_KEY=sk-你的DashScope密钥
  EMBED_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
  ```
- [ ] **填写 Qdrant 配置**
  ```ini
  QDRANT_URL=https://你的实例地址.cloud.qdrant.io
  QDRANT_API_KEY=你的Qdrant密钥
  ```
- [ ] **验证 QDRANT_VECTOR_SIZE 一致性**
  - `.env` 中填写 `QDRANT_VECTOR_SIZE=1024`
  - 如果已有 Qdrant 集合且维度不同，系统启动时会自动检测并警告
  - 首次创建集合时使用 1024 维度
- [ ] **Neo4j 配置（可选）**
  ```ini
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USERNAME=neo4j
  NEO4J_PASSWORD=你的密码
  ```

---

## 4. Python 虚拟环境

- [ ] **创建虚拟环境**
  ```bash
  python -m venv .venv
  ```
- [ ] **激活虚拟环境**
  ```bash
  # Windows PowerShell:
  .venv\Scripts\Activate.ps1
  # Windows CMD:
  .venv\Scripts\activate.bat
  # macOS / Linux:
  source .venv/bin/activate
  ```
- [ ] **确认已激活**（终端提示符前出现 `(.venv)`）

---

## 5. 安装依赖

- [ ] **安装 Python 依赖**
  ```bash
  pip install -r requirements.txt
  ```
- [ ] **验证关键依赖已安装**
  ```bash
  python -c "import openai; import dashscope; import qdrant_client; import jieba; print('关键依赖 OK')"
  ```
- [ ] **可选：安装 sentence-transformers（重排序模型）**
  - 已在 `requirements.txt` 中声明，`pip install` 时会自动安装
  - 如果安装失败（如缺少 PyTorch），系统会自动降级并跳过重排序
  - 如需手动指定 PyTorch 版本：先安装 PyTorch，再安装 `sentence-transformers`

- [ ] **慢速网络替代方案：使用国内镜像**
  ```bash
  pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
  ```

---

## 6. 启动前自检

- [ ] **Python 可找到项目模块**
  ```bash
  cd 项目根目录
  python -c "from config import config; print(f'QDRANT_VECTOR_SIZE={config.QDRANT_VECTOR_SIZE}')"
  ```
  预期输出：`QDRANT_VECTOR_SIZE=1024`

- [ ] **API 密钥不为空**
  ```bash
  python -c "from config import config; print(f'LLM: {\"✅\" if config.LLM_API_KEY else \"❌\"} Embed: {\"✅\" if config.EMBED_API_KEY else \"❌\"} Qdrant: {\"✅\" if config.QDRANT_API_KEY else \"❌\"}')"
  ```

- [ ] **Qdrant 集合维度无冲突**
  - 如果之前用不同维度创建过集合，系统会输出警告：
    ```
    [TreeStorage] WARNING: Qdrant collection 'hierarchical_chunks' exists with dimension 384, but config.QDRANT_VECTOR_SIZE=1024. Using existing dimension 384.
    ```
  - 这不是错误，系统会自动适配已有集合的维度
  - 如需重新创建，需到 Qdrant Cloud 控制台删除旧集合

---

## 7. 启动系统

- [ ] **启动应用**
  ```bash
  python app.py
  ```

- [ ] **检查启动日志，确认以下关键信息**
  ```
  [TreeStorage] Qdrant cloud connected: hierarchical_chunks       ← Qdrant 连接成功
  [Reranker] Cross-Encoder model loaded: BAAI/bge-reranker-v2-m3  ← 重排序加载成功（可选）
  [App] 系统初始化完成，后台 Agent 已启动                         ← 初始化完成
  * Running on local URL:  http://127.0.0.1:7861                  ← Web 服务已运行
  ```

- [ ] **浏览器自动打开** `http://127.0.0.1:7861`
  - 如未自动打开，手动在浏览器访问该地址

---

## 8. 功能验证

### 8.1 基础问答
- [ ] 在聊天框输入 "你好"，确认收到回复
- [ ] 上传一份 Markdown 或 PDF 文档
- [ ] 提问文档相关内容（如 "这篇文章讲了什么？"）
- [ ] 确认回答底部有来源标注 `[来源 1]`、`[来源 2]`

### 8.2 智能出题
- [ ] 输入 "出几道题考考我"
- [ ] 确认选择题选项可点击选择
- [ ] 提交答案后确认系统给出评分

### 8.3 知识图谱
- [ ] 在知识图谱页签搜索实体名称
- [ ] 确认图谱有实体和关系展示

### 8.4 学习进度
- [ ] 在学习进度页签确认有数据展示
- [ ] 开始复习后确认进度更新

---

## 9. Docker 部署（可选）

- [ ] **安装 Docker Desktop**
  ```bash
  docker --version
  ```
- [ ] **确保 `.env` 已正确配置**
- [ ] **创建数据目录**
  ```bash
  mkdir data
  ```
- [ ] **构建并启动**
  ```bash
  docker compose up --build
  ```
- [ ] **确认不需要 volumes 覆盖**
  - 当前配置只挂载了 `./data:/app/data`（持久化数据）和 `./.env:/app/.env`（配置）
  - 容器内代码来自 Docker build，不会被本地覆盖
  - 如果修改了代码，需要重新 `docker compose up --build`

---

## 10. 常见问题排查

| 现象 | 原因 | 解决方法 |
|------|------|---------|
| `ModuleNotFoundError: No module named 'xxx'` | 依赖未安装 | `pip install -r requirements.txt` |
| `[LLM调用失败: ...]` | API Key 无效或网络不通 | 检查 `LLM_API_KEY` 和 `LLM_BASE_URL` |
| `[TreeStorage] Qdrant cloud failed` | Qdrant 连接失败 | 检查 `QDRANT_URL` 和 `QDRANT_API_KEY`；首次运行会自动降级到内存模式 |
| `sqlite3.OperationalError: database is locked` | 多线程写入冲突 | 已内置自动重试机制，稍后重试即可 |
| Reranker 未生效（不走 rerank） | `sentence-transformers` 未安装 | `pip install sentence-transformers`；或等自动降级 |
| 端口 7861 被占用 | 其他程序已占用该端口 | 修改 `app.py` 中的 `server_port` 参数 |
| `.venv\Scripts\Activate.ps1` 无法执行 | PowerShell 执行策略限制 | 以管理员运行 `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |

---

## 完整启动速查命令（Windows PowerShell）

```powershell
# 1. 克隆并进入目录
git clone https://github.com/your-username/QAsystem.git
cd QAsystem

# 2. 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量（手动编辑 .env）
cp .env.example .env
notepad .env

# 5. 启动
python app.py
```

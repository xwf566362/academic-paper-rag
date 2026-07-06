# PaperMind - 学术智析

> 基于本地向量检索 + 远程大模型的学术论文智能分析平台。

上传论文 -> AI 自动解析 -> 智能问答、摘要生成、文献导出。

---

## 功能特性

### 论文管理
- **PDF 批量上传**：支持多文件同时上传，每篇独立处理
- **自动分片**：按章节（摘要/引言/方法/实验/结论）智能分割，保留段落边界
- **元数据存储**：SQLite 持久化，支持摘要、关键词、BibTeX 元数据管理
- **论文删除**：按文件名删除，连带清除向量和元数据

### RAG 智能问答
- **语义检索**：Faiss 近似最近邻搜索 + BGE 中文嵌入模型
- **结果重排**：BGE 重排模型对候选结果二次排序
- **学术防幻觉**：内置学术 Prompt，严格基于论文片段回答，强制溯源
- **流式输出**：SSE 流式传输，逐 token 显示，无需等待完整响应
- **限定论文**：可指定在单篇论文范围内搜索

### 论文分析
- **一键研究方法解析**：AI 自动提取创新点、数据集、技术链路、基线对比、局限性
- **一键算法流程分析**：AI 提取算法架构，输出文字步骤 + Mermaid 流程图

### 多论文对比
- 选择多篇论文进行对比分析
- AI 自动对比方法差异、实验设置、结论异同

### 批量操作
- **批量上传**：一次选择多个 PDF，每篇独立处理
- **批量生成摘要**：LLM 自动生成中文摘要 + 关键词
- **批量对比**：支持多篇论文横向对比

### 导出功能
- **BibTeX 导出**：自动提取参考文献，调用大模型格式化为标准 BibTeX
- **摘要导出**：论文摘要和关键词持久化存储

### 对话历史
- 对话自动保存到 SQLite，页面刷新不丢失
- 支持多对话管理

### 配置持久化
- API Key 一次配置，持久化到本地文件
- 页面刷新和服务器重启后自动加载，无需重复配置

---

## 技术架构

```
用户 (Gradio UI) -> FastAPI 后端 -> Faiss 向量检索 -> LLM API (DeepSeek/通义千问/智谱)
                    -> SQLite 持久化 -> BGE 嵌入 -> PyMuPDF PDF 解析
```

### 核心依赖

| 组件 | 技术选型 |
|------|----------|
| PDF 解析 | PyMuPDF |
| 向量嵌入 | BGE-large-zh-v1.5 (sentence-transformers) |
| 向量索引 | Faiss (IndexFlatIP) |
| 数据持久化 | SQLite + WAL 模式 |
| 后端框架 | FastAPI + Uvicorn |
| 前端界面 | Gradio |
| LLM 调用 | httpx + OpenAI 兼容接口 |
| 容器化 | Docker + docker-compose + Nginx |

### 向量检索演进
- v1: numpy 暴力全量扫描 (O(N*d))
- v2: Faiss IndexFlatIP (SIMD 优化，约 10x 提升)
- 冷热数据分离架构设计，支持未来扩展 ANN 索引

---

## 项目结构

```
paper-rag/
├── README.md
├── Dockerfile                       # 容器构建
├── docker-compose.yml               # 多服务编排
├── config.yaml                      # 主配置文件
├── requirements.txt                 # 依赖清单
├── run.py                           # 启动入口
├── docker/
│   ├── nginx.conf                   # Nginx 反向代理
│   └── start.sh                     # 容器启动脚本
├── scripts/
│   └── download_models.py           # BGE 模型预下载
├── tests/
│   ├── conftest.py                  # 测试夹具 + Mock
│   ├── unit/
│   │   ├── test_vector_store.py     # 向量存储测试 (20)
│   │   ├── test_pdf_parser.py       # PDF 解析测试 (11)
│   │   ├── test_chunker.py          # 分片策略测试 (5)
│   │   └── test_llm_client.py      # LLM 客户端测试 (5)
│   └── integration/
│       └── __init__.py
├── app/
│   ├── main.py                      # FastAPI 入口
│   ├── config.py                    # 配置加载 (+ config.local.yaml 合并)
│   ├── types.py                     # TypedDict 类型定义
│   ├── api/
│   │   ├── routers.py               # 核心 API 路由
│   │   ├── config.py                # 配置 API (动态设置 Key)
│   │   ├── conversations.py         # 对话历史 API
│   │   ├── analysis_routes.py       # 论文分析路由
│   │   ├── paper_analyzer.py        # 论文分析引擎
│   │   └── llm_client.py            # 多厂商 LLM 客户端
│   ├── database/
│   │   ├── vector_store.py          # Faiss + SQLite 向量存储
│   │   ├── conversations.py         # 对话持久化
│   │   └── papers.py                # 论文元数据 (摘要/关键词/BibTeX)
│   ├── processing/
│   │   ├── pdf_parser.py            # PDF 解析 + 章节识别
│   │   ├── chunker.py               # 论文分片策略
│   │   └── embeddings.py            # BGE 嵌入 + 重排模型
│   └── frontend/
│       └── gradio_app.py            # Gradio 前端界面
└── data/
    ├── papers/                      # 上传的 PDF 存储
    └── chroma/
        ├── store.db                 # 向量 + 分片数据
        └── app.db                   # 用户配置 + 对话 + 论文元数据
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

编辑 `config.yaml`，填入你的 API Key。支持三种厂商：

| 厂商 | 获取地址 | 默认模型 |
|------|----------|----------|
| DeepSeek | https://platform.deepseek.com/ | deepseek-chat |
| 通义千问 | https://dashscope.aliyun.com/ | qwen-plus |
| 智谱 | https://open.bigmodel.cn/ | glm-4-plus |

也可以在启动后在 Gradio 界面中通过配置 Tab 动态设置。

### 3. 下载嵌入模型

首次使用需下载 BGE 模型：

```bash
# 使用国内镜像加速
$env:HF_ENDPOINT="https://hf-mirror.com"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-zh-v1.5')"
```

### 4. 启动

```bash
# 前后端同时启动
python run.py

# 或分别启动
python run.py --mode backend    # FastAPI :8000
python run.py --mode frontend   # Gradio :7860
```

启动后打开 `http://127.0.0.1:7861` 即可使用。

### Docker 部署

```bash
docker-compose up -d --build
```

首次启动会自动下载 BGE 模型（约 2GB），缓存到 named volume。

---

## 使用指南

### 配置（首次使用）
1. 打开页面 -> 切到 配置 Tab
2. 选择厂商，输入 API Key，点 Save
3. 配置持久化，下次打开无需重新填写

### 论文管理
1. 切到 论文管理 Tab
2. 点击选择 PDF 文件（支持多选）-> 点 批量上传
3. 上传后在右侧选择论文，可查看详情、生成摘要、导出参考文献

### 问答
1. 切到 论文问答 Tab
2. 输入问题，支持限定单篇论文
3. 回答会逐字显示（SSE 流式），附带论文溯源

### 对比
1. 切到 多论文对比 Tab
2. 选择两篇或多篇论文（需先上传）
3. 输入对比问题，AI 横向分析

### 分析
1. 切到 论文分析 Tab
2. 选择论文，点击一键分析研究方法或算法流程

---

## 开发

### 运行测试

```bash
python -m pytest tests/unit -v
```

41 个单元测试覆盖：向量存储 CRUD、PDF 解析、分片策略、LLM 客户端异常处理。

### 代码检查

```bash
python -m mypy app --ignore-missing-imports
```

类型配置见 `pyproject.toml`（strict 模式）。

---

## 常见问题

**Q: API 调用返回认证失败？**
A: 检查 API Key 是否正确，或在配置 Tab 中重新设置。DeepSeek 等厂商的 API 额度用完也会返回 401。

**Q: PDF 解析乱码或章节识别不准？**
A: PyMuPDF 对部分 CJK 编码 PDF 的文本提取可能有编码问题，导致章节标题乱码。可以通过改善 PDF 源文件质量来缓解。

**Q: 向量检索太慢？**
A: 当前使用 Faiss IndexFlatIP（精确检索），适用于万级分片。如需更高性能，可扩展为 IVF 或 HNSW 索引。

**Q: BGE 模型下载太慢？**
A: 使用国内镜像 `HF_ENDPOINT=https://hf-mirror.com` 加速，或改用 ModelScope 下载。

---

## License

MIT

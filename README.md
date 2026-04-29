# CP_RAG

面向算法题学习场景的多层检索问答系统（Cache + Exact Retrieval + RAG + LLM）。

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Milvus](https://img.shields.io/badge/Milvus-Vector%20DB-00B3A4)
![BGE-M3](https://img.shields.io/badge/Embedding-BGE--M3-6A5ACD)
![PaddleOCR](https://img.shields.io/badge/OCR-PaddleOCR-D32F2F)
![FastAPI](https://img.shields.io/badge/FastAPI-Async%20API-009688?logo=fastapi&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-Cache%20%26%20Session-DC382D?logo=redis&logoColor=white)
![Status](https://img.shields.io/badge/Project%20Status-Active-success)
![API](https://img.shields.io/badge/API-FastAPI%20Running-00C4B3)
![RAG](https://img.shields.io/badge/RAG-Tier1%2F2%2F3-blue)
![Interview](https://img.shields.io/badge/Use%20Case-Interview%20Demo-8A2BE2)

## 项目简介

`CP_RAG` 的目标是把“题号精确命中”和“语义检索补充”结合起来，在保证回答质量的同时降低延迟与调用成本。

系统将用户请求按三层策略处理：

1. **Tier 1 语义缓存（Redis）**：高相似问题直接返回缓存答案。
2. **Tier 2 精确检索（MySQL）**：识别题号后查询结构化题解。
3. **Tier 3 语义检索（Milvus）**：对通用问题补充相关文档上下文。

最终将多源上下文交给 LLM（OpenAI-compatible，如 DeepSeek）生成苏格拉底助教风格回答。

---

## 核心功能

- 多平台题号识别：LeetCode / 洛谷 / Codeforces / AtCoder
- 多轮对话会话管理：`session_id`、历史消息、当前题锁定
- “完成当前问题”判定：防止用户未完成当前题就切换题目
- 本地 Embedding：`BAAI/bge-m3`（sentence-transformers）
- Milvus Top-K 语义检索（当前默认 `K=2`）
- 前端演示页面（轻量 Web UI）
- 原始文档解析入库脚本（文本优先 + OCR回退）

---

## 技术栈

- **Backend**: FastAPI (async)
- **Cache / Session**: Redis
- **Structured Data**: MySQL
- **Vector DB**: Milvus
- **Embedding**: sentence-transformers (BGE-M3)
- **LLM**: OpenAI-compatible API
- **Doc Parsing**: python-docx / python-pptx / pdfplumber / PyMuPDF / PaddleOCR

---

## 项目结构

```text
CP_RAG/
├─ api/
│  ├─ router.py
│  └─ routes/
│     ├─ health.py
│     └─ query.py
├─ api_v2/
│  ├─ router.py
│  └─ routes/
│     └─ query.py
├─ core/
│  ├─ config.py
│  ├─ llm_client.py
│  ├─ router.py
│  └─ session_manager.py
├─ core_v2/
│  ├─ intent_classifier.py
│  ├─ pipeline.py
│  ├─ qwen_client.py
│  └─ retriever.py
├─ db/
│  ├─ redis_client.py
│  ├─ mysql_pool.py
│  └─ milvus_client.py
├─ utils/
│  ├─ insert_data.py
│  └─ ingest_raw_to_milvus.py
├─ utils_v2/
│  └─ rebuild_milvus_langchain.py
├─ web/
│  ├─ index.html
│  └─ v2_test.html
├─ docs/
│  ├─ PROJECT_STRUCTURE.md
│  └─ INTERVIEW_PROJECT_OVERVIEW.md
├─ .env.example
├─ app_run.py
├─ app_run_v2.py
├─ app_run_v2_web.py
├─ main_v2.py
├─ start.bat
└─ requirements.txt
```

---

## 快速开始

### 1) 安装依赖

建议在 `conda` 环境中运行（Python 3.10）。

```bash
pip install -r requirements.txt
```

### 2) 配置环境变量

复制并修改：

```bash
cp .env.example .env
```

重点配置项：

- `MYSQL_*`
- `REDIS_*`
- `MILVUS_*`
- `EMBEDDING_MODEL_NAME`（可填本地模型目录）
- `LLM_API_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`

### 3) 启动服务

方式 A（推荐）：

```bash
python app_run.py
```

方式 B（Windows 一键）：

```bash
start.bat
```

启动后访问：

- 前端页面：`http://127.0.0.1:8000/`
- 健康检查：`http://127.0.0.1:8000/api/v1/health`

---

## API 示例

### 查询接口

`POST /api/v1/query`

请求体：

```json
{
  "query": "LeetCode 1 two sum 怎么做",
  "session_id": "default",
  "finish_current": false
}
```

响应示例：

```json
{
  "answer": "...",
  "from_cache": false,
  "cache_similarity": null,
  "matched_problem_id": "LC0001",
  "rag_hits": 2,
  "session_id": "default",
  "waiting_for_completion": true,
  "active_problem_id": "LC0001"
}
```

---

## 文档入库（Milvus）

使用脚本把 `source_data/raw_data` 中的文档解析并写入 Milvus：

```bash
python utils/ingest_raw_to_milvus.py \
  --source source_data/raw_data \
  --collection cp_rag_document \
  --batch-size 32
```

说明：

- 支持：`txt / doc / docx / ppt / pptx / pdf`
- 策略：文本抽取优先，文本不足时 OCR 回退（方案 B）

---

## 已实现的工程特性

- 异步连接池与生命周期管理（Redis/MySQL/Milvus）
- 字段兼容处理（`context/content`、`id/doc_id`）
- 会话状态隔离（按 `session_id`）
- 缓存、检索、生成链路闭环

---

## Roadmap

- [ ] 检索重排（Reranker）增强
- [ ] 评测集与自动化指标（命中率/延迟/成本）
- [ ] 更精细的 OCR 资源调度与限流
- [ ] 生产监控（Tracing、慢请求、失败分类）

---

## CP_RAG v2（LangChain 重构）

v2 保持与 v1 并行，不改动原有接口与启动方式。新增能力：

- `multipart/form-data` 输入：`query` + 可选 `jpg/png` 图片
- Qwen 多模态识别并与文本 query 融合
- 本地 BERT 意图识别 + Qwen query 改写后再检索
- Markdown 父子文档切分重建 Milvus（子块检索 + 父块语义补全）

### 启动 v2

```bash
python app_run_v2.py
```

默认接口：

- `POST http://127.0.0.1:8001/api/v2/query`

### v2 网页测试（文本 + 图片）

```bash
python app_run_v2_web.py
```

启动后会自动打开：

- `http://127.0.0.1:8001/`

页面支持：

- 文本 query
- 上传 `jpg/png`
- 展示 `intent`、`rewritten_query`、`rag_hits`

### 重建 v2 Milvus

```bash
python utils_v2/rebuild_milvus_langchain.py \
  --source-dir source_data/raw_data \
  --selected-pdf-dir source_data/selected_pdf_100 \
  --max-pdf 100 \
  --batch-size 256
```

说明：

- 脚本会先从 `raw_data` 中筛选并复制前 `100` 个 `pdf`
- 再执行父（默认 512）子（默认 128）切分
- 子块批量写入 `V2_MILVUS_COLLECTION`（默认 `cp_rag_document_v2`）
- 控制台会打印解析、切分和分批入库进度

---

## License

This project is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for details.

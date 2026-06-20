# RAG 集成设计文档

> 日期：2026-06-17
> 状态：设计确认，待实现
> 关联文档：[MCP 集成设计](2026-06-17-mcp-integration-design.md) · [技术路线图](../tech-roadmap.md)
> 上次 Review：2026-06-17（P0×1 + P1×2 + 实现注意×6，全部修正）

---

## 1. 目标

在现有 ChromaDB + LocalEmbedding 基础设施上，补全 RAG（Retrieval-Augmented Generation）管道。
让 Agent 能对用户上传的文档和抓取的网页内容进行语义检索，生成带来源引用的回答。

## 2. 现状

| 组件 | 状态 | 位置 |
|------|------|------|
| ChromaDB | ✅ 运行中 | `backend/memory/episodic.py` + `semantic.py` |
| LocalEmbedding (BGE-small-zh-v1.5, 512-dim) | ✅ 运行中 | `backend/memory/embedding.py` |
| MCP memory_server | ✅ 运行中 | `backend/protocols/mcp/servers/memory_server.py` |
| RAGTool | ❌ 空壳 | `backend/memory/tools/rag_tool.py` — 仅 `class RAGTool: pass` |

## 3. 设计原则

1. **复用基础设施** — 不加新进程、不加载第二份嵌入模型。RAG 工具注册在 `memory_server.py`
2. **用户隔离** — 每个用户的文档互不可见，通过 `_meta.user_id` 过滤
3. **自动入库** — 用户上传文件和 Agent `web_fetch` 抓取结果自动进入知识库
4. **Agent 自主检索** — LLM 判断需要查知识库时主动调用 `rag_search`

### 3.1 Embedding 和 ChromaDB 单例（P0 修正）

### 问题

`memory_server.py` 的现有 handler 每次调用 `new MemoryManager()` → `new LocalEmbedding()`。
LocalEmbedding 惰性加载 BGE 模型（~100MB），意味着**每次工具调用都重新创建实例**。虽然 `SentenceTransformer` 有进程级缓存，但 Python 对象的重复创建浪费内存且不可靠。

RAG 的 `rag_search` 被高频调用时，不能每次都 new。

### 修正

在 `memory_server.py` 模块级建两个单例：

```python
# memory_server.py 模块顶层

_embedding: LocalEmbedding = None
_chroma_client = None
_chroma_collections: dict[str, object] = {}  # collection 缓存

def _get_embedding() -> LocalEmbedding:
    global _embedding
    if _embedding is None:
        _embedding = LocalEmbedding()
    return _embedding

def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(
            path=os.path.join(PROJECT_ROOT, "data", "chroma")
        )
    return _chroma_client

def _get_rag_collection():
    collection_name = "rag_documents"
    if collection_name not in _chroma_collections:
        client = _get_chroma_client()
        _chroma_collections[collection_name] = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collections[collection_name]
```

现有 `handle_memory_search` 和 `handle_list_downloads` 也改用 `_get_embedding()`，不做 breaking change 但受益于单例。

## 4. 数据流

```
文档入库:
  用户上传 PDF/txt ──> /api/upload ──> rag_ingest(content, source)
                                          │
  用户粘贴内容 ──────────────────────────> │
                                          │
  Agent web_fetch ──> handler ───────────> │
                                          │
                                     ┌────▼────────────────────┐
                                     │ _chunk_text(content)     │
                                     │   → 按段落/句号切分       │
                                     │   → ~400字/块, 80字重叠  │
                                     │                          │
                                     │ embed(chunks) → ChromaDB │
                                     │ collection: rag_documents│
                                     └──────────────────────────┘

知识检索:
  Agent 提问 ──> rag_search(query, top_k=5)
                    │
                    ├── embed(query)
                    ├── ChromaDB.query(rag_documents,
                    │       where={"user_id": $user_id})
                    ├── 格式化结果块 + 来源 + 相关度
                    └── Agent 整合到最终回答
```

## 5. MCP 工具定义

### 5.1 `rag_ingest` — 文档入库

```json
{
  "name": "rag_ingest",
  "description": "将文本内容存入知识库。支持用户上传的文档和Agent抓取的网页。支持追加：同一source再次调用会追加新分块。",
  "inputSchema": {
    "type": "object",
    "properties": {
      "content": {"type": "string", "description": "要入库的文本内容"},
      "source": {"type": "string", "description": "来源标识（文件名或URL）"},
      "title": {"type": "string", "description": "文档标题（可选，默认取source）"}
    },
    "required": ["content", "source"]
  }
}
```

**处理逻辑**：

1. 分块：`_chunk_text(content)` → `list[(chunk_id, chunk_text)]`
2. 嵌入：`self._emb.embed(chunks)` → `list[list[float]]`
3. 存储：每条写入 ChromaDB `rag_documents` collection
4. 返回：`{"ok": true, "source": "...", "chunks": N}`

### 5.2 `rag_search` — 知识检索

```json
{
  "name": "rag_search",
  "description": "在知识库中语义搜索相关文档片段。返回top-k个最相关的文本块及来源信息。",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "搜索查询"},
      "top_k": {"type": "integer", "description": "返回结果数，默认5，最大10"}
    },
    "required": ["query"]
  }
}
```

**返回格式**：

```json
{
  "results": [
    {
      "rank": 1,
      "score": 0.92,
      "source": "深度学习综述.pdf",
      "chunk_id": "chunk_3",
      "chunk_index": "3/12",
      "text": "Transformer架构自2017年提出以来..."
    }
  ],
  "query": "transformer architecture",
  "count": 3
}
```

## 6. 分块策略

### BGE token 上限

BGE-small-zh-v1.5 的 `max_seq_length = 512 tokens`。中英文混合时 1 token ≈ 1.5-2 字符，512 tokens ≈ 750-1000 字符。但加 overlap 后实际块长 = chunk_size + overlap，需在 512 token 内。安全值：`chunk_size=400, overlap=80`，即最长 480 字符 ≈ 300 tokens。

```python
def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[dict]:
    """
    按段落边界优先切分，段落过长则按句号切分。
    chunk_size=400 确保加 overlap 后不超过 BGE 512 token 上限。
    返回: [{"chunk_id": "chunk_0", "text": "...", "index": 0, "total": N}, ...]
    """
    paragraphs = text.split("\n\n")
    chunks = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            # 按句号切长段落
            sentences = para.replace("。", "。|||").replace(".", ".|||").split("|||")
            current = ""
            for sent in sentences:
                if len(current) + len(sent) <= chunk_size:
                    current += sent
                else:
                    if current.strip():
                        chunks.append(current.strip())
                    current = sent
            if current.strip():
                chunks.append(current.strip())

    # 组装结果（相邻块带重叠）
    result = []
    for i, chunk_text in enumerate(chunks):
        if i > 0:
            # 前一块的后 overlap 字符作为上下文前缀
            prev_tail = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]
            chunk_text = prev_tail + "\n" + chunk_text
        result.append({
            "chunk_id": f"chunk_{i}",
            "text": chunk_text,
            "index": i,
            "total": len(chunks),
        })

    return result
```

## 7. ChromaDB Collection

```
collection_name: rag_documents

每条记录:
  id:          "{user_id}_{source_hash}_chunk_{index}"
  embedding:   [512] float  (BGE-small-zh-v1.5, L2归一化)
  document:    "chunk_text"  (实际文本)
  metadata:
    user_id:        int         ← 用于过滤
    source:         str         ← 文件名/URL
    title:          str         ← 文档标题
    chunk_index:    int
    total_chunks:   int
    content_hash:   str         ← 全量 content 的 MD5（用于去重）
    ingested_at:    float       ← timestamp
```

### 7.1 与现有 Collection 隔离

| Collection | 用途 | 数据来源 |
|-----------|------|---------|
| `semantic_all` | 用户偏好/知识 | Agent 通过 `memory_search` 保存 |
| `episodic_all` | 任务事件记录 | Agent 自动记录 |
| `rag_documents` | 🆕 外部文档 | 用户上传 + web_fetch |

三者共享同一 ChromaDB 实例，互不干扰。检索时各查各的 collection。

## 8. 自动入库触发

### 8.1 用户上传 → `/api/upload`

**关键**：现有 upload handler 在返回前把 PDF 文本截断到 `[:80000]`。`rag_ingest` 需要用**截断前的全量文本**，否则长文档后半段丢失。

```python
# chat.py upload handler 修改：
if content_type == "application/pdf":
    # ... 解析逻辑 ...
    pdf_text = "\n--- 分页 ---\n".join(text_parts)  # 全量
    if pdf_text.strip():
        # 1. 异步入库（全量文本）
        asyncio.create_task(
            mcp_manager.call_tool("memory", "rag_ingest", {
                "content": pdf_text,
                "source": file.filename,
            })
        )
        # 2. 返回给前端的用截断版
        return {"content": f"[PDF: {file.filename}]\n{pdf_text[:80000]}", ...}

elif content_type in ("text/plain", "text/markdown", ...):
    # 同理：text 是全量的，直接传入
    asyncio.create_task(
        mcp_manager.call_tool("memory", "rag_ingest", {
            "content": text,
            "source": file.filename,
        })
    )
    return {"content": text, ...}
```

异步入库，不阻塞上传响应。入库用全量文本，前端返回照旧截断。

### 8.2 Agent web_fetch → LLM 决定入库

LLM 看到 `web_fetch` 返回的网页内容后，判断是否值得保留。如果值得，调用 `rag_ingest`：

```
Agent: 我抓到了一篇关于GRPO的论文，内容很有价值
→ call rag_ingest(content="...", source="https://arxiv.org/...", title="GRPO论文")
```

不自动入库——只有 LLM 认为有价值的内容才保留，避免知识库被垃圾网页污染。

## 9. 检索与引用

### Agent 侧集成

`rag_search` 注册为 MCP 工具后，LLM 看到它的描述里有"在知识库中语义搜索"，遇到需要查文档的问题时会主动调用。

### 引用格式

检索结果包含 `source` 和 `chunk_index`，Agent 在回答中标注引用：

> 根据《深度学习综述.pdf》第 3 节，Transformer 架构的核心创新在于自注意力机制... [来源: 深度学习综述.pdf, chunk 3/12]

格式由 LLM 自主决定，不由系统强制。

### 去重

同一 `source` 的新内容再次 `rag_ingest` 时，先检查是否已有该 source 的分块。如果存在：
- **方案 A（推荐）**：追加新分块（`chunk_index` 续排），旧分块保留
- **方案 B**：删除旧分块，重新分块入库

方案 A 更安全——不会因为重新分块策略变化导致检索结果不稳定。

## 10. 实现范围

### 改动文件

| 文件 | 改动 |
|------|------|
| `backend/protocols/mcp/servers/memory_server.py` | 新增 `rag_ingest` + `rag_search` 两个工具 handler |
| `backend/routes/chat.py` | upload 完成后异步调 `rag_ingest` |
| `backend/memory/tools/rag_tool.py` | 更新注释（标记为"已由 MCP memory_server 实现"） |

### 不新增文件

所有 RAG 逻辑放在 `memory_server.py` 内——它已经持有 ChromaDB 连接和 LocalEmbedding 实例，加两个 handler 函数即可。

### 不做的（第二期）

- 不支持图片/扫描件 PDF（OCR 需要额外依赖）
- 不支持分块策略的高级配置（段落长度/重叠大小可调，但硬编码默认值）
- 不自动同步文件系统目录

## 11. 错误处理

| 场景 | 处理 |
|------|------|
| 文本为空 | `rag_ingest` 返回 `{"ok": False, "error": "content is empty"}` |
| 分块后无有效内容 | 同上，error 说明 |
| ChromaDB 写入失败 | 捕获异常，返回错误，不影响主流程 |
| 检索无结果 | `rag_search` 返回 `{"results": [], "count": 0}` |
| 用户隔离 | 检索时自动带 `where={"user_id": user_id}` |

## 12. 实现注意事项

### 12.1 追加入库的 chunk_index 续排

```python
def _get_next_chunk_index(collection, source: str, user_id: int) -> int:
    """查询已有分块，新块从 max+1 开始"""
    existing = collection.get(
        where={"$and": [{"source": source}, {"user_id": user_id}]}
    )
    if existing and existing["ids"]:
        indices = [int(m["chunk_index"]) for m in existing["metadatas"]]
        return max(indices) + 1
    return 0
```

### 12.2 Score 转换

ChromaDB 余弦距离返回值：`distance`（0=完全相同，2=完全相反）。

```python
score = round(1.0 - distance, 4)  # 转换到 [0, 1]，越高越相关
```

与 `semantic.py:97` 格式一致。

### 12.3 大文档批量嵌入

50+ 块一次性 `embed(chunks)` 可能内存溢出。分批处理：

```python
def _embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    emb = _get_embedding()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        all_embeddings.extend(emb.embed(batch))
    return all_embeddings
```

### 12.4 Collection 创建参数

```python
collection = client.get_or_create_collection(
    name="rag_documents",
    metadata={"hnsw:space": "cosine"},  # 与 semantic_all/episodic_all 一致
)
```

### 12.5 文本存储位置

现有 memory collection（semantic_all/episodic_all）把原文存 SQLite、ChromaDB 只存向量+metadata。RAG 设计把原文直接存 ChromaDB 的 `documents` 字段——RAG 不需要 SQLite 的 importance/access_count/ttl 体系，纯检索场景 ChromaDB 的 `documents` 字段更直接。

### 12.6 web_fetch → rag_ingest 的 token 消耗

LLM 调用 `rag_ingest` 时需把全文塞进 tool call 的 `content` 参数。超长网页（>2 万字）可能触发 API 参数长度限制。v1 可接受（大多数网页 <1 万字），后续可考虑：

- 加 `ingest_from_cache` 工具：Agent 不传全文，只传 URL，memory_server 自己从缓存或重新 fetch 取内容
- 自动截断：content 传前 5000 字，剩余的异步补充

### 12.7 文本去重

每次 `rag_ingest` 都重新分块+嵌入。同一 source 重复调用会浪费存储。

**hash 粒度**：对全量 `content` 做 MD5（不是 per-chunk）。相同内容 → 分块结果必然相同 → 不需要重新嵌入。

**存储**：metadata 字段加 `content_hash`。

**命中行为**：跳过整个 ingest，返回：

```json
{"ok": true, "skipped": true, "reason": "duplicate content"}
```

**唯一性约束**：`(source, content_hash, user_id)` 组合唯一。同一 source 不同内容（更新版本文档）不触发去重。

---

## 13. 测试策略

| 层级 | 内容 |
|------|------|
| 单元测试 | `_chunk_text()` 各种输入（空文本、短文本、长段落、混合中英文） |
| 集成测试 | `rag_ingest` → `rag_search` 端到端，验证检索到的文本块正确 |
| 手动测试 | 上传一个 PDF，在对话中问相关问题，检查回答是否引用文档 |

---

## 版本记录

| 日期 | 内容 | 作者 |
|------|------|------|
| 2026-06-17 | Review v3：修正 P0(ChromaDB 路径) + P2×3(数据流图/章节编号/MD5去重规格) | Xin-Zho + Claude |

#!/usr/bin/env python3
"""MCP Server: memory — memory_search, list_downloads, rag_ingest, rag_search"""
import hashlib
import os
import sys

# §4.2: 环境变量传递 PROJECT_ROOT
PROJECT_ROOT = os.environ.get("AGENT_PROJECT_ROOT")
if not PROJECT_ROOT:
    raise RuntimeError("AGENT_PROJECT_ROOT environment variable required")

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool as MCPToolType, TextContent

server = Server("memory")

# ── 模块级单例（Embedding + ChromaDB）────────────────────────────────
_embedding = None
_chroma_client = None
_chroma_collections: dict[str, object] = {}


def _get_embedding():
    """单例：BGE-small-zh-v1.5 嵌入模型"""
    global _embedding
    if _embedding is None:
        sys.path.insert(0, PROJECT_ROOT)
        from backend.memory.embedding import LocalEmbedding
        _embedding = LocalEmbedding()
    return _embedding


def _get_chroma_client():
    """单例：ChromaDB PersistentClient，与 episodic/semantic 共享同一持久化目录"""
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        chroma_path = os.path.join(PROJECT_ROOT, "data", "chroma")
        _chroma_client = chromadb.PersistentClient(path=chroma_path)
    return _chroma_client


def _get_rag_collection():
    """获取或创建 rag_documents collection"""
    collection_name = "rag_documents"
    if collection_name not in _chroma_collections:
        client = _get_chroma_client()
        _chroma_collections[collection_name] = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collections[collection_name]


# ── 分块 ─────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[dict]:
    """段落优先切分。chunk_size=400 确保加 overlap 后不超 BGE 512 token 上限。"""
    paragraphs = text.split("\n\n")
    chunks = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
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

    result = []
    for i, chunk_text in enumerate(chunks):
        if i > 0:
            prev_tail = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]
            chunk_text = prev_tail + "\n" + chunk_text
        result.append({"chunk_id": f"chunk_{i}", "text": chunk_text, "index": i, "total": len(chunks)})
    return result


# ── 批量嵌入 ─────────────────────────────────────────────────────────

def _embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """分批嵌入，避免大文档内存溢出"""
    emb = _get_embedding()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        all_embeddings.extend(emb.embed(texts[i:i + batch_size]))
    return all_embeddings


# ── 工具定义 ──────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools():
    return [
        MCPToolType(
            name="memory_search",
            description="搜索/保存/列出跨会话长期记忆。action='search'搜索历史(需query) / 'save'保存(key+value) / 'list'列出最近。用于记住用户偏好、历史结论、项目上下文。",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "search / save / list"},
                    "query": {"type": "string", "description": "搜索关键词（action=search时必填）"},
                    "key": {"type": "string", "description": "记忆标题（action=save时必填）"},
                    "value": {"type": "string", "description": "记忆内容（action=save时必填）"},
                    "user_id": {"type": "integer", "description": "internal: user id"},
                },
                "required": ["action"],
            },
        ),
        MCPToolType(
            name="list_downloads",
            description="列出/搜索当前用户生成的所有可下载文件（文档/Excel/报告等）。可选user_search参数搜索文件名。返回文件名、大小、时间、下载链接。",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_search": {"type": "string", "description": "可选，搜索文件名关键词"},
                    "user_id": {"type": "integer", "description": "internal: user id"},
                },
                "required": [],
            },
        ),
        MCPToolType(
            name="rag_ingest",
            description="将文档内容存入知识库，支持追加。用于保存用户上传的文档、Agent抓取的网页等。同一内容不会重复入库。",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "要入库的文本内容"},
                    "source": {"type": "string", "description": "来源标识（文件名或URL）"},
                    "title": {"type": "string", "description": "文档标题（可选，默认取source）"},
                    "user_id": {"type": "integer", "description": "internal: user id"},
                },
                "required": ["content", "source"],
            },
        ),
        MCPToolType(
            name="rag_search",
            description="在知识库中语义搜索相关文档片段。返回top-k个最相关的文本块及来源。用于回答需要引用文档内容的问题。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询"},
                    "top_k": {"type": "integer", "description": "返回结果数，默认5，最大10"},
                    "user_id": {"type": "integer", "description": "internal: user id"},
                },
                "required": ["query"],
            },
        ),
    ]


# ── Handler ──────────────────────────────────────────────────────────

async def handle_memory_search(action: str, query: str = "", key: str = "", value: str = "", **kwargs) -> dict:
    """搜索/保存/列出跨会话记忆"""
    # user_id 从 arguments 获取（MCPClientManager 注入）
    user_id = kwargs.get("user_id", 0)

    sys.path.insert(0, PROJECT_ROOT)
    from backend.memory.manager import MemoryManager
    mgr = MemoryManager(user_id=user_id)

    if action == "save" and key and value:
        mid = await mgr.add(content=f"{key}: {value}", memory_type="semantic", importance=0.7)
        return {"action": "save", "key": key, "value": value[:200], "memory_id": mid, "status": "saved"}

    elif action == "list":
        items = await mgr.search(query="", memory_types=["episodic", "semantic"], limit=10)
        return {"action": "list", "memories": [i.to_dict() for i in items], "count": len(items)}

    else:  # search
        if not query:
            items = await mgr.search(query="", memory_types=["episodic", "semantic"], limit=5)
        else:
            items = await mgr.search(query=query, memory_types=["episodic", "semantic"], limit=5)
        return {"action": "search", "query": query, "memories": [i.to_dict() for i in items], "count": len(items)}


async def handle_list_downloads(user_search: str = "", **kwargs) -> dict:
    """列出当前用户生成的可下载文件"""
    user_id = kwargs.get("user_id", 0)

    sys.path.insert(0, PROJECT_ROOT)
    from backend.database import get_db

    conn = get_db("memory")
    if user_search:
        rows = conn.execute(
            "SELECT filename, size_bytes, created_at FROM downloads WHERE user_id = ? AND filename LIKE ? ORDER BY created_at DESC",
            (user_id, f"%{user_search}%"),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT filename, size_bytes, created_at FROM downloads WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    conn.close()

    from urllib.parse import quote as _quote
    files = [
        {"name": r["filename"], "size_kb": round(r["size_bytes"]/1024, 1) if r["size_bytes"] else 0,
         "created": r["created_at"]}
        for r in rows
    ]
    dl_base = "/api/download/"
    for f in files:
        f["url"] = dl_base + _quote(f["name"], safe='/')
    return {"files": files, "count": len(files), "user_id": user_id}


async def handle_rag_ingest(content: str, source: str, title: str = "", **kwargs) -> dict:
    """文档入库：分块 → 嵌入 → 存 ChromaDB"""
    user_id = kwargs.get("user_id", 0)

    if not content or not content.strip():
        return {"ok": False, "error": "content is empty"}

    # 去重：对全量 content 做 MD5
    content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
    collection = _get_rag_collection()

    # 查询是否已有同 hash 的记录
    try:
        existing = collection.get(
            where={"$and": [{"source": source}, {"content_hash": content_hash}, {"user_id": user_id}]}
        )
        if existing and existing["ids"]:
            return {"ok": True, "skipped": True, "reason": "duplicate content", "source": source}
    except Exception:
        pass  # collection 为空时 get() 可能抛异常，忽略

    # 查询已有分块的最大 index
    try:
        existing_chunks = collection.get(
            where={"$and": [{"source": source}, {"user_id": user_id}]}
        )
        if existing_chunks and existing_chunks["metadatas"]:
            max_idx = max(int(m.get("chunk_index", -1)) for m in existing_chunks["metadatas"])
            start_index = max_idx + 1
        else:
            start_index = 0
    except Exception:
        start_index = 0

    # 分块
    chunks = _chunk_text(content.strip())
    if not chunks:
        return {"ok": False, "error": "no valid chunks after splitting"}

    # 准备数据
    doc_title = title or source
    ingested_at = __import__('time').time()

    ids = []
    documents = []
    metadatas = []
    for c in chunks:
        chunk_idx = start_index + c["index"]
        ids.append(f"{user_id}_{hashlib.md5(source.encode()).hexdigest()[:8]}_chunk_{chunk_idx}")
        documents.append(c["text"])
        metadatas.append({
            "user_id": user_id,
            "source": source,
            "title": doc_title,
            "chunk_index": chunk_idx,
            "total_chunks": start_index + c["total"],
            "content_hash": content_hash,
            "ingested_at": ingested_at,
        })

    # 批量嵌入 + 写入
    embeddings = _embed_batch([c["text"] for c in chunks])
    collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    return {
        "ok": True,
        "source": source,
        "title": doc_title,
        "chunks": len(chunks),
        "chunk_range": f"{start_index}-{start_index + len(chunks) - 1}",
    }


async def handle_rag_search(query: str, top_k: int = 5, **kwargs) -> dict:
    """知识库语义检索"""
    user_id = kwargs.get("user_id", 0)
    top_k = min(max(top_k, 1), 10)

    if not query or not query.strip():
        return {"results": [], "query": query, "count": 0}

    collection = _get_rag_collection()
    emb = _get_embedding()
    query_vec = emb.embed([query.strip()])[0]

    try:
        chroma_result = collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where={"user_id": user_id},
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return {"results": [], "query": query, "count": 0}

    results = []
    if chroma_result and chroma_result["ids"] and chroma_result["ids"][0]:
        for i, doc_id in enumerate(chroma_result["ids"][0]):
            meta = chroma_result["metadatas"][0][i] if chroma_result["metadatas"] else {}
            distance = chroma_result["distances"][0][i] if chroma_result["distances"] else 0.0
            score = round(1.0 - distance, 4)
            results.append({
                "rank": i + 1,
                "score": score,
                "source": meta.get("source", ""),
                "title": meta.get("title", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "chunk_range": f"{meta.get('chunk_index', 0) + 1}/{meta.get('total_chunks', 0)}",
                "text": chroma_result["documents"][0][i] if chroma_result["documents"] else "",
            })

    return {"results": results, "query": query, "count": len(results)}


TOOL_HANDLERS = {
    "memory_search": handle_memory_search,
    "list_downloads": handle_list_downloads,
    "rag_ingest": handle_rag_ingest,
    "rag_search": handle_rag_search,
}


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    import json
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False))]

    try:
        result = await handler(**arguments)
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]


async def main():
    # 预热：提前加载并触发嵌入模型初始化（首次 ~30s，后续秒级）
    emb = _get_embedding()
    emb.embed(["warmup"])
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
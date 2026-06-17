"""
RAGTool — 已实现（2026-06-17）
文档解析 → 分块 → 嵌入 → ChromaDB → 问答。

实现位置：backend/protocols/mcp/servers/memory_server.py
  - rag_ingest: 文档入库（分块 + 嵌入 + ChromaDB）
  - rag_search: 知识检索（语义搜索 + top-k 返回）

此文件保留作为 Python API 入口的预留位置。
若需要进程内调用（非 MCP），可在此实现 Thin Wrapper。
"""


class RAGTool:
    """RAG 管道已由 MCP memory_server 实现。
    使用方式:
      - Agent: 直接调用 rag_ingest / rag_search MCP 工具
      - 进程内: from backend.memory.tools.rag_tool import RAGTool (待实现)
    """
    pass

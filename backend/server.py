"""
统一 FastAPI 后端 — 融合 agent_learning + codex_test

Agent 引擎（2 种模式）:
  - react (DeepSeek ReAct — 增强版 Token 预算 + 并行执行 + 自动反思)
  - plan_solve (先规划再执行)

接口 (18 REST + 1 WebSocket):
  POST /api/register              — 注册
  POST /api/login                 — 登录
  POST /api/chat                  — 非流式对话
  POST /api/chat/stream           — 流式对话 (SSE)
  POST /api/upload                — 文件上传 (含 PDF 解析)
  GET  /api/conversations         — 对话列表
  POST /api/conversations         — 创建对话
  GET  /api/conversations/{id}    — 对话详情
  POST /api/agent/tasks           — 创建 Agent 任务
  GET  /api/agent/tasks           — 任务列表
  GET  /api/agent/tasks/{id}      — 任务详情
  DELETE /api/agent/tasks/{id}    — 删除任务
  GET  /api/claude-config         — Claude Code 配置
  PUT  /api/claude-config         — 更新 Claude Code 配置
  GET  /api/logs/stats            — 日志统计
  GET  /api/logs/suggestions      — Prompt 优化建议
  GET  /api/memory                — 长期记忆列表
  POST /api/memory                — 保存记忆
  DELETE /api/memory/{key}        — 删除记忆
  WS   /ws/agent/{task_id}        — Agent 实时进度

启动方式:
  python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
"""
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from fastapi.responses import RedirectResponse

from .database import init_db
from .dependencies import (
    PROJECT_ROOT, STATIC_DIR, deepseek, tool_registry, mcp_manager, ws_manager,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ─────────────────────────────────────────────
    init_db()

    # 1. 注册本地工具（frozen 之前）
    from .agent.tools import LOCAL_TOOLS
    for tool in LOCAL_TOOLS:
        tool_registry.register_local(tool)

    # 2. 连接 MCP Server + 发现工具 → freeze
    try:
        await tool_registry.initialize(mcp_manager)
        mcp_ok = True
    except Exception as e:
        print(f"[WARN] MCP Server 初始化失败（仅本地工具可用）: {e}")
        mcp_ok = False

    total_tools = len(tool_registry.get_all_tools())
    mcp_count = total_tools - len(LOCAL_TOOLS)
    print(f"数据库已初始化 (chat/agent/memory)")
    print(f"工具注册完成: {total_tools} 个 ({len(LOCAL_TOOLS)} local + {mcp_count} MCP{' [部分失败]' if not mcp_ok else ''})")
    print(f"Agent 模式: react / plan_solve")
    print(f"LLM: Ollama (qwen3:14b)")
    print(f"API 文档: http://127.0.0.1:8000/docs")

    yield

    # ── Shutdown ────────────────────────────────────────────
    await tool_registry.shutdown()


# ── App ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Agent Learning — Unified", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Security headers middleware
@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-cache"
    # Static files get long cache
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "public, max-age=86400"
    return response

# ── Mount routers ──────────────────────────────────────────────────────

from .routes.auth import router as auth_router
from .routes.chat import router as chat_router
from .routes.conversations import router as conversations_router
from .routes.agent import router as agent_router
from .routes.websocket import router as ws_router
from .routes.memory import router as memory_router
from .routes.downloads import router as downloads_router
from .routes.logs import router as logs_router
from .evaluation.router import router as eval_router

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(agent_router)
app.include_router(ws_router)
app.include_router(memory_router)
app.include_router(downloads_router)
app.include_router(logs_router)
app.include_router(eval_router)


# ── Root redirect ───────────────────────────────────────────────────────

@app.get("/")
async def root():
    return RedirectResponse(url="/app")


# ── Static files mount ─────────────────────────────────────────────────

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.mount("/app", StaticFiles(directory=STATIC_DIR, html=True), name="app")


# ── Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run("backend.server:app", host=args.host, port=args.port, reload=True)
    # If above fails with relative import error, run instead:
    #   cd D:\agent_learning && python -m backend.server

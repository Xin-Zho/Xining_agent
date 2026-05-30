"""
FastAPI 后端 — 为聊天前端提供 API 服务。

两个核心接口：
  POST /api/chat         — 非流式对话（一次性返回完整回复）
  POST /api/chat/stream  — 流式对话（SSE 逐 token 推送，打字机效果）

启动方式：
  cd step2_web
  uvicorn backend:app --reload --port 8000
"""
import sys
import os
import json

# 把 chat/ 目录加到 sys.path，这样能导入 LLMClient
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "chat"
))

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llm_client import LLMClient

app = FastAPI(title="Agent Chat")

# CORS：允许任何来源访问（开发阶段宽松，生产环境应限制）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局单例 LLM 客户端
client = LLMClient()


class ChatRequest(BaseModel):
    """前端发来的请求体格式"""
    messages: list[dict]  # [{"role": "user", "content": "..."}, ...]


# ============================================================
# 接口 1：非流式对话（兜底方案）
# ============================================================
@app.post("/api/chat")
async def chat(req: ChatRequest):
    """发送 messages，返回完整回复。简单但用户要等。"""
    reply = client.chat(req.messages)
    return {"reply": reply}


# ============================================================
# 接口 2：流式对话（主力方案）
# ============================================================
@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    流式对话 — SSE (Server-Sent Events) 格式。
    模型每生成一个 token，就推送给前端。
    前端实时显示，体验像 ChatGPT 打字。
    """
    def generate():
        full_reply = ""
        for token in client.chat_stream(req.messages):
            full_reply += token
            # SSE 格式：data: <json>\n\n
            yield f"data: {json.dumps({'token': token})}\n\n"
        # 发送完成信号
        yield f"data: {json.dumps({'done': True, 'full_reply': full_reply})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",       # 不要缓存流
            "Connection": "keep-alive",         # 保持连接
            "X-Accel-Buffering": "no",          # 禁用代理缓冲
        }
    )


# ============================================================
# 托管前端静态文件
# 注：必须在路由定义之后 mount，FastAPI 按注册顺序匹配
# ============================================================
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

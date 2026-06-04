"""
统一 FastAPI 后端 — 融合 agent_learning + codex_test

Agent 引擎（2 种模式）:
  - react (DeepSeek ReAct — 增强版 Token 预算 + 并行执行 + 自动反思)
  - plan_solve (先规划再执行)

接口列表 (18 REST + 1 WebSocket):
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
import io
import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

from .database import get_db, init_db
from .auth import (
    create_token, get_current_user, verify_token,
    hash_password, verify_password, security, SECRET_KEY,
)
from .models import (
    AuthRequest, ChatRequest, CreateConversationRequest, CreateAgentTaskRequest,
)
from .agent import AgentEngine, PlanSolveEngine, TOOLS, WebSocketManager
from .llm_client import LLMClient
from .context_manager import ContextManager
from .training import DialogueLogger, RuleExtractor
from .memory import LongTermMemory

# ── Config ──────────────────────────────────────────────────────────────

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
ALLOW_REGISTRATION = os.environ.get("ALLOW_REGISTRATION", "false").strip().lower() != "false"
INVITE_CODE = os.environ.get("INVITE_CODE", "xin-agent-2026")
MAX_HISTORY_ROUNDS = 20

SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely in Chinese. IMPORTANT: If asked about real-time events, specific dates, or factual data you are unsure about, you MUST tell the user you don't have real-time access and suggest switching to Agent mode for tool-based verification. Never fabricate earthquake reports, stock prices, news events, or weather data. 用中文回复。"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ALLOWED_IMAGE = {"image/jpeg", "image/png", "image/gif", "image/webp"}
ALLOWED_TEXT = {"text/plain", "application/json", "text/csv",
                "text/markdown", "text/x-python", "application/pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def _has_valid_deepseek_key() -> bool:
    return bool(DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your-api-key-here")


# ── Dependencies ────────────────────────────────────────────────────────

deepseek = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com") \
    if _has_valid_deepseek_key() else None

ws_manager = WebSocketManager()
llm_client = LLMClient() if deepseek else None
ctx_manager = ContextManager(llm_client)
dialogue_logger = DialogueLogger()

if deepseek is None:
    raise RuntimeError("DEEPSEEK_API_KEY is required to run Agent engine.")


def _get_engine(agent_mode: str):
    """为每个请求创建独立的引擎实例，保证会话隔离"""
    if agent_mode == "plan_solve":
        return PlanSolveEngine(deepseek, TOOLS, ws_manager)
    else:
        return AgentEngine(deepseek, TOOLS, ws_manager)


# ── App ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Agent Learning — Unified")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth routes ─────────────────────────────────────────────────────────

@app.post("/api/register")
def register(body: AuthRequest):
    if not ALLOW_REGISTRATION:
        raise HTTPException(status_code=403, detail="注册已关闭")
    if getattr(body, 'invite_code', '') != INVITE_CODE:
        raise HTTPException(status_code=403, detail="认证码错误")
    if not body.username.strip() or len(body.password) < 8:
        raise HTTPException(status_code=422, detail="用户名不能为空，密码至少4位")

    conn = get_db("chat")
    existing = conn.execute(
        "SELECT id FROM users WHERE username = ?", (body.username.strip(),)
    ).fetchone()
    if existing:
        conn.close()
        raise HTTPException(status_code=409, detail="用户名已被注册")

    hash_ = hash_password(body.password)
    cur = conn.execute(
        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
        (body.username.strip(), hash_),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return {"token": create_token(user_id), "username": body.username.strip()}


@app.post("/api/login")
def login(body: AuthRequest):
    conn = get_db("chat")
    user = conn.execute(
        "SELECT id, username, password_hash FROM users WHERE username = ?",
        (body.username.strip(),),
    ).fetchone()
    conn.close()

    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    return {"token": create_token(user["id"]), "username": user["username"]}


# ── Chat routes ─────────────────────────────────────────────────────────

@app.post("/api/chat")
def chat(body: ChatRequest, user: dict = Depends(get_current_user)):
    conn = get_db("chat")

    if deepseek is None:
        conn.close()
        raise HTTPException(status_code=503, detail="DeepSeek chat is not configured.")

    conv = conn.execute(
        "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
        (body.conversation_id, user["id"]),
    ).fetchone()
    if not conv:
        conn.close()
        raise HTTPException(status_code=404, detail="对话不存在")

    conn.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (?, 'user', ?)",
        (body.conversation_id, body.message),
    )

    current_title = conn.execute(
        "SELECT title FROM conversations WHERE id = ?", (body.conversation_id,)
    ).fetchone()["title"]
    if current_title == "新对话":
        title = body.message[:30] + ("..." if len(body.message) > 30 else "")
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (title, body.conversation_id),
        )

    rows = conn.execute(
        """SELECT role, content FROM messages
           WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?""",
        (body.conversation_id, MAX_HISTORY_ROUNDS * 2),
    ).fetchall()
    rows = list(reversed(rows))

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for r in rows:
        messages.append({"role": r["role"], "content": r["content"]})

    try:
        resp = deepseek.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
        )
        reply = resp.choices[0].message.content
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=502, detail=f"AI API 调用失败: {e}")

    conn.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (?, 'assistant', ?)",
        (body.conversation_id, reply),
    )
    conn.commit()
    conn.close()

    dialogue_logger.log(
        user=user["username"], question=body.message, answer=reply,
        source="chat", model="deepseek-chat",
        tokens=resp.usage.total_tokens if resp.usage else 0,
    )

    return {"reply": reply, "conversation_id": body.conversation_id}


@app.post("/api/chat/stream")
async def chat_stream(req: dict | ChatRequest):
    """
    流式对话 — SSE 格式
    兼容两种格式:
      - 旧前端: {messages: [...], model: "chat"|"reasoner"}
      - 新移动端: {conversation_id: int, message: str}
    """
    if deepseek is None:
        raise HTTPException(status_code=503, detail="DeepSeek chat is not configured.")

    # 判断请求格式
    if isinstance(req, dict) and "messages" in req:
        # 旧前端格式
        messages = list(req["messages"])
        model_id = "deepseek-reasoner" if req.get("model") == "reasoner" else "deepseek-chat"
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, list):
                    user_msg = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
                else:
                    user_msg = str(content)
                break
    elif hasattr(req, 'conversation_id'):
        # 新格式
        conn = get_db("chat")
        conv = conn.execute(
            "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
            (req.conversation_id, get_current_user.__wrapped__),
        ).fetchone()
        conn.close()
        if not conv:
            raise HTTPException(status_code=404, detail="对话不存在")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        conn2 = get_db("chat")
        rows = conn2.execute(
            """SELECT role, content FROM messages
               WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?""",
            (req.conversation_id, MAX_HISTORY_ROUNDS * 2),
        ).fetchall()
        conn2.close()
        for r in reversed(rows):
            messages.append({"role": r["role"], "content": r["content"]})
        user_msg = req.message
        model_id = "deepseek-chat"
    else:
        raise HTTPException(status_code=422, detail="无效的请求格式")

    cl = deepseek

    def generate():
        full_reply = ""
        try:
            stream = cl.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.7,
                max_tokens=4096,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_reply += delta.content
                    yield f"data: {json.dumps({'token': delta.content})}\n\n"

            dialogue_logger.log(
                user="anonymous", question=user_msg[:200], answer=full_reply[:2000],
                source="chat", model=model_id,
            )
            yield f"data: {json.dumps({'done': True, 'full_reply': full_reply})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ── File upload ─────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload(file: UploadFile = File(...),
                 user: dict = Depends(get_current_user)):
    content_bytes = await file.read()

    if len(content_bytes) > MAX_FILE_SIZE:
        raise HTTPException(400, f"文件过大，最大 {MAX_FILE_SIZE // 1024 // 1024}MB")

    content_type = file.content_type or "application/octet-stream"

    # 图片：返回 base64
    if content_type in ALLOWED_IMAGE:
        import base64
        b64 = base64.b64encode(content_bytes).decode("ascii")
        return {
            "ok": True,
            "filename": file.filename,
            "content_type": content_type,
            "is_image": True,
            "content": f"data:{content_type};base64,{b64}",
        }

    # 文本文件
    if content_type in ALLOWED_TEXT:
        try:
            text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content_bytes.decode("gbk")
            except Exception:
                raise HTTPException(400, "无法解码文件内容，请确认文件编码为 UTF-8")

        # PDF 特殊处理
        if content_type == "application/pdf":
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(io.BytesIO(content_bytes))
                text_parts = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                pdf_text = "\n--- 分页 ---\n".join(text_parts)
                if not pdf_text.strip():
                    raise HTTPException(400, "PDF 文件中未提取到文字（可能是扫描件或图片型 PDF）")
                return {
                    "ok": True,
                    "filename": file.filename,
                    "content_type": content_type,
                    "is_image": False,
                    "content": f"[PDF: {file.filename}]\n{pdf_text[:80000]}",
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(400, f"PDF 解析失败：{str(e)[:100]}")

        return {
            "ok": True,
            "filename": file.filename,
            "content_type": content_type,
            "is_image": False,
            "content": text,
        }

    allowed = ALLOWED_IMAGE | ALLOWED_TEXT
    allowed_str = ", ".join(sorted(allowed))
    raise HTTPException(400, f"不支持的文件类型。允许: {allowed_str}")


# ── Conversation routes ─────────────────────────────────────────────────

@app.get("/api/conversations")
def list_conversations(user: dict = Depends(get_current_user)):
    conn = get_db("chat")
    rows = conn.execute(
        """SELECT id, title, created_at FROM conversations
           WHERE user_id = ? ORDER BY created_at DESC""",
        (user["id"],),
    ).fetchall()
    conn.close()
    return [
        {"id": r["id"], "title": r["title"], "created_at": r["created_at"]}
        for r in rows
    ]


@app.post("/api/conversations")
def create_conversation(
    body: CreateConversationRequest, user: dict = Depends(get_current_user)
):
    conn = get_db("chat")
    cur = conn.execute(
        "INSERT INTO conversations (user_id, title) VALUES (?, ?)",
        (user["id"], body.title),
    )
    conn.commit()
    conv_id = cur.lastrowid
    conn.close()
    return {"id": conv_id, "title": body.title}


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: int, user: dict = Depends(get_current_user)):
    conn = get_db("chat")
    conv = conn.execute(
        "SELECT id, title, created_at FROM conversations WHERE id = ? AND user_id = ?",
        (conversation_id, user["id"]),
    ).fetchone()
    if not conv:
        conn.close()
        raise HTTPException(status_code=404, detail="对话不存在")

    msgs = conn.execute(
        """SELECT role, content, created_at FROM messages
           WHERE conversation_id = ? ORDER BY created_at""",
        (conversation_id,),
    ).fetchall()
    conn.close()
    return {
        "id": conv["id"],
        "title": conv["title"],
        "created_at": conv["created_at"],
        "messages": [
            {"role": m["role"], "content": m["content"], "created_at": m["created_at"]}
            for m in msgs
        ],
    }


# ── Agent task routes ───────────────────────────────────────────────────

@app.post("/api/agent/tasks")
async def create_agent_task(
    body: CreateAgentTaskRequest, user: dict = Depends(get_current_user)
):
    agent_mode = body.agent_mode or "react"
    conn = get_db("agent")
    cur = conn.execute(
        """INSERT INTO agent_tasks (user_id, title, description, status, agent_mode, conversation_id)
           VALUES (?, ?, ?, 'pending', ?, ?)""",
        (user["id"], body.description[:50], body.description, agent_mode, body.conversation_id),
    )
    conn.commit()
    task_id = cur.lastrowid
    conn.close()

    engine = _get_engine(agent_mode)

    asyncio.create_task(engine.run(
        task_description=body.description,
        user_id=user["id"],
        task_id=task_id,
    ))

    return {"task_id": task_id, "status": "pending", "agent_mode": agent_mode}


@app.get("/api/agent/tasks")
def list_agent_tasks(user: dict = Depends(get_current_user)):
    conn = get_db("agent")
    rows = conn.execute(
        """SELECT id, title, status, agent_mode, total_tokens, duration_ms, created_at
           FROM agent_tasks WHERE user_id = ? ORDER BY created_at DESC""",
        (user["id"],),
    ).fetchall()
    conn.close()
    return [
        {
            "id": r["id"], "title": r["title"], "status": r["status"],
            "agent_mode": r["agent_mode"],
            "total_tokens": r["total_tokens"], "duration_ms": r["duration_ms"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


@app.get("/api/agent/tasks/{task_id}")
def get_agent_task(task_id: int, user: dict = Depends(get_current_user)):
    conn = get_db("agent")
    task = conn.execute(
        "SELECT * FROM agent_tasks WHERE id = ? AND user_id = ?",
        (task_id, user["id"]),
    ).fetchone()
    if not task:
        conn.close()
        raise HTTPException(status_code=404, detail="任务不存在")

    steps = conn.execute(
        """SELECT step_number, status, step_type, tool_name, tool_args,
                  tool_result, thought, duration_ms, created_at
           FROM agent_steps WHERE task_id = ? ORDER BY step_number""",
        (task_id,),
    ).fetchall()
    conn.close()

    return {
        "id": task["id"],
        "title": task["title"],
        "description": task["description"],
        "status": task["status"],
        "agent_mode": task["agent_mode"],
        "plan_json": task["plan_json"],
        "final_answer": task["final_answer"],
        "conversation_id": task["conversation_id"],
        "total_tokens": task["total_tokens"],
        "duration_ms": task["duration_ms"],
        "created_at": task["created_at"],
        "steps": [
            {
                "step_number": s["step_number"], "status": s["status"],
                "step_type": s["step_type"], "tool_name": s["tool_name"],
                "tool_args": s["tool_args"], "tool_result": s["tool_result"],
                "thought": s["thought"], "duration_ms": s["duration_ms"],
            }
            for s in steps
        ],
    }


@app.delete("/api/agent/tasks/{task_id}")
def delete_agent_task(task_id: int, user: dict = Depends(get_current_user)):
    conn = get_db("agent")
    task = conn.execute(
        "SELECT id FROM agent_tasks WHERE id = ? AND user_id = ?",
        (task_id, user["id"]),
    ).fetchone()
    if not task:
        conn.close()
        raise HTTPException(status_code=404, detail="任务不存在")
    conn.execute("DELETE FROM agent_tasks WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


# ── Tool confirmation ───────────────────────────────────────────────────

@app.post("/api/agent/confirm/{task_id}")
def confirm_tool(task_id: int, body: dict):
    """确认或拒绝工具执行"""
    step = body.get("step_number")
    approved = body.get("approved", False)
    key = f"{task_id}_{step}"
    ws_manager.confirmations[key] = {"approved": approved}
    return {"ok": True, "task_id": task_id, "step": step, "approved": approved}


# ── WebSocket ───────────────────────────────────────────────────────────

@app.websocket("/ws/agent/{task_id}")
async def agent_websocket(websocket: WebSocket, task_id: int):
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4001, reason="Missing token")
        return
    try:
        user = verify_token(token)
    except HTTPException:
        await websocket.close(code=4002, reason="Invalid token")
        return

    conn = get_db("agent")
    task = conn.execute(
        "SELECT id, agent_mode FROM agent_tasks WHERE id = ? AND user_id = ?",
        (task_id, user["id"]),
    ).fetchone()
    conn.close()
    if not task:
        await websocket.close(code=4004, reason="Task not found")
        return

    await ws_manager.connect(task_id, websocket, user["id"])

    try:
        while True:
            msg = await websocket.receive_json()
            action = msg.get("action")
            if action == "cancel":
                engine = _get_engine(task["agent_mode"])
                await engine.cancel(task_id)
            elif action == "confirm":
                ws_manager.confirmations[task_id] = msg
    except WebSocketDisconnect:
        ws_manager.disconnect(task_id)


# ── Logs & Insights routes ──────────────────────────────────────────────

@app.get("/api/logs/stats")
def logs_stats(user: dict = Depends(get_current_user)):
    return dialogue_logger.stats()


@app.get("/api/logs/suggestions")
def logs_suggestions(user: dict = Depends(get_current_user)):
    logs = dialogue_logger.get_agent_logs(days=30)
    extractor = RuleExtractor(logs)
    suggestions = extractor.generate_prompt_suggestions()
    return {"ok": True, "total_logs": len(logs), "suggestions": suggestions}


# ── Memory routes ───────────────────────────────────────────────────────

@app.get("/api/memory")
def list_memory(user: dict = Depends(get_current_user)):
    ltm = LongTermMemory(user["id"])
    return {"memories": ltm.list_all()}


@app.post("/api/memory")
def save_memory(body: dict, user: dict = Depends(get_current_user)):
    key = body.get("key", "").strip()
    value = body.get("value", "").strip()
    if not key or not value:
        raise HTTPException(400, "key 和 value 不能为空")
    ltm = LongTermMemory(user["id"])
    ltm.save(key, value)
    return {"ok": True, "key": key}


@app.delete("/api/memory/{key}")
def delete_memory(key: str, user: dict = Depends(get_current_user)):
    ltm = LongTermMemory(user["id"])
    ltm.delete(key)
    return {"ok": True}


# ── Agent modes info ────────────────────────────────────────────────────

@app.get("/api/agent/modes")
def agent_modes():
    return {
        "default": "react",
        "available": [
            {"id": "react", "name": "ReAct Agent", "description": "思考→行动→观察循环，Token预算 + 并行执行 + 自动反思"},
            {"id": "plan_solve", "name": "Plan-Solve", "description": "先制定计划，再逐步执行，最后汇总"},
        ],
    }


# ── Downloads (user-scoped) ────────────────────────────────────────────

@app.get("/api/download/{filename:path}")
def download_file(filename: str, token: str = None, user: dict = Depends(get_current_user)):
    """下载文件（检查归属权限）。支持 ?token= 查询参数或 Authorization header。"""
    # 兼容查询参数 token（浏览器直接打开链接时无 header）
    if not user and token:
        try:
            user = verify_token(token)
        except Exception:
            raise HTTPException(401, "无效的下载链接，请刷新页面重新生成")

    dl_dir = os.path.join(STATIC_DIR, "downloads")
    filepath = os.path.join(dl_dir, os.path.basename(filename))
    if not os.path.isfile(filepath):
        raise HTTPException(404, "文件不存在")

    if user.get("username") == "admin":
        return FileResponse(filepath, filename=os.path.basename(filename))

    conn = get_db("memory")
    row = conn.execute(
        "SELECT user_id FROM downloads WHERE filepath = ? ORDER BY id DESC LIMIT 1",
        (filepath,),
    ).fetchone()
    conn.close()

    if not row or row["user_id"] != user["id"]:
        raise HTTPException(403, "无权访问此文件")

    return FileResponse(filepath, filename=os.path.basename(filename))


@app.get("/api/downloads")
def list_downloads(user: dict = Depends(get_current_user)):
    """列出当前用户的可下载文件"""
    conn = get_db("memory")
    rows = conn.execute(
        "SELECT filename, filepath, size_bytes, created_at FROM downloads WHERE user_id = ? ORDER BY created_at DESC",
        (user["id"],),
    ).fetchall()
    conn.close()
    from urllib.parse import quote
    files = [
        {"name": r["filename"], "size": r["size_bytes"],
         "url": f"/api/download/{quote(r['filename'], safe='/')}",
         "created": r["created_at"]}
        for r in rows
    ]
    return {"files": files, "count": len(files)}


# ── Static files ────────────────────────────────────────────────────────

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "static")

if os.path.isdir(STATIC_DIR):
    @app.get("/app")
    async def web_app():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    @app.get("/downloads")
    async def downloads_page(user: dict = Depends(get_current_user)):
        from urllib.parse import quote
        conn = get_db("memory")
        rows = conn.execute(
            "SELECT filename, size_bytes, created_at FROM downloads WHERE user_id = ? ORDER BY created_at DESC",
            (user["id"],),
        ).fetchall()
        conn.close()
        files_html = ""
        for r in rows:
            sz = r["size_bytes"]
            sz_str = f"{sz/1024:.0f}KB" if sz > 1024 else f"{sz}B"
            encoded = quote(r["filename"], safe='/')
            files_html += f'<tr><td><a href="/api/download/{encoded}">📥 {r["filename"]}</a></td><td>{sz_str}</td><td>{r["created_at"]}</td></tr>'
        html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><title>下载文件</title>
<style>body{{font-family:system-ui,sans-serif;max-width:800px;margin:40px auto;padding:0 20px;background:#FAFAFB;color:#18181B}}
h1{{font-size:24px;margin-bottom:8px}}table{{width:100%;border-collapse:collapse;margin-top:20px}}
th{{text-align:left;padding:10px 12px;border-bottom:2px solid #E5E7EB;font-size:12px;color:#71717A;text-transform:uppercase;letter-spacing:0.5px}}td{{padding:10px 12px;border-bottom:1px solid #E5E7EB}}a{{color:#5B6AF0;text-decoration:none}}a:hover{{text-decoration:underline}}
.back{{display:inline-block;margin-top:24px;color:#71717A;font-size:14px}}</style></head><body>
<h1>📂 下载文件</h1><p>Agent 生成的所有可下载文件</p>
<table><tr><th>文件</th><th>大小</th><th>时间</th></tr>{files_html or '<tr><td colspan=3>暂无文件</td></tr>'}</table>
<a class="back" href="/app">← 返回聊天</a></body></html>"""
        return HTMLResponse(content=html)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
    return {
        "name": "Agent Learning — Unified",
        "version": "2.0.0",
        "agent_mode": "react",
        "endpoints": {
            "auth": ["/api/register", "/api/login"],
            "chat": ["/api/chat", "/api/chat/stream"],
            "upload": ["/api/upload"],
            "conversations": ["/api/conversations", "/api/conversations/{id}"],
            "agent_tasks": ["/api/agent/tasks", "/api/agent/tasks/{id}"],
            "agent_modes": ["/api/agent/modes"],
            "logs": ["/api/logs/stats", "/api/logs/suggestions"],
            "memory": ["/api/memory"],
            "websocket": ["/ws/agent/{task_id}"],
        },
    }


# ── 前端兼容路由 (agent_learning Apple HIG 前端) ────────────────────
# 原 agent_learning 前端使用 /api/auth/* 和 /api/agent/stream 等路径

class LegacyChatRequest(BaseModel):
    messages: list[dict]
    model: str = "chat"


class LegacyAgentRequest(BaseModel):
    messages: list[dict]
    model: str = "chat"
    mode: str = "react"


def _verify_token_from_header(authorization: str | None) -> dict | None:
    """从 Authorization header 解析 token，返回 user dict 或 None"""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:]
    try:
        return verify_token(token)
    except HTTPException:
        return None


@app.post("/api/auth/register")
def compat_register(body: AuthRequest):
    """兼容前端 /api/auth/register 路径，返回 {ok, token, username} 格式"""
    if not ALLOW_REGISTRATION:
        return JSONResponse({"ok": False, "error": "注册已关闭"}, 403)
    if getattr(body, 'invite_code', '') != INVITE_CODE:
        return JSONResponse({"ok": False, "error": "认证码错误"}, 403)
    if not body.username.strip() or len(body.password) < 8:
        return JSONResponse({"ok": False, "error": "用户名不能为空，密码至少8位"}, 422)

    conn = get_db("chat")
    existing = conn.execute(
        "SELECT id FROM users WHERE username = ?", (body.username.strip(),)
    ).fetchone()
    if existing:
        conn.close()
        return JSONResponse({"ok": False, "error": "用户名已存在"}, 409)

    hash_ = hash_password(body.password)
    cur = conn.execute(
        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
        (body.username.strip(), hash_),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return {"ok": True, "token": create_token(user_id), "username": body.username.strip()}


@app.post("/api/auth/login")
def compat_login(body: AuthRequest):
    """兼容前端 /api/auth/login 路径，返回 {ok, token, username} 格式"""
    conn = get_db("chat")
    user = conn.execute(
        "SELECT id, username, password_hash FROM users WHERE username = ?",
        (body.username.strip(),),
    ).fetchone()
    conn.close()

    if not user or not verify_password(body.password, user["password_hash"]):
        return JSONResponse({"ok": False, "error": "用户名或密码错误"}, 401)

    return {"ok": True, "token": create_token(user["id"]), "username": user["username"]}


@app.get("/api/auth/me")
def compat_me(authorization: str | None = Header(None)):
    """兼容前端 /api/auth/me 路径"""
    user = _verify_token_from_header(authorization)
    if not user:
        return JSONResponse({"ok": False, "error": "未登录或 token 已过期"}, 401)
    return {"ok": True, "username": user["username"]}


@app.post("/api/agent/stream")
async def compat_agent_stream(req: LegacyAgentRequest, authorization: str | None = Header(None)):
    """
    兼容前端 /api/agent/stream — SSE 格式
    前端发送 {messages, model, mode} + Authorization header
    后台异步执行 Agent，SSE 推送步骤和结果
    """
    # 从 token 获取当前用户
    current_user = _verify_token_from_header(authorization) if authorization else None
    if deepseek is None:
        return StreamingResponse(
            iter([f"data: {json.dumps({'error': 'DeepSeek 未配置，Agent 不可用'})}\n\n"]),
            media_type="text/event-stream",
        )

    # 提取用户消息 + 对话历史上下文
    user_msg = ""
    history_context = ""
    # 分离 system 消息和对话消息
    system_msgs = [m for m in req.messages if m.get("role") == "system"]
    chat_msgs = [m for m in req.messages if m.get("role") in ("user", "assistant")]

    # 取最近 6 轮对话作为上下文
    recent = chat_msgs[-12:]  # 最多 6 轮 (user+assistant 各一)
    if len(recent) > 1:
        history_lines = []
        for m in recent[:-1]:  # 除了最后一条 user 消息
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
            role_label = "用户" if m.get("role") == "user" else "助手"
            if str(content).strip():
                history_lines.append(f"{role_label}: {str(content)[:300]}")
        if history_lines:
            history_context = "## 对话历史\n" + "\n".join(history_lines) + "\n\n"

    # 提取最后一条 user 消息
    for m in reversed(req.messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                user_msg = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
            else:
                user_msg = str(content)
            if user_msg.strip():
                break

    if not user_msg or not user_msg.strip():
        user_msg = "(empty)"

    # 拼接 Agent 的完整任务描述
    agent_system = ""
    if system_msgs:
        agent_system = "## 自定义角色\n" + system_msgs[-1].get("content", "")[:500] + "\n\n"

    full_task = f"{agent_system}{history_context}## 当前任务\n{user_msg}"

    agent_mode = req.mode or "react"
    engine = _get_engine(agent_mode)

    async def generate():
        yield f"data: {json.dumps({'type': 'start', 'mode': agent_mode})}\n\n"

        # 创建临时任务记录
        conn = get_db("agent")

        # 从 token 取当前用户 ID
        user_id = current_user["id"] if current_user else 1

        # 设置当前用户上下文（供 memory_search 工具使用）
        from .agent.tools import set_current_user
        set_current_user(user_id)

        cur = conn.execute(
            "INSERT INTO agent_tasks (user_id, title, description, status, agent_mode) VALUES (?, ?, ?, 'executing', ?)",
            (user_id, user_msg[:50], full_task, agent_mode),
        )
        conn.commit()
        task_id = cur.lastrowid
        conn.close()

        try:
            # 后台执行 Agent
            bg_task = asyncio.create_task(engine.run(full_task, user_id, task_id))

            # 轮询步骤更新，推送给前端
            last_step = 0
            while not bg_task.done():
                await asyncio.sleep(0.5)
                conn = get_db("agent")
                steps = conn.execute(
                    "SELECT * FROM agent_steps WHERE task_id = ? AND step_number > ? ORDER BY step_number",
                    (task_id, last_step),
                ).fetchall()
                conn.close()
                for s in steps:
                    last_step = s["step_number"]

                    # 检测需要确认的步骤
                    if s["status"] == "confirming":
                        yield f"data: {json.dumps({'type': 'confirmation_required', 'task_id': task_id, 'step_num': s['step_number'], 'tool_name': s['tool_name'], 'args': s['tool_args']})}\n\n"
                        # 等待用户确认（轮询 confirmations dict）
                        key = f"{task_id}_{s['step_number']}"
                        waited = 0
                        while key not in ws_manager.confirmations and waited < 120:
                            await asyncio.sleep(0.5)
                            waited += 1
                            if bg_task.done():
                                break
                        confirm = ws_manager.confirmations.pop(key, None)
                        if confirm and confirm.get("approved"):
                            yield f"data: {json.dumps({'type': 'step', 'step': {'turn': s['step_number'], 'type': 'tool_call', 'tool_name': s['tool_name'], 'thought': None, 'observation': '✅ 已允许执行'}})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'step', 'step': {'turn': s['step_number'], 'type': 'tool_call', 'tool_name': s['tool_name'], 'thought': None, 'observation': '⛔ 已取消'}})}\n\n"
                        continue

                    # 解析 tool_result 为可读文本
                    obs_text = ""
                    raw = s["tool_result"]
                    if raw:
                        try:
                            parsed = json.loads(raw)
                            # 提取嵌套的 observation
                            inner = parsed.get("observation", "")
                            if isinstance(inner, str) and inner.startswith("{"):
                                try:
                                    inner_parsed = json.loads(inner)
                                    # 针对不同工具格式化
                                    if inner_parsed.get("type") == "directory":
                                        obs_text = f"📁 {inner_parsed.get('path','.')} ({inner_parsed.get('count',0)} 项): " + ", ".join(inner_parsed.get("items", [])[:10])
                                    elif "results" in inner_parsed:
                                        items = inner_parsed["results"][:3]
                                        obs_text = f"🔍 搜索 '{inner_parsed.get('query','')}' ({inner_parsed.get('count',0)} 条)\n" + "\n".join(f"  {r.get('title','')} - {r.get('url','')}" for r in items)
                                    elif "content" in inner_parsed:
                                        obs_text = inner_parsed["content"][:500]
                                    elif "result" in inner_parsed:
                                        obs_text = f"🧮 {inner_parsed.get('expression','')} = {inner_parsed.get('result','')}"
                                    elif "output" in inner_parsed:
                                        obs_text = inner_parsed["output"][:500]
                                    elif "error" in inner_parsed:
                                        obs_text = f"❌ {inner_parsed['error']}"
                                    else:
                                        obs_text = inner[:500]
                                except (json.JSONDecodeError, TypeError):
                                    obs_text = inner[:500]
                            elif isinstance(inner, str):
                                obs_text = inner[:500]
                            else:
                                obs_text = str(inner)[:500]
                        except (json.JSONDecodeError, TypeError):
                            obs_text = raw[:500]

                    step_data = {
                        "turn": s["step_number"],
                        "type": s["step_type"],
                        "tool_name": s["tool_name"],
                        "thought": s["thought"],
                        "observation": obs_text,
                    }
                    yield f"data: {json.dumps({'type': 'step', 'step': step_data})}\n\n"

            # Agent 完成
            conn = get_db("chat")
            task = conn.execute("SELECT * FROM agent_tasks WHERE id = ?", (task_id,)).fetchone()
            conn.close()

            if task and task["final_answer"]:
                final = task["final_answer"]
            else:
                final = "任务已完成。"

            dialogue_logger.log(
                user="anonymous", question=user_msg, answer=final,
                source="agent", model=req.model,
                tokens=task["total_tokens"] if task else 0,
            )
            yield f"data: {json.dumps({'type': 'answer', 'answer': final})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ── 无状态聊天流式 (兼容前端直接传 messages 数组) ──────────────────

@app.post("/api/chat/stream-legacy")
async def chat_stream_legacy(req: LegacyChatRequest):
    """兼容前端直接传 messages 数组的流式对话"""
    if deepseek is None:
        return StreamingResponse(
            iter([f"data: {json.dumps({'error': 'DeepSeek 未配置'})}\n\n"]),
            media_type="text/event-stream",
        )

    cl = deepseek
    model_id = "deepseek-reasoner" if req.model == "reasoner" else "deepseek-chat"

    # Context 管理
    compressed, token_info = ctx_manager.maybe_compress(req.messages)

    def generate():
        yield f"data: {json.dumps({'token_info': token_info})}\n\n"
        full_reply = ""
        try:
            stream = cl.chat.completions.create(
                model=model_id,
                messages=compressed,
                temperature=0.7,
                max_tokens=4096,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_reply += delta.content
                    yield f"data: {json.dumps({'token': delta.content})}\n\n"
            dialogue_logger.log(
                user="anonymous", question=str(req.messages[-1].get("content", ""))[:200],
                answer=full_reply[:2000], source="chat", model=model_id,
            )
            yield f"data: {json.dumps({'done': True, 'full_reply': full_reply, 'token_info': token_info})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ── Startup ─────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    init_db()
    print(f"数据库已初始化 (chat/agent/memory)")
    print(f"Agent 模式: react")
    print(f"DeepSeek: {'已配置' if deepseek else '未配置'}")
    print(f"API 文档: http://127.0.0.1:8000/docs")


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

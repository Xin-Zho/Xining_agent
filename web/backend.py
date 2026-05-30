"""
FastAPI 后端 — 聊天 + 用户认证 + 文件上传

接口列表：
  POST /api/auth/register   — 用户注册
  POST /api/auth/login      — 用户登录
  GET  /api/auth/me         — 获取当前用户
  POST /api/chat            — 非流式对话
  POST /api/chat/stream     — 流式对话（SSE）
  POST /api/upload          — 文件上传

启动方式：
  cd web
  uvicorn backend:app --reload --port 8000
"""
import sys
import os
import json
import hashlib
import secrets
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from chat.llm_client import LLMClient
from chat.config import LLM_REASONER_ID
from chat.context_manager import ContextManager, count_messages_tokens, MAX_TOKENS

from agent_framework.tools.registry import ToolRegistry
from agent_framework.tools.builtin_tools import read_file, execute_command, web_search
from agent_framework.core.agent_loop import ReactAgent
from agent_framework.core.planner import PlanAndSolveAgent
from agent_framework.core.reflector import ReflectionAgent

# ============================================================
# 初始化
# ============================================================
app = FastAPI(title="Agent Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# LLM 客户端
fast_client = LLMClient()                     # deepseek-chat
reasoner_client = LLMClient(LLM_REASONER_ID)  # deepseek-reasoner

# 上下文管理器
ctx_manager = ContextManager(fast_client)

# 工具注册表 + 注册内置工具
tool_registry = ToolRegistry()
tool_registry.register(
    "read_file", "读取指定路径的文件内容",
    {"path": {"type": "string", "description": "文件路径（相对或绝对）"}},
    read_file
)
tool_registry.register(
    "execute_command", "执行 Shell 命令（白名单限制，危险命令被阻止）",
    {"command": {"type": "string", "description": "要执行的命令"}},
    execute_command
)
tool_registry.register(
    "web_search", "搜索互联网获取信息",
    {"query": {"type": "string", "description": "搜索关键词"}},
    web_search
)

# Agent 实例
react_agent = ReactAgent(fast_client, tool_registry, max_turns=10)
plan_agent = PlanAndSolveAgent(fast_client, tool_registry, max_steps=5)
reflection_agent = ReflectionAgent(fast_client, tool_registry, max_iterations=3)

# ============================================================
# 用户数据存储
# ============================================================
USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")
sessions = {}  # {token: username}  内存存储，重启失效

ALLOWED_IMAGE = {"image/jpeg", "image/png", "image/gif", "image/webp"}
ALLOWED_TEXT = {"text/plain", "application/json", "text/csv",
                "text/markdown", "text/x-python", "application/pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def load_users() -> dict:
    """从 JSON 文件读取用户"""
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(users: dict):
    """写入 JSON 文件"""
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def hash_password(password: str) -> str:
    """SHA-256 密码哈希"""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def get_username_from_token(authorization: Optional[str]) -> Optional[str]:
    """从 Authorization header 解析 token，返回用户名或 None"""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:]
    return sessions.get(token)


# ============================================================
# Request/Response 模型
# ============================================================
class AuthRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    messages: list[dict]
    model: str = "chat"  # "chat" = 快速, "reasoner" = 深度思考


# ============================================================
# 用户认证接口
# ============================================================

@app.post("/api/auth/register")
async def register(req: AuthRequest):
    """注册新用户"""
    username = req.username.strip()
    password = req.password.strip()

    if not username or not password:
        raise HTTPException(400, "用户名和密码不能为空")
    if len(username) < 2 or len(username) > 20:
        raise HTTPException(400, "用户名长度 2-20 个字符")
    if len(password) < 4:
        raise HTTPException(400, "密码至少 4 位")

    users = load_users()
    if username in users:
        return JSONResponse({"ok": False, "error": "用户名已存在"}, 409)

    users[username] = {
        "password_hash": hash_password(password),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    save_users(users)

    # 自动登录
    token = secrets.token_hex(32)
    sessions[token] = username

    return {"ok": True, "token": token, "username": username}


@app.post("/api/auth/login")
async def login(req: AuthRequest):
    """用户登录"""
    username = req.username.strip()
    password = req.password.strip()

    users = load_users()
    user = users.get(username)

    if not user or user["password_hash"] != hash_password(password):
        return JSONResponse({"ok": False, "error": "用户名或密码错误"}, 401)

    token = secrets.token_hex(32)
    sessions[token] = username

    return {"ok": True, "token": token, "username": username}


@app.get("/api/auth/me")
async def me(authorization: Optional[str] = Header(None)):
    """校验 token，返回当前用户"""
    username = get_username_from_token(authorization)
    if not username:
        return JSONResponse({"ok": False, "error": "未登录或 token 已过期"}, 401)
    return {"ok": True, "username": username}


# ============================================================
# 聊天接口
# ============================================================

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """非流式对话"""
    cl = reasoner_client if req.model == "reasoner" else fast_client
    # Context 管理：压缩超长消息
    compressed, token_info = ctx_manager.maybe_compress(req.messages)
    reply = cl.chat(compressed)
    cache = cl.get_cache_info()
    return {"reply": reply, "token_info": token_info, "cache": cache}


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式对话 — SSE 格式"""
    cl = reasoner_client if req.model == "reasoner" else fast_client
    # Context 管理：压缩超长消息
    compressed, token_info = ctx_manager.maybe_compress(req.messages)

    def generate():
        # 先发 token 用量信息
        yield f"data: {json.dumps({'token_info': token_info})}\n\n"
        full_reply = ""
        for token in cl.chat_stream(compressed):
            full_reply += token
            yield f"data: {json.dumps({'token': token})}\n\n"
        cache = cl.get_cache_info()
        yield f"data: {json.dumps({'done': True, 'full_reply': full_reply, 'token_info': token_info, 'cache': cache})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ============================================================
# Agent 接口（ReAct / Plan-Solve / Reflection）
# ============================================================

class AgentRequest(BaseModel):
    messages: list[dict]
    model: str = "chat"
    mode: str = "react"  # "react" | "plan_solve" | "reflection"


@app.post("/api/agent/run")
async def agent_run(req: AgentRequest):
    """Agent 非流式执行"""
    cl = reasoner_client if req.model == "reasoner" else fast_client
    user_msg = req.messages[-1]["content"] if req.messages else ""

    # 重建 Agent 实例（使用正确的 LLM 客户端）
    agent_reg = ToolRegistry()
    agent_reg.register("read_file", "读取文件", {"path": {"type": "string", "description": "文件路径"}}, read_file)
    agent_reg.register("execute_command", "执行命令", {"command": {"type": "string", "description": "命令"}}, execute_command)
    agent_reg.register("web_search", "搜索网页", {"query": {"type": "string", "description": "关键词"}}, web_search)

    if req.mode == "plan_solve":
        agent = PlanAndSolveAgent(cl, agent_reg)
        result = agent.run(user_msg)
    elif req.mode == "reflection":
        agent = ReflectionAgent(cl, agent_reg)
        result = agent.run(user_msg)
        result["plan"] = []
    else:
        agent = ReactAgent(cl, agent_reg)
        result = agent.run(user_msg)
        result["plan"] = []
        result["reflections"] = []

    return {"ok": True, **result}


@app.post("/api/agent/stream")
async def agent_stream(req: AgentRequest):
    """Agent 流式执行 — SSE 格式，每步思考过程实时推送给前端"""
    cl = reasoner_client if req.model == "reasoner" else fast_client
    user_msg = req.messages[-1]["content"] if req.messages else ""

    # 重建 Agent
    agent_reg = ToolRegistry()
    agent_reg.register("read_file", "读取文件", {"path": {"type": "string", "description": "文件路径"}}, read_file)
    agent_reg.register("execute_command", "执行命令", {"command": {"type": "string", "description": "命令"}}, execute_command)
    agent_reg.register("web_search", "搜索网页", {"query": {"type": "string", "description": "关键词"}}, web_search)

    if req.mode == "plan_solve":
        agent = PlanAndSolveAgent(cl, agent_reg)
    elif req.mode == "reflection":
        agent = ReflectionAgent(cl, agent_reg)
    else:
        agent = ReactAgent(cl, agent_reg)

    def generate():
        # 发送开始信号
        yield f"data: {json.dumps({'type': 'start', 'mode': req.mode})}\n\n"

        import threading
        result_holder = {}
        error_holder = {}

        def run_agent():
            try:
                result_holder["data"] = agent.run(user_msg)
            except Exception as e:
                error_holder["error"] = str(e)

        # 在后台线程运行 Agent
        thread = threading.Thread(target=run_agent)
        thread.start()

        # 等待并检查进度（简化版：等线程完成）
        thread.join(timeout=120)

        if error_holder.get("error"):
            yield f"data: {json.dumps({'type': 'error', 'error': error_holder['error']})}\n\n"
            return

        result = result_holder.get("data", {})

        # 发送思考步骤
        if "plan" in result and result["plan"]:
            yield f"data: {json.dumps({'type': 'plan', 'plan': result['plan']})}\n\n"

        for step in result.get("steps", []):
            yield f"data: {json.dumps({'type': 'step', 'step': step})}\n\n"

        if "reflections" in result and result.get("reflections"):
            yield f"data: {json.dumps({'type': 'reflections', 'reflections': result['reflections']})}\n\n"

        # 发送最终回答
        answer = result.get("answer", "")
        yield f"data: {json.dumps({'type': 'answer', 'answer': answer})}\n\n"
        cache = cl.get_cache_info()
        yield f"data: {json.dumps({'done': True, 'full_reply': answer, 'cache': cache})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


# ============================================================
# 文件上传接口
# ============================================================

@app.post("/api/upload")
async def upload(file: UploadFile = File(...),
                 authorization: Optional[str] = Header(None)):
    """上传文件，返回 base64（图片）或文本内容"""
    # 认证检查
    username = get_username_from_token(authorization)
    if not username:
        raise HTTPException(401, "请先登录")

    # 读取文件内容
    content_bytes = await file.read()

    # 大小检查
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
            "content": f"data:{content_type};base64,{b64}"
        }

    # 文本文件：返回文本内容
    if content_type in ALLOWED_TEXT:
        try:
            text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # 尝试 GBK（Windows 常见编码）
            try:
                text = content_bytes.decode("gbk")
            except Exception:
                raise HTTPException(400, "无法解码文件内容，请确认文件编码为 UTF-8")

        # PDF 特殊处理：暂时返回提示
        if content_type == "application/pdf":
            raise HTTPException(400,
                "暂不支持 PDF 解析，请将内容复制粘贴为文本消息发送")

        return {
            "ok": True,
            "filename": file.filename,
            "content_type": content_type,
            "is_image": False,
            "content": text
        }

    # 不支持的类型
    allowed = ALLOWED_IMAGE | ALLOWED_TEXT
    allowed_str = ", ".join(sorted(allowed))
    raise HTTPException(400, f"不支持的文件类型。允许: {allowed_str}")


# ============================================================
# 托管前端静态文件
# ============================================================
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

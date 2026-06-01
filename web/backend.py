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
from agent_framework.tools.builtin_tools import read_file, execute_command, web_search, grep_files, edit_file, glob_files
from agent_framework.training.dialogue_logger import DialogueLogger
from agent_framework.training.rule_extractor import RuleExtractor
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

# 工具注册表 + 注册内置工具（描述要精准，带示例）
tool_registry = ToolRegistry()
tool_registry.register(
    "read_file",
    "读取文件内容或列出目录。传文件路径返回内容，传目录路径返回文件列表。示例：read_file(path='README.md')",
    {"path": {"type": "string", "description": "要读取的文件路径或目录路径，例如 'README.md' 或 '.'"}},
    read_file
)
tool_registry.register(
    "execute_command",
    "执行系统命令。允许的命令：ls/dir/cat/echo/head/tail/python/git/find/grep 等。示例：execute_command(command='ls *.py')",
    {"command": {"type": "string", "description": "要执行的命令字符串，如 'ls -la' 或 'cat app.py'"}},
    execute_command
)
tool_registry.register(
    "web_search",
    "搜索互联网获取信息。用来查实时信息、API 文档、解决方案。示例：web_search(query='Python asyncio tutorial')",
    {"query": {"type": "string", "description": "搜索关键词，英文更精准，如 'python read file example'"}},
    web_search
)
# 代码专用工具
tool_registry.register(
    "grep_files",
    "用正则表达式搜索文件内容，返回 file:line:content 格式。比 execute_command('grep') 更快更准确。示例：grep_files(pattern='TODO', glob='*.py')",
    {
        "pattern": {"type": "string", "description": "正则表达式，如 'TODO|FIXME' 或 'class\\\\s+\\\\w+'"},
        "glob": {"type": "string", "description": "可选，文件名过滤，如 '*.py' 或 '*.{js,ts}'"},
        "path": {"type": "string", "description": "可选，搜索目录，默认项目根目录。如 'src/'"}
    },
    grep_files
)
tool_registry.register(
    "edit_file",
    "精确字符串替换：在文件中找到 old_string 并替换为 new_string。old_string 必须唯一（防止误改）。示例：edit_file(file_path='app.py', old_string='print(1)', new_string='print(2)')",
    {
        "file_path": {"type": "string", "description": "要编辑的文件路径，如 'web/backend.py'"},
        "old_string": {"type": "string", "description": "要替换的原字符串，必须与文件中完全一致（含空白），且在文件中唯一"},
        "new_string": {"type": "string", "description": "替换后的新字符串"}
    },
    edit_file
)
tool_registry.register(
    "glob_files",
    "用 glob 模式匹配文件名。示例：glob_files(pattern='**/*.py') 找出所有 Python 文件",
    {
        "pattern": {"type": "string", "description": "glob 模式，如 '*.py'、'**/*.md'、'src/**/*.ts'"},
        "path": {"type": "string", "description": "可选，搜索起始目录，默认项目根目录"}
    },
    glob_files
)

# Agent 实例
react_agent = ReactAgent(fast_client, tool_registry, max_turns=10)
plan_agent = PlanAndSolveAgent(fast_client, tool_registry, max_steps=5)
reflection_agent = ReflectionAgent(fast_client, tool_registry, max_iterations=3)

# 对话日志记录器
dialogue_logger = DialogueLogger()

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

    # 使用全局工具注册表（已注册好，描述优化过）

    if req.mode == "plan_solve":
        agent = PlanAndSolveAgent(cl, tool_registry)
        result = agent.run(user_msg)
    else:
        agent = ReactAgent(cl, tool_registry)
        result = agent.run(user_msg)
        result["plan"] = []

    # 记录日志
    dialogue_logger.log(
        user="anonymous", question=user_msg, answer=result.get("answer", ""),
        source="agent", model=req.model,
        tokens=result.get("total_tokens", 0),
        steps=result.get("steps", []),
        reflection=result.get("reflection", "")
    )

    return {"ok": True, **result}


@app.post("/api/agent/stream")
async def agent_stream(req: AgentRequest):
    """Agent 流式执行 — SSE 格式，每步思考过程实时推送给前端"""
    cl = reasoner_client if req.model == "reasoner" else fast_client
    user_msg = req.messages[-1]["content"] if req.messages else ""

    # 重建 Agent
    tool_registry = ToolRegistry()
    tool_registry.register("read_file", "读取文件", {"path": {"type": "string", "description": "文件路径"}}, read_file)
    tool_registry.register("execute_command", "执行命令", {"command": {"type": "string", "description": "命令"}}, execute_command)
    tool_registry.register("web_search", "搜索网页", {"query": {"type": "string", "description": "关键词"}}, web_search)

    if req.mode == "plan_solve":
        agent = PlanAndSolveAgent(cl, tool_registry)
    else:
        agent = ReactAgent(cl, tool_registry)

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

        # 记录日志
        dialogue_logger.log(
            user="anonymous", question=user_msg, answer=result.get("answer", ""),
            source="agent", model=req.model,
            tokens=result.get("total_tokens", 0),
            steps=result.get("steps", []),
            reflection=result.get("reflection", "")
        )

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
# 日志 & Prompt 优化接口
# ============================================================

@app.get("/api/logs/stats")
async def logs_stats():
    """日志统计概览"""
    return dialogue_logger.stats()


@app.get("/api/logs/suggestions")
async def logs_suggestions():
    """从 Claude 日志中提取 Prompt 优化建议"""
    logs = dialogue_logger.get_claude_logs(days=30)
    extractor = RuleExtractor(logs)
    suggestions = extractor.generate_prompt_suggestions()
    return {"ok": True, "total_logs": len(logs), "suggestions": suggestions}


# ============================================================
# Claude Code CLI 管道（直通，不走 Agent 循环）
# ============================================================

@app.post("/api/claude/stream")
async def claude_stream(req: ChatRequest):
    """
    Claude Code CLI 管道模式：
    直接把用户消息喂给 claude -p，stdout 实时流回前端。
    跳过 Agent 循环，纯管道。
    """
    import subprocess
    import re

    # 取最后一条用户消息
    user_msg = ""
    for m in reversed(req.messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                user_msg = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
            else:
                user_msg = str(content)
            break

    if not user_msg:
        user_msg = "(empty)"

    def generate():
        try:
            proc = subprocess.Popen(
                [os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "npm", "claude.cmd"),
                 "-p", user_msg, "--output-format", "text"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            ansi_re = re.compile(r'\x1b\[[0-9;]*m')
            full_output = []
            for line in proc.stdout:
                clean = ansi_re.sub('', line)
                if clean.strip():
                    full_output.append(clean)
                    yield f"data: {json.dumps({'token': clean})}\n\n"
            proc.wait()
            # 记录日志
            full_reply = "".join(full_output)
            dialogue_logger.log(
                user="anonymous", question=user_msg, answer=full_reply,
                source="claude", model="claude-code",
                tokens=sum(len(t)//3 for t in full_output)
            )
            yield f"data: {json.dumps({'done': True})}\n\n"
        except FileNotFoundError:
            yield f"data: {json.dumps({'error': 'Claude Code CLI 未安装，请运行 npm install -g @anthropic-ai/claude-code'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

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

        # PDF 特殊处理：用 PyPDF2 解析
        if content_type == "application/pdf":
            try:
                from PyPDF2 import PdfReader
                import io
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
                    "content": f"[PDF: {file.filename}]\n{pdf_text[:80000]}"  # 限制 80KB
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

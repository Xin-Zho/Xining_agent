"""
共享依赖 — 所有路由文件从此模块导入单例，避免与 server.py 循环引用
"""
import os
from pathlib import Path

from openai import OpenAI

from .database import init_db
from .agent import WebSocketManager
from .agent.intervention import InterventionHandler
from .protocols.mcp import ToolRegistry, MCPClientManager
from .llm_client import LLMClient
from .memory.context import ContextBuilder
from .training import DialogueLogger

# ── Config ──────────────────────────────────────────────────────────────

ALLOW_REGISTRATION = os.environ.get("ALLOW_REGISTRATION", "false").strip().lower() != "false"
INVITE_CODE = os.environ.get("INVITE_CODE", "xin-agent-2026")
MAX_HISTORY_ROUNDS = 20

SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely in Chinese. IMPORTANT: If asked about real-time events, specific dates, or factual data you are unsure about, you MUST tell the user you don't have real-time access and suggest switching to Agent mode for tool-based verification. Never fabricate earthquake reports, stock prices, news events, or weather data. 用中文回复。"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ALLOWED_IMAGE = {"image/jpeg", "image/png", "image/gif", "image/webp"}
ALLOWED_TEXT = {"text/plain", "application/json", "text/csv",
                "text/markdown", "text/x-python", "application/pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

STATIC_DIR = os.path.join(str(PROJECT_ROOT), "web", "static")

# ── Singletons ──────────────────────────────────────────────────────────

import httpx as _httpx
from .llm_client import OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_API_KEY

_LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "300"))
_LLM_CONNECT_TIMEOUT = float(os.environ.get("LLM_CONNECT_TIMEOUT", "15"))
_STEP_TIMEOUT = float(os.environ.get("STEP_TIMEOUT", "120"))
_AGENT_STREAM_TIMEOUT = float(os.environ.get("AGENT_STREAM_TIMEOUT", "300"))

# Ollama OpenAI-compatible client (replaces DeepSeek)
deepseek = OpenAI(
    api_key=LLM_API_KEY,
    base_url=OLLAMA_BASE_URL,
    timeout=_httpx.Timeout(_LLM_TIMEOUT, connect=_LLM_CONNECT_TIMEOUT),
)

ws_manager = WebSocketManager()
intervention_handler = InterventionHandler()
llm_client = LLMClient()
ctx_manager = ContextBuilder(llm_client=llm_client)
dialogue_logger = DialogueLogger()

# Tool Registry & MCP Manager
tool_registry = ToolRegistry()
mcp_manager = MCPClientManager(project_root=str(PROJECT_ROOT))


def get_engine(agent_mode: str):
    """为每个请求创建独立的引擎实例，保证会话隔离"""
    from .agent import AgentEngine, PlanSolveEngine
    tools = tool_registry.get_all_tools()
    if agent_mode == "plan_solve":
        return PlanSolveEngine(deepseek, tools, ws_manager)
    else:
        return AgentEngine(deepseek, tools, ws_manager, intervention_handler)


def verify_token_from_header(authorization: str | None) -> dict | None:
    """从 Authorization header 解析 token，返回 user dict 或 None"""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:]
    try:
        from .auth import verify_token as _verify
        return _verify(token)
    except Exception:
        return None

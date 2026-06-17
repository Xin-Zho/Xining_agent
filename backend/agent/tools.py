"""
融合工具系统 — 本地进程内工具 + MCP Server 共享模块

本地进程内保留：
  - calculator: 安全数学计算
  - timer_set: 异步延时

已迁移到 MCP Server：
  - web_search, web_fetch, stock_query  → network_server
  - read_file, read_pdf, grep_files, glob_files  → filesystem_read_server
  - edit_file  → filesystem_write_server
  - execute_command  → shell_server
  - create_excel, create_docx, create_document  → document_server
  - memory_search, list_downloads  → memory_server

本文件保留的安全函数和常量，供 MCP Server import 共享：
  - _is_public_url: SSRF 防护
  - _safe_path: 路径安全检查
  - ALLOWED_COMMANDS: 命令白名单
  - PROJECT_ROOT: 项目根目录
  - _current_user_id / set_current_user: ContextVar
  - _record_download: 文件归属记录（由 MCPTool.handler() 主进程侧调用）
"""
import asyncio
import ipaddress
import math
import os
import socket
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Awaitable
from urllib.parse import urlparse


# 项目根目录 (agent_learning/)
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

# 命令白名单（shell_server import 共享）
ALLOWED_COMMANDS = {
    "ls", "dir", "cat", "type", "echo", "head", "tail", "wc",
    "python", "python3", "pip", "curl", "wget", "git", "find",
    "grep", "sort", "uniq", "date", "whoami", "pwd", "which",
    "mkdir", "touch", "cp", "mv", "tree", "nano", "vim", "code",
}


# ── Tool dataclass（满足 ToolProtocol）───────────────────────────────

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., Awaitable[dict]]
    require_confirmation: bool = False

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ── ContextVar（主进程侧使用，MCPClientManager 读取注入到 arguments）────

_current_user_id: ContextVar[int] = ContextVar('current_user_id', default=1)


def set_current_user(user_id: int):
    _current_user_id.set(user_id)


# ── 安全函数（MCP Server import 共享）───────────────────────────────

def _is_public_url(url: str) -> tuple[bool, str]:
    """检查 URL 是否指向公网地址（防 SSRF）。
    返回 (is_safe, reason)。阻止内网、环回、链路本地、多云元数据地址。
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False, f"无法解析主机名：{url}"

        try:
            ip = ipaddress.ip_address(hostname)
        except ValueError:
            try:
                ip = ipaddress.ip_address(socket.gethostbyname(hostname))
            except (socket.gaierror, ValueError) as e:
                return False, f"DNS 解析失败：{hostname} — {e}"

        if ip.is_loopback:
            return False, f"禁止访问环回地址：{hostname} ({ip})"
        if ip.is_private:
            return False, f"禁止访问内网地址：{hostname} ({ip})"
        if ip.is_link_local:
            return False, f"禁止访问链路本地地址：{hostname} ({ip})"
        if ip.is_multicast:
            return False, f"禁止访问多播地址：{hostname} ({ip})"
        if ip.is_reserved:
            return False, f"禁止访问保留地址：{hostname} ({ip})"

        blocked_ips = {
            "169.254.169.254",  # AWS / GCP / Azure 云元数据
            "100.100.100.200",  # 里云元数据
            "169.254.0.0",      # 链路本地范围起始
        }
        if str(ip) in blocked_ips:
            return False, f"禁止访问云元数据地址：{hostname} ({ip})"

        return True, ""
    except Exception as e:
        return False, f"URL 安全检查失败：{e}"


def _safe_path(path: str) -> str:
    """安全检查：确保路径在项目目录内"""
    full = os.path.abspath(os.path.join(PROJECT_ROOT, path))
    if not full.startswith(PROJECT_ROOT):
        raise PermissionError(f"安全限制：只能访问项目目录 {PROJECT_ROOT} 内的文件")
    return full


# ── _record_download（主进程侧后处理，MCPTool.handler() 调用）──────────

def _record_download(user_id: int, filename: str, filepath: str, size: int):
    """记录文件归属（在主进程侧执行，ContextVar 有效）"""
    from ..database import get_db
    try:
        conn = get_db("memory")
        conn.execute(
            "INSERT INTO downloads (user_id, filename, filepath, size_bytes) VALUES (?, ?, ?, ?)",
            (user_id, filename, filepath, size),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # 记录失败不阻塞文件生成


# ── 本地进程内工具 handler ────────────────────────────────────────────

async def _calculator(expression: str) -> dict:
    """安全数学计算"""
    allowed = set("0123456789+-*/().%^ eEpPiI")
    if not all(c in allowed for c in expression):
        return {"error": "表达式包含不允许的字符", "expression": expression}
    try:
        safe_dict = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "pi": math.pi, "e": math.e,
            "ceil": math.ceil, "floor": math.floor,
        }
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e), "expression": expression}


async def _timer_set(seconds: int, label: str = "计时器") -> dict:
    """异步延时计时器"""
    await asyncio.sleep(min(seconds, 300))
    return {"label": label, "seconds": seconds, "done": True}


# ── 本地工具注册表（供 lifespan 注册到 ToolRegistry）───────────────────

LOCAL_TOOLS: list[Tool] = [
    Tool(
        name="calculator",
        description="执行数学计算。支持基本运算、三角函数、对数等。",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式，例如 'sqrt(16) + 3*5'"},
            },
            "required": ["expression"],
        },
        handler=_calculator,
    ),
    Tool(
        name="timer_set",
        description="设置一个计时器，等待指定秒数后返回。用于需要延时等待的场景。",
        parameters={
            "type": "object",
            "properties": {
                "seconds": {"type": "integer", "description": "等待秒数，最大300"},
                "label": {"type": "string", "description": "计时器标签"},
            },
            "required": ["seconds"],
        },
        handler=_timer_set,
    ),
]
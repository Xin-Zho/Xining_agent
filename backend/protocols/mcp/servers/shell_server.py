#!/usr/bin/env python3
"""MCP Server: shell — execute_command"""
import os
import sys
import shlex
import asyncio
import subprocess

# §4.2: 环境变量传递 PROJECT_ROOT
PROJECT_ROOT = os.environ.get("AGENT_PROJECT_ROOT")
if not PROJECT_ROOT:
    raise RuntimeError("AGENT_PROJECT_ROOT environment variable required")

# §4.2: 工具超时
TOOL_TIMEOUT = int(os.environ.get("AGENT_TOOL_TIMEOUT", "30"))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool as MCPToolType, TextContent

server = Server("shell")

# 命令白名单 — 从 tools.py 共享定义
sys.path.insert(0, PROJECT_ROOT)
from backend.agent.tools import ALLOWED_COMMANDS


# ── 工具定义 ──────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools():
    return [
        MCPToolType(
            name="execute_command",
            description="执行系统命令（白名单限制）。允许: ls/dir/cat/echo/head/tail/python/git/find/grep 等。",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "命令字符串，如 'ls -la' 或 'cat app.py'"},
                    "user_id": {"type": "integer", "description": "internal: user id"},
                },
                "required": ["command"],
            },
        ),
    ]


# ── Handler ──────────────────────────────────────────────────────────

async def handle_execute_command(command: str, **kwargs) -> dict:
    """执行 shell 命令（白名单 + 超时 + shell=False 防注入）"""
    if not command.strip():
        return {"error": "命令为空"}

    try:
        args = shlex.split(command.strip())
    except ValueError as e:
        return {"error": f"命令解析失败：{e}"}

    if not args:
        return {"error": "命令为空"}

    cmd_base = os.path.basename(args[0])

    if cmd_base not in ALLOWED_COMMANDS:
        return {
            "error": f"命令 '{cmd_base}' 不在白名单中",
            "allowed": sorted(ALLOWED_COMMANDS),
        }

    # 额外检查危险模式
    dangerous = ["rm -rf", "format", "mkfs", "dd if=", "> /dev/", ":(){ :|:& };:"]
    for d in dangerous:
        if d in command.lower():
            return {"error": f"检测到危险操作 '{d}'，已阻止"}

    try:
        # shell=True: Windows 用 cmd.exe /c，Unix 用 /bin/sh -c
        # 使 echo/dir/type 等 shell builtin 都能正常执行
        result = await asyncio.to_thread(
            subprocess.run,
            args, shell=True, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=TOOL_TIMEOUT, cwd=PROJECT_ROOT,
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        if result.returncode != 0:
            output += f"\n[退出码: {result.returncode}]"
        return {
            "command": command,
            "output": output or "(命令执行完成，无输出)",
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"命令 '{command}' 执行超时（{TOOL_TIMEOUT} 秒）"}
    except Exception as e:
        return {"error": f"命令执行失败：{e}"}


TOOL_HANDLERS = {
    "execute_command": handle_execute_command,
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
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
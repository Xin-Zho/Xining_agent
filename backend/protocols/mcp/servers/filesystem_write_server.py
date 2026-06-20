#!/usr/bin/env python3
"""MCP Server: filesystem-write — edit_file (require_confirmation=True)"""
import os
import sys

# §4.2: 环境变量传递 PROJECT_ROOT
PROJECT_ROOT = os.environ.get("AGENT_PROJECT_ROOT")
if not PROJECT_ROOT:
    raise RuntimeError("AGENT_PROJECT_ROOT environment variable required")

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool as MCPToolType, TextContent

server = Server("filesystem-write")


def _safe_path(path: str) -> str:
    """安全检查：确保路径在项目目录内"""
    full = os.path.abspath(os.path.join(PROJECT_ROOT, path))
    if not full.startswith(PROJECT_ROOT):
        raise PermissionError(f"安全限制：只能访问项目目录 {PROJECT_ROOT} 内的文件")
    return full


# ── 工具定义 ──────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools():
    return [
        MCPToolType(
            name="edit_file",
            description="精确字符串替换：在文件中找到 old_string 并替换为 new_string。old_string 必须唯一匹配。",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "要编辑的文件路径"},
                    "old_string": {"type": "string", "description": "要替换的原字符串（必须唯一）"},
                    "new_string": {"type": "string", "description": "替换后的新字符串"},
                    "user_id": {"type": "integer", "description": "internal: user id"},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        ),
    ]


# require_confirmation 标记：由 MCPTool 在 list_tools 时设置
# 实际确认拦截在引擎侧 _execute_tool() 执行


# ── Handler ──────────────────────────────────────────────────────────

async def handle_edit_file(file_path: str, old_string: str, new_string: str, **kwargs) -> dict:
    """精确字符串替换"""
    try:
        full_path = _safe_path(file_path)

        if not os.path.exists(full_path):
            return {"error": f"文件不存在：{file_path}"}

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return {"error": f"读取文件失败：{e}"}

        count = content.count(old_string)
        if count == 0:
            return {"error": "未找到匹配的字符串。请确认 old_string 是否与文件中完全一致（包括空格和换行）"}
        if count > 1:
            lines = content.split("\n")
            occurrences = []
            for i, line in enumerate(lines, 1):
                if old_string.strip() in line:
                    occurrences.append(f"  {file_path}:{i}: {line.strip()[:100]}")
            return {
                "error": f"old_string 出现了 {count} 次，必须唯一",
                "occurrences": occurrences[:10],
            }

        new_content = content.replace(old_string, new_string, 1)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        old_lines = old_string.count("\n") + 1
        new_lines = new_string.count("\n") + 1
        return {
            "file": file_path,
            "old_lines": old_lines,
            "new_lines": new_lines,
            "preview": old_string[:60] + ("..." if len(old_string) > 60 else ""),
        }
    except PermissionError as e:
        return {"error": str(e)}


TOOL_HANDLERS = {
    "edit_file": handle_edit_file,
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
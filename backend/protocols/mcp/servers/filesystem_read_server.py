#!/usr/bin/env python3
"""MCP Server: filesystem-read — read_file, read_pdf, grep_files, glob_files"""
import os
import sys

# §4.2: 环境变量传递 PROJECT_ROOT
PROJECT_ROOT = os.environ.get("AGENT_PROJECT_ROOT")
if not PROJECT_ROOT:
    raise RuntimeError("AGENT_PROJECT_ROOT environment variable required")
DOWNLOADS_DIR = os.path.join(PROJECT_ROOT, "web", "static", "downloads")

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool as MCPToolType, TextContent

server = Server("filesystem-read")

# ── 安全路径函数 ──────────────────────────────────────────────────────

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
            name="read_file",
            description="读取文件内容或列出目录。传文件路径返回内容，传目录路径返回文件列表。",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件或目录路径，如 'README.md' 或 '.'"},
                    "user_id": {"type": "integer", "description": "internal: user id"},
                },
                "required": ["path"],
            },
        ),
        MCPToolType(
            name="read_pdf",
            description="解析 PDF 文件，提取所有页面的文本内容。支持多页文档。",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "PDF 文件路径，如 'docs/report.pdf'"},
                    "user_id": {"type": "integer", "description": "internal: user id"},
                },
                "required": ["path"],
            },
        ),
        MCPToolType(
            name="grep_files",
            description="用正则表达式搜索文件内容，返回 file:line:content。示例：grep_files(pattern='TODO', glob='*.py')",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "正则表达式，如 'TODO|FIXME' 或 'class\\s+\\w+'"},
                    "glob": {"type": "string", "description": "可选，文件名过滤，如 '*.py'"},
                    "path": {"type": "string", "description": "可选，搜索目录，默认项目根目录"},
                    "user_id": {"type": "integer", "description": "internal: user id"},
                },
                "required": ["pattern"],
            },
        ),
        MCPToolType(
            name="glob_files",
            description="用 glob 模式匹配文件名。示例：glob_files(pattern='**/*.py') 找出所有 Python 文件。",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "glob 模式，如 '*.py'、'**/*.md'"},
                    "path": {"type": "string", "description": "可选，搜索起始目录"},
                    "user_id": {"type": "integer", "description": "internal: user id"},
                },
                "required": ["pattern"],
            },
        ),
    ]


# ── Handler ──────────────────────────────────────────────────────────

async def handle_read_file(path: str, **kwargs) -> dict:
    """读取文件内容或列出目录"""
    try:
        full_path = _safe_path(path)

        if not os.path.exists(full_path):
            return {"error": f"文件不存在：{path}"}

        if os.path.isdir(full_path):
            items = os.listdir(full_path)
            return {
                "type": "directory",
                "path": path,
                "items": items[:100],
                "count": len(items),
            }

        size = os.path.getsize(full_path)
        if size > 50 * 1024:
            return {"error": f"文件过大（{size} bytes），限制 50KB"}

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            return {"path": path, "content": content, "size": size}
        except UnicodeDecodeError:
            return {"error": "无法读取：二进制文件或编码不是 UTF-8"}
    except PermissionError as e:
        return {"error": str(e)}


async def handle_read_pdf(path: str, **kwargs) -> dict:
    """用 PyPDF2 解析 PDF 文件，提取文本内容"""
    # lazy import
    from PyPDF2 import PdfReader

    full_path = os.path.abspath(os.path.join(PROJECT_ROOT, path))
    if not full_path.startswith(PROJECT_ROOT):
        return {"error": "安全限制：只能读取项目目录内的文件"}
    if not os.path.exists(full_path):
        return {"error": f"文件不存在：{path}"}
    if not full_path.lower().endswith('.pdf'):
        return {"error": f"不是 PDF 文件：{path}"}

    try:
        with open(full_path, "rb") as f:
            reader = PdfReader(f)
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages.append(f"--- 第{i+1}页 ---\n{text}")
            if not pages:
                return {"error": "PDF 中未提取到文字（可能是扫描件或图片型 PDF）"}
            content = "\n\n".join(pages)
            return {
                "path": path,
                "pages": len(reader.pages),
                "content": content[:12000],
                "truncated": len(content) > 12000,
            }
    except Exception as e:
        return {"error": f"PDF 解析失败：{str(e)[:200]}"}


async def handle_grep_files(pattern: str, glob: str = None, path: str = None, **kwargs) -> dict:
    """用正则表达式搜索文件内容"""
    import re

    search_dir = os.path.abspath(os.path.join(PROJECT_ROOT, path)) if path else PROJECT_ROOT
    if not search_dir.startswith(PROJECT_ROOT):
        return {"error": f"安全限制：只能搜索项目目录 {PROJECT_ROOT} 内的文件"}
    if not os.path.isdir(search_dir):
        return {"error": f"目录不存在：{path}"}

    try:
        pattern_re = re.compile(pattern)
    except re.error as e:
        return {"error": f"正则表达式错误：{e}"}

    import fnmatch
    results = []
    max_results = 50
    max_line_len = 200

    for root, dirs, files in os.walk(search_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')
                   and d not in ('venv', 'node_modules', '__pycache__')]
        for fname in files:
            if glob and not fnmatch.fnmatch(fname, glob):
                continue
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, PROJECT_ROOT)
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_no, line in enumerate(f, 1):
                        if pattern_re.search(line):
                            results.append({
                                "file": rel_path,
                                "line": line_no,
                                "content": line.rstrip()[:max_line_len],
                            })
                            if len(results) >= max_results:
                                break
                    if len(results) >= max_results:
                        break
            except (PermissionError, OSError):
                continue
        if len(results) >= max_results:
            break

    return {
        "pattern": pattern, "glob": glob,
        "results": results, "count": len(results),
        "truncated": len(results) >= max_results,
    }


async def handle_glob_files(pattern: str, path: str = None, **kwargs) -> dict:
    """用 glob 模式匹配文件名"""
    import glob as glob_module

    search_dir = os.path.abspath(os.path.join(PROJECT_ROOT, path)) if path else PROJECT_ROOT
    if not search_dir.startswith(PROJECT_ROOT):
        return {"error": f"安全限制：只能搜索项目目录 {PROJECT_ROOT} 内的文件"}

    full_pattern = os.path.join(search_dir, pattern)
    matches = glob_module.glob(full_pattern, recursive=True)
    matches = [m for m in matches
               if not os.path.basename(m).startswith('.')
               and '__pycache__' not in m
               and 'node_modules' not in m]

    if not matches:
        return {"pattern": pattern, "files": [], "dirs": [], "count": 0}

    rel_paths = [os.path.relpath(m, PROJECT_ROOT) for m in matches]
    rel_paths.sort()

    files_list = []
    dirs_list = []
    for rp in rel_paths:
        if os.path.isdir(os.path.join(PROJECT_ROOT, rp)):
            dirs_list.append(rp)
        else:
            files_list.append(rp)

    return {
        "pattern": pattern,
        "files": files_list[:50],
        "dirs": dirs_list[:20],
        "count": len(rel_paths),
        "truncated": len(rel_paths) > 50,
    }


TOOL_HANDLERS = {
    "read_file": handle_read_file,
    "read_pdf": handle_read_pdf,
    "grep_files": handle_grep_files,
    "glob_files": handle_glob_files,
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
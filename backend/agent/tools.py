"""
融合工具系统 — 7 个工具，合并两个项目的精华

从 codex_test 继承 (async):
  - web_search: DuckDuckGo 搜索
  - web_fetch: httpx 异步抓取
  - calculator: 安全数学计算
  - timer_set: 异步延时

从 agent_learning 移植 (路径安全 + 白名单):
  - read_file: 读取文件/目录
  - execute_command: 命令白名单 + 超时
  - grep_files: 正则内容搜索
  - edit_file: 精确字符串替换
  - glob_files: glob 文件名匹配
"""
import asyncio
import math
import fnmatch
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable

import httpx
from duckduckgo_search import DDGS

# 项目根目录 (agent_learning/)
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

# 命令白名单
ALLOWED_COMMANDS = {
    "ls", "dir", "cat", "type", "echo", "head", "tail", "wc",
    "python", "python3", "pip", "curl", "wget", "git", "find",
    "grep", "sort", "uniq", "date", "whoami", "pwd", "which",
    "mkdir", "touch", "cp", "mv", "tree", "nano", "vim", "code",
}


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


# ── 通用工具 (来自 codex_test) ────────────────────────────────────────

async def _web_search(query: str, max_results: int = 5, fresh: str = "") -> dict:
    """
    智能搜索：自动感知时效需求，优先返回最新结果。

    fresh: 时间范围 'd'(一天内) / 'w'(一周内) / 'm'(一月内)，留空自动判断。
    检测到"昨日/今天/最新/实时"等关键词时自动启用 d 级时效过滤。
    """
    from datetime import datetime, timezone

    # 自动检测时效需求
    time_keywords = ["今日", "今天", "昨天", "昨日", "最新", "实时", "刚刚",
                     "涨幅", "跌", "股票", "行情", "新闻", "发布", "公布",
                     "today", "latest", "news", "stock", "breaking"]
    if not fresh:
        for kw in time_keywords:
            if kw in query:
                fresh = "d"
                break

    # 注入日期上下文，提升搜索精度
    today_str = datetime.now(timezone.utc).strftime("%Y年%m月%d日")
    if any(kw in query for kw in ["昨天", "昨日", "今日", "今天", "最新"]):
        query = f"{query} {today_str}"

    try:
        kwargs = {"max_results": max_results}
        if fresh:
            kwargs["timedelta"] = fresh

        with DDGS() as ddgs:
            results = list(ddgs.text(query, **kwargs))

        items = []
        for r in results:
            items.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", "")[:300],
                "date": r.get("date", ""),
            })

        # 无结果时回退：去掉时效限制再试
        if not items and fresh:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            items = [
                {"title": r.get("title", ""), "url": r.get("href", ""),
                 "snippet": r.get("body", "")[:300]}
                for r in results
            ]

        return {
            "query": query,
            "fresh": fresh or "auto",
            "search_date": today_str,
            "results": items,
            "count": len(items),
        }
    except Exception as e:
        return {"error": str(e), "query": query, "hint": "换短关键词重试，或换用 web_fetch 直接抓取URL"}


async def _web_fetch(url: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers={"User-Agent": "AI-Agent/1.0"})
            resp.raise_for_status()
            text = resp.text[:8000]
        return {"url": url, "content": text, "status_code": resp.status_code}
    except Exception as e:
        return {"error": str(e), "url": url}


async def _calculator(expression: str) -> dict:
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
    await asyncio.sleep(min(seconds, 300))
    return {"label": label, "seconds": seconds, "done": True}


# ── 代码工具 (来自 agent_learning，适配 async) ─────────────────────────

def _safe_path(path: str) -> str:
    """安全检查：确保路径在项目目录内"""
    full = os.path.abspath(os.path.join(PROJECT_ROOT, path))
    if not full.startswith(PROJECT_ROOT):
        raise PermissionError(f"安全限制：只能访问项目目录 {PROJECT_ROOT} 内的文件")
    return full


async def _read_file(path: str) -> dict:
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


async def _execute_command(command: str) -> dict:
    """执行 shell 命令（白名单 + 超时 + 危险参数检查）"""
    if not command.strip():
        return {"error": "命令为空"}

    cmd_name = command.strip().split()[0]
    cmd_base = os.path.basename(cmd_name)

    if cmd_base not in ALLOWED_COMMANDS:
        return {
            "error": f"命令 '{cmd_base}' 不在白名单中",
            "allowed": sorted(ALLOWED_COMMANDS),
        }

    dangerous = ["rm -rf", "format", "mkfs", "dd if=", "> /dev/", ":(){ :|:& };:"]
    for d in dangerous:
        if d in command.lower():
            return {"error": f"检测到危险操作 '{d}'，已阻止"}

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            command, shell=True, capture_output=True, text=True,
            timeout=10, cwd=PROJECT_ROOT,
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
        return {"error": f"命令 '{command}' 执行超时（10 秒）"}
    except Exception as e:
        return {"error": f"命令执行失败：{e}"}


async def _grep_files(pattern: str, glob: str = None, path: str = None) -> dict:
    """用正则表达式搜索文件内容"""
    search_dir = os.path.abspath(os.path.join(PROJECT_ROOT, path)) if path else PROJECT_ROOT

    if not search_dir.startswith(PROJECT_ROOT):
        return {"error": f"安全限制：只能搜索项目目录 {PROJECT_ROOT} 内的文件"}

    if not os.path.isdir(search_dir):
        return {"error": f"目录不存在：{path}"}

    try:
        pattern_re = re.compile(pattern)
    except re.error as e:
        return {"error": f"正则表达式错误：{e}"}

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
        "pattern": pattern,
        "glob": glob,
        "results": results,
        "count": len(results),
        "truncated": len(results) >= max_results,
    }


async def _edit_file(file_path: str, old_string: str, new_string: str) -> dict:
    """精确字符串替换：在文件中找到 old_string 并替换为 new_string"""
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


async def _glob_files(pattern: str, path: str = None) -> dict:
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


# ── 股票工具 ─────────────────────────────────────────────────────────────

async def _stock_query(action: str = "top", market: str = "a", count: int = 10) -> dict:
    """
    查询 A 股实时行情。
    action: "top" (涨幅榜), "down" (跌幅榜), "volume" (成交量榜)
    market: "a" (A股), "kcb" (科创板), "cyb" (创业板)
    """
    try:
        market_map = {
            "a": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
            "kcb": "m:1+t:23",
            "cyb": "m:0+t:80",
        }
        sort_map = {"top": "f3", "down": "f3", "volume": "f5"}
        order_map = {"top": 0, "down": 1, "volume": 0}

        fs = market_map.get(market, market_map["a"])
        fid = sort_map.get(action, "f3")
        po = order_map.get(action, 0)

        url = (
            f"https://push2.eastmoney.com/api/qt/clist/get?"
            f"pn=1&pz={count}&po={po}&np=1&fltt=2&invt=2&fid={fid}&fs={fs}"
            f"&fields=f2,f3,f4,f5,f6,f12,f14,f15,f16,f17,f18,f20"
        )

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers={"Referer": "https://quote.eastmoney.com/"})
            data = resp.json()

        if not data or "data" not in data or not data["data"]:
            return {"error": "未获取到股票数据", "action": action}

        stocks = []
        for item in data["data"].get("diff", []):
            stocks.append({
                "code": item.get("f12", ""),
                "name": item.get("f14", ""),
                "price": item.get("f2", None),
                "change_pct": item.get("f3", None),
                "change_amount": item.get("f4", None),
                "volume_hand": item.get("f5", None),
                "turnover_yuan": item.get("f6", None),
                "high": item.get("f15", None),
                "low": item.get("f16", None),
                "open": item.get("f17", None),
                "pre_close": item.get("f18", None),
            })

        action_names = {"top": "涨幅榜", "down": "跌幅榜", "volume": "成交量榜"}
        return {
            "action": action_names.get(action, action),
            "market": market,
            "count": len(stocks),
            "stocks": stocks,
            "update_time": data["data"].get("total", 0),
        }
    except Exception as e:
        return {"error": str(e), "action": action}


# ── 工具注册表 ──────────────────────────────────────────────────────────

TOOLS: list[Tool] = [
    # 股票工具
    Tool(
        name="stock_query",
        description="查询A股实时行情。action='top'涨幅榜/'down'跌幅榜/'volume'成交量榜。市场: a=A股/kcb=科创板/cyb=创业板。返回实时价格、涨跌幅、成交量。",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "查询类型: top(涨幅榜), down(跌幅榜), volume(成交量榜)"},
                "market": {"type": "string", "description": "市场: a(A股), kcb(科创板), cyb(创业板)"},
                "count": {"type": "integer", "description": "返回数量，默认10"},
            },
            "required": ["action"],
        },
        handler=_stock_query,
    ),
    # 通用工具
    Tool(
        name="web_search",
        description="智能搜索引擎，自动感知时效需求。查行情/新闻/最新信息自动开启24h过滤，无结果自动回退。需要精确信息时配合 web_fetch 抓取详情页。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词。英文更精准，中文亦可。"},
                "max_results": {"type": "integer", "description": "返回结果数，默认5，最多10"},
                "fresh": {"type": "string", "description": "时效过滤: d(24h内) / w(一周) / m(一月)。查实时信息用d，留空自动判断"},
            },
            "required": ["query"],
        },
        handler=_web_search,
    ),
    Tool(
        name="web_fetch",
        description="抓取指定URL的网页内容，返回页面文本。用于获取搜索结果的详情页面。",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "要抓取的网页URL"},
            },
            "required": ["url"],
        },
        handler=_web_fetch,
    ),
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
    # 代码工具
    Tool(
        name="read_file",
        description="读取文件内容或列出目录。传文件路径返回内容，传目录路径返回文件列表。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件或目录路径，如 'README.md' 或 '.'"},
            },
            "required": ["path"],
        },
        handler=_read_file,
    ),
    Tool(
        name="execute_command",
        description="执行系统命令（白名单限制）。允许: ls/dir/cat/echo/head/tail/python/git/find/grep 等。",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "命令字符串，如 'ls -la' 或 'cat app.py'"},
            },
            "required": ["command"],
        },
        handler=_execute_command,
    ),
    Tool(
        name="grep_files",
        description="用正则表达式搜索文件内容，返回 file:line:content。示例：grep_files(pattern='TODO', glob='*.py')",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "正则表达式，如 'TODO|FIXME' 或 'class\\s+\\w+'"},
                "glob": {"type": "string", "description": "可选，文件名过滤，如 '*.py'"},
                "path": {"type": "string", "description": "可选，搜索目录，默认项目根目录"},
            },
            "required": ["pattern"],
        },
        handler=_grep_files,
    ),
    Tool(
        name="edit_file",
        description="精确字符串替换：在文件中找到 old_string 并替换为 new_string。old_string 必须唯一匹配。",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "要编辑的文件路径"},
                "old_string": {"type": "string", "description": "要替换的原字符串（必须唯一）"},
                "new_string": {"type": "string", "description": "替换后的新字符串"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
        handler=_edit_file,
    ),
    Tool(
        name="glob_files",
        description="用 glob 模式匹配文件名。示例：glob_files(pattern='**/*.py') 找出所有 Python 文件。",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "glob 模式，如 '*.py'、'**/*.md'"},
                "path": {"type": "string", "description": "可选，搜索起始目录"},
            },
            "required": ["pattern"],
        },
        handler=_glob_files,
    ),
]

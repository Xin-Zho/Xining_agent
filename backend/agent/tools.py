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
import ipaddress
import math
import fnmatch
import json
import os
import re
import shlex
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable
from urllib.parse import urlparse

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
    根据用户提问中的时间词（昨天/上周/本月）自动推算目标日期注入搜索词。
    """
    from datetime import datetime, timedelta, timezone

    # 北京时间
    CST = timezone(timedelta(hours=8))
    now = datetime.now(CST)

    # 根据用户提问自动推算目标日期
    target_date = now
    if "前天" in query:
        target_date = now - timedelta(days=2)
    elif "昨天" in query or "昨日" in query:
        target_date = now - timedelta(days=1)
    elif "今天" in query or "今日" in query:
        target_date = now
    elif "明天" in query or "明日" in query:
        target_date = now + timedelta(days=1)
    elif "上周" in query:
        target_date = now - timedelta(weeks=1)
    elif "上个月" in query or "上月" in query:
        target_date = now - timedelta(days=30)
    elif "本周" in query:
        target_date = now

    # 自动检测时效需求 → 决定 fresh 级别
    time_keywords = ["今日", "今天", "昨天", "昨日", "最新", "实时", "刚刚",
                     "涨幅", "跌", "股票", "行情", "新闻", "发布", "公布",
                     "today", "latest", "news", "stock", "breaking", "本周", "这周"]
    if not fresh:
        for kw in time_keywords:
            if kw in query:
                fresh = "d"
                break

    # 差超过 1 天就用 w 级过滤（避免 d 太窄漏结果）
    if fresh == "d" and abs((target_date - now).days) > 1:
        fresh = "w"

    # 注入目标日期上下文，提升搜索精度
    date_str = target_date.strftime("%Y年%m月%d日")
    has_time_word = any(kw in query for kw in
        ["昨天", "昨日", "今日", "今天", "最新", "前天", "明天", "上周", "本周", "这周"])
    if has_time_word:
        query = f"{date_str} {query}"

    items = []

    # ── 百度搜索（国内优先，快）───
    try:
        import urllib.request as _ureq, re as _re
        baidu_url = f"https://www.baidu.com/s?wd={_ureq.quote(query)}&rn={max_results}"
        req = _ureq.Request(baidu_url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        with _ureq.urlopen(req, timeout=6) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        for m in _re.finditer(r'<h3[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, _re.DOTALL):
            url = m.group(1)
            title = _re.sub(r'<[^>]+>', '', m.group(2)).strip()
            if url.startswith("http") and title and "百度" not in title:
                items.append({"title": title, "url": url, "snippet": "", "source": "baidu"})
                if len(items) >= max_results: break
    except Exception:
        pass  # 百度挂了不阻塞

    # ── DuckDuckGo 兜底 ────────
    if not items:
        try:
            kwargs = {"max_results": max_results, "backend": "html"}
            if fresh:
                kwargs["timelimit"] = fresh
            with DDGS() as ddgs:
                results = list(ddgs.text(query, **kwargs))
            for r in results:
                items.append({"title": r.get("title",""), "url": r.get("href",""),
                              "snippet": (r.get("body","") or "")[:300], "source": "ddg"})
        except Exception:
            pass

    if not items:
        return {"query": query, "results": [], "count": 0,
                "hint": "均无结果，换短关键词或用 web_fetch 直接抓取URL"}

    return {"query": query, "fresh": fresh or "auto", "search_date": date_str,
            "results": items[:max_results], "count": len(items)}


def _is_public_url(url: str) -> tuple[bool, str]:
    """检查 URL 是否指向公网地址（防 SSRF）。
    返回 (is_safe, reason)。阻止内网、环回、链路本地、多云元数据地址。
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False, f"无法解析主机名：{url}"

        # 解析主机名到 IP
        try:
            ip = ipaddress.ip_address(hostname)
        except ValueError:
            # 不是 IP 字面量，DNS 解析
            try:
                ip = ipaddress.ip_address(socket.gethostbyname(hostname))
            except (socket.gaierror, ValueError) as e:
                return False, f"DNS 解析失败：{hostname} — {e}"

        # 检查是否为私有/保留地址
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

        # 额外检查常见的云元数据地址
        blocked_ips = {
            "169.254.169.254",  # AWS / GCP / Azure 云元数据
            "100.100.100.200",  # 阿里云元数据
            "169.254.0.0",      # 链路本地范围起始
        }
        if str(ip) in blocked_ips:
            return False, f"禁止访问云元数据地址：{hostname} ({ip})"

        return True, ""
    except Exception as e:
        return False, f"URL 安全检查失败：{e}"


async def _web_fetch(url: str) -> dict:
    # SSRF 防护：仅允许访问公网地址
    is_safe, reason = _is_public_url(url)
    if not is_safe:
        return {"error": reason, "url": url}

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
    """执行 shell 命令（白名单 + 超时 + shell=False 防注入）"""
    if not command.strip():
        return {"error": "命令为空"}

    # 使用 shlex 安全解析命令，防止 shell 注入（|、$()、;、&& 等均无效）
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

    # 额外检查原始命令字符串中的危险模式（即使 shlex 已阻止 shell 注入）
    dangerous = ["rm -rf", "format", "mkfs", "dd if=", "> /dev/", ":(){ :|:& };:"]
    for d in dangerous:
        if d in command.lower():
            return {"error": f"检测到危险操作 '{d}'，已阻止"}

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            args, shell=False, capture_output=True, text=True,
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
    查询 A 股实时行情（新浪财经接口）。
    action: "top" (涨幅榜), "down" (跌幅榜), "volume" (成交量榜)
    market: "a" (A股), "kcb" (科创板), "cyb" (创业板)
    """
    try:
        # 新浪财经 A 股涨幅排行 JSON API
        sort_map = {"top": "changepercent", "down": "changepercent", "volume": "volume"}
        order_map = {"top": 0, "down": 1, "volume": 0}
        market_nodes = {
            "a": "hs_a",       # 沪深A股
            "kcb": "kcb",      # 科创板
            "cyb": "cyb",      # 创业板
        }

        node = market_nodes.get(market, "hs_a")
        sort_field = sort_map.get(action, "changepercent")
        asc = order_map.get(action, 0)

        url = (
            f"http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
            f"Market_Center.getHQNodeData?"
            f"page=1&num={count}&sort={sort_field}&asc={asc}"
            f"&node={node}&symbol=&_s_r_a=auto"
        )

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://finance.sina.com.cn/",
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers)
            data = resp.json()

        stocks = []
        for item in data:
            stocks.append({
                "code": item.get("code", ""),
                "name": item.get("name", ""),
                "price": float(item.get("trade", 0)),
                "change_pct": float(item.get("changepercent", 0)),
                "change_amount": float(item.get("pricechange", 0)),
                "volume_hand": int(item.get("volume", 0)),
                "turnover_yuan": int(item.get("amount", 0)),
                "high": float(item.get("high", 0)),
                "low": float(item.get("low", 0)),
                "open": float(item.get("open", 0)),
                "pre_close": float(item.get("settlement", 0)),
            })
        if not data or len(stocks) == 0:
            return {"error": "stock_query仅覆盖A股（沪深/科创/创业板）", "hint": "该股票不在A股范围。请立即改用 web_search 搜索美股/港股行情", "stocks": []}


        action_names = {"top": "涨幅榜", "down": "跌幅榜", "volume": "成交量榜"}
        return {
            "action": action_names.get(action, action),
            "market": market,
            "count": len(stocks),
            "stocks": stocks[:count],
        }
    except Exception as e:
        return {"error": str(e), "action": action, "hint": "新浪接口可能暂时不可用，建议用 web_search 搜索股票行情替代"}


# ── Excel / Word 生成工具 ────────────────────────────────────────────────

async def _create_excel(filename: str, data_json: str) -> dict:
    """生成真正的 .xlsx Excel 文件（支持多 Sheet、表头、自动列宽）"""
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    import json as _json
    import os as _os

    downloads_dir = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
        "web", "static", "downloads"
    )
    _os.makedirs(downloads_dir, exist_ok=True)

    safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ()[]")
    if not safe_name.endswith('.xlsx'):
        safe_name += '.xlsx'
    filepath = _os.path.join(downloads_dir, safe_name)

    try:
        wb = openpyxl.Workbook()
        ws = wb.active

        # 解析 JSON 数据
        data = _json.loads(data_json)
        sheets_data = data if isinstance(data, list) else [{"title": data.get("title", "Sheet1"), "headers": data.get("headers", []), "rows": data.get("rows", [])}]

        header_font = Font(bold=True, color="FFFFFF", size=12)
        header_fill = PatternFill(start_color="5B6AF0", end_color="5B6AF0", fill_type="solid")
        thin_border = Border(
            left=Side(style='thin'), right=Side(style='thin'),
            top=Side(style='thin'), bottom=Side(style='thin')
        )

        for idx, sheet_data in enumerate(sheets_data):
            if idx > 0:
                ws = wb.create_sheet(title=sheet_data.get("title", f"Sheet{idx+1}"))
            else:
                ws.title = sheet_data.get("title", "Sheet1")

            # 写表头
            headers = sheet_data.get("headers", [])
            for col, h in enumerate(headers, 1):
                cell = ws.cell(row=1, column=col, value=h)
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal='center')
                cell.border = thin_border

            # 写数据行
            for row_idx, row in enumerate(sheet_data.get("rows", []), 2):
                for col_idx, val in enumerate(row, 1):
                    cell = ws.cell(row=row_idx, column=col_idx, value=val)
                    cell.border = thin_border
                    cell.alignment = Alignment(vertical='center')

            # 自动列宽
            for col in ws.columns:
                max_len = 0
                for cell in col:
                    if cell.value:
                        max_len = max(max_len, len(str(cell.value)))
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)

        wb.save(filepath)
        wb.close()
        _record_download(_current_user_id.get(), safe_name, filepath, _os.path.getsize(filepath))

        from urllib.parse import quote as _quote
        encoded_url = "/api/download/" + _quote(safe_name, safe='/')
        return {
            "filename": safe_name, "file_type": "xlsx",
            "size_bytes": _os.path.getsize(filepath),
            "download_url": encoded_url,
            "clickable_link": f"[📥 下载 {safe_name}]({encoded_url})",
        }
    except Exception as e:
        return {"error": str(e), "hint": "data_json 格式: {\"headers\":[\"列1\",\"列2\"],\"rows\":[[\"a\",1],[\"b\",2]]} 或数组形式"}


async def _create_docx(filename: str, markdown_content: str) -> dict:
    """生成 .docx Word 文档（从 Markdown 转换）"""
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    import os as _os, re as _re

    downloads_dir = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
        "web", "static", "downloads"
    )
    _os.makedirs(downloads_dir, exist_ok=True)

    safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ()[]")
    if not safe_name.endswith('.docx'):
        safe_name += '.docx'
    filepath = _os.path.join(downloads_dir, safe_name)

    try:
        doc = Document()

        # 解析 Markdown → Word 段落
        for line in markdown_content.split('\n'):
            line = line.strip()
            if not line:
                doc.add_paragraph()
                continue

            if line.startswith('# ') or line.startswith('## ') or line.startswith('### '):
                level = line.count('#')
                heading = doc.add_heading(line.lstrip('# ').strip(), level=min(level, 3))
            elif line.startswith('|') and '|' in line[1:]:
                # 跳过表格分隔行
                if _re.match(r'\|[\s\-:|]+\|', line):
                    continue
                cells = [c.strip() for c in line.split('|')[1:-1]]
                if not hasattr(_create_docx, '_table'):
                    _create_docx._table = doc.add_table(rows=0, cols=len(cells))
                    _create_docx._table.style = 'Light Shading Accent 1'
                row = _create_docx._table.add_row()
                for i, cell_text in enumerate(cells):
                    row.cells[i].text = cell_text
            elif line.startswith('- ') or line.startswith('* '):
                doc.add_paragraph(line[2:], style='List Bullet')
            elif _re.match(r'^\d+[\.、]', line):
                doc.add_paragraph(_re.sub(r'^\d+[\.、]\s*', '', line), style='List Number')
            elif line.startswith('> '):
                p = doc.add_paragraph(line[2:])
                p.style = 'Intense Quote'
            elif line.startswith('---'):
                doc.add_paragraph('─' * 40)
            else:
                p = doc.add_paragraph(line)
                # 解析行内 Markdown: **bold**, *italic*, [link](url)
                for run in p.runs:
                    if _re.search(r'\*\*(.+?)\*\*', run.text):
                        run.bold = True
                        run.text = _re.sub(r'\*\*(.+?)\*\*', r'\1', run.text)

        # 清除表格状态
        if hasattr(_create_docx, '_table'):
            del _create_docx._table

        doc.save(filepath)
        _record_download(_current_user_id.get(), safe_name, filepath, _os.path.getsize(filepath))

        from urllib.parse import quote as _quote
        encoded_url = "/api/download/" + _quote(safe_name, safe='/')
        return {
            "filename": safe_name, "file_type": "docx",
            "size_bytes": _os.path.getsize(filepath),
            "download_url": encoded_url,
            "clickable_link": f"[📥 下载 {safe_name}]({encoded_url})",
        }
    except Exception as e:
        return {"error": str(e), "hint": "提供 Markdown 格式的内容，会自动转换为 Word 文档"}


# ── PDF 读取工具 ──────────────────────────────────────────────────────────

async def _read_pdf(path: str) -> dict:
    """用 PyPDF2 解析 PDF 文件，提取文本内容"""
    from PyPDF2 import PdfReader
    import io as _io, os as _os

    full_path = _os.path.abspath(_os.path.join(PROJECT_ROOT, path))
    if not full_path.startswith(PROJECT_ROOT):
        return {"error": f"安全限制：只能读取项目目录内的文件"}
    if not _os.path.exists(full_path):
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


# ── 文档管理工具 ──────────────────────────────────────────────────────

async def _list_downloads(user_search: str = "") -> dict:
    """列出当前用户生成的可下载文件"""
    from ..database import get_db
    user_id = _current_user_id.get()
    conn = get_db()
    if user_search:
        rows = conn.execute(
            "SELECT filename, size_bytes, created_at FROM downloads WHERE user_id = ? AND filename LIKE ? ORDER BY created_at DESC",
            (user_id, f"%{user_search}%"),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT filename, size_bytes, created_at FROM downloads WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    conn.close()
    files = [
        {"name": r["filename"], "size_kb": round(r["size_bytes"]/1024, 1) if r["size_bytes"] else 0,
         "created": r["created_at"]}
        for r in rows
    ]
    dl_base = "/api/download/"
    from urllib.parse import quote as _quote
    for f in files:
        f["url"] = dl_base + _quote(f["name"], safe='/')
    return {"files": files, "count": len(files), "user_id": user_id}


# ── 跨会话记忆工具 ──────────────────────────────────────────────────────

from contextvars import ContextVar
_current_user_id: ContextVar[int] = ContextVar('current_user_id', default=1)

def set_current_user(user_id: int):
    _current_user_id.set(user_id)

async def _memory_search(query: str = "", action: str = "search", key: str = "", value: str = "") -> dict:
    """搜索/保存/列出跨会话记忆。支持三层记忆（working/episodic/semantic）。"""
    from ..memory.manager import MemoryManager
    mgr = MemoryManager(user_id=_current_user_id.get())

    if action == "save" and key and value:
        mid = await mgr.add(content=f"{key}: {value}", memory_type="semantic", importance=0.7)
        return {"action": "save", "key": key, "value": value[:200], "memory_id": mid, "status": "saved"}

    elif action == "list":
        items = await mgr.search(query="", memory_types=["episodic", "semantic"], limit=10)
        return {"action": "list", "memories": [i.to_dict() for i in items], "count": len(items)}

    else:  # search
        if not query:
            items = await mgr.search(query="", memory_types=["episodic", "semantic"], limit=5)
        else:
            items = await mgr.search(query=query, memory_types=["episodic", "semantic"], limit=5)
        return {"action": "search", "query": query, "memories": [i.to_dict() for i in items], "count": len(items)}


# ── 文件生成工具 ──────────────────────────────────────────────────────────

def _record_download(user_id: int, filename: str, filepath: str, size: int):
    """记录文件归属"""
    from ..database import get_db
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO downloads (user_id, filename, filepath, size_bytes) VALUES (?, ?, ?, ?)",
            (user_id, filename, filepath, size),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # 记录失败不阻塞文件生成


async def _create_document(filename: str, content: str, file_type: str = "md") -> dict:
    """
    创建可下载文件。支持 Markdown 表格、CSV、HTML、Python 脚本等。
    文件保存在 /static/downloads/ 目录，返回直接下载链接。
    """
    import os as _os
    downloads_dir = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
        "web", "static", "downloads"
    )
    _os.makedirs(downloads_dir, exist_ok=True)

    # 安全检查文件名
    safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ()[]")
    if not safe_name:
        safe_name = f"document.{file_type}"

    filepath = _os.path.join(downloads_dir, safe_name)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        return {"error": str(e), "filename": safe_name}

    _record_download(_current_user_id.get(), safe_name, filepath, _os.path.getsize(filepath))
    download_url = f"/api/download/{safe_name}"
    from urllib.parse import quote as _quote
    encoded_url = "/api/download/" + _quote(safe_name, safe='/')
    return {
        "filename": safe_name,
        "file_type": file_type,
        "size_bytes": _os.path.getsize(filepath),
        "download_url": encoded_url,
        "clickable_link": f"[📥 下载 {safe_name}]({encoded_url})",
    }


# ── 工具注册表 ──────────────────────────────────────────────────────────

TOOLS: list[Tool] = [
    # 文档管理
    Tool(
        name="list_downloads",
        description="列出/搜索当前用户生成的所有可下载文件（文档/Excel/报告等）。可选user_search参数搜索文件名。返回文件名、大小、时间、下载链接。",
        parameters={
            "type": "object",
            "properties": {
                "user_search": {"type": "string", "description": "可选，搜索文件名关键词"},
            },
            "required": [],
        },
        handler=_list_downloads,
    ),
    # 记忆工具
    Tool(
        name="memory_search",
        description="搜索/保存/列出跨会话长期记忆。action='search'搜索历史(需query) / 'save'保存(key+value) / 'list'列出最近。用于记住用户偏好、历史结论、项目上下文。",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "search / save / list"},
                "query": {"type": "string", "description": "搜索关键词（action=search时必填）"},
                "key": {"type": "string", "description": "记忆标题（action=save时必填）"},
                "value": {"type": "string", "description": "记忆内容（action=save时必填）"},
            },
            "required": ["action"],
        },
        handler=_memory_search,
    ),
    # Office 文件生成
    Tool(
        name="create_excel",
        description="生成真正的 .xlsx Excel 文件（可多Sheet、带表头样式、自动列宽）。适合报表、数据导出、表格。",
        parameters={
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "文件名，如 'stock_report.xlsx'"},
                "data_json": {"type": "string", "description": "JSON数据: {\"headers\":[\"列1\",\"列2\"],\"rows\":[[\"a\",1],[\"b\",2]]} 或 [{\"title\":\"Sheet名\",\"headers\":[...],\"rows\":[...]}]"},
            },
            "required": ["filename", "data_json"],
        },
        handler=_create_excel,
    ),
    Tool(
        name="create_docx",
        description="生成 .docx Word 文档。传入 Markdown 格式内容，自动转换为标题/表格/列表/引用。适合报告、方案、说明书。",
        parameters={
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "文件名，如 'report.docx'"},
                "markdown_content": {"type": "string", "description": "Markdown 格式的文档内容（支持 #标题/|表格|/列表/**加粗**）"},
            },
            "required": ["filename", "markdown_content"],
        },
        handler=_create_docx,
    ),
    # 通用文件生成
    Tool(
        name="create_document",
        description="创建可下载文件（Markdown表格、CSV、HTML、Python脚本等）。生成后返回下载链接给用户。适合做报表、数据汇总、文档。",
        parameters={
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "文件名，如 'stock_report.md' 或 'data.csv'"},
                "content": {"type": "string", "description": "文件内容。Markdown用|表格|、HTML用标签、CSV用逗号分隔"},
                "file_type": {"type": "string", "description": "文件类型: md/csv/html/py/txt，默认md"},
            },
            "required": ["filename", "content"],
        },
        handler=_create_document,
    ),
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
        name="read_pdf",
        description="解析 PDF 文件，提取所有页面的文本内容。支持多页文档。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "PDF 文件路径，如 'docs/report.pdf'"},
            },
            "required": ["path"],
        },
        handler=_read_pdf,
    ),
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

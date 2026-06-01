"""
内置工具：读文件、执行命令、搜索网页、搜索代码、精确编辑、文件匹配

安全限制：
- 所有文件操作限制在项目目录内
- execute_command: 白名单命令
- edit_file: 只改已存在的文件
"""
import fnmatch
from typing import Optional
import os
import subprocess
import urllib.request
import urllib.parse
import json
import re

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 命令白名单
ALLOWED_COMMANDS = {
    "ls", "dir", "cat", "type", "echo", "head", "tail", "wc",
    "python", "python3", "pip", "curl", "wget", "git", "find",
    "grep", "sort", "uniq", "date", "whoami", "pwd", "which",
    "mkdir", "touch", "cp", "mv", "tree", "nano", "vim", "code",
}


def read_file(path: str) -> str:
    """
    读取文件内容（限制在项目目录内）。
    path: 相对或绝对路径
    """
    # 路径安全检查
    full_path = os.path.abspath(path)
    if not full_path.startswith(PROJECT_ROOT):
        return f"安全限制：只能读取项目目录 {PROJECT_ROOT} 内的文件"

    if not os.path.exists(full_path):
        return f"文件不存在：{path}"

    if os.path.isdir(full_path):
        # 列目录代替读目录
        items = os.listdir(full_path)
        return f"目录 {path} 内容（{len(items)} 项）：\n" + "\n".join(f"  {i}" for i in items[:100])

    size = os.path.getsize(full_path)
    if size > 50 * 1024:
        return f"文件过大（{size} bytes），限制 50KB。请用 head/tail 命令查看部分内容。"

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except UnicodeDecodeError:
        return "无法读取：二进制文件或编码不是 UTF-8"


def execute_command(command: str) -> str:
    """
    执行 shell 命令（白名单 + 超时限制）。
    command: 要执行的命令字符串
    """
    # 提取命令名（第一个词）
    cmd_name = command.strip().split()[0] if command.strip() else ""
    cmd_base = os.path.basename(cmd_name)

    if cmd_base not in ALLOWED_COMMANDS:
        return f"安全限制：命令 '{cmd_base}' 不在白名单中。允许的命令：{', '.join(sorted(ALLOWED_COMMANDS))}"

    # 危险参数检查
    dangerous = ["rm -rf", "format", "mkfs", "dd if=", "> /dev/", ":(){ :|:& };:"]
    for d in dangerous:
        if d in command.lower():
            return f"安全限制：检测到危险操作 '{d}'，已阻止"

    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=10, cwd=PROJECT_ROOT  # 降到 10 秒，避免 Agent 卡住
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        if result.returncode != 0:
            output += f"\n[退出码: {result.returncode}]"
        return output or "(命令执行完成，无输出)"
    except subprocess.TimeoutExpired:
        return f"命令 '{command}' 执行超时（超过 10 秒）。请尝试简化命令或分步执行。"
    except Exception as e:
        return f"命令执行失败：{e}"


def web_search(query: str) -> str:
    """
    网页搜索。优先用 Bing（国内可访问），不行动用 DuckDuckGo。
    query: 搜索关键词
    返回前 5 条结果。
    """
    # 先试 Bing（国内直接访问，速度快）
    try:
        url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}&setlang=zh-cn"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        with urllib.request.urlopen(req, timeout=5) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # 提取 Bing 搜索结果
        results = re.findall(
            r'<li class="b_algo".*?<h2><a href="([^"]+)".*?>(.*?)</a></h2>.*?<p.*?>(.*?)</p>',
            html, re.DOTALL
        )

        if results:
            lines = [f"搜索：{query}（来源：Bing）\n"]
            for i, (url, title, snippet) in enumerate(results[:5], 1):
                title_clean = re.sub(r'<[^>]+>', '', title).strip()
                snippet_clean = re.sub(r'<[^>]+>', '', snippet).strip()
                lines.append(f"{i}. {title_clean}")
                lines.append(f"   {snippet_clean[:200]}")
                lines.append(f"   {url}\n")
            return "\n".join(lines)
    except Exception:
        pass

    # Bing 失败，回退到 DuckDuckGo
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        with urllib.request.urlopen(req, timeout=3) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        results = re.findall(
            r'<a rel="nofollow" class="result__a" href="([^"]+)".*?>(.*?)</a>.*?<a class="result__snippet".*?>(.*?)</a>',
            html, re.DOTALL
        )

        if not results:
            return f"搜索 '{query}' 无结果。请尝试更具体的关键词，或换用 execute_command 执行 curl 来搜索。"

        lines = [f"搜索：{query}（来源：DuckDuckGo）\n"]
        for i, (url, title, snippet) in enumerate(results[:5], 1):
            title_clean = re.sub(r'<[^>]+>', '', title).strip()
            snippet_clean = re.sub(r'<[^>]+>', '', snippet).strip()
            lines.append(f"{i}. {title_clean}")
            lines.append(f"   {snippet_clean}")
            lines.append(f"   {url}\n")
        return "\n".join(lines)

    except Exception:
        return (
            f"搜索失败：网络不可用，Bing 和 DuckDuckGo 均无法连接。\n"
            f"建议：将查询改为使用 execute_command 工具，执行 "
            f"'curl -s \"https://www.google.com/search?q={urllib.parse.quote(query)}\"' "
            f"来手动搜索。"
        )


# ============================================================
# 代码专用工具
# ============================================================

def grep_files(pattern: str, glob: Optional[str] = None, path: Optional[str] = None) -> str:
    """
    用正则表达式搜索文件内容（类似 ripgrep）。
    pattern: 正则表达式
    glob: 文件名过滤，如 "*.py" 或 "*.{js,ts}"
    path: 搜索目录，默认项目根目录
    返回：file:line: content 格式
    """
    import re as re_module
    search_dir = os.path.abspath(os.path.join(PROJECT_ROOT, path)) if path else PROJECT_ROOT

    if not search_dir.startswith(PROJECT_ROOT):
        return f"安全限制：只能搜索项目目录 {PROJECT_ROOT} 内的文件"

    if not os.path.isdir(search_dir):
        return f"目录不存在：{path}"

    try:
        pattern_re = re_module.compile(pattern)
    except re_module.error as e:
        return f"正则表达式错误：{e}"

    results = []
    max_results = 50
    max_line_len = 200

    for root, dirs, files in os.walk(search_dir):
        # 跳过隐藏目录和虚拟环境
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('venv', 'node_modules', '__pycache__')]

        for fname in files:
            if glob and not fnmatch.fnmatch(fname, glob):
                continue

            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, PROJECT_ROOT)

            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_no, line in enumerate(f, 1):
                        if pattern_re.search(line):
                            line_stripped = line.rstrip()[:max_line_len]
                            results.append(f"{rel_path}:{line_no}: {line_stripped}")
                            if len(results) >= max_results:
                                break
                    if len(results) >= max_results:
                        break
            except (PermissionError, OSError):
                continue

        if len(results) >= max_results:
            break

    if not results:
        return f"未找到匹配 '{pattern}' 的内容。提示：检查正则是否正确，或用 grep_files(path='.') 扩大范围。"

    header = f"搜索 '{pattern}'" + (f" (glob: {glob})" if glob else "") + f" — {len(results)} 条结果"
    if len(results) >= max_results:
        header += "（已达上限 50 条，请缩小范围）"
    return header + "\n" + "\n".join(results)


def edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """
    精确字符串替换：在文件中找到 old_string 并替换为 new_string。
    old_string 必须在文件中恰好出现一次（防止误改）。
    类似 Claude Code 的 Edit 工具。

    file_path: 文件路径（相对于项目根目录）
    old_string: 要替换的原字符串（必须唯一匹配）
    new_string: 替换后的新字符串
    """
    full_path = os.path.abspath(os.path.join(PROJECT_ROOT, file_path))

    if not full_path.startswith(PROJECT_ROOT):
        return f"安全限制：只能编辑项目目录 {PROJECT_ROOT} 内的文件"

    if not os.path.exists(full_path):
        return f"文件不存在：{file_path}"

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return f"读取文件失败：{e}"

    count = content.count(old_string)
    if count == 0:
        return (
            f"未找到匹配的字符串。请确认 old_string 的内容是否与文件中完全一致（包括空格和换行）。\n"
            f"提示：用 read_file('{file_path}') 查看文件内容后再试。"
        )
    if count > 1:
        # 显示上下文帮助定位
        lines = content.split("\n")
        occurrences = []
        for i, line in enumerate(lines, 1):
            if old_string.strip() in line:
                occurrences.append(f"  {file_path}:{i}: {line.strip()[:100]}")
        occ_info = "\n".join(occurrences[:10])
        return (
            f"old_string 在文件中出现了 {count} 次，必须唯一。"
            f"请包含更多上下文以确保唯一匹配。\n"
            f"匹配位置：\n{occ_info}"
        )

    new_content = content.replace(old_string, new_string, 1)

    try:
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except Exception as e:
        return f"写入文件失败：{e}"

    # 返回修改摘要
    old_lines = old_string.count("\n") + 1
    new_lines = new_string.count("\n") + 1
    return (
        f"✅ 已修改 {file_path}\n"
        f"  替换: {old_lines} 行 → {new_lines} 行\n"
        f"  位置: {old_string[:60]}{'...' if len(old_string)>60 else ''}"
    )


def glob_files(pattern: str, path: Optional[str] = None) -> str:
    """
    用 glob 模式匹配文件名（类似 ls 但支持通配符）。
    pattern: glob 模式，如 "*.py"、"**/*.md"、"src/**/*.ts"
    path: 搜索起始目录，默认项目根目录
    返回匹配的文件列表
    """
    import glob as glob_module
    search_dir = os.path.abspath(os.path.join(PROJECT_ROOT, path)) if path else PROJECT_ROOT

    if not search_dir.startswith(PROJECT_ROOT):
        return f"安全限制：只能搜索项目目录 {PROJECT_ROOT} 内的文件"

    full_pattern = os.path.join(search_dir, pattern)
    matches = glob_module.glob(full_pattern, recursive=True)

    # 过滤掉隐藏文件和缓存
    matches = [m for m in matches
               if not os.path.basename(m).startswith('.')
               and '__pycache__' not in m
               and 'node_modules' not in m]

    if not matches:
        return f"未找到匹配 '{pattern}' 的文件"

    rel_paths = [os.path.relpath(m, PROJECT_ROOT) for m in matches]
    rel_paths.sort()

    # 区分文件和目录
    files_list = []
    dirs_list = []
    for rp in rel_paths:
        if os.path.isdir(os.path.join(PROJECT_ROOT, rp)):
            dirs_list.append(f"  📁 {rp}/")
        else:
            files_list.append(f"  📄 {rp}")

    result = f"匹配 '{pattern}' — {len(rel_paths)} 项\n"
    if dirs_list:
        result += "\n".join(dirs_list[:20]) + "\n"
    if files_list:
        result += "\n".join(files_list[:50])
    if len(rel_paths) > 50:
        result += f"\n... 还有 {len(rel_paths) - 50} 项，请缩小范围"
    return result

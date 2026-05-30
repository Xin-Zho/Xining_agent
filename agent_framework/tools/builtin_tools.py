"""
内置工具：读文件、执行命令、搜索网页

安全限制：
- read_file: 限制路径在项目目录内，最大 50KB
- execute_command: 白名单命令，禁止 rm/del/format 等危险操作
- web_search: 用 urllib 访问 DuckDuckGo（免费，不需 API key）
"""
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
            timeout=30, cwd=PROJECT_ROOT
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        if result.returncode != 0:
            output += f"\n[退出码: {result.returncode}]"
        return output or "(命令执行完成，无输出)"
    except subprocess.TimeoutExpired:
        return "命令执行超时（30 秒限制）"
    except Exception as e:
        return f"命令执行失败：{e}"


def web_search(query: str) -> str:
    """
    网页搜索（DuckDuckGo）。
    query: 搜索关键词
    返回前 5 条结果的标题和摘要。
    """
    try:
        # DuckDuckGo HTML 搜索（免费，无 API key）
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # 简单正则提取搜索结果
        results = re.findall(
            r'<a rel="nofollow" class="result__a" href="([^"]+)".*?>(.*?)</a>.*?<a class="result__snippet".*?>(.*?)</a>',
            html, re.DOTALL
        )

        if not results:
            return f"搜索 '{query}' 无结果，或 DuckDuckGo 暂时不可用"

        lines = [f"搜索：{query}\n"]
        for i, (url, title, snippet) in enumerate(results[:5], 1):
            title_clean = re.sub(r'<[^>]+>', '', title).strip()
            snippet_clean = re.sub(r'<[^>]+>', '', snippet).strip()
            lines.append(f"{i}. {title_clean}")
            lines.append(f"   {snippet_clean}")
            lines.append(f"   {url}\n")

        return "\n".join(lines)

    except Exception as e:
        return f"搜索失败：{e}\n提示：DuckDuckGo 可能需要科学上网。可以改用 'https://www.baidu.com/s?wd={urllib.parse.quote(query)}' 手动搜索。"

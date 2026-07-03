"""
融合工具系统 — 本地进程内工具 + MCP Server 共享模块

本地进程内保留：
  - calculator: 安全数学计算
  - timer_set: 异步延时

已迁移到 MCP Server：
  - web_search, web_fetch  → network_server
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
import re as _re
import sympy as _sympy
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

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

# ── sympy 科学计算器 ─────────────────────────────────────────────────

# 预定义符号
_SYM_NAMES = {
    'x', 'y', 'z', 't', 'a', 'b', 'c', 'n', 'm', 'k', 'h', 'r', 'T', 'P', 'V',
    'theta', 'alpha', 'beta', 'gamma', 'omega', 'lambda_', 'sigma',
}
_SYM_TABLE = {name: _sympy.symbols(name) for name in _SYM_NAMES}

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)

_SYMPY_FUNCTIONS = {
    'diff': _sympy.diff, 'integrate': _sympy.integrate, 'solve': _sympy.solve,
    'dsolve': _sympy.dsolve, 'limit': _sympy.limit, 'series': _sympy.series,
    'Matrix': _sympy.Matrix, 'pi': _sympy.pi, 'E': _sympy.E,
    'sin': _sympy.sin, 'cos': _sympy.cos, 'tan': _sympy.tan,
    'log': _sympy.log, 'exp': _sympy.exp, 'sqrt': _sympy.sqrt, 'Abs': _sympy.Abs,
    'evalf': lambda expr, n=15: _sympy.N(expr, n),
    'simplify': _sympy.simplify,
    'expand': _sympy.expand, 'factor': _sympy.factor,
    'apart': _sympy.apart, 'together': _sympy.together,
}

_BLOCKED_NAMES = frozenset({
    '__import__', 'eval', 'exec', 'open', 'compile', 'getattr',
    'setattr', 'delattr', 'globals', 'locals', '__builtins__',
    '__builtin__', '__class__', '__bases__', '__subclasses__',
    '__mro__', '__dict__', '__globals__', '__code__', '__closure__',
    'os', 'sys', 'subprocess', 'importlib', 'builtins',
    'pty', 'posix', 'shutil', 'socket', 'ctypes',
})


def _safe_parse(expr_str: str) -> _sympy.Expr:
    """安全解析 sympy 表达式。返回 sympy.Expr 或抛出 ValueError。"""
    tokens = set(_re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr_str))
    blocked = tokens & _BLOCKED_NAMES
    if blocked:
        raise ValueError(f"Blocked identifiers: {', '.join(sorted(blocked))}")
    local_dict = dict(_SYMPY_FUNCTIONS)
    local_dict.update(_SYM_TABLE)
    return parse_expr(expr_str, local_dict=local_dict, transformations=_TRANSFORMATIONS)


def _safe_parse_and_eval(expr_str: str) -> str:
    """在线程中执行 sympy 计算（可被超时取消）"""
    parsed = _safe_parse(expr_str)
    if isinstance(parsed, _sympy.Expr):
        simplified = _sympy.simplify(parsed)
        return str(simplified)
    return str(parsed)


async def _calculator(expression: str) -> dict:
    """科学计算器 — sympy 符号引擎 + pint 单位处理

    支持的表达式格式：
      diff(x**3 + sin(x), x)          — 符号微分
      integrate(x**2, (x, 0, 1))     — 定积分
      solve(x**2 - 4, x)             — 方程求解
      evalf(pi, 50)                  — 高精度数值
      unit(5*meter/second, km/hour)  — 单位转换
    """
    import asyncio as _asyncio

    expr = expression.strip()

    # ── 单位转换路径 ──
    if expr.startswith("unit("):
        try:
            import pint as _pint
            ureg = _pint.UnitRegistry()
            inner = expr[5:].rstrip(")")
            parts = [p.strip() for p in inner.split(",", 1)]
            if len(parts) != 2:
                return {"status": "error", "expression": expression,
                        "error": "unit() requires 2 arguments: unit(value, target_unit)"}
            value_str, target_str = parts
            quantity = eval(value_str, {"__builtins__": {}}, {"ureg": ureg, **_SYM_TABLE})
            target = eval(target_str, {"__builtins__": {}}, {"ureg": ureg})
            result = quantity.to(target)
            return {"status": "ok", "expression": expression, "result": str(result)}
        except Exception as e:
            return {"status": "error", "expression": expression, "error": str(e)}

    # ── sympy 符号计算路径 ──
    try:
        result = await _asyncio.wait_for(
            _asyncio.to_thread(_safe_parse_and_eval, expr),
            timeout=30.0,
        )
        return {"status": "ok", "expression": expression, "result": result}
    except _asyncio.TimeoutError:
        return {"status": "error", "expression": expression,
                "error": "计算超时（30s），请简化表达式"}
    except ValueError as e:
        return {"status": "error", "expression": expression, "error": str(e)}
    except Exception as e:
        return {"status": "error", "expression": expression, "error": str(e)}


async def _timer_set(seconds: int, label: str = "计时器") -> dict:
    """异步延时计时器"""
    await asyncio.sleep(min(seconds, 300))
    return {"label": label, "seconds": seconds, "done": True}


# ── 本地 web_search（进程内，不经过 MCP，稳定不反爬）──────────────────

async def _web_search(query: str, max_results: int = 5, fresh: str = "") -> dict:
    """多引擎网页搜索，百度优先 → cn.bing.com → 搜狗 → DuckDuckGo"""
    import re as _re
    import urllib.request as _ureq
    items = []

    def _do_fetch(url, timeout, parser):
        nonlocal items
        req = _ureq.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        })
        with _ureq.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        parser(html)

    # 1) 百度
    if not items:
        try:
            _do_fetch(
                f"https://www.baidu.com/s?wd={_ureq.quote(query)}&rn={max_results}", 8,
                lambda html: [
                    items.append({"title": _re.sub(r'<[^>]+>', '', m.group(2)).strip(), "url": m.group(1), "snippet": "", "source": "baidu"})
                    for m in _re.finditer(r'<h3[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, _re.DOTALL)
                    if m.group(1).startswith("http") and "百度" not in _re.sub(r'<[^>]+>', '', m.group(2)).strip()
                    and len(items) < max_results
                ]
            )
        except Exception:
            pass

    # 2) cn.bing.com
    if not items:
        try:
            _do_fetch(
                f"https://cn.bing.com/search?q={_ureq.quote(query)}&count={max_results}&setlang=zh-cn", 8,
                lambda html: [
                    items.append({"title": _re.sub(r'<[^>]+>', '', m.group(2)).strip(), "url": m.group(1), "snippet": "", "source": "cn-bing"})
                    for m in _re.finditer(r'<li[^>]*class="[^"]*b_algo[^"]*"[^>]*>.*?<h2[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, _re.DOTALL)
                    if m.group(1).startswith("http") and _re.sub(r'<[^>]+>', '', m.group(2)).strip()
                    and len(items) < max_results
                ]
            )
        except Exception:
            pass

    # 3) 搜狗
    if not items:
        try:
            _do_fetch(
                f"https://www.sogou.com/web?query={_ureq.quote(query)}", 8,
                lambda html: [
                    items.append({"title": _re.sub(r'<[^>]+>', '', m.group(2)).strip(), "url": m.group(1), "snippet": "", "source": "sogou"})
                    for m in _re.finditer(r'<a[^>]*href="([^"]+)"[^>]*id="[^"]*result[^"]*"[^>]*>(.*?)</a>', html, _re.DOTALL)
                    if m.group(1).startswith("http") and "sogou.com" not in m.group(1)
                    and _re.sub(r'<[^>]+>', '', m.group(2)).strip() and len(items) < max_results
                ]
            )
        except Exception:
            pass

    # 4) DuckDuckGo
    if not items:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            for r in results:
                items.append({"title": r.get("title", ""), "url": r.get("href", ""),
                              "snippet": (r.get("body", "") or "")[:300], "source": "ddg"})
        except Exception:
            pass

    if not items:
        return {"query": query, "results": [], "count": 0, "hint": "所有搜索引擎均无结果，请缩短关键词重试"}
    return {"query": query, "results": items[:max_results], "count": len(items)}


async def _web_fetch(url: str) -> dict:
    """抓取指定URL的网页内容（进程内，不经过 MCP）"""
    from urllib.parse import urlparse

    # SSRF 防护
    hostname = urlparse(url).hostname or ""
    if not hostname:
        return {"error": "无效的 URL", "url": url}
    # 调用共享安全函数
    is_safe, reason = _is_public_url(url)
    if not is_safe:
        return {"error": reason, "url": url}

    try:
        import httpx
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers={"User-Agent": "AI-Agent/1.0"})
            resp.raise_for_status()
            text = resp.text[:8000]
        return {"url": url, "content": text, "status_code": resp.status_code}
    except Exception as e:
        return {"error": str(e), "url": url}


# ── 本地工具注册表（供 lifespan 注册到 ToolRegistry）───────────────────

# ── Chemistry + Physics 本地 handler（从 MCP Server 导入纯函数）─────

from ..protocols.mcp.servers.chemistry_server import (
    _balance_equation, _element_lookup, _calc_ph, _kinetics, _nernst,
)

async def _chem_balance(equation: str) -> dict:
    return _balance_equation(equation)

async def _chem_element(query: str) -> dict:
    return _element_lookup(query)

async def _chem_ph(acid: str = "HCl", concentration: float = 0.1) -> dict:
    return _calc_ph(acid, concentration)

async def _chem_kinetics(order: int, k: float, concentration: float, time: float) -> dict:
    return _kinetics(order, k, concentration, time)

async def _chem_nernst(half_reaction: str, concentration: float, temperature: float = 298) -> dict:
    return _nernst(half_reaction, concentration, temperature)

from ..protocols.mcp.servers.physics_server import (
    _mechanics_kinematics, _coulomb_force, _harmonic_oscillator,
    _infinite_well_ground, _quantum_handler, _carnot_efficiency,
    _error_propagation,
)

async def _phys_mechanics(u: float = None, v: float = None, a: float = None,
                          t: float = None, s: float = None) -> dict:
    return _mechanics_kinematics(u, v, a, t, s)

async def _phys_coulomb(q1: float, q2: float, r: float) -> dict:
    return _coulomb_force(q1, q2, r)

async def _phys_quantum(system: str, L: float = None, mass: float = None,
                        k: float = None, n: int = None) -> dict:
    kw = {k: v for k, v in [("L", L), ("mass", mass), ("k", k), ("n", n)] if v is not None}
    return _quantum_handler(system, **kw)

async def _phys_thermo(T_hot: float, T_cold: float) -> dict:
    return _carnot_efficiency(T_hot, T_cold)

async def _phys_error(values: str, uncertainties: str, operation: str) -> dict:
    return _error_propagation(values, uncertainties, operation)


LOCAL_TOOLS: list[Tool] = [
    Tool(
        name="web_search",
        description="智能网页搜索。百度优先→cn.bing.com→搜狗→DuckDuckGo。查行情/新闻/最新信息自动多引擎兜底。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词。英文更精准，中文亦可。"},
                "max_results": {"type": "integer", "description": "返回结果数，默认5，最多10"},
                "fresh": {"type": "string", "description": "时效过滤: d(24h) / w(一周) / m(一月)。留空自动判断"},
            },
            "required": ["query"],
        },
        handler=_web_search,
    ),
    Tool(
        name="web_fetch",
        description="抓取指定URL的网页内容，返回页面文本。用于获取搜索结果的详情页面。含SSRF防护。",
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
        description="科学计算器。符号计算: diff(表达式,变量)/integrate(表达式,(变量,下限,上限))/solve(方程,变量)/dsolve/l/series/evalf(expr,n)。单位转换: unit(值*单位, 目标单位)。示例: diff(x**3+sin(x),x) / integrate(x**2,(x,0,1)) / solve(x**2-4,x) / evalf(pi,50) / unit(5*m/s,km/h)",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "sympy表达式，如 'diff(x**3+sin(x),x)' 或 'unit(5*m/s,km/h)'"},
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
    # ── Chemistry tools ─────────────────────────────────────────────
    Tool(
        name="balance_equation",
        description="配平化学方程式。输入如 'CH4 + O2 -> CO2 + H2O'。",
        parameters={"type": "object", "properties": {"equation": {"type": "string"}}, "required": ["equation"]},
        handler=_chem_balance,
    ),
    Tool(
        name="element_lookup",
        description="查询元素或化合物属性。元素符号如'H'/'Fe'返回原子序数/原子量/电负性；化学式如'H2SO4'返回分子量和组成。",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        handler=_chem_element,
    ),
    Tool(
        name="solution_chem",
        description="计算溶液pH。acid='HCl'/'CH3COOH', concentration in mol/L。",
        parameters={"type": "object", "properties": {"acid": {"type": "string"}, "concentration": {"type": "number"}}, "required": ["acid", "concentration"]},
        handler=_chem_ph,
    ),
    Tool(
        name="kinetics",
        description="反应动力学计算。order (0/1/2), k (速率常数), concentration (mol/L), time (s)。",
        parameters={"type": "object", "properties": {"order": {"type": "integer"}, "k": {"type": "number"}, "concentration": {"type": "number"}, "time": {"type": "number"}}, "required": ["order", "k", "concentration", "time"]},
        handler=_chem_kinetics,
    ),
    Tool(
        name="electrochem",
        description="电化学Nernst方程。half_reaction如'Cu2+ + 2e- -> Cu', concentration (mol/L)。",
        parameters={"type": "object", "properties": {"half_reaction": {"type": "string"}, "concentration": {"type": "number"}, "temperature": {"type": "number"}}, "required": ["half_reaction", "concentration"]},
        handler=_chem_nernst,
    ),
    # ── Physics tools ───────────────────────────────────────────────
    Tool(
        name="mechanics",
        description="运动学计算。提供u/v/a/t/s中至少3个求其余。",
        parameters={"type": "object", "properties": {"u": {"type": "number"}, "v": {"type": "number"}, "a": {"type": "number"}, "t": {"type": "number"}, "s": {"type": "number"}}},
        handler=_phys_mechanics,
    ),
    Tool(
        name="electromagnetism",
        description="库仑力计算 F=k*q1*q2/r²。",
        parameters={"type": "object", "properties": {"q1": {"type": "number"}, "q2": {"type": "number"}, "r": {"type": "number"}}, "required": ["q1", "q2", "r"]},
        handler=_phys_coulomb,
    ),
    Tool(
        name="quantum",
        description="量子力学计算（仅解析可解模型）。system: infinite_well/harmonic_oscillator/hydrogen_atom。多电子体系(He,Li等)返回拒绝信息。",
        parameters={"type": "object", "properties": {"system": {"type": "string"}, "L": {"type": "number"}, "mass": {"type": "number"}, "k": {"type": "number"}, "n": {"type": "integer"}}, "required": ["system"]},
        handler=_phys_quantum,
    ),
    Tool(
        name="thermodynamics",
        description="热力学卡诺循环效率 eta=1-Tc/Th。",
        parameters={"type": "object", "properties": {"T_hot": {"type": "number"}, "T_cold": {"type": "number"}}, "required": ["T_hot", "T_cold"]},
        handler=_phys_thermo,
    ),
    Tool(
        name="error_propagation",
        description="误差传递计算。values/uncertainties为JSON数组字符串，operation: add/subtract/multiply/divide。",
        parameters={"type": "object", "properties": {"values": {"type": "string"}, "uncertainties": {"type": "string"}, "operation": {"type": "string"}}, "required": ["values", "uncertainties", "operation"]},
        handler=_phys_error,
    ),
]
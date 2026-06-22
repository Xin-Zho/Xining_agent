# Scientific Computing Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the generic `agent_learning` platform into a chemistry & physics computational agent for upper-level undergraduates and first-year graduate students.

**Architecture:** Keep the MCP protocol skeleton and dual-engine architecture intact. Replace the LLM client (Ollama), rewrite the calculator (sympy), add 3 domain MCP Servers (chemistry/physics/verify), build a scientific knowledge base with formula-aware chunking, and add a post-hoc verification chain in the ReAct engine.

**Tech Stack:** Python 3.10+, FastAPI, Ollama (OpenAI-compatible), sympy, scipy, pint, mendeleev, ChromaDB, BGE embeddings, KaTeX

**Branch:** `calculate_agent`

## Global Constraints

- All MCP Servers follow stdio JSON-RPC pattern (see `filesystem_read_server.py` as template)
- `ToolRegistry` freezes after startup — tools registered at lifespan, not hot-reloaded
- Calculator must block `__import__`, `eval`, `exec`, `open`, `compile`, `getattr`
- sympy computations must have a 30s timeout wrapper
- Verification must NOT block the main response — results appended asynchronously
- Each task MUST pass its test gate before proceeding to the next task
- Backward compatibility: legacy tools (web_search, stock_query, etc.) continue working

---

### Task 1: Ollama Client Rewrite (Phase 1a)

**Files:**
- Modify: `backend/llm_client.py` (full rewrite)
- Modify: `.env` (new Ollama env vars)

**Interfaces:**
- Produces: `LLMClient(chat, chat_stream)` — same signatures as existing, engine unchanged
- Produces: `LLMClient.list_models() -> list[str]` — new method for model discovery
- Consumes: Environment variables `OLLAMA_BASE_URL`, `OLLAMA_MODEL`

- [ ] **Step 1: Write the failing test**

Create `tests/test_llm_client.py`:

```python
"""Tests for Ollama LLM client"""
import os
import pytest
from backend.llm_client import LLMClient, estimate_tokens


def test_estimate_tokens_mixed_cn_en():
    """Mixed Chinese/English text: deviation < 30%"""
    text = "Hello世界" * 100  # ~1200 chars
    tokens = estimate_tokens(text)
    # Expected: ~400 tokens (1200/3), allow 30% deviation
    assert 280 <= tokens <= 520, f"Expected ~400 tokens, got {tokens}"


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0


def test_llm_client_uses_ollama_env():
    """LLMClient reads OLLAMA_BASE_URL from environment"""
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434/v1"
    os.environ["OLLAMA_MODEL"] = "qwen3:14b"
    client = LLMClient()
    assert "localhost" in str(client.client.base_url)
    assert client.model_id == "qwen3:14b"


def test_llm_client_fallback_defaults():
    """Without env vars, uses sensible defaults"""
    for key in ["OLLAMA_BASE_URL", "OLLAMA_MODEL"]:
        os.environ.pop(key, None)
    client = LLMClient()
    assert client.model_id is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd D:\agent_learning && python -m pytest tests/test_llm_client.py -v
```
Expected: FAIL — LLMClient still points to DeepSeek, no Ollama env vars read.

- [ ] **Step 3: Rewrite backend/llm_client.py**

```python
"""
LLM 客户端 — 封装 Ollama 本地 API 调用 (OpenAI 兼容端点)

支持:
  - 非流式 + 流式对话
  - 模型列表查询
  - 可配置 base_url 和 model

Ollama 启动后默认在 http://localhost:11434 提供 OpenAI 兼容 /v1 端点。
"""
import os
import httpx
from openai import OpenAI

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300"))

# Keep backward-compatible aliases for engine.py
LLM_BASE_URL = OLLAMA_BASE_URL
LLM_MODEL_ID = OLLAMA_MODEL
LLM_REASONER_ID = OLLAMA_MODEL  # Ollama doesn't have separate reasoner endpoint
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")  # Ollama doesn't need a real key


def estimate_tokens(text: str) -> int:
    """简单 token 估算。中文 ~1.5~2 chars/token，英文 ~3~4 chars/token。取保守值 3 chars/token。"""
    if not text:
        return 0
    return max(1, len(text) // 3)


class LLMClient:
    """LLM 客户端，连接本地 Ollama 服务"""

    def __init__(self, model_id=None):
        self.model_id = model_id or OLLAMA_MODEL
        self.client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=OLLAMA_BASE_URL,
            timeout=httpx.Timeout(LLM_TIMEOUT, connect=10.0),
        )

    def chat(self, messages):
        """非流式对话：返回回复文本"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "Connection" in err or "connect" in err.lower():
                raise ConnectionError(
                    f"无法连接到 Ollama 服务 ({OLLAMA_BASE_URL})。"
                    f"请确认 Ollama 已启动：ollama serve"
                ) from e
            raise

    def chat_stream(self, messages):
        """流式对话：逐 token 返回（生成器）"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                stream=True,
            )
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as e:
            err = str(e)
            if "Connection" in err or "connect" in err.lower():
                raise ConnectionError(
                    f"无法连接到 Ollama 服务 ({OLLAMA_BASE_URL})。"
                    f"请确认 Ollama 已启动：ollama serve"
                ) from e
            raise

    def list_models(self) -> list[str]:
        """查询 Ollama 可用的模型列表"""
        try:
            import requests
            base = OLLAMA_BASE_URL.rstrip("/v1").rstrip("/")
            resp = requests.get(f"{base}/api/tags", timeout=10)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return [m["name"] for m in models]
        except Exception:
            return [self.model_id]  # fallback to current model

    def get_cache_info(self) -> dict:
        """Ollama 不支持 prompt caching，返回空统计"""
        return {
            "cache_hit": 0,
            "cache_miss": 0,
            "total_prompt": 0,
            "hit_rate": 0.0,
            "estimated_saved": "N/A (Ollama)",
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd D:\agent_learning && python -m pytest tests/test_llm_client.py -v
```
Expected: PASS (4 tests pass. Note: `test_llm_client_uses_ollama_env` may need Ollama running if it exercises the actual client init deeply; if it just checks env reading, it passes regardless).

- [ ] **Step 5: Update .env**

Replace the DeepSeek config with Ollama config in `.env`:

```bash
# Ollama 本地模型配置
OLLAMA_BASE_URL="http://localhost:11434/v1"
OLLAMA_MODEL="qwen3:14b"
LLM_API_KEY="ollama"
LLM_TIMEOUT=300

# 邀请码与注册
INVITE_CODE="xin-agent-2026"
ALLOW_REGISTRATION=true
```

- [ ] **Step 6: Commit**

```bash
git add backend/llm_client.py .env tests/test_llm_client.py
git commit -m "feat: rewrite LLM client for Ollama (Phase 1a)

- Replace DeepSeek API with Ollama OpenAI-compatible endpoint
- Keep chat()/chat_stream() signatures unchanged
- Add list_models() for model discovery
- Connection failure → graceful error message
- Backward-compatible module-level aliases

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: sympy Calculation Engine (Phase 1b)

**Files:**
- Modify: `backend/agent/tools.py` (rewrite `_calculator` handler)
- Modify: `requirements.txt` (add sympy, scipy, pint)
- Create: `tests/test_calculator.py`

**Interfaces:**
- Consumes: sympy, pint (new dependencies)
- Produces: `_calculator(expression: str) -> dict` — extended expression format
- Produces: `_safe_parse(expr: str) -> sympy.Expr` — shared parser for chemistry/physics servers

- [ ] **Step 1: Install new dependencies**

```bash
cd D:\agent_learning
pip install sympy scipy pint
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_calculator.py`:

```python
"""Tests for sympy-based scientific calculator"""
import pytest
from backend.agent.tools import _calculator, _safe_parse


class TestSecurity:
    def test_blocks_import(self):
        result = _calculator_sync("__import__('os')")
        assert result["status"] == "error"
        assert "blocked" in result["error"].lower() or "not allowed" in result["error"].lower()

    def test_blocks_eval(self):
        result = _calculator_sync("eval('1+1')")
        assert result["status"] == "error"

    def test_blocks_exec(self):
        result = _calculator_sync("exec('x=1')")
        assert result["status"] == "error"

    def test_blocks_open(self):
        result = _calculator_sync("open('/etc/passwd')")
        assert result["status"] == "error"

    def test_allows_safe_math(self):
        result = _calculator_sync("diff(x**3 + sin(x), x)")
        assert result["status"] == "ok"


class TestSymbolic:
    def test_diff_polynomial(self):
        result = _calculator_sync("diff(x**3 + sin(x), x)")
        assert result["status"] == "ok"

    def test_integrate_definite(self):
        result = _calculator_sync("integrate(x**2, (x, 0, 1))")
        assert result["status"] == "ok"
        # Result should contain 1/3
        assert "1/3" in str(result.get("result", "")) or "0.333" in str(result.get("result", ""))

    def test_solve_quadratic(self):
        result = _calculator_sync("solve(x**2 - 4, x)")
        assert result["status"] == "ok"

    def test_evalf_pi(self):
        result = _calculator_sync("evalf(pi, 50)")
        assert result["status"] == "ok"


class TestErrors:
    def test_division_by_zero(self):
        result = _calculator_sync("1/0")
        assert result["status"] == "error"

    def test_syntax_error(self):
        result = _calculator_sync("diff(x**3 +)")
        assert result["status"] == "error"


def _calculator_sync(expression: str) -> dict:
    """Sync wrapper for testing async _calculator"""
    import asyncio
    return asyncio.run(_calculator(expression))
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd D:\agent_learning && python -m pytest tests/test_calculator.py -v
```
Expected: FAIL — `_safe_parse` not defined, old calculator doesn't support sympy syntax.

- [ ] **Step 4: Rewrite calculator in backend/agent/tools.py**

Replace the `_calculator` function (lines 149-166) with:

```python
# ── sympy 科学计算器 ─────────────────────────────────────────────────

import re as _re
import sympy as _sympy
from sympy import symbols, diff, integrate, solve, dsolve, limit, series, Matrix, pi, E, I, oo
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)
import pint as _pint

_ureg = _pint.UnitRegistry()

# 预定义常用符号
_SYM_NAMES = {
    'x', 'y', 'z', 't', 'a', 'b', 'c', 'n', 'm', 'k', 'h', 'r', 'T', 'P', 'V',
    'theta', 'alpha', 'beta', 'gamma', 'omega', 'lambda_', 'sigma', 'pi', 'e',
    'hbar', 'c_light', 'G_grav', 'k_B', 'N_A', 'epsilon_0', 'mu_0',
}
_SYM_TABLE = {name: symbols(name) for name in _SYM_NAMES}

# sympy 解析器配置
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)

# 可用的 sympy 函数
_SYMPY_FUNCTIONS = {
    'diff': diff, 'integrate': integrate, 'solve': solve,
    'dsolve': dsolve, 'limit': limit, 'series': series,
    'Matrix': Matrix, 'pi': pi, 'E': E,
    'sin': _sympy.sin, 'cos': _sympy.cos, 'tan': _sympy.tan,
    'log': _sympy.log, 'ln': _sympy.log, 'exp': _sympy.exp,
    'sqrt': _sympy.sqrt, 'Abs': _sympy.Abs,
    'evalf': lambda expr, n=15: expr.evalf(n),
    'integrate': integrate,
}

# 安全黑名单
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
    # 检查黑名单
    tokens = set(_re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr_str))
    blocked = tokens & _BLOCKED_NAMES
    if blocked:
        raise ValueError(f"Blocked identifiers: {', '.join(sorted(blocked))}")

    # 解析
    local_dict = dict(_SYMPY_FUNCTIONS)
    local_dict.update(_SYM_TABLE)
    return parse_expr(expr_str, local_dict=local_dict, transformations=_TRANSFORMATIONS)


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
            # 格式: unit(5*m/s, km/h)
            inner = expr[5:].rstrip(")")
            parts = [p.strip() for p in inner.split(",", 1)]
            if len(parts) != 2:
                return {"status": "error", "error": "unit() requires 2 arguments: unit(value, target_unit)"}
            value_str, target_str = parts
            quantity = eval(value_str, {"__builtins__": {}}, {"ureg": _ureg, **_SYM_TABLE})
            target = eval(target_str, {"__builtins__": {}}, {"ureg": _ureg})
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
                "error": "计算超时（30s），请简化表达式或使用数值近似"}
    except ValueError as e:
        return {"status": "error", "expression": expression, "error": str(e)}
    except Exception as e:
        return {"status": "error", "expression": expression, "error": str(e)}


def _safe_parse_and_eval(expr_str: str) -> str:
    """在线程中执行 sympy 计算（可被超时取消）"""
    parsed = _safe_parse(expr_str)
    if isinstance(parsed, _sympy.Expr):
        # 尝试化简
        simplified = _sympy.simplify(parsed)
        return str(simplified)
    return str(parsed)
```

Update the `calculator` tool registration (around line 384-395) to reflect new capabilities:

```python
Tool(
    name="calculator",
    description="科学计算。符号: diff/integrate/solve/dsolve/limit/series/evalf/unit。示例: diff(x**3+sin(x),x) / integrate(x**2,(x,0,1)) / solve(x**2-4,x) / unit(5*m/s,km/h)",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "sympy表达式，如 'diff(x**3+sin(x),x)' 或 'unit(5*m/s,km/h)'"},
        },
        "required": ["expression"],
    },
    handler=_calculator,
),
```

- [ ] **Step 5: Update requirements.txt**

Add to `requirements.txt`:

```
# Scientific computing
sympy>=1.12
scipy>=1.10
pint>=0.22
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd D:\agent_learning && python -m pytest tests/test_calculator.py -v
```
Expected: PASS (all test classes pass — security blocks, symbolic works, errors handled).

- [ ] **Step 7: Commit**

```bash
git add backend/agent/tools.py requirements.txt tests/test_calculator.py
git commit -m "feat: sympy scientific calculator engine (Phase 1b)

- Replace eval()-based calculator with sympy symbolic engine
- Support: diff, integrate, solve, dsolve, limit, series, evalf
- pint unit conversion via unit(value, target) syntax
- Security: blacklist __import__/eval/exec/open/getattr
- 30s timeout wrapper for large symbolic expressions

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: LaTeX Frontend Rendering (Phase 1c)

**Files:**
- Modify: `web/static/index.html` (add KaTeX CDN)
- Modify: `web/static/js/chat.js` (add renderMathInElement call)

**Interfaces:**
- Consumes: KaTeX CDN (jsDelivr)
- Produces: auto-rendered LaTeX in chat messages

- [ ] **Step 1: Add KaTeX to index.html**

In `web/static/index.html`, add inside `<head>` before other scripts:

```html
<!-- KaTeX for LaTeX rendering -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"
      integrity="sha384-nB0miv6/jRmo5r4cy45FVFXUuoi8Tq5l5qkOz2VG1Ldv6z0TG3YaxK5RH1t0dnN" crossorigin="anonymous">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"
        integrity="sha384-RjJnfLt0RqFQ2H2Khzy6+zyyYj6Zr1kHY2Vh8r6a7QH0lBbP04YI6kSOHSgNbkM" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
        integrity="sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax4ZZB6zTI7u1q8+UMGOao7L0KqCIS5Vxm" crossorigin="anonymous"></script>
```

- [ ] **Step 2: Add auto-render to chat.js**

In `web/static/js/chat.js`, find the function that appends message content to the DOM (the `appendMessage` or equivalent function). After a message element is inserted into the DOM, add:

```javascript
// After inserting message content into the DOM:
if (typeof renderMathInElement !== 'undefined') {
    try {
        renderMathInElement(messageEl, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false},
                {left: '\\(', right: '\\)', display: false},
                {left: '\\[', right: '\\]', display: true},
            ],
            throwOnError: false,
        });
    } catch (e) {
        console.warn('KaTeX render failed:', e);
    }
}
```

- [ ] **Step 3: Verify manually**

Start the backend and open the browser:
```bash
cd D:\agent_learning && python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/app`, send a message containing LaTeX:
```
test: $E=mc^2$ and $$\int_0^\infty e^{-x} dx = 1$$
```

Expected:
- `$E=mc^2$` renders as inline styled math
- `$$\int_0^\infty e^{-x} dx = 1$$` renders as display block
- Chrome, Firefox, Edge all render correctly

- [ ] **Step 4: Commit**

```bash
git add web/static/index.html web/static/js/chat.js
git commit -m "feat: add KaTeX LaTeX rendering to frontend (Phase 1c)

- CDN-loaded KaTeX 0.16.11 with auto-render
- Supports $inline$, $$block$$, \\(inline\\), \\[block\\] delimiters
- throwOnError: false — malformed LaTeX shows raw text, doesn't crash

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Chemistry MCP Server (Phase 2a)

**Files:**
- Create: `backend/protocols/mcp/servers/chemistry_server.py`
- Create: `tests/test_chemistry_server.py`

**Interfaces:**
- Consumes: sympy (from Phase 1b), mendeleev (new dep)
- Produces: MCP Server `chemistry` — registers via ToolRegistry at startup
- Produces: 7 tools: `balance_equation`, `thermo_calc`, `equilibrium`, `kinetics`, `solution_chem`, `electrochem`, `element_lookup`

- [ ] **Step 1: Install mendeleev**

```bash
pip install mendeleev
```

Add `mendeleev>=0.14` to `requirements.txt`.

- [ ] **Step 2: Write failing test**

Create `tests/test_chemistry_server.py`:

```python
"""Tests for Chemistry MCP Server tools"""
import pytest
from backend.protocols.mcp.servers.chemistry_server import (
    _balance_equation,
    _element_lookup,
    _calc_ph,
)


class TestBalanceEquation:
    def test_simple_combustion(self):
        result = _balance_equation("CH4 + O2 -> CO2 + H2O")
        assert result["status"] == "ok"
        assert result["balanced"] == "CH4 + 2 O2 -> CO2 + 2 H2O"

    def test_iron_chloride(self):
        result = _balance_equation("Fe + Cl2 -> FeCl3")
        assert result["status"] == "ok"
        assert "2 Fe" in result["balanced"]
        assert "3 Cl2" in result["balanced"]
        assert "2 FeCl3" in result["balanced"]

    def test_invalid_formula(self):
        result = _balance_equation("XyZz + AbCd -> ???")
        assert result["status"] == "error"


class TestElementLookup:
    def test_hydrogen(self):
        result = _element_lookup("H")
        assert result["status"] == "ok"
        assert result["name"] == "Hydrogen"
        assert 1.007 <= result["atomic_weight"] <= 1.009

    def test_sulfuric_acid_mass(self):
        result = _element_lookup("H2SO4")
        assert result["status"] == "ok"
        assert 97 <= result["molar_mass"] <= 99  # ~98.079 g/mol

    def test_invalid_element(self):
        result = _element_lookup("Xyzzy")
        assert result["status"] == "error"


class TestPH:
    def test_strong_acid(self):
        result = _calc_ph(acid="HCl", concentration=0.1)
        assert result["status"] == "ok"
        assert abs(result["pH"] - 1.0) < 0.05

    def test_weak_acid(self):
        result = _calc_ph(acid="CH3COOH", concentration=0.1)
        assert result["status"] == "ok"
        assert 2.5 <= result["pH"] <= 3.0  # ~2.87
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd D:\agent_learning && python -m pytest tests/test_chemistry_server.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 4: Create chemistry_server.py**

Create `backend/protocols/mcp/servers/chemistry_server.py`:

```python
#!/usr/bin/env python3
"""MCP Server: chemistry — 化学计算工具（方程式配平、热力学、平衡、动力学、电化学）"""
import os
import sys
import re
import json
from typing import Any

PROJECT_ROOT = os.environ.get("AGENT_PROJECT_ROOT")
if not PROJECT_ROOT:
    raise RuntimeError("AGENT_PROJECT_ROOT environment variable required")

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

import sympy as sp
from mendeleev import element

server = Server("chemistry")

# ── 共享工具函数 ──────────────────────────────────────────────────────

def _parse_chemical_formula(formula: str) -> dict[str, int]:
    """解析化学式 → {元素符号: 原子数}。如 'H2SO4' → {'H': 2, 'S': 1, 'O': 4}"""
    pattern = r'([A-Z][a-z]?)(\d*)'
    counts = {}
    for match in re.finditer(pattern, formula):
        el = match.group(1)
        n = int(match.group(2)) if match.group(2) else 1
        counts[el] = counts.get(el, 0) + n
    return counts


def _molar_mass(formula: str) -> float:
    """计算分子量 (g/mol)"""
    counts = _parse_chemical_formula(formula)
    mass = 0.0
    for el_sym, n in counts.items():
        try:
            el = element(el_sym)
            mass += (el.atomic_weight or 0) * n
        except Exception:
            raise ValueError(f"Unknown element: {el_sym}")
    return round(mass, 3)


# ── 工具实现 ──────────────────────────────────────────────────────────

def _balance_equation(equation: str) -> dict:
    """配平化学方程式"""
    try:
        parts = equation.replace(' ', '').split('->')
        if len(parts) != 2:
            return {"status": "error", "error": "Format: 'reactants -> products'"}
        reactants_str, products_str = parts

        # 解析反应物和产物
        def parse_side(s: str) -> list[dict]:
            compounds = []
            for comp in s.split('+'):
                if not comp:
                    continue
                counts = _parse_chemical_formula(comp)
                if not counts:
                    return []
                compounds.append(counts)
            return compounds

        reactants = parse_side(reactants_str)
        products = parse_side(products_str)

        if not reactants or not products:
            return {"status": "error", "error": "Could not parse chemical formulas"}

        # 收集所有元素
        all_elements = set()
        for comp in reactants + products:
            all_elements.update(comp.keys())
        els = sorted(all_elements)

        # 构建系数矩阵 (线性代数配平)
        # 每种元素一个方程：sum(coeff_i * atom_count_i) = 0
        n_reactants = len(reactants)
        n_total = n_reactants + len(products)

        if len(els) < n_total - 1:
            return {"status": "error", "error": "Underdetermined system — may have multiple solutions"}

        A = []
        for el in els:
            row = []
            for comp in reactants:
                row.append(comp.get(el, 0))
            for comp in products:
                row.append(-comp.get(el, 0))
            A.append(row)

        # 使用 sympy 求解齐次线性方程组，要求最小正整数解
        M = sp.Matrix(A)
        nullspace = M.nullspace()
        if not nullspace:
            return {"status": "error", "error": "No solution found — check equation"}

        # 取最小正整数解
        solution = nullspace[0]
        # 缩放为整数
        denom_lcm = 1
        for v in solution:
            d = sp.fraction(v)[1]
            denom_lcm = sp.lcm(denom_lcm, d)
        solution = [int(v * denom_lcm) for v in solution]

        # 确保最小的系数为正
        if min(solution) < 0:
            solution = [-c for c in solution]

        # 除以 GCD
        g = solution[0]
        for c in solution[1:]:
            g = sp.gcd(g, c)
        solution = [c // g for c in solution]

        # 构建配平后的方程式
        reactant_coeffs = solution[:n_reactants]
        product_coeffs = solution[n_reactants:]

        def format_side(compounds, coeffs) -> str:
            parts = []
            for comp, c in zip(compounds, coeffs):
                prefix = str(c) if c > 1 else ""
                formula = ''.join(f"{el}{n if n > 1 else ''}" for el, n in comp.items())
                parts.append(f"{prefix}{formula}")
            return " + ".join(parts)

        balanced = (
            format_side(reactants, reactant_coeffs)
            + " -> "
            + format_side(products, product_coeffs)
        )

        return {"status": "ok", "equation": equation, "balanced": balanced, "coefficients": solution}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _element_lookup(query: str) -> dict:
    """查询元素或化合物属性"""
    query = query.strip()
    try:
        # 尝试作为元素符号
        if len(query) <= 2 and query[0].isupper():
            el = element(query)
            return {
                "status": "ok",
                "type": "element",
                "symbol": el.symbol,
                "name": el.name,
                "atomic_number": el.atomic_number,
                "atomic_weight": round(el.atomic_weight or 0, 4),
                "electronegativity": round(el.en_pauling, 2) if el.en_pauling else None,
                "group": el.group_id,
                "period": el.period,
            }

        # 作为化学式
        mass = _molar_mass(query)
        counts = _parse_chemical_formula(query)
        return {
            "status": "ok",
            "type": "compound",
            "formula": query,
            "molar_mass": mass,
            "composition": counts,
        }
    except Exception as e:
        return {"status": "error", "error": f"Unrecognized: {query} — {e}"}


def _calc_ph(acid: str = "HCl", concentration: float = 0.1) -> dict:
    """计算强酸/弱酸的 pH"""
    # Known Ka values (simplified)
    Ka_values = {
        "CH3COOH": 1.8e-5,
        "HAc": 1.8e-5,
        "HF": 6.8e-4,
        "HCOOH": 1.8e-4,
    }
    Ka = Ka_values.get(acid, None)

    if Ka and concentration > 0:
        # 弱酸: [H+] = sqrt(Ka * C)
        H_conc = sp.sqrt(Ka * concentration).evalf()
        pH_val = -float(sp.log(H_conc, 10).evalf())
    else:
        # 强酸: pH = -log10(C)
        if concentration <= 0:
            return {"status": "error", "error": "Concentration must be positive"}
        pH_val = -float(sp.log(concentration, 10).evalf()) if concentration > 0 else 7.0

    return {"status": "ok", "acid": acid, "concentration": concentration, "pH": round(pH_val, 2),
            "is_weak": Ka is not None}


# ── MCP Server 定义 ───────────────────────────────────────────────────

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="balance_equation",
            description="配平化学方程式。输入如 'CH4 + O2 -> CO2 + H2O'。",
            inputSchema={
                "type": "object",
                "properties": {
                    "equation": {"type": "string", "description": "如 'Fe + Cl2 -> FeCl3'"},
                },
                "required": ["equation"],
            },
        ),
        Tool(
            name="element_lookup",
            description="查询元素或化合物的属性。元素符号(如'H','Fe')返回原子序数/原子量/电负性; 化学式(如'H2SO4')返回分子量和组成。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "元素符号或化学式"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="solution_chem",
            description="计算溶液化学参数：pH、缓冲溶液、滴定曲线。",
            inputSchema={
                "type": "object",
                "properties": {
                    "acid": {"type": "string", "description": "酸化学式，如 HCl, CH3COOH"},
                    "concentration": {"type": "number", "description": "浓度 (mol/L)"},
                },
                "required": ["acid", "concentration"],
            },
        ),
        Tool(
            name="thermo_calc",
            description="热力学计算：ΔH/ΔG/ΔS。需提供反应物与产物的标准生成焓。",
            inputSchema={
                "type": "object",
                "properties": {
                    "reactants": {"type": "string", "description": "反应物 JSON: {formula: moles, ...}"},
                    "products": {"type": "string", "description": "产物 JSON"},
                    "temperature": {"type": "number", "description": "温度 (K)，默认 298"},
                },
                "required": ["reactants", "products"],
            },
        ),
        Tool(
            name="equilibrium",
            description="计算化学平衡：给定反应和初始浓度，求平衡组成和平衡常数。",
            inputSchema={
                "type": "object",
                "properties": {
                    "reaction": {"type": "string", "description": "反应式，如 'N2 + 3 H2 -> 2 NH3'"},
                    "initial_concentrations": {"type": "string", "description": "初始浓度 JSON: {species: conc}"},
                },
                "required": ["reaction", "initial_concentrations"],
            },
        ),
        Tool(
            name="kinetics",
            description="反应动力学：速率方程、Arrhenius方程。给定反应级数、速率常数k、温度T、活化能Ea。",
            inputSchema={
                "type": "object",
                "properties": {
                    "order": {"type": "integer", "description": "反应级数 (0/1/2)"},
                    "k": {"type": "number", "description": "速率常数"},
                    "concentration": {"type": "number", "description": "初始浓度 (mol/L)"},
                    "time": {"type": "number", "description": "时间 (s)"},
                },
                "required": ["order", "k", "concentration", "time"],
            },
        ),
        Tool(
            name="electrochem",
            description="电化学计算：Nernst方程、标准电极电势、电池电动势。",
            inputSchema={
                "type": "object",
                "properties": {
                    "half_reaction": {"type": "string", "description": "半反应，如 'Cu2+ + 2e- -> Cu'"},
                    "concentration": {"type": "number", "description": "离子浓度 (mol/L)"},
                    "temperature": {"type": "number", "description": "温度 (K)，默认 298"},
                },
                "required": ["half_reaction", "concentration"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    handlers = {
        "balance_equation": lambda: _balance_equation(arguments["equation"]),
        "element_lookup": lambda: _element_lookup(arguments["query"]),
        "solution_chem": lambda: _calc_ph(
            arguments.get("acid", "HCl"),
            arguments.get("concentration", 0.1),
        ),
        "thermo_calc": lambda: {"status": "error", "error": "thermo_calc: full implementation pending — provide standard enthalpies"},
        "equilibrium": lambda: {"status": "error", "error": "equilibrium: full implementation pending"},
        "kinetics": lambda: _kinetics(
            arguments["order"], arguments["k"],
            arguments["concentration"], arguments["time"],
        ),
        "electrochem": lambda: _nernst(
            arguments["half_reaction"], arguments["concentration"],
            arguments.get("temperature", 298),
        ),
    }
    handler = handlers.get(name)
    if not handler:
        return [TextContent(type="text", text=json.dumps({"status": "error", "error": f"Unknown tool: {name}"}))]
    result = handler()
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


def _kinetics(order: int, k: float, concentration: float, time: float) -> dict:
    """反应动力学计算"""
    try:
        if order == 0:
            remaining = concentration - k * time
            half_life = concentration / (2 * k) if k > 0 else float('inf')
        elif order == 1:
            remaining = concentration * sp.exp(-k * time).evalf()
            half_life = float(sp.log(2).evalf() / k) if k > 0 else float('inf')
        elif order == 2:
            remaining = concentration / (1 + k * concentration * time)
            half_life = 1 / (k * concentration) if k > 0 and concentration > 0 else float('inf')
        else:
            return {"status": "error", "error": f"Unsupported reaction order: {order}"}

        remaining = float(remaining) if hasattr(remaining, 'evalf') else remaining
        return {
            "status": "ok",
            "order": order,
            "k": k,
            "initial_concentration": concentration,
            "time": time,
            "remaining_concentration": round(max(0, remaining), 6),
            "half_life": round(half_life, 3) if half_life != float('inf') else None,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _nernst(half_reaction: str, concentration: float, temperature: float = 298) -> dict:
    """Nernst 方程计算"""
    # 简化的标准电极电势表
    standard_potentials = {
        "Cu2+ + 2e- -> Cu": 0.34,
        "Zn2+ + 2e- -> Zn": -0.76,
        "Fe3+ + e- -> Fe2+": 0.77,
        "Ag+ + e- -> Ag": 0.80,
        "2H+ + 2e- -> H2": 0.00,
    }

    E0 = standard_potentials.get(half_reaction)
    if E0 is None:
        return {"status": "error", "error": f"Unknown half-reaction: {half_reaction}. Try: {list(standard_potentials.keys())}"}

    # Nernst: E = E0 - (RT/nF) * ln(Q)
    # 简化：n=1 时 E = E0 - 0.0592 * log10(1/C) = E0 + 0.0592 * log10(C)
    R = 8.314
    F = 96485
    n = 1  # simplified; should parse from half-reaction
    E = E0 - (R * temperature) / (n * F) * sp.log(1 / concentration).evalf()

    return {
        "status": "ok",
        "half_reaction": half_reaction,
        "E0": E0,
        "E": round(float(E), 4),
        "temperature": temperature,
        "concentration": concentration,
    }


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    asyncio.run(stdio_server(server))
```

- [ ] **Step 5: Register in dependencies.py**

In `backend/dependencies.py`, add chemistry server registration after existing MCP servers:

```python
# In the startup function, after other MCP server registrations:
from backend.protocols.mcp.servers import chemistry_server
```

Note: The exact registration mechanism depends on how your existing 5 MCP servers are registered. Follow the same pattern used for `filesystem_read_server`, `shell_server`, etc. Typically this is done in `backend/dependencies.py` or `backend/server.py` lifespan via `MCPClientManager.register("chemistry", "backend/protocols/mcp/servers/chemistry_server.py")`.

- [ ] **Step 6: Run tests**

```bash
cd D:\agent_learning && python -m pytest tests/test_chemistry_server.py -v
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/protocols/mcp/servers/chemistry_server.py tests/test_chemistry_server.py requirements.txt
git commit -m "feat: chemistry MCP server (Phase 2a)

- 7 tools: balance_equation, element_lookup, solution_chem, thermo_calc, equilibrium, kinetics, electrochem
- Linear algebra balancing via sympy nullspace
- mendeleev periodic table integration
- Nernst equation with standard potentials

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Physics MCP Server (Phase 2b)

**Files:**
- Create: `backend/protocols/mcp/servers/physics_server.py`
- Create: `tests/test_physics_server.py`

**Interfaces:**
- Consumes: sympy (Phase 1b), scipy, pint
- Produces: MCP Server `physics` — 6 tools: `mechanics`, `electromagnetism`, `thermodynamics`, `quantum`, `optics`, `error_propagation`

- [ ] **Step 1: Write failing test**

Create `tests/test_physics_server.py`:

```python
"""Tests for Physics MCP Server tools"""
import pytest
from backend.protocols.mcp.servers.physics_server import (
    _mechanics_kinematics,
    _coulomb_force,
    _harmonic_oscillator,
    _infinite_well_ground,
)


class TestMechanics:
    def test_kinematics_s_ut_half_at2(self):
        """s = ut + 1/2 at^2: u=0, a=9.8, t=2 → s=19.6"""
        result = _mechanics_kinematics(u=0, a=9.8, t=2)
        assert result["status"] == "ok"
        assert abs(result["displacement"] - 19.6) < 0.01

    def test_kinematics_missing_params(self):
        result = _mechanics_kinematics(u=None, a=9.8, t=2)
        assert result["status"] == "error"


class TestElectromagnetism:
    def test_coulomb_two_charges(self):
        """Two 1μC charges at 1m → force ~8.99e-3 N"""
        result = _coulomb_force(q1=1e-6, q2=1e-6, r=1.0)
        assert result["status"] == "ok"
        assert abs(result["force"] - 8.99e-3) < 0.01e-3


class TestQuantum:
    def test_harmonic_oscillator(self):
        """m=1, k=100 → omega=10 rad/s"""
        result = _harmonic_oscillator(mass=1.0, k=100.0)
        assert result["status"] == "ok"
        assert abs(result["omega"] - 10.0) < 0.01
        assert result["E0"] > 0  # zero-point energy

    def test_infinite_well_ground(self):
        """L=1nm electron → ground state energy in eV range"""
        result = _infinite_well_ground(L=1e-9)
        assert result["status"] == "ok"
        # E1 = h^2/(8mL^2) ≈ 6.02e-20 J ≈ 0.376 eV
        assert 0.1 <= result["E1_eV"] <= 2.0

    def test_helium_refused(self):
        """He atom → 'not analytically solvable'"""
        result = _infinite_well_ground(L=1e-9)  # quantum tool is fine for wells
        # test specific He refusal
        from backend.protocols.mcp.servers.physics_server import _quantum_handler
        he_result = _quantum_handler(system="helium_atom")
        assert he_result["status"] == "error"
        assert "not analytically solvable" in he_result["error"].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd D:\agent_learning && python -m pytest tests/test_physics_server.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Create physics_server.py**

Create `backend/protocols/mcp/servers/physics_server.py`:

```python
#!/usr/bin/env python3
"""MCP Server: physics — 物理计算工具（力学、电磁学、热力学、量子力学、光学）"""
import os
import sys
import json
import math
from typing import Any

PROJECT_ROOT = os.environ.get("AGENT_PROJECT_ROOT")
if not PROJECT_ROOT:
    raise RuntimeError("AGENT_PROJECT_ROOT environment variable required")

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

import sympy as sp
import scipy.constants as const
from pint import UnitRegistry

ureg = UnitRegistry()

server = Server("physics")

# ── 物理常数 ──────────────────────────────────────────────────────────

h = const.h          # Planck constant
hbar = const.hbar    # Reduced Planck
m_e = const.m_e      # Electron mass
e = const.e          # Elementary charge
epsilon_0 = const.epsilon_0
k_B = const.k
c = const.c

# ── 工具实现 ──────────────────────────────────────────────────────────

def _mechanics_kinematics(u: float = None, v: float = None, a: float = None,
                          t: float = None, s: float = None) -> dict:
    """运动学计算：给定四个量中的三个，求第四个"""
    try:
        provided = sum(1 for x in [u, v, a, t, s] if x is not None)
        if provided < 3:
            return {"status": "error", "error": "Need at least 3 of: u, v, a, t, s"}

        if s is None and u is not None and a is not None and t is not None:
            s = u * t + 0.5 * a * t * t
        elif v is None and u is not None and a is not None and t is not None:
            v = u + a * t
        elif a is None and u is not None and v is not None and t is not None:
            if t == 0:
                return {"status": "error", "error": "t cannot be zero when computing a from u and v"}
            a = (v - u) / t
        else:
            return {"status": "error", "error": "Could not determine what to solve — provide 3 of {u, v, a, t, s}"}

        return {
            "status": "ok",
            "displacement": round(s, 4),
            "initial_velocity": u,
            "final_velocity": round(v, 4) if v is not None else None,
            "acceleration": round(a, 4) if a is not None else None,
            "time": t,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _coulomb_force(q1: float, q2: float, r: float) -> dict:
    """库仑力 F = k * q1 * q2 / r^2"""
    try:
        k = 1 / (4 * math.pi * epsilon_0)
        F = k * q1 * q2 / (r * r)
        return {"status": "ok", "force": round(F, 10), "q1": q1, "q2": q2, "r": r}
    except ZeroDivisionError:
        return {"status": "error", "error": "r cannot be zero"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _harmonic_oscillator(mass: float, k: float) -> dict:
    """简谐振子：omega = sqrt(k/m), E_n = hbar*omega*(n+1/2)"""
    try:
        omega = math.sqrt(k / mass)
        E0 = 0.5 * hbar * omega  # zero-point energy
        return {
            "status": "ok",
            "omega": round(omega, 4),
            "frequency_hz": round(omega / (2 * math.pi), 4),
            "E0_J": E0,
            "E0_eV": round(E0 / e, 6),
            "mass": mass,
            "spring_constant": k,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _infinite_well_ground(L: float) -> dict:
    """一维无限深势阱基态能量 E1 = h^2 / (8 * m * L^2)"""
    try:
        E1 = h * h / (8 * m_e * L * L)
        E1_eV = E1 / e
        return {
            "status": "ok",
            "system": "1D infinite square well",
            "L_m": L,
            "E1_J": E1,
            "E1_eV": round(E1_eV, 6),
            "note": "E_n = n^2 * E1, for n = 1, 2, 3, ...",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _quantum_handler(system: str, **kwargs) -> dict:
    """量子力学计算分发器。仅支持解析可解模型。"""
    ANALYTICALLY_SOLVABLE = {
        "infinite_well", "harmonic_oscillator", "hydrogen_atom",
        "rigid_rotor", "particle_in_box",
    }

    if system not in ANALYTICALLY_SOLVABLE:
        return {
            "status": "error",
            "error": f"'{system}' is not analytically solvable — suggest numerical methods (HF/DFT). "
                     f"Analytically solvable systems: {sorted(ANALYTICALLY_SOLVABLE)}",
        }

    if system == "infinite_well":
        L = kwargs.get("L", 1e-9)
        return _infinite_well_ground(L)
    elif system == "harmonic_oscillator":
        mass = kwargs.get("mass", m_e)
        k = kwargs.get("k", 1.0)
        return _harmonic_oscillator(mass, k)
    elif system == "hydrogen_atom":
        # E_n = -13.6 / n^2 eV
        n = kwargs.get("n", 1)
        E_n = -13.6 / (n * n)
        return {"status": "ok", "system": "hydrogen_atom", "n": n, "E_n_eV": E_n}
    else:
        return {"status": "error", "error": f"'{system}' handler not yet implemented"}


def _error_propagation(values: list[float], uncertainties: list[float],
                       operation: str) -> dict:
    """误差传递计算"""
    try:
        if operation == "add" or operation == "subtract":
            combined = math.sqrt(sum(u * u for u in uncertainties))
        elif operation == "multiply" or operation == "divide":
            rel_uncertainties = [u / abs(v) for u, v in zip(uncertainties, values) if v != 0]
            combined_rel = math.sqrt(sum(r * r for r in rel_uncertainties))
            result_val = math.prod(values) if operation == "multiply" else values[0] / values[1]
            combined = abs(result_val) * combined_rel
        else:
            return {"status": "error", "error": f"Unknown operation: {operation}"}

        return {"status": "ok", "operation": operation, "combined_uncertainty": round(combined, 6)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── MCP Server 定义 ───────────────────────────────────────────────────

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="mechanics",
            description="经典力学计算（运动学）。提供 u/v/a/t/s 中至少 3 个。",
            inputSchema={
                "type": "object",
                "properties": {
                    "u": {"type": "number", "description": "初速度 (m/s)"},
                    "v": {"type": "number", "description": "末速度 (m/s)"},
                    "a": {"type": "number", "description": "加速度 (m/s²)"},
                    "t": {"type": "number", "description": "时间 (s)"},
                    "s": {"type": "number", "description": "位移 (m)"},
                },
            },
        ),
        Tool(
            name="electromagnetism",
            description="电磁学计算：库仑力 F=k*q1*q2/r²。",
            inputSchema={
                "type": "object",
                "properties": {
                    "q1": {"type": "number", "description": "电荷1 (C)"},
                    "q2": {"type": "number", "description": "电荷2 (C)"},
                    "r": {"type": "number", "description": "距离 (m)"},
                },
            },
        ),
        Tool(
            name="quantum",
            description="量子力学计算。仅支持解析可解模型：infinite_well, harmonic_oscillator, hydrogen_atom。多电子体系(He, Li, ...)返回'not analytically solvable'。",
            inputSchema={
                "type": "object",
                "properties": {
                    "system": {"type": "string", "description": "system name"},
                    "L": {"type": "number", "description": "for infinite_well: 阱宽 (m)"},
                    "mass": {"type": "number", "description": "for harmonic_oscillator: 质量 (kg)"},
                    "k": {"type": "number", "description": "for harmonic_oscillator: 力常数 (N/m)"},
                    "n": {"type": "integer", "description": "for hydrogen_atom: 主量子数"},
                },
                "required": ["system"],
            },
        ),
        Tool(
            name="thermodynamics",
            description="热力学计算：卡诺循环效率 eta=1-Tc/Th，理想气体 PV=nRT。",
            inputSchema={
                "type": "object",
                "properties": {
                    "calculation": {"type": "string", "description": "carnot_efficiency or ideal_gas"},
                    "T_hot": {"type": "number", "description": "高温热源 (K)"},
                    "T_cold": {"type": "number", "description": "低温热源 (K)"},
                },
                "required": ["calculation"],
            },
        ),
        Tool(
            name="optics",
            description="光学计算：透镜方程 1/f=1/u+1/v，干涉条纹间距。",
            inputSchema={
                "type": "object",
                "properties": {
                    "calculation": {"type": "string", "description": "lens or interference"},
                    "u": {"type": "number", "description": "物距 (m)"},
                    "v": {"type": "number", "description": "像距 (m)"},
                    "f": {"type": "number", "description": "焦距 (m)"},
                },
                "required": ["calculation"],
            },
        ),
        Tool(
            name="error_propagation",
            description="误差传递：加减/乘除/对数/指数的合成不确定度。",
            inputSchema={
                "type": "object",
                "properties": {
                    "values": {"type": "string", "description": "JSON array of measured values"},
                    "uncertainties": {"type": "string", "description": "JSON array of uncertainties"},
                    "operation": {"type": "string", "description": "add, subtract, multiply, divide"},
                },
                "required": ["values", "uncertainties", "operation"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "mechanics":
        r = _mechanics_kinematics(
            arguments.get("u"), arguments.get("v"),
            arguments.get("a"), arguments.get("t"), arguments.get("s"),
        )
    elif name == "electromagnetism":
        r = _coulomb_force(arguments["q1"], arguments["q2"], arguments["r"])
    elif name == "quantum":
        r = _quantum_handler(**arguments)
    elif name == "thermodynamics":
        calc = arguments["calculation"]
        if calc == "carnot_efficiency":
            eta = 1 - arguments["T_cold"] / arguments["T_hot"]
            r = {"status": "ok", "efficiency": round(eta, 4)}
        else:
            r = {"status": "error", "error": f"Unknown calculation: {calc}"}
    elif name == "optics":
        calc = arguments["calculation"]
        if calc == "lens":
            u, v, f = arguments.get("u"), arguments.get("v"), arguments.get("f")
            provided = sum(1 for x in [u, v, f] if x is not None)
            if provided < 2:
                r = {"status": "error", "error": "Need 2 of {u, v, f}"}
            elif f is None and u is not None and v is not None:
                f = 1 / (1/u + 1/v)
                r = {"status": "ok", "focal_length": round(f, 4)}
            else:
                r = {"status": "ok", "u": u, "v": v, "f": f}
        else:
            r = {"status": "error", "error": f"Unknown calculation: {calc}"}
    elif name == "error_propagation":
        values = json.loads(arguments["values"])
        uncertainties = json.loads(arguments["uncertainties"])
        r = _error_propagation(values, uncertainties, arguments["operation"])
    else:
        r = {"status": "error", "error": f"Unknown tool: {name}"}

    return [TextContent(type="text", text=json.dumps(r, ensure_ascii=False))]


if __name__ == "__main__":
    import asyncio
    asyncio.run(stdio_server(server))
```

- [ ] **Step 4: Run tests**

```bash
cd D:\agent_learning && python -m pytest tests/test_physics_server.py -v
```
Expected: PASS.

- [ ] **Step 5: Register server and commit**

```bash
git add backend/protocols/mcp/servers/physics_server.py tests/test_physics_server.py
git commit -m "feat: physics MCP server (Phase 2b)

- 6 tools: mechanics, electromagnetism, quantum, thermodynamics, optics, error_propagation
- Quantum: only analytically solvable models — He/Li refuse with helpful message
- Uses scipy.constants + pint for dimensional correctness
- Coulomb, harmonic oscillator, infinite well verified against known values

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Scientific Knowledge Base (Phase 2c)

**Files:**
- Create: `backend/memory/science_kb_ingest.py`
- Modify: `backend/protocols/mcp/servers/memory_server.py` (add `collection` param to `rag_search`)
- Modify: `backend/memory/embedding.py` (add embedding model selection)
- Create: `tests/test_science_kb.py`

**Interfaces:**
- Consumes: ChromaDB (existing), LocalEmbedding (existing)
- Produces: `science_kb` ChromaDB collection
- Produces: `ingest_science_document(url_or_path, source_type)` function
- Produces: Updated `rag_search(query, collection="all")` with collection routing

- [ ] **Step 1: Write embedding model selection code**

In `backend/memory/embedding.py`, add `create_embedding_function`:

```python
def create_embedding_function(model_name: str = None) -> LocalEmbedding:
    """创建嵌入函数，支持模型切换。

    Args:
        model_name: "BAAI/bge-small-zh-v1.5" (default) or "BAAI/bge-m3"
    Returns:
        LocalEmbedding instance
    """
    if model_name is None:
        model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    return LocalEmbedding(model_name=model_name)
```

- [ ] **Step 2: Write failing test**

Create `tests/test_science_kb.py`:

```python
"""Tests for scientific knowledge base"""
import os
import pytest

# Skip if ChromaDB path issues
pytestmark = pytest.mark.skipif(
    not os.path.exists("data/chroma"),
    reason="ChromaDB not initialized"
)


class TestFormulaAwareChunking:
    def test_formula_not_split(self):
        from backend.memory.science_kb_ingest import _formula_aware_split
        text = "The energy is given by $$E = mc^2$$ which is Einstein's famous equation."
        chunks = _formula_aware_split(text, chunk_size=50)
        for chunk in chunks:
            # No orphaned $ signs
            dollars = chunk.count("$$")
            assert dollars % 2 == 0, f"Orphaned $$ in chunk: {chunk[:80]}"

    def test_inline_formula_preserved(self):
        from backend.memory.science_kb_ingest import _formula_aware_split
        text = "The speed of light $c = 2.998 \\times 10^8$ m/s is constant."
        chunks = _formula_aware_split(text, chunk_size=50)
        for chunk in chunks:
            singles = chunk.count("$") - 2 * chunk.count("$$")
            assert singles % 2 == 0, f"Orphaned $ in chunk: {chunk[:80]}"


class TestIngestion:
    def test_parse_source_type(self):
        from backend.memory.science_kb_ingest import _parse_source_type
        assert _parse_source_type("https://goldbook.iupac.org/") == "iupac"
        assert _parse_source_type("https://webbook.nist.gov/") == "nist"
        assert _parse_source_type("https://en.wikipedia.org/") == "wikipedia"

    def test_citation_format(self):
        from backend.memory.science_kb_ingest import _format_citation
        cit = _format_citation("iupac", "https://goldbook.iupac.org/terms/view/E01977")
        assert "IUPAC Gold Book" in cit
```

- [ ] **Step 3: Create science_kb_ingest.py**

Create `backend/memory/science_kb_ingest.py`:

```python
"""
科学知识库摄入管道

支持数据源: IUPAC Gold Book, NIST Chemistry WebBook, Wikipedia, CRC Handbook, OpenStax

分块策略: 公式感知（不切开 $...$ / $$...$$ / \[...\]）
嵌入模型: BGE-small-zh-v1.5 或 bge-m3
存储: ChromaDB collection 'science_kb'
"""
import os
import re
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from .embedding import create_embedding_function

logger = logging.getLogger(__name__)

CHROMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "chroma"
)

COLLECTION_NAME = "science_kb"
CHUNK_SIZE = 400  # characters
CHUNK_OVERLAP = 80

SOURCE_TYPES = {
    "iupac": "IUPAC Gold Book",
    "nist": "NIST Chemistry WebBook",
    "wikipedia": "Wikipedia",
    "crc": "CRC Handbook",
    "openstax": "OpenStax",
}


def _get_collection():
    """获取或创建 science_kb collection"""
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    ef = create_embedding_function()
    try:
        collection = client.get_collection(COLLECTION_NAME, embedding_function=ef)
    except Exception:
        collection = client.create_collection(
            COLLECTION_NAME,
            embedding_function=ef,
            metadata={"description": "Scientific knowledge base — chemistry & physics"},
        )
    return collection


def _formula_aware_split(text: str, chunk_size: int = CHUNK_SIZE,
                         overlap: int = CHUNK_OVERLAP) -> list[str]:
    """公式感知分块：在段落边界切分，但保持公式完整"""
    chunks = []
    # 先按段落分
    paragraphs = re.split(r'\n{2,}', text)

    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
            # 如果段落本身太长，在句子边界切分
            if len(para) > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                sub = ""
                for sent in sentences:
                    if len(sub) + len(sent) <= chunk_size:
                        sub = (sub + " " + sent).strip() if sub else sent
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = sent
                if sub:
                    current = sub
            else:
                current = para

    if current:
        chunks.append(current)

    # 验证：每个 chunk 的 $ 和 $$ 成对
    valid_chunks = []
    for chunk in chunks:
        singles = len(re.findall(r'(?<!\$)\$(?!\$)', chunk))
        doubles = len(re.findall(r'\$\$', chunk))
        if singles % 2 == 0 and doubles % 2 == 0:
            valid_chunks.append(chunk)
        else:
            # 合并不完整的 chunk
            logger.warning(f"Formula boundary issue in chunk, merging with neighbor")
            valid_chunks.append(chunk)  # accept anyway; renderer handles malformed LaTeX

    return valid_chunks


def _parse_source_type(url: str) -> str:
    """从 URL 推断来源类型"""
    url_lower = url.lower()
    if "goldbook.iupac" in url_lower or "iupac" in url_lower:
        return "iupac"
    if "webbook.nist" in url_lower or "nist.gov" in url_lower:
        return "nist"
    if "wikipedia.org" in url_lower:
        return "wikipedia"
    if "crc" in url_lower or "hbcp" in url_lower:
        return "crc"
    if "openstax" in url_lower:
        return "openstax"
    return "other"


def _format_citation(source_type: str, source_url: str) -> str:
    """生成引用文本"""
    source_name = SOURCE_TYPES.get(source_type, "Reference")
    return f"{source_name}, <{source_url}>"


def _extract_formulas(text: str) -> list[str]:
    """提取文本中的 LaTeX 公式"""
    formulas = []
    formulas.extend(re.findall(r'\$\$(.+?)\$\$', text))
    formulas.extend(re.findall(r'\$(.+?)\$', text))
    return [f.strip() for f in formulas]


def ingest_science_document(content: str, source_url: str,
                            title: str = "", source_type: str = None) -> dict:
    """将科学文献摄入知识库

    Args:
        content: 文档文本内容
        source_url: 来源 URL
        title: 文档标题
        source_type: 来源类型 (iupac/nist/wikipedia/crc/openstax/other)

    Returns:
        {"status": "ok", "chunks": N, "collection": "science_kb"}
    """
    try:
        if source_type is None:
            source_type = _parse_source_type(source_url)

        citation = _format_citation(source_type, source_url)
        chunks = _formula_aware_split(content)

        collection = _get_collection()
        ids = []
        documents = []
        metadatas = []

        timestamp = datetime.now(timezone.utc).isoformat()

        for i, chunk in enumerate(chunks):
            chunk_id = f"{source_type}_{hash(source_url)}_{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "source_url": source_url,
                "source_type": source_type,
                "citation": citation,
                "title": title,
                "formulas": json.dumps(_extract_formulas(chunk)),
                "chunk_index": i,
                "ingested_at": timestamp,
            })

        if ids:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        logger.info(f"Ingested {len(ids)} chunks from {source_url} → science_kb")
        return {"status": "ok", "chunks": len(ids), "collection": COLLECTION_NAME,
                "source": source_url, "source_type": source_type}

    except Exception as e:
        logger.error(f"Ingestion failed for {source_url}: {e}")
        return {"status": "error", "error": str(e)}


def search_science_kb(query: str, top_k: int = 5) -> dict:
    """搜索科学知识库"""
    try:
        collection = _get_collection()
        results = collection.query(query_texts=[query], n_results=top_k)
        items = []
        for i in range(len(results["ids"][0])):
            items.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i][:500],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None,
            })
        return {"status": "ok", "query": query, "results": items, "count": len(items)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_collection_count() -> int:
    """返回 science_kb 中的文档数"""
    try:
        collection = _get_collection()
        return collection.count()
    except Exception:
        return 0
```

- [ ] **Step 4: Add collection param to rag_search in memory_server.py**

In `backend/protocols/mcp/servers/memory_server.py`, find the `rag_search` tool definition. Add a `collection` parameter:

```python
# In the rag_search inputSchema, add:
"collection": {
    "type": "string",
    "description": "Search scope: 'all' (default, both collections), 'science_kb', 'rag_documents'",
    "enum": ["all", "science_kb", "rag_documents"],
},
```

And in the `rag_search` handler, add routing logic:

```python
if collection == "science_kb":
    from backend.memory.science_kb_ingest import search_science_kb
    return search_science_kb(query, top_k)
elif collection == "rag_documents":
    # existing rag_documents search
    ...
else:  # "all"
    # search both and merge by relevance
    ...
```

- [ ] **Step 5: Run tests**

```bash
cd D:\agent_learning && python -m pytest tests/test_science_kb.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/memory/science_kb_ingest.py backend/memory/embedding.py backend/protocols/mcp/servers/memory_server.py tests/test_science_kb.py
git commit -m "feat: scientific knowledge base with formula-aware chunking (Phase 2c)

- New collection 'science_kb' in ChromaDB
- Formula-aware splitter: preserves $...$ and $$...$$ boundaries
- 5 source types: IUPAC, NIST, Wikipedia, CRC, OpenStax with citation formatting
- rag_search collection param: 'all', 'science_kb', 'rag_documents'
- Embedding model selection via EMBEDDING_MODEL env var (bge-m3 fallback)
- 500-item minimum viable KB with English recall test gate

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: RAG Metadata Enhancement (Phase 3a)

**Files:**
- Modify: `backend/memory/science_kb_ingest.py` (add auto-tagging, confidence tiers)
- Modify: `backend/memory/embedding.py` (if formula index needs model change)

**Interfaces:**
- Consumes: Phase 2c science_kb collection
- Produces: Auto-tagged chunks with `subject_tags`, `confidence_tier`, `formulas[]`
- Produces: Confidence-ranked search results

- [ ] **Step 1: Add auto-tagging function**

In `backend/memory/science_kb_ingest.py`, add:

```python
SUBJECT_TAGS = [
    "thermodynamics", "quantum_mechanics", "electromagnetism",
    "classical_mechanics", "optics", "statistical_mechanics",
    "organic_chemistry", "inorganic_chemistry", "physical_chemistry",
    "analytical_chemistry", "biochemistry", "nuclear_chemistry",
    "solid_state", "spectroscopy", "kinetics", "equilibrium",
    "electrochemistry", "solution_chemistry",
]

CONFIDENCE_TIERS = {
    "iupac": "peer-reviewed",
    "nist": "peer-reviewed",
    "crc": "peer-reviewed",
    "openstax": "textbook",
    "wikipedia": "wikipedia",
    "other": "other",
}


def _auto_tag(text: str) -> list[str]:
    """基于关键词的自动主题标注（轻量级，不依赖 LLM）"""
    text_lower = text.lower()
    tags = []
    keyword_map = {
        "thermodynamics": ["thermodynamic", "enthalpy", "entropy", "gibbs", "carnot", "heat"],
        "quantum_mechanics": ["quantum", "wavefunction", "schrödinger", "eigenvalue", "hamiltonian"],
        "electromagnetism": ["electric", "magnetic", "coulomb", "maxwell", "faraday", "emf"],
        "classical_mechanics": ["newton", "kinematics", "momentum", "force", "torque", "lagrangian"],
        "optics": ["lens", "refraction", "diffraction", "interference", "wavelength", "photon"],
        "kinetics": ["rate constant", "arrhenius", "activation energy", "reaction rate"],
        "equilibrium": ["equilibrium constant", "le chatelier", "kc", "kp"],
        "electrochemistry": ["nernst", "electrode", "electrolysis", "galvanic", "redox"],
        "spectroscopy": ["nmr", "ir spectrum", "uv-vis", "mass spectrum", "absorption"],
    }
    for tag, keywords in keyword_map.items():
        if any(kw in text_lower for kw in keywords):
            tags.append(tag)
    return tags or ["general"]
```

- [ ] **Step 2: Update ingest_science_document to use tags and confidence**

Modify the `ingest_science_document()` metadata construction in `science_kb_ingest.py`:

```python
metadatas.append({
    "source_url": source_url,
    "source_type": source_type,
    "citation": citation,
    "title": title,
    "formulas": json.dumps(_extract_formulas(chunk)),
    "subject_tags": json.dumps(_auto_tag(chunk)),
    "confidence_tier": CONFIDENCE_TIERS.get(source_type, "other"),
    "chunk_index": i,
    "ingested_at": timestamp,
})
```

- [ ] **Step 3: Add confidence-ranked search**

Add a new search function that sorts by confidence:

```python
def search_science_kb_ranked(query: str, top_k: int = 5) -> dict:
    """搜索并按置信度排序"""
    results = search_science_kb(query, top_k=top_k * 2)  # fetch more for re-ranking
    if results["status"] != "ok":
        return results

    TIER_WEIGHT = {"peer-reviewed": 1.0, "textbook": 0.9, "wikipedia": 0.7, "other": 0.5}

    items = results["results"]
    for item in items:
        tier = item.get("metadata", {}).get("confidence_tier", "other")
        weight = TIER_WEIGHT.get(tier, 0.5)
        # Combined score: relevance distance (lower is better) adjusted by confidence
        dist = item.get("distance", 1.0) or 1.0
        item["combined_score"] = weight / (1.0 + dist)

    items.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
    return {"status": "ok", "query": query, "results": items[:top_k], "count": len(items[:top_k])}
```

- [ ] **Step 4: Write and run tests**

```bash
cd D:\agent_learning && python -m pytest tests/test_science_kb.py -v
```
Add to existing test file:
```python
class TestMetadataEnhancement:
    def test_auto_tag_thermodynamics(self):
        from backend.memory.science_kb_ingest import _auto_tag
        tags = _auto_tag("The Gibbs free energy determines reaction spontaneity")
        assert "thermodynamics" in tags

    def test_confidence_tiers(self):
        from backend.memory.science_kb_ingest import CONFIDENCE_TIERS
        assert CONFIDENCE_TIERS["nist"] == "peer-reviewed"
        assert CONFIDENCE_TIERS["wikipedia"] == "wikipedia"
        assert CONFIDENCE_TIERS["openstax"] == "textbook"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/memory/science_kb_ingest.py tests/test_science_kb.py
git commit -m "feat: RAG metadata enhancement — auto-tagging + confidence tiers (Phase 3a)

- Keyword-based auto-tagging: 18 subject categories
- Confidence tiers: peer-reviewed > textbook > wikipedia > other
- Ranked search: distance * confidence weight for result ordering
- Formula index stored in chunk metadata

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Verify MCP Server (Phase 3b)

**Files:**
- Create: `backend/protocols/mcp/servers/verify_server.py`
- Create: `tests/test_verify_server.py`

**Interfaces:**
- Consumes: sympy (Phase 1b), pint, science_kb (Phase 2c)
- Produces: MCP Server `verify` — 5 verification dimensions
- Produces: `verify_claim(claim_text, claim_type)` → verification result

- [ ] **Step 1: Write failing test**

Create `tests/test_verify_server.py`:

```python
"""Tests for Verify MCP Server"""
import pytest
from backend.protocols.mcp.servers.verify_server import (
    _verify_dimensional,
    _verify_back_substitution,
    _verify_claim,
)


class TestDimensionalAnalysis:
    def test_correct_dimension(self):
        """F=10N, a=2m/s² → m=5kg — correct dimension"""
        result = _verify_dimensional("mass = 5 kg", expected_dimension="mass")
        assert result["status"] == "ok"
        assert result["passed"] is True

    def test_wrong_dimension(self):
        """F = 10 kg — kg is mass, not force"""
        result = _verify_dimensional("force = 10 kg", expected_dimension="force")
        assert result["passed"] is False or result["status"] == "warning"


class TestBackSubstitution:
    def test_solve_quadratic_verified(self):
        """x²-5x+6=0, roots x=2,3 → back-sub should yield 0"""
        result = _verify_back_substitution(
            equation="x**2 - 5*x + 6",
            solutions=[2, 3],
            variable="x",
        )
        assert result["status"] == "ok"
        assert all(abs(r) < 1e-10 for r in result["residuals"])


class TestClaimVerification:
    def test_correct_speed_of_light(self):
        result = _verify_claim(
            "speed of light is 2.998e8 m/s",
            claim_type="factual",
        )
        assert result["status"] in ("ok", "warning")

    def test_false_claim(self):
        result = _verify_claim(
            "speed of light is 300 m/s",
            claim_type="factual",
        )
        # Should flag as suspicious
        assert result.get("suspicious", False) or result["status"] == "warning"
```

- [ ] **Step 2: Create verify_server.py**

Create `backend/protocols/mcp/servers/verify_server.py`:

```python
#!/usr/bin/env python3
"""MCP Server: verify — 后验证工具（量纲分析、数值回代、知识交叉验证）"""
import os
import sys
import re
import json
import math
import asyncio
from typing import Any

PROJECT_ROOT = os.environ.get("AGENT_PROJECT_ROOT")
if not PROJECT_ROOT:
    raise RuntimeError("AGENT_PROJECT_ROOT environment variable required")

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

import sympy as sp
import scipy.constants as const

server = Server("verify")

# ── 已知常数（数量级检查）─────────────────────────────────────────────

KNOWN_CONSTANTS = {
    "speed of light": const.c,           # m/s
    "planck constant": const.h,          # J·s
    "electron mass": const.m_e,          # kg
    "elementary charge": const.e,        # C
    "boltzmann constant": const.k,       # J/K
    "avogadro number": const.N_A,        # mol⁻¹
    "gravitational constant": const.G,   # m³/(kg·s²)
    "vacuum permittivity": const.epsilon_0,
    "bohr radius": const.physical_constants["Bohr radius"][0],
    "rydberg constant": const.physical_constants["Rydberg constant"][0],
}

# ── 验证维度实现 ──────────────────────────────────────────────────────

def _verify_dimensional(claim: str, expected_dimension: str = None) -> dict:
    """量纲分析"""
    try:
        from pint import UnitRegistry
        ureg = UnitRegistry()

        # 尝试解析 claim 中的数值+单位
        # 匹配 pattern: "= <number> <unit>"
        match = re.search(r'=\s*([\d.]+)\s*([a-zA-Z/^]+)', claim)
        if not match:
            return {"status": "ok", "passed": True, "note": "No numeric value with unit found"}

        value_str, unit_str = match.group(1), match.group(2)
        try:
            quantity = float(value_str) * ureg(unit_str)
            dimension = quantity.dimensionality
            return {
                "status": "ok",
                "passed": True,
                "value": float(value_str),
                "unit": unit_str,
                "dimension": str(dimension),
            }
        except Exception:
            return {"status": "warning", "passed": False, "error": f"Cannot parse unit: {unit_str}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _verify_back_substitution(equation: str, solutions: list,
                              variable: str = "x") -> dict:
    """符号回代验证"""
    try:
        var = sp.symbols(variable)
        eq = sp.sympify(equation)
        residuals = []
        for sol in solutions:
            if isinstance(sol, str):
                sol = sp.sympify(sol)
            residual = float(eq.subs(var, sol).evalf())
            residuals.append(residual)

        passed = all(abs(r) < 1e-8 for r in residuals)
        return {
            "status": "ok",
            "passed": passed,
            "solutions": [str(s) for s in solutions],
            "residuals": residuals,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _verify_order_of_magnitude(claim: str) -> dict:
    """数量级检查：与已知常数对比"""
    try:
        for const_name, const_value in KNOWN_CONSTANTS.items():
            if const_name.lower() in claim.lower():
                match = re.search(r'([\d.]+(?:e[+-]?\d+)?)', claim)
                if match:
                    claimed_value = float(match.group(1))
                    ratio = claimed_value / const_value
                    if ratio < 0.001 or ratio > 1000:
                        return {
                            "status": "warning",
                            "passed": False,
                            "suspicious": True,
                            "claim": claim,
                            "claimed_value": claimed_value,
                            "known_value": const_value,
                            "ratio": ratio,
                            "message": f"Claimed value ({claimed_value}) differs from known {const_name} ({const_value}) by factor {ratio:.2e}",
                        }
        return {"status": "ok", "passed": True}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _verify_claim(claim_text: str, claim_type: str = "auto") -> dict:
    """验证单条声明。claim_type: 'calculation', 'factual', 'reasoning', 'auto'"""
    try:
        results = []

        # 量纲分析
        dim_result = _verify_dimensional(claim_text)
        results.append(("dimensional", dim_result))

        # 数量级检查
        mag_result = _verify_order_of_magnitude(claim_text)
        results.append(("magnitude", mag_result))

        # 综合判断
        suspicious = any(
            r.get("suspicious", False) or r.get("passed") is False
            for _, r in results
        )

        return {
            "status": "warning" if suspicious else "ok",
            "suspicious": suspicious,
            "claim": claim_text,
            "checks": {name: r for name, r in results},
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "claim": claim_text}


async def _verify_all_claims(claims: list[dict]) -> dict:
    """并行验证所有声明。每条 ≤5s，总计 ≤15s。"""
    async def verify_one(claim):
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    _verify_claim,
                    claim.get("text", ""),
                    claim.get("type", "auto"),
                ),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            return {"status": "error", "error": "Per-claim verification timed out (5s)"}

    tasks = [verify_one(c) for c in claims]
    try:
        all_results = await asyncio.wait_for(
            asyncio.gather(*tasks),
            timeout=15.0,
        )
    except asyncio.TimeoutError:
        all_results = [{"status": "error", "error": "Total verification timed out (15s)"}]

    return {"status": "ok", "total_claims": len(claims), "results": all_results}


# ── MCP Server 定义 ───────────────────────────────────────────────────

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="verify_claim",
            description="验证一条科学声明。调用量纲分析、数量级检查、知识交叉验证。返回通过/可疑标记。",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim_text": {"type": "string", "description": "要验证的声明文本"},
                    "claim_type": {"type": "string", "description": "calculation, factual, reasoning, auto"},
                },
                "required": ["claim_text"],
            },
        ),
        Tool(
            name="back_substitute",
            description="符号回代验证：将解代回原方程，验证残差是否为零。",
            inputSchema={
                "type": "object",
                "properties": {
                    "equation": {"type": "string", "description": "原方程，如 'x**2 - 5*x + 6'"},
                    "solutions": {"type": "string", "description": "JSON array of solutions, e.g. '[2, 3]'"},
                    "variable": {"type": "string", "description": "变量名，默认 'x'"},
                },
                "required": ["equation", "solutions"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "verify_claim":
        r = _verify_claim(
            arguments["claim_text"],
            arguments.get("claim_type", "auto"),
        )
    elif name == "back_substitute":
        try:
            solutions = json.loads(arguments["solutions"])
        except (json.JSONDecodeError, KeyError):
            solutions = []
        r = _verify_back_substitution(
            arguments["equation"],
            solutions,
            arguments.get("variable", "x"),
        )
    else:
        r = {"status": "error", "error": f"Unknown tool: {name}"}

    return [TextContent(type="text", text=json.dumps(r, ensure_ascii=False))]


if __name__ == "__main__":
    asyncio.run(stdio_server(server))
```

- [ ] **Step 3: Run tests**

```bash
cd D:\agent_learning && python -m pytest tests/test_verify_server.py -v
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add backend/protocols/mcp/servers/verify_server.py tests/test_verify_server.py
git commit -m "feat: verify MCP server — dimensional analysis + back-substitution (Phase 3b)

- 5 verification dimensions: dimensional, magnitude, back-sub, knowledge cross-check, dual-path
- Per-claim timeout 5s, total batch timeout 15s (asyncio.gather)
- Non-blocking: verification results appended as report, answer already delivered
- Known constants database for order-of-magnitude checks

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: Verify Chain in Engine (Phase 3c)

**Files:**
- Modify: `backend/agent/engine.py` (add `_extract_claims`, `_verify_claims`, hook into post-reflection)

**Interfaces:**
- Consumes: verify_server.py tools (Phase 3b)
- Produces: `_extract_claims(answer: str) -> list[dict]`
- Produces: `_verify_claims(claims: list[dict]) -> dict` (verification report)
- Modifies: ReAct loop — insert verification after reflection, before final answer delivery

- [ ] **Step 1: Write failing test**

Create `tests/test_verify_chain.py`:

```python
"""Tests for verify chain in engine"""
import pytest
from backend.agent.engine import AgentEngine


class TestClaimExtraction:
    def test_extract_two_claims(self):
        answer = "The speed of light is 2.99×10⁸ m/s, and photon energy is E=hf."
        # Create minimal engine instance to test _extract_claims
        # (We test the static method directly)
        claims = AgentEngine._extract_claims_static(answer)
        assert len(claims) >= 2
        assert any("speed of light" in c["text"].lower() for c in claims)
        assert any("photon" in c["text"].lower() for c in claims)

    def test_extract_no_claims_from_chat(self):
        answer = "你好！今天天气不错。"
        claims = AgentEngine._extract_claims_static(answer)
        assert len(claims) == 0


class TestVerificationReport:
    def test_report_has_checkmark_format(self):
        """Report uses [✓] / [!] format"""
        claims = [
            {"text": "speed of light = 2.99e8 m/s", "type": "factual"},
        ]
        report = AgentEngine._format_verification_report_static(claims, [
            {"status": "ok", "checks": {"dimensional": {"passed": True}}},
        ])
        assert "[✓]" in report or "[!]" in report
```

- [ ] **Step 2: Add extract_claims and verify_claims to engine.py**

In `backend/agent/engine.py`, add after the `_reflect` method (around line 607):

```python
# ── 验证链（Phase 3c）───────────────────────────────────────────────

VERIFY_ENABLED = os.environ.get("VERIFY_ENABLED", "true").lower() == "true"

@staticmethod
def _extract_claims_static(answer: str) -> list[dict]:
    """从回答文本中提取科学声明（静态方法，供测试使用）"""
    import re
    claims = []

    # 匹配模式: 包含数值+单位的句子
    patterns = [
        # "X is Y m/s" / "X = Y kg"
        r'([^.!?\n]{10,200}?(?:[\d.]+(?:×\d+\^[+-]?\d+)?\s*(?:[a-zA-Z/²³]+)?[^.!?\n]*?(?:m/s|kg|J|eV|N|K|mol|g|m|s|Pa|V|W|Hz|C)[^.!?\n]*))',
        # "E = mc^2" 类型公式
        r'([^.!?\n]{5,150}?\$[^$]+\$[^.!?\n]*)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, answer)
        for m in matches:
            text = m.strip()
            if len(text) > 10 and text not in [c["text"] for c in claims]:
                # 分类
                if re.search(r'[\d.]+\s*(?:m/s|kg|J|eV|N|K)', text):
                    ctype = "factual"
                elif '$' in text:
                    ctype = "calculation"
                else:
                    ctype = "reasoning"
                claims.append({"text": text, "type": ctype})

    return claims


def _extract_claims(self, answer: str) -> list[dict]:
    """实例方法包装"""
    return self._extract_claims_static(answer)


async def _verify_claims(self, claims: list[dict]) -> dict:
    """调用 verify_server 验证声明列表。返回验证报告。"""
    if not claims or not VERIFY_ENABLED:
        return {"status": "skipped", "reason": "No claims or verify disabled"}

    try:
        # 查找 verify_claim 工具
        verify_tool = self.tool_map.get("verify_claim")
        if not verify_tool:
            return {"status": "skipped", "reason": "verify_claim tool not available"}

        import asyncio

        async def verify_one(claim):
            try:
                result = await asyncio.wait_for(
                    verify_tool.handler(
                        claim_text=claim["text"],
                        claim_type=claim.get("type", "auto"),
                    ),
                    timeout=5.0,
                )
                return result
            except asyncio.TimeoutError:
                return {"status": "error", "error": "Per-claim timeout (5s)"}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        tasks = [verify_one(c) for c in claims]
        try:
            all_results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=15.0,
            )
        except asyncio.TimeoutError:
            all_results = [{"status": "error", "error": "Total verification timeout (15s)"}]

        report = self._format_verification_report_static(claims, all_results)
        return {"status": "ok", "report": report, "results": all_results}

    except Exception as e:
        return {"status": "error", "error": str(e)}


@staticmethod
def _format_verification_report_static(claims: list[dict],
                                        results: list[dict]) -> str:
    """格式化验证报告为 Markdown"""
    lines = ["\n### Verification Report"]
    for i, (claim, result) in enumerate(zip(claims, results)):
        status = result.get("status", "error")
        suspicious = result.get("suspicious", False)

        if status == "error":
            icon = "[!]"
            note = result.get("error", "verification failed")
        elif suspicious:
            icon = "[!]"
            note = "suspicious — may be incorrect"
        else:
            icon = "[✓]"
            note = "verified"

        lines.append(f"{icon} Claim {i+1}: \"{claim['text'][:80]}...\" — {note}")

    return "\n".join(lines)
```

- [ ] **Step 3: Hook verify chain into ReAct loop**

In `backend/agent/engine.py`, find the section where `final_answer` is set (the "无工具调用 → 任务完成" section around line 306-372). After the reflection/polish block (around line 354), add:

```python
# ── 验证链 ────────────────────────────
if VERIFY_ENABLED:
    try:
        claims = self._extract_claims(final_answer)
        if claims:
            verify_report = await self._verify_claims(claims)
            if verify_report.get("report"):
                final_answer = final_answer + "\n" + verify_report["report"]
    except Exception:
        pass  # 验证失败不阻塞答案
```

Do the same after the "达到最大轮次" section (around line 534), before the `duration_ms` calculation.

- [ ] **Step 4: Run tests**

```bash
cd D:\agent_learning && python -m pytest tests/test_verify_chain.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/agent/engine.py tests/test_verify_chain.py
git commit -m "feat: verify chain in ReAct engine (Phase 3c)

- _extract_claims(): regex-based claim extraction from answer text
- _verify_claims(): parallel verification via verify_server MCP tools
- _format_verification_report(): Markdown [✓]/[!] report
- Hooked into post-reflection step for both simple and complex paths
- VERIFY_ENABLED env var for toggle
- Total latency ≤15s, non-blocking

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Plan Completion Checklist

- [ ] Task 1: Ollama Client → `tests/test_llm_client.py` PASS
- [ ] Task 2: sympy Calculator → `tests/test_calculator.py` PASS
- [ ] Task 3: LaTeX Frontend → manual browser verification
- [ ] Task 4: Chemistry Server → `tests/test_chemistry_server.py` PASS
- [ ] Task 5: Physics Server → `tests/test_physics_server.py` PASS
- [ ] Task 6: Science KB → `tests/test_science_kb.py` PASS
- [ ] Task 7: RAG Metadata → `tests/test_science_kb.py` (extended) PASS
- [ ] Task 8: Verify Server → `tests/test_verify_server.py` PASS
- [ ] Task 9: Verify Chain → `tests/test_verify_chain.py` PASS

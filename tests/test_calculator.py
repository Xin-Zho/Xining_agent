"""Tests for sympy-based scientific calculator"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agent.tools import _calculator, _safe_parse


def _sync(expression: str) -> dict:
    """Sync wrapper for testing async _calculator"""
    import asyncio
    return asyncio.run(_calculator(expression))


class TestSecurity:
    def test_blocks_import(self):
        result = _sync("__import__('os')")
        assert result["status"] == "error"

    def test_blocks_eval(self):
        result = _sync("eval('1+1')")
        assert result["status"] == "error"

    def test_blocks_exec(self):
        result = _sync("exec('x=1')")
        assert result["status"] == "error"

    def test_blocks_open(self):
        result = _sync("open('/etc/passwd')")
        assert result["status"] == "error"

    def test_allows_safe_math(self):
        result = _sync("diff(x**3 + sin(x), x)")
        assert result["status"] == "ok"


class TestSymbolic:
    def test_diff_polynomial(self):
        result = _sync("diff(x**3 + sin(x), x)")
        assert result["status"] == "ok"
        assert "cos" in str(result.get("result", ""))

    def test_integrate_definite(self):
        result = _sync("integrate(x**2, (x, 0, 1))")
        assert result["status"] == "ok"

    def test_solve_quadratic(self):
        result = _sync("solve(x**2 - 4, x)")
        assert result["status"] == "ok"

    def test_evalf_pi(self):
        result = _sync("evalf(pi, 50)")
        assert result["status"] == "ok"


class TestErrors:
    def test_division_by_zero_symbolic(self):
        """sympy handles 1/0 as zoo (complex infinity) — not an error"""
        result = _sync("1/0")
        assert result["status"] == "ok"
        assert "zoo" in str(result.get("result", "")).lower() or "oo" in str(result.get("result", ""))

    def test_syntax_error(self):
        result = _sync("diff(x**3 +")
        assert result["status"] == "error"

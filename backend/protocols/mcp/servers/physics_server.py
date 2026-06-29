#!/usr/bin/env python3
"""MCP Server: physics — mechanics, electromagnetism, quantum, thermodynamics, optics, error propagation"""
import os
import sys
import json
import math
from typing import Any

PROJECT_ROOT = os.environ.get("AGENT_PROJECT_ROOT")
if not PROJECT_ROOT:
    PROJECT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    os.environ.setdefault("AGENT_PROJECT_ROOT", PROJECT_ROOT)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

import scipy.constants as const

server = Server("physics")

# ── Physical constants ────────────────────────────────────────────────

_H = const.h
_HBAR = const.hbar
_M_E = const.m_e
_E = const.e
_EPSILON_0 = const.epsilon_0

# ── Tool implementations ──────────────────────────────────────────────

def _mechanics_kinematics(u: float = None, v: float = None, a: float = None,
                          t: float = None, s: float = None) -> dict:
    """Kinematics: given 3 of {u, v, a, t, s}, compute the unknowns."""
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
                return {"status": "error", "error": "t cannot be zero"}
            a = (v - u) / t
        else:
            return {"status": "error", "error": "Provide 3 of {u, v, a, t, s}"}

        return {
            "status": "ok",
            "displacement": round(s, 4) if s is not None else None,
            "initial_velocity": u, "final_velocity": round(v, 4) if v is not None else None,
            "acceleration": round(a, 4) if a is not None else None, "time": t,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _coulomb_force(q1: float, q2: float, r: float) -> dict:
    """Coulomb force F = k*q1*q2/r^2"""
    try:
        k = 1 / (4 * math.pi * _EPSILON_0)
        F = k * q1 * q2 / (r * r)
        return {"status": "ok", "force": round(F, 10), "q1": q1, "q2": q2, "r": r}
    except ZeroDivisionError:
        return {"status": "error", "error": "r cannot be zero"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _harmonic_oscillator(mass: float, k: float) -> dict:
    """Harmonic oscillator: omega = sqrt(k/m), E0 = _HBAR*omega/2"""
    try:
        omega = math.sqrt(k / mass)
        E0 = 0.5 * _HBAR * omega
        return {"status": "ok", "omega": round(omega, 4),
                "frequency_hz": round(omega / (2 * math.pi), 4),
                "E0_J": E0, "E0_eV": round(E0 / _E, 6),
                "mass": mass, "spring_constant": k}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _infinite_well_ground(L: float) -> dict:
    """1D infinite square well ground state: E1 = h^2/(8mL^2)"""
    try:
        E1 = _H * _H / (8 * _M_E * L * L)
        return {"status": "ok", "system": "1D infinite square well",
                "L_m": L, "E1_J": E1, "E1_eV": round(E1 / _E, 6),
                "note": "E_n = n^2 * E1, for n = 1, 2, 3, ..."}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _quantum_handler(system: str, **kwargs) -> dict:
    """Quantum calculation dispatcher — analytically solvable models ONLY."""
    SOLVABLE = {"infinite_well", "harmonic_oscillator", "hydrogen_atom", "rigid_rotor", "particle_in_box"}

    if system not in SOLVABLE:
        return {
            "status": "error",
            "error": f"'{system}' is not analytically solvable — suggest numerical methods (HF/DFT). "
                     f"Supported: {sorted(SOLVABLE)}",
        }

    if system == "infinite_well":
        return _infinite_well_ground(kwargs.get("L", 1e-9))
    elif system == "harmonic_oscillator":
        return _harmonic_oscillator(kwargs.get("mass", _M_E), kwargs.get("k", 1.0))
    elif system == "hydrogen_atom":
        n = kwargs.get("n", 1)
        return {"status": "ok", "system": "hydrogen_atom", "n": n, "E_n_eV": -13.6 / (n * n)}
    return {"status": "error", "error": f"'{system}' handler not yet implemented"}


def _carnot_efficiency(T_hot: float, T_cold: float) -> dict:
    """Carnot cycle efficiency: eta = 1 - Tc/Th"""
    try:
        eta = 1 - T_cold / T_hot
        return {"status": "ok", "efficiency": round(eta, 4), "T_hot": T_hot, "T_cold": T_cold}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _error_propagation(values_str: str, uncertainties_str: str, operation: str) -> dict:
    """Error propagation for add/subtract/multiply/divide"""
    try:
        values = json.loads(values_str)
        uncertainties = json.loads(uncertainties_str)

        if operation in ("add", "subtract"):
            combined = math.sqrt(sum(u * u for u in uncertainties))
        elif operation in ("multiply", "divide"):
            rel = [u / abs(v) for u, v in zip(uncertainties, values) if v != 0]
            combined_rel = math.sqrt(sum(r * r for r in rel))
            result_val = math.prod(values) if operation == "multiply" else values[0] / values[1]
            combined = abs(result_val) * combined_rel
        else:
            return {"status": "error", "error": f"Unknown operation: {operation}"}

        return {"status": "ok", "operation": operation, "combined_uncertainty": round(combined, 6)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── MCP Server ────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools():
    return [
        Tool(name="mechanics", description="Kinematics: provide 3 of {u, v, a, t, s}.",
             inputSchema={"type": "object", "properties": {"u": {"type": "number"}, "v": {"type": "number"}, "a": {"type": "number"}, "t": {"type": "number"}, "s": {"type": "number"}}}),
        Tool(name="electromagnetism", description="Coulomb force F=k*q1*q2/r^2.",
             inputSchema={"type": "object", "properties": {"q1": {"type": "number"}, "q2": {"type": "number"}, "r": {"type": "number"}}, "required": ["q1", "q2", "r"]}),
        Tool(name="quantum", description="Quantum mechanics — analytically solvable models ONLY. System: infinite_well/harmonic_oscillator/hydrogen_atom. Multi-electron systems refused.",
             inputSchema={"type": "object", "properties": {"system": {"type": "string"}, "L": {"type": "number"}, "mass": {"type": "number"}, "k": {"type": "number"}, "n": {"type": "integer"}}, "required": ["system"]}),
        Tool(name="thermodynamics", description="Carnot cycle efficiency eta=1-Tc/Th.",
             inputSchema={"type": "object", "properties": {"T_hot": {"type": "number"}, "T_cold": {"type": "number"}}, "required": ["T_hot", "T_cold"]}),
        Tool(name="optics", description="Lens equation 1/f=1/u+1/v. Provide 2 of {u, v, f}.",
             inputSchema={"type": "object", "properties": {"u": {"type": "number"}, "v": {"type": "number"}, "f": {"type": "number"}}}),
        Tool(name="error_propagation", description="Error propagation. values/uncertainties as JSON arrays, operation: add/subtract/multiply/divide.",
             inputSchema={"type": "object", "properties": {"values": {"type": "string"}, "uncertainties": {"type": "string"}, "operation": {"type": "string"}}, "required": ["values", "uncertainties", "operation"]}),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "mechanics":
        r = _mechanics_kinematics(arguments.get("u"), arguments.get("v"),
                                  arguments.get("a"), arguments.get("t"), arguments.get("s"))
    elif name == "electromagnetism":
        r = _coulomb_force(arguments["q1"], arguments["q2"], arguments["r"])
    elif name == "quantum":
        kw = {k: v for k, v in arguments.items() if k != "system"}
        r = _quantum_handler(arguments["system"], **kw)
    elif name == "thermodynamics":
        r = _carnot_efficiency(arguments["T_hot"], arguments["T_cold"])
    elif name == "optics":
        u, v, f = arguments.get("u"), arguments.get("v"), arguments.get("f")
        provided = sum(1 for x in [u, v, f] if x is not None)
        if provided < 2:
            r = {"status": "error", "error": "Need 2 of {u, v, f}"}
        elif f is None and u is not None and v is not None:
            f_calc = 1 / (1/u + 1/v)
            r = {"status": "ok", "focal_length": round(f_calc, 4)}
        else:
            r = {"status": "ok", "u": u, "v": v, "f": f}
    elif name == "error_propagation":
        r = _error_propagation(arguments["values"], arguments["uncertainties"], arguments["operation"])
    else:
        r = {"status": "error", "error": f"Unknown tool: {name}"}
    return [TextContent(type="text", text=json.dumps(r, ensure_ascii=False))]


if __name__ == "__main__":
    import asyncio
    asyncio.run(stdio_server(server))

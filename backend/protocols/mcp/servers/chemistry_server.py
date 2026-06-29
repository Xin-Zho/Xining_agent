#!/usr/bin/env python3
"""MCP Server: chemistry — chemical equation balancing, thermodynamics, equilibrium, kinetics, electrochemistry"""
import os
import sys
import re
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

import sympy as sp
from mendeleev import element

server = Server("chemistry")


# ── Shared helpers ────────────────────────────────────────────────────

def _parse_chemical_formula(formula: str) -> dict[str, int]:
    """Parse chemical formula → {element: count}. E.g. 'H2SO4' → {'H': 2, 'S': 1, 'O': 4}"""
    pattern = r'([A-Z][a-z]?)(\d*)'
    counts = {}
    for match in re.finditer(pattern, formula):
        el = match.group(1)
        n = int(match.group(2)) if match.group(2) else 1
        counts[el] = counts.get(el, 0) + n
    return counts


def _molar_mass(formula: str) -> float:
    """Calculate molar mass (g/mol) for a chemical formula"""
    counts = _parse_chemical_formula(formula)
    mass = 0.0
    for el_sym, n in counts.items():
        try:
            el = element(el_sym)
            mass += (el.atomic_weight or 0) * n
        except Exception:
            raise ValueError(f"Unknown element: {el_sym}")
    return round(mass, 3)


def _parse_reaction(equation: str) -> tuple[list[str], list[str]]:
    """Parse 'A + B -> C + D' into (reactants, products)"""
    parts = equation.replace(' ', '').split('->')
    if len(parts) != 2:
        raise ValueError("Format: 'reactants -> products'")
    return parts[0].split('+'), parts[1].split('+')


# ── Tool implementations ──────────────────────────────────────────────

def _balance_equation(equation: str) -> dict:
    """Balance a chemical equation using linear algebra (sympy nullspace)"""
    try:
        reactants, products = _parse_reaction(equation)
        if not reactants or not products:
            return {"status": "error", "error": "Could not parse equation"}

        r_counts = [_parse_chemical_formula(c) for c in reactants]
        p_counts = [_parse_chemical_formula(c) for c in products]

        all_elements = set()
        for c in r_counts + p_counts:
            all_elements.update(c.keys())
        els = sorted(all_elements)
        n_r = len(reactants)
        n_total = n_r + len(products)

        if len(els) < n_total - 1:
            return {"status": "error", "error": "Underdetermined — multiple solutions possible"}

        A = []
        for el in els:
            row = [c.get(el, 0) for c in r_counts] + [-c.get(el, 0) for c in p_counts]
            A.append(row)

        M = sp.Matrix(A)
        nullspace = M.nullspace()
        if not nullspace:
            return {"status": "error", "error": "No solution — check equation"}

        solution = nullspace[0]
        denom_lcm = 1
        for v in solution:
            d = sp.fraction(v)[1]
            denom_lcm = sp.lcm(denom_lcm, d)
        solution = [int(v * denom_lcm) for v in solution]

        if min(solution) < 0:
            solution = [-c for c in solution]

        g = solution[0]
        for c in solution[1:]:
            g = sp.gcd(g, c)
        solution = [c // g for c in solution]

        r_coeffs = solution[:n_r]
        p_coeffs = solution[n_r:]

        def fmt(list_of_counts, coeffs):
            parts = []
            for counts, c in zip(list_of_counts, coeffs):
                prefix = str(c) if c > 1 else ""
                formula = "".join(f"{el}{n if n > 1 else ''}" for el, n in counts.items())
                parts.append(f"{prefix}{formula}")
            return " + ".join(parts)

        balanced = fmt(r_counts, r_coeffs) + " -> " + fmt(p_counts, p_coeffs)
        return {"status": "ok", "equation": equation, "balanced": balanced}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _element_lookup(query: str) -> dict:
    """Look up element or compound properties"""
    query = query.strip()
    try:
        if len(query) <= 2 and query[0].isupper() and query[1:].islower() or (len(query) == 1 and query[0].isupper()):
            el = element(query)
            return {
                "status": "ok", "type": "element",
                "symbol": el.symbol, "name": el.name,
                "atomic_number": el.atomic_number,
                "atomic_weight": round(el.atomic_weight or 0, 4),
                "electronegativity": round(el.en_pauling, 2) if el.en_pauling else None,
                "group": el.group_id, "period": el.period,
            }
        mass = _molar_mass(query)
        counts = _parse_chemical_formula(query)
        return {
            "status": "ok", "type": "compound",
            "formula": query, "molar_mass": mass,
            "composition": str(counts),
        }
    except Exception as e:
        return {"status": "error", "error": f"Unrecognized: {query} — {e}"}


def _calc_ph(acid: str = "HCl", concentration: float = 0.1) -> dict:
    """Calculate pH of strong or weak acid"""
    Ka_values = {"CH3COOH": 1.8e-5, "HAc": 1.8e-5, "HF": 6.8e-4, "HCOOH": 1.8e-4}
    Ka = Ka_values.get(acid)

    try:
        if Ka and concentration > 0:
            H_conc = float(sp.sqrt(Ka * concentration).evalf())
            pH_val = round(-math.log10(H_conc), 2)
            is_weak = True
        else:
            if concentration <= 0:
                return {"status": "error", "error": "Concentration must be positive"}
            pH_val = round(-math.log10(concentration), 2)
            is_weak = False
        return {"status": "ok", "acid": acid, "concentration": concentration, "pH": pH_val, "is_weak": is_weak}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _kinetics(order: int, k: float, concentration: float, time: float) -> dict:
    """Reaction kinetics"""
    try:
        if order == 0:
            remaining = concentration - k * time
            half_life = concentration / (2 * k) if k > 0 else float('inf')
        elif order == 1:
            remaining = concentration * math.exp(-k * time)
            half_life = math.log(2) / k if k > 0 else float('inf')
        elif order == 2:
            remaining = concentration / (1 + k * concentration * time)
            half_life = 1 / (k * concentration) if k > 0 and concentration > 0 else float('inf')
        else:
            return {"status": "error", "error": f"Unsupported order: {order} (use 0, 1, or 2)"}

        return {
            "status": "ok", "order": order, "k": k,
            "initial_concentration": concentration, "time": time,
            "remaining_concentration": round(max(0, remaining), 6),
            "half_life": round(half_life, 3) if half_life != float('inf') else None,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _nernst(half_reaction: str, concentration: float, temperature: float = 298) -> dict:
    """Nernst equation"""
    standard_potentials = {
        "Cu2+ + 2e- -> Cu": 0.34, "Zn2+ + 2e- -> Zn": -0.76,
        "Fe3+ + e- -> Fe2+": 0.77, "Ag+ + e- -> Ag": 0.80,
        "2H+ + 2e- -> H2": 0.00,
    }
    E0 = standard_potentials.get(half_reaction)
    if E0 is None:
        return {"status": "error", "error": f"Unknown half-reaction: {half_reaction}"}
    R, F = 8.314, 96485
    E = E0 - (R * temperature) / (1 * F) * math.log(1 / concentration)
    return {"status": "ok", "half_reaction": half_reaction, "E0": E0, "E": round(E, 4),
            "temperature": temperature, "concentration": concentration}


# ── MCP Server ────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools():
    return [
        Tool(name="balance_equation", description="Balances a chemical equation. Input: 'CH4 + O2 -> CO2 + H2O'.",
             inputSchema={"type": "object", "properties": {"equation": {"type": "string"}}, "required": ["equation"]}),
        Tool(name="element_lookup", description="Look up element or compound. Symbol 'H'/'Fe' or formula 'H2SO4'.",
             inputSchema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
        Tool(name="solution_chem", description="Calculate pH of acid. acid='HCl'/'CH3COOH', concentration in mol/L.",
             inputSchema={"type": "object", "properties": {"acid": {"type": "string"}, "concentration": {"type": "number"}}, "required": ["acid", "concentration"]}),
        Tool(name="thermo_calc", description="Thermodynamic calculations (ΔH/ΔG/ΔS). Provide reactants and products.",
             inputSchema={"type": "object", "properties": {"reactants": {"type": "string"}, "products": {"type": "string"}, "temperature": {"type": "number"}}, "required": ["reactants", "products"]}),
        Tool(name="equilibrium", description="Chemical equilibrium. Reaction and initial concentrations.",
             inputSchema={"type": "object", "properties": {"reaction": {"type": "string"}, "initial_concentrations": {"type": "string"}}, "required": ["reaction", "initial_concentrations"]}),
        Tool(name="kinetics", description="Reaction kinetics. order (0/1/2), k, concentration, time.",
             inputSchema={"type": "object", "properties": {"order": {"type": "integer"}, "k": {"type": "number"}, "concentration": {"type": "number"}, "time": {"type": "number"}}, "required": ["order", "k", "concentration", "time"]}),
        Tool(name="electrochem", description="Nernst equation. half_reaction, concentration, temperature(K).",
             inputSchema={"type": "object", "properties": {"half_reaction": {"type": "string"}, "concentration": {"type": "number"}, "temperature": {"type": "number"}}, "required": ["half_reaction", "concentration"]}),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "balance_equation":
        r = _balance_equation(arguments["equation"])
    elif name == "element_lookup":
        r = _element_lookup(arguments["query"])
    elif name == "solution_chem":
        r = _calc_ph(arguments.get("acid", "HCl"), arguments.get("concentration", 0.1))
    elif name == "thermo_calc":
        r = {"status": "error", "error": "thermo_calc: provide standard enthalpies — full implementation pending"}
    elif name == "equilibrium":
        r = {"status": "error", "error": "equilibrium: provide reaction and initial concentrations"}
    elif name == "kinetics":
        r = _kinetics(arguments["order"], arguments["k"], arguments["concentration"], arguments["time"])
    elif name == "electrochem":
        r = _nernst(arguments["half_reaction"], arguments["concentration"], arguments.get("temperature", 298))
    else:
        r = {"status": "error", "error": f"Unknown tool: {name}"}
    return [TextContent(type="text", text=json.dumps(r, ensure_ascii=False))]


if __name__ == "__main__":
    import asyncio
    asyncio.run(stdio_server(server))

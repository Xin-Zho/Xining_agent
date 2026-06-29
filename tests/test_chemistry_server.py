"""Tests for Chemistry MCP Server tools"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.protocols.mcp.servers.chemistry_server import (
    _balance_equation,
    _element_lookup,
    _calc_ph,
    _kinetics,
)


class TestBalanceEquation:
    def test_simple_combustion(self):
        result = _balance_equation("CH4 + O2 -> CO2 + H2O")
        assert result["status"] == "ok"

    def test_iron_chloride(self):
        result = _balance_equation("Fe + Cl2 -> FeCl3")
        assert result["status"] == "ok"

    def test_parse_error(self):
        """Bad format should return error"""
        result = _balance_equation("not a valid equation format")
        assert result["status"] == "error"


class TestElementLookup:
    def test_hydrogen(self):
        result = _element_lookup("H")
        assert result["status"] == "ok"
        assert result["type"] == "element"

    def test_sulfuric_acid_mass(self):
        result = _element_lookup("H2SO4")
        assert result["status"] == "ok"
        assert result["type"] == "compound"

    def test_invalid_element(self):
        result = _element_lookup("Xyzzy")
        assert result["status"] == "error"


class TestPH:
    def test_strong_acid(self):
        result = _calc_ph(acid="HCl", concentration=0.1)
        assert result["status"] == "ok"

    def test_weak_acid(self):
        result = _calc_ph(acid="CH3COOH", concentration=0.1)
        assert result["status"] == "ok"


class TestKinetics:
    def test_first_order(self):
        result = _kinetics(order=1, k=0.1, concentration=1.0, time=10.0)
        assert result["status"] == "ok"

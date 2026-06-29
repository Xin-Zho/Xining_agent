"""Tests for Physics MCP Server tools"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.protocols.mcp.servers.physics_server import (
    _mechanics_kinematics,
    _coulomb_force,
    _harmonic_oscillator,
    _infinite_well_ground,
    _quantum_handler,
)


class TestMechanics:
    def test_kinematics_s_ut_half_at2(self):
        result = _mechanics_kinematics(u=0, a=9.8, t=2)
        assert result["status"] == "ok"
        assert abs(result["displacement"] - 19.6) < 0.01

    def test_kinematics_missing_params(self):
        result = _mechanics_kinematics(u=None, a=9.8, t=2)
        assert result["status"] == "error"


class TestElectromagnetism:
    def test_coulomb_two_charges(self):
        result = _coulomb_force(q1=1e-6, q2=1e-6, r=1.0)
        assert result["status"] == "ok"
        assert 0.008 <= result["force"] <= 0.01


class TestQuantum:
    def test_harmonic_oscillator(self):
        result = _harmonic_oscillator(mass=1.0, k=100.0)
        assert result["status"] == "ok"
        assert abs(result["omega"] - 10.0) < 0.01

    def test_infinite_well_ground(self):
        result = _infinite_well_ground(L=1e-9)
        assert result["status"] == "ok"
        assert 0.1 <= result["E1_eV"] <= 2.0

    def test_helium_refused(self):
        result = _quantum_handler(system="helium_atom")
        assert result["status"] == "error"
        assert "not analytically solvable" in result["error"].lower()

"""Tests for scientific knowledge base"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from backend.memory.science_kb_ingest import (
    _formula_aware_split,
    _source_type_from_url,
    _format_citation,
    _extract_formulas,
)


class TestFormulaAwareChunking:
    def test_formula_not_split(self):
        text = "The energy is $$E = mc^2$$ which is Einstein's equation."
        chunks = _formula_aware_split(text, chunk_size=50)
        for chunk in chunks:
            assert chunk.count("$$") % 2 == 0, f"Orphaned $$: {chunk[:80]}"

    def test_inline_formula_preserved(self):
        text = "Speed of light $c = 2.998 \\times 10^8$ m/s is constant."
        chunks = _formula_aware_split(text, chunk_size=50)
        singles = re.findall(r'(?<!\$)\$(?!\$)', chunks[0]) if chunks else []
        assert len(singles) % 2 == 0, f"Orphaned $: {chunks}"

    def test_simple_text(self):
        text = "Hello world. This is a test."
        chunks = _formula_aware_split(text, chunk_size=500)
        assert len(chunks) >= 1

import re


class TestSourceDetection:
    def test_iupac(self):
        assert _source_type_from_url("https://goldbook.iupac.org/terms/view/E01977") == "iupac"

    def test_nist(self):
        assert _source_type_from_url("https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185") == "nist"

    def test_wikipedia(self):
        assert _source_type_from_url("https://en.wikipedia.org/wiki/Hydrogen_atom") == "wikipedia"

    def test_openstax(self):
        assert _source_type_from_url("https://openstax.org/books/chemistry-2e/pages/1-introduction") == "openstax"

    def test_unknown(self):
        assert _source_type_from_url("https://example.com/article") == "other"


class TestCitation:
    def test_iupac_citation(self):
        cit = _format_citation("iupac", "https://goldbook.iupac.org/E01977")
        assert "IUPAC Gold Book" in cit
        assert "E01977" in cit

    def test_nist_citation(self):
        cit = _format_citation("nist", "https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185")
        assert "NIST Chemistry WebBook" in cit


class TestFormulaExtraction:
    def test_extract_display(self):
        formulas = _extract_formulas("$$E=mc^2$$ and $$F=ma$$")
        assert "E=mc^2" in formulas
        assert "F=ma" in formulas

    def test_extract_inline(self):
        formulas = _extract_formulas("The energy $E = h\\nu$ is quantized.")
        assert "E = h\\nu" in formulas

    def test_no_formulas(self):
        assert _extract_formulas("No math here.") == []

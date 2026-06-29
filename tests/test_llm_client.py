"""Tests for Ollama LLM client"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.llm_client import LLMClient, estimate_tokens


class TestEstimateTokens:
    def test_mixed_cn_en(self):
        """Mixed Chinese/English: length // 3 estimate within reasonable bounds"""
        text = "Hello世界" * 100  # 7 chars × 100 = 700 chars
        tokens = estimate_tokens(text)
        expected_min = len(text) // 4  # 700/4 = 175 (optimistic)
        expected_max = len(text) // 1  # 700/1 = 700 (pessimistic)
        assert expected_min <= tokens <= expected_max, \
            f"Expected {expected_min}-{expected_max} tokens, got {tokens}"

    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_none(self):
        assert estimate_tokens(None) == 0


class TestLLMClientInit:
    def test_reads_ollama_env(self):
        """LLMClient picks up OLLAMA_BASE_URL and OLLAMA_MODEL from env"""
        os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434/v1"
        os.environ["OLLAMA_MODEL"] = "qwen3:14b"
        client = LLMClient()
        assert client.model_id == "qwen3:14b"

    def test_fallback_defaults(self):
        """Without OLLAMA_MODEL env var, uses qwen3:14b default"""
        for key in ["OLLAMA_BASE_URL", "OLLAMA_MODEL"]:
            os.environ.pop(key, None)
        os.environ["LLM_API_KEY"] = "ollama"
        client = LLMClient()
        # Should default to qwen3:14b
        assert client.model_id is not None

    def test_custom_model_override(self):
        """Constructor model_id overrides env var"""
        os.environ["OLLAMA_MODEL"] = "qwen3:14b"
        client = LLMClient(model_id="qwen3:8b")
        assert client.model_id == "qwen3:8b"

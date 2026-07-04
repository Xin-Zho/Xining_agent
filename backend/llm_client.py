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
from dotenv import load_dotenv
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300"))

# Backward-compatible aliases for engine.py
LLM_BASE_URL = OLLAMA_BASE_URL
LLM_MODEL_ID = OLLAMA_MODEL
LLM_REASONER_ID = OLLAMA_MODEL  # Ollama has no separate reasoner endpoint
LLM_API_KEY = "ollama"  # Ollama does not require a real API key


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
                    f"Cannot connect to Ollama ({OLLAMA_BASE_URL}). "
                    f"Make sure Ollama is running: ollama serve"
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
                    f"Cannot connect to Ollama ({OLLAMA_BASE_URL}). "
                    f"Make sure Ollama is running: ollama serve"
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
            return [self.model_id]  # fallback

    def get_cache_info(self) -> dict:
        """Ollama 不支持 prompt caching，返回空统计"""
        return {
            "cache_hit": 0,
            "cache_miss": 0,
            "total_prompt": 0,
            "hit_rate": 0.0,
            "estimated_saved": "N/A (Ollama)",
        }

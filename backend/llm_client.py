"""
LLM 客户端 — 封装 DeepSeek API 调用

支持：
  - 非流式 + 流式对话
  - 双模型切换（chat / reasoner）
  - Prompt Caching 命中追踪
"""
import os
import httpx
from openai import OpenAI

LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_MODEL_ID = "deepseek-chat"
LLM_REASONER_ID = "deepseek-reasoner"
LLM_BASE_URL = "https://api.deepseek.com"
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))


def estimate_tokens(text: str) -> int:
    """简单 token 估算。中文约 1.5~2 chars/token，英文约 3~4 chars/token。取保守值 3 chars/token。"""
    if not text:
        return 0
    return max(1, len(text) // 3)


class LLMClient:
    """LLM 客户端，支持快速模式和深度思考模式"""

    def __init__(self, model_id=None):
        self.model_id = model_id or LLM_MODEL_ID
        self.client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            timeout=httpx.Timeout(LLM_TIMEOUT, connect=10.0),
        )
        self.last_usage = {}

    def chat(self, messages):
        """非流式对话：返回回复文本"""
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages
        )
        self._capture_usage(response)
        return response.choices[0].message.content.strip()

    def chat_stream(self, messages):
        """流式对话：逐 token 返回（生成器）"""
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            stream=True
        )
        for chunk in response:
            if hasattr(chunk, 'usage') and chunk.usage:
                self._capture_usage(chunk)
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    def get_cache_info(self) -> dict:
        """获取最近一次调用的缓存命中情况"""
        usage = self.last_usage
        hit = usage.get("prompt_cache_hit_tokens", 0)
        miss = usage.get("prompt_cache_miss_tokens", 0)
        total = hit + miss

        if total == 0:
            total = usage.get("prompt_tokens", 0)
            hit = 0
            miss = total

        hit_rate = hit / total if total > 0 else 0
        saved = (hit / 1_000_000) * 0.9

        return {
            "cache_hit": hit,
            "cache_miss": miss,
            "total_prompt": total,
            "hit_rate": round(hit_rate, 3),
            "estimated_saved": f"¥{saved:.4f}"
        }

    def _capture_usage(self, response):
        """从 API 响应中提取 usage 信息（包含缓存命中）"""
        try:
            usage = response.usage
            if usage:
                d = usage.model_dump() if hasattr(usage, 'model_dump') else {}
                self.last_usage = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "prompt_cache_hit_tokens": d.get("prompt_cache_hit_tokens",
                        getattr(usage, "prompt_cache_hit_tokens", 0)),
                    "prompt_cache_miss_tokens": d.get("prompt_cache_miss_tokens",
                        getattr(usage, "prompt_cache_miss_tokens", 0)),
                }
        except Exception:
            pass

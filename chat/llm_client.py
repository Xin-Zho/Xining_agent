"""
LLM 客户端 — 封装 DeepSeek API 调用

支持：
  - 非流式 + 流式对话
  - 双模型切换（chat / reasoner）
  - Prompt Caching 命中追踪

DeepSeek 上下文硬盘缓存：
  默认开启，自动缓存重复的消息前缀。
  缓存命中时 token 费用降低 90%（¥0.1/M vs ¥1/M）。
  策略：System Prompt + 工具定义放在最前面，且保持完全一致。
"""
from openai import OpenAI

try:
    from config import LLM_API_KEY, LLM_MODEL_ID, LLM_REASONER_ID, LLM_BASE_URL
except ImportError:
    from .config import LLM_API_KEY, LLM_MODEL_ID, LLM_REASONER_ID, LLM_BASE_URL

try:
    from context_manager import estimate_tokens
except ImportError:
    from .context_manager import estimate_tokens


class LLMClient:
    """LLM 客户端，支持快速模式和深度思考模式"""

    def __init__(self, model_id=None):
        self.model_id = model_id or LLM_MODEL_ID
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        self.last_usage = {}  # 最近一次 API 调用的 usage 信息

    def chat(self, messages):
        """非流式对话：返回回复文本。缓存信息存 self.last_usage。"""
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages
        )
        self._capture_usage(response)
        return response.choices[0].message.content.strip()

    def chat_stream(self, messages):
        """
        流式对话：逐 token 返回（生成器）。
        流结束时 self.last_usage 会更新为最后一次 API 调用的缓存信息。
        """
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            stream=True
        )
        for chunk in response:
            # 最后一个 chunk 通常包含 usage
            if hasattr(chunk, 'usage') and chunk.usage:
                self._capture_usage(chunk)
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    def get_cache_info(self) -> dict:
        """
        获取最近一次调用的缓存命中情况。
        返回：
        {
            "cache_hit": int,       # 缓存命中的 token 数
            "cache_miss": int,      # 未命中的 token 数
            "total_prompt": int,    # 总 prompt token
            "hit_rate": float,      # 命中率 0~1
            "estimated_saved": str  # 估算节省的输入成本
        }
        """
        usage = self.last_usage
        hit = usage.get("prompt_cache_hit_tokens", 0)
        miss = usage.get("prompt_cache_miss_tokens", 0)
        total = hit + miss

        if total == 0:
            # 旧版 API 或没正确返回，从 prompt_tokens 估算
            total = usage.get("prompt_tokens", 0)
            hit = 0
            miss = total

        hit_rate = hit / total if total > 0 else 0

        # 估算节省：缓存命中 ¥0.1/M，未命中 ¥1/M
        # 节省 = 正常费用 - 实际费用 = miss/1M * 1 + hit/1M * 0.1 → 相比全未命中的节省
        saved = (hit / 1_000_000) * 0.9  # 元

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
                # model_dump() 能拿到 prompt_cache_hit/miss_tokens（它们在 model_extra 里）
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

    def analyze_prefix_efficiency(self, messages: list[dict]) -> str:
        """
        分析当前消息结构的缓存效率。
        返回优化建议。
        """
        if len(messages) < 2:
            return "消息太少，单轮对话缓存收益有限。多轮对话时前缀缓存的优势才会体现。"

        # 找出可以在多轮对话中复用的前缀部分
        system_msgs = [m for m in messages if m["role"] == "system"]
        prefix_tokens = sum(estimate_tokens(m.get("content", "")) for m in system_msgs)

        tips = []
        if system_msgs:
            tips.append(f"✓ System Prompt ({prefix_tokens} tokens) 在多轮对话中作为前缀会被缓存")
        else:
            tips.append("⚠ 没有 System Prompt，建议使用智能体或固定提示词来增加可缓存前缀")

        # 检查工具定义
        has_tools = any("function" in str(m) for m in messages)
        if has_tools:
            tips.append("✓ 工具定义作为前缀的一部分会被缓存")

        tips.append("⚠ 对话历史（user/assistant 消息）每轮都在变，无法缓存")
        tips.append("💡 建议：固定的 System Prompt + 工具定义放在最前面，不要在前面插入变动内容")

        return "\n".join(tips)

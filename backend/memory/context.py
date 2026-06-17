"""
ContextBuilder — 上下文工程层。
将记忆 + 对话历史 + 任务 按时间分窗策略组装进 LLM 上下文窗口，
不超出 Token 预算。
"""
import time

from .base import MemoryConfig, MemoryItem
from ..llm_client import estimate_tokens


class ContextBuilder:
    """记忆与 LLM 之间的最后一公里。"""

    def __init__(self, config: MemoryConfig = None, llm_client=None):
        self.config = config or MemoryConfig()
        self._llm = llm_client  # 用于温窗口摘要压缩

    def build(self, system_prompt: str, memories: list[MemoryItem],
              history: list[dict], task: str,
              hot_window_sec: int = None) -> list[dict]:
        """
        组装最终 messages 列表。

        history: 带 timestamp 的消息列表，格式:
            [{"role": "user", "content": "...", "ts": 1700000000.0}, ...]
        """
        if hot_window_sec is None:
            hot_window_sec = self.config.hot_window_sec

        budget = self.config.history_budget_tokens
        messages = []

        # ① System Prompt
        system_tokens = estimate_tokens(system_prompt)
        messages.append({"role": "system", "content": system_prompt})
        used = system_tokens

        # ② 记忆上下文
        if memories:
            memory_block = self._format_memories(memories)
            mem_tokens = estimate_tokens(memory_block)
            max_mem = int(budget * self.config.memory_context_max_ratio)
            if mem_tokens > max_mem:
                memory_block = self._truncate_to_tokens(memory_block, max_mem)
            messages.append({"role": "system", "content": memory_block})
            used += estimate_tokens(memory_block)

        # ③ 切分对话历史
        now = time.time()
        hot_msgs = []
        warm_msgs = []
        for msg in history:
            ts = msg.get("ts", 0)
            if ts and (now - ts) < hot_window_sec:
                hot_msgs.append(msg)
            else:
                warm_msgs.append(msg)

        # ④ 压缩温窗口
        if warm_msgs:
            compressed_block = self._compress_warm(warm_msgs)
            warm_tokens = estimate_tokens(compressed_block)
            messages.append({"role": "system", "content": compressed_block})
            used += warm_tokens

        # ⑤ Token 总量兜底检查
        hot_tokens = sum(estimate_tokens(m.get("content", "")) for m in hot_msgs)
        if used + hot_tokens + estimate_tokens(task) > budget:
            # 全量二次压缩
            all_to_compress = warm_msgs + hot_msgs[:max(0, len(hot_msgs) - self.config.keep_min_hot_rounds)]
            if all_to_compress:
                mega_summary = self._compress_warm(all_to_compress)
                # Replace warm block
                messages = [m for m in messages if "[会话早期摘要]" not in str(m.get("content", ""))]
                messages.append({"role": "system", "content": mega_summary})
                hot_msgs = hot_msgs[-self.config.keep_min_hot_rounds:]

        # ⑥ 热窗口 + 任务
        for msg in hot_msgs:
            messages.append({"role": msg.get("role", "user"),
                             "content": msg.get("content", "")})
        messages.append({"role": "user", "content": task})

        return messages

    def _format_memories(self, memories: list[MemoryItem]) -> str:
        """格式化记忆为文本块。"""
        lines = ["[Agent 记忆]"]
        for item in memories:
            label = "偏好" if item.memory_type.value == "semantic" else "事件"
            lines.append(f"- [{label}] {item.content[:200]}")
        return "\n".join(lines)

    def _compress_warm(self, warm_msgs: list[dict]) -> str:
        """将温窗口消息压缩为摘要。"""
        if not warm_msgs:
            return ""

        if self._llm:
            try:
                text = "\n".join(
                    f"[{m.get('role', '?')}]: {str(m.get('content', ''))[:300]}"
                    for m in warm_msgs[-20:]
                )
                prompt = (
                    f"将以下对话历史总结为一段简洁文字（中文，200字以内），"
                    f"保留关键信息和结论：\n\n{text[:6000]}"
                )
                summary = self._llm.chat([{"role": "user", "content": prompt}])
                return f"[会话早期摘要]\n{summary}"
            except Exception:
                pass

        # Fallback: 简单截断拼接
        lines = []
        for m in warm_msgs[-5:]:
            content = str(m.get("content", ""))[:100]
            if content.strip():
                lines.append(f"[{m.get('role', '?')}]: {content}")
        return "[会话早期摘要]\n" + "\n".join(lines) if lines else ""

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """截断文本到指定 token 数，保留前 70% + 尾 10%。"""
        current = estimate_tokens(text)
        if current <= max_tokens:
            return text
        head_size = int(len(text) * 0.7)
        tail_size = int(len(text) * 0.1)
        return (
            text[:head_size]
            + "\n...(记忆已截断)...\n"
            + (text[-tail_size:] if tail_size > 0 else "")
        )

"""
ContextBuilder — 上下文工程层。
将记忆 + 对话历史 + 任务 按时间分窗策略组装进 LLM 上下文窗口，
不超出 Token 预算。
"""
import time

from .base import MemoryConfig, MemoryItem
from ..llm_client import estimate_tokens


class ContextBuilder:
    """记忆与 LLM 之间的最后一公里。同时提供 Chat 层的消息压缩能力。

    两种使用模式：
    ── Agent 模式 ──
        cb = ContextBuilder(llm_client=llm)
        messages = cb.build(system_prompt, memories, history, task)

    ── Chat 模式 ──
        cb = ContextBuilder(llm_client=llm)
        compressed, token_info = cb.maybe_compress(messages)
    """

    # ── Chat 压缩常量 ──
    MAX_TOKENS = 100_000          # 触发压缩的阈值（128K 的 ~80%）
    KEEP_RECENT = 8               # 保留最近 N 条消息
    SUMMARY_PROMPT = (
        "请将以下对话历史总结为一段简洁的文字，保留：\n"
        "- 用户的核心问题和需求\n"
        "- 已经达成的结论或决定\n"
        "- 重要的代码/数据/文件信息\n"
        "- 尚未解决的事项\n\n"
        "不要遗漏关键信息。用中文总结，控制在 300 字以内。\n\n"
        "对话历史：\n---\n{conversation}\n---\n"
    )

    def __init__(self, config: MemoryConfig = None, llm_client=None):
        self.config = config or MemoryConfig()
        self._llm = llm_client  # 用于摘要压缩
        self.compression_count = 0

    # ── Chat 模式：消息压缩 ─────────────────────────────────

    def _count_messages_tokens(self, messages: list[dict]) -> int:
        """计算消息列表的估算 token 数（处理多模态内容）"""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += estimate_tokens(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total += estimate_tokens(part.get("text", ""))
                    elif isinstance(part, dict) and part.get("type") == "image_url":
                        total += 1000
            total += 2  # role 字段约 2 token
        return total

    def _truncate_messages(self, messages: list[dict]) -> list[dict]:
        """截断消息内容，每条保留前 2000 字符"""
        result = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 2000:
                msg = dict(msg)
                msg["content"] = content[:2000] + "\n...(内容过长已截断)"
            result.append(msg)
        return result

    def _summarize_old_messages(self, old_messages: list[dict]) -> str:
        """调用 LLM 将旧消息总结为一段文字"""
        lines = []
        for msg in old_messages:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                content = " ".join(text_parts)
            lines.append(f"[{role}]: {content}")

        conversation = "\n".join(lines)
        if len(conversation) > 8000:
            conversation = conversation[:8000] + "\n...(内容过长已截断)"

        prompt = self.SUMMARY_PROMPT.format(conversation=conversation)

        try:
            summary_msgs = [{"role": "user", "content": prompt}]
            summary = self._llm.chat(summary_msgs)
            return summary
        except Exception:
            return f"（对话历史过长已截断，保留了最近 {self.KEEP_RECENT} 条消息）"

    def maybe_compress(self, messages: list[dict]) -> tuple[list[dict], dict]:
        """检查并压缩消息列表（Chat 模式入口）。

        返回: (处理后的消息列表, token_info)
        """
        before_tokens = self._count_messages_tokens(messages)
        token_info = {
            "before": before_tokens,
            "after": before_tokens,
            "max": self.MAX_TOKENS,
            "compressed": False,
        }

        if before_tokens <= self.MAX_TOKENS:
            return messages, token_info

        # 需要压缩：分离 system 消息
        system_msgs = [m for m in messages if m["role"] == "system"]
        other_msgs = [m for m in messages if m["role"] != "system"]

        if len(other_msgs) <= self.KEEP_RECENT:
            compressed = system_msgs + self._truncate_messages(other_msgs)
        else:
            recent = other_msgs[-self.KEEP_RECENT:]
            old = other_msgs[:-self.KEEP_RECENT]

            if self._llm:
                summary = self._summarize_old_messages(old)
                compressed = system_msgs + [
                    {"role": "system", "content": f"[历史摘要]\n{summary}"}
                ] + recent
            else:
                compressed = system_msgs + recent

        after_tokens = self._count_messages_tokens(compressed)
        self.compression_count += 1
        token_info.update(
            after=after_tokens,
            compressed=True,
            compression_count=self.compression_count,
        )

        return compressed, token_info

    # ── Agent 模式：完整 Prompt 组装 ─────────────────────────

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

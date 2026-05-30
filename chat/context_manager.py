"""
Context 窗口管理 — 防止消息过长超出模型限制

三层策略：
  1. Token 估算（简单字符换算法）
  2. 超限警告
  3. 自动摘要压缩（调 LLM 把旧消息压成一段）

DeepSeek 上下文窗口：chat=128K, reasoner=128K
"""
import json

# ============================================================
# 配置
# ============================================================
MAX_TOKENS = 100_000   # 触发压缩的阈值（128K 的 ~80%）
KEEP_RECENT = 8        # 保留最近 N 条消息（约 4 轮对话）
SUMMARY_PROMPT = """请将以下对话历史总结为一段简洁的文字，保留：
- 用户的核心问题和需求
- 已经达成的结论或决定
- 重要的代码/数据/文件信息
- 尚未解决的事项

不要遗漏关键信息。用中文总结，控制在 300 字以内。

对话历史：
---
{conversation}
---
"""


def estimate_tokens(text: str) -> int:
    """
    简单 token 估算。
    中文约 1.5~2 chars/token，英文约 3~4 chars/token。
    取保守值 3 chars/token，宁可多估不少估。
    """
    if not text:
        return 0
    return max(1, len(text) // 3)


def count_messages_tokens(messages: list[dict]) -> int:
    """计算消息列表的估算 token 数"""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            # 多模态消息：[{type: "text", text: "..."}, {type: "image_url", ...}]
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += estimate_tokens(part.get("text", ""))
                elif isinstance(part, dict) and part.get("type") == "image_url":
                    # 图片估算约 1000 token（DeepSeek 官方数据）
                    total += 1000
        # role 字段本身约占 2 token
        total += 2
    return total


def format_tokens(n: int) -> str:
    """格式化 token 数为可读字符串"""
    if n >= 1000:
        return f"{n/1000:.1f}K"
    return str(n)


class ContextManager:
    """
    上下文管理器：检测消息是否超限，超了就压缩旧消息。

    用法：
        manager = ContextManager(llm_client)
        compressed = manager.maybe_compress(messages)
    """

    def __init__(self, llm_client=None):
        """
        llm_client: LLMClient 实例（用于生成摘要）。
                    传 None 则只做简单截断，不做摘要。
        """
        self.client = llm_client
        self.compression_count = 0  # 统计压缩次数

    def maybe_compress(self, messages: list[dict]) -> tuple[list[dict], dict]:
        """
        检查并压缩消息列表。
        返回: (处理后的消息列表, token_info)
        """
        before_tokens = count_messages_tokens(messages)
        token_info = {
            "before": before_tokens,
            "after": before_tokens,
            "max": MAX_TOKENS,
            "compressed": False
        }

        if before_tokens <= MAX_TOKENS:
            return messages, token_info

        # 需要压缩
        system_msgs = [m for m in messages if m["role"] == "system"]
        other_msgs = [m for m in messages if m["role"] != "system"]

        if len(other_msgs) <= KEEP_RECENT:
            # 消息不多但 token 超了（可能是超长文件）
            # 简单截断每条消息内容
            compressed = system_msgs + self._truncate_messages(other_msgs)
        else:
            recent = other_msgs[-KEEP_RECENT:]
            old = other_msgs[:-KEEP_RECENT]

            if self.client:
                # 调 LLM 生成摘要
                summary = self._summarize(old)
                compressed = system_msgs + [
                    {"role": "system", "content": f"[历史摘要]\n{summary}"}
                ] + recent
            else:
                # 没有 LLM 客户端，丢弃旧消息
                compressed = system_msgs + recent

        after_tokens = count_messages_tokens(compressed)
        self.compression_count += 1
        token_info["after"] = after_tokens
        token_info["compressed"] = True
        token_info["compression_count"] = self.compression_count

        return compressed, token_info

    def _summarize(self, old_messages: list[dict]) -> str:
        """调用 LLM 将旧消息总结为一段文字"""
        # 把旧消息转成可读文本
        lines = []
        for msg in old_messages:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                content = " ".join(text_parts)
            lines.append(f"[{role}]: {content}")

        conversation = "\n".join(lines)

        # 如果内容太多，先截断再让 LLM 总结
        if len(conversation) > 8000:
            conversation = conversation[:8000] + "\n...(内容过长已截断)"

        prompt = SUMMARY_PROMPT.format(conversation=conversation)

        try:
            summary_msgs = [{"role": "user", "content": prompt}]
            summary = self.client.chat(summary_msgs)
            return summary
        except Exception as e:
            # 摘要失败 → 直接截断
            return f"（对话历史过长已截断，保留了最近 {KEEP_RECENT} 条消息）"

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

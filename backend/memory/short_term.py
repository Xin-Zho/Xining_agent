"""
短期记忆 — 对话上下文窗口管理

三层结构：
  System Prompt（永远保留）
  + 最近 N 条完整消息
  + 旧消息压缩摘要
"""
from ..context_manager import ContextManager, count_messages_tokens, MAX_TOKENS


class ConversationMemory:
    """
    短期记忆管理器：封装 ContextManager，提供统一的消息管理接口。

    用法：
        memory = ConversationMemory(llm_client)
        memory.add_message("user", "你好")
        memory.add_message("assistant", "你好！有什么可以帮你的？")
        messages = memory.get_messages()  # 自动压缩超长内容
        memory.clear()  # 清空
    """

    def __init__(self, llm_client=None, max_tokens: int = None, keep_recent: int = 8):
        self.messages: list[dict] = []
        self.ctx = ContextManager(llm_client)
        if max_tokens:
            self.ctx.MAX_TOKENS = max_tokens
        self.keep_recent = keep_recent
        self.compression_count = 0

    def add_message(self, role: str, content: str):
        """追加一条消息"""
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        """获取消息列表（超长自动压缩）"""
        compressed, info = self.ctx.maybe_compress(self.messages)
        if info.get("compressed"):
            self.messages = compressed
            self.compression_count += 1
        return compressed

    def get_token_count(self) -> int:
        """估算当前 token 数"""
        return count_messages_tokens(self.messages)

    def trim(self, max_messages: int):
        """按条数截断：只保留最近 N 条"""
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def summarize(self) -> str:
        """生成当前对话的摘要（不修改消息列表）"""
        if not self.messages:
            return ""
        return self.ctx._summarize(self.messages)

    def clear(self):
        """清空记忆"""
        self.messages = []

    def to_list(self) -> list[dict]:
        """兼容旧接口：等同于 get_messages()"""
        return self.get_messages()

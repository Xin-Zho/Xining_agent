from openai import OpenAI

# 兼容两种导入方式：
#   - 在 chat/ 内部运行时：from config import ...
#   - 从外部以包形式导入时：from .config import ...
try:
    from config import LLM_API_KEY, LLM_MODEL_ID, LLM_REASONER_ID, LLM_BASE_URL
except ImportError:
    from .config import LLM_API_KEY, LLM_MODEL_ID, LLM_REASONER_ID, LLM_BASE_URL


class LLMClient:
    """LLM 客户端，支持快速模式和深度思考模式"""

    def __init__(self, model_id=None):
        """
        model_id: 不传则用默认 LLM_MODEL_ID (deepseek-chat)
                  传 LLM_REASONER_ID 则用深度思考 (deepseek-reasoner)
        """
        self.model_id = model_id or LLM_MODEL_ID
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    def chat(self, messages):
        """非流式对话：发送全部消息，返回完整回复"""
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages
        )
        return response.choices[0].message.content.strip()

    def chat_stream(self, messages):
        """
        流式对话：逐 token 返回，适合前端打字机效果。
        这是一个生成器函数，用 for token in client.chat_stream(...) 来消费。
        """
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            stream=True
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

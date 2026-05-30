from openai import OpenAI
from config import LLM_API_KEY, LLM_MODEL_ID, LLM_BASE_URL

class LLMClient:
    def __init__(self):
        self.model_id = LLM_MODEL_ID
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

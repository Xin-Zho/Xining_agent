from config import LLM_API_KEY, LLM_MODEL_ID, LLM_BASE_URL
from llm_client import LLMClient


def test_basic_chat():
    """测试基本的对话功能"""
    client = LLMClient()
    messages = [{"role": "user", "content": "你好，请用一句话介绍你自己"}]
    reply = client.chat(messages)

      # 验证：回复不是空的
    assert reply is not None, "错误：回复为 None"
    assert len(reply) > 0, "错误：回复为空字符串"

    print(f"✅ 测试通过！")
    print(f"回复: {reply}")


if __name__ == "__main__":
    test_basic_chat()
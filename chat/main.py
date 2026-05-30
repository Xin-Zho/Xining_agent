from config import LLM_API_KEY, LLM_MODEL_ID, LLM_BASE_URL
from llm_client import LLMClient

def main():
    client = LLMClient()
    messages = []
    print("="*20)
    print("欢迎使用聊天机器人！输入 'exit' 退出。")
    print("="*20)

    while True:
        user_input = input("你: ")
        if user_input.lower() == "exit":
            print("再见！")
            break
        
        messages.append({"role": "user", "content": user_input})
        reply = client.chat(messages)
        print(f"机器人: {reply}")
        messages.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()
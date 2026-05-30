import os
from dotenv import load_dotenv

load_dotenv()  # 自动找 .env 并加载

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "deepseek-chat")
LLM_REASONER_ID = os.getenv("LLM_REASONER_ID", "deepseek-reasoner")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
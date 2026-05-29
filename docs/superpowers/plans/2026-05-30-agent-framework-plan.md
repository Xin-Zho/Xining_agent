# Agent 框架从零搭建 — 实现计划

> **对于执行者：** 用户希望自己动手写代码来学习。本计划提供结构引导和验证标准，不替用户写核心实现。

**目标：** 从环境搭建开始，分 8 步从零构建一个能部署到服务器的 Agent 平台

**架构：** 渐进式——每步在独立目录产出可运行代码，最后整合为统一框架

**技术栈：** Python 3.10+, openai, gradio, fastapi, chromadb (可选), Docker

---

### Task 0: 环境搭建 & 项目初始化

**文件：**
- 创建: `.gitignore`, `.env.example`, `.env`, `requirements.txt`, `README.md`

**目标：** 项目骨架就位，Git 仓库可用，依赖可安装

---

- [ ] **Step 0.1: 确认 Python 版本**

```bash
python --version
```

预期：`Python 3.10.x` 或更高。若版本低于 3.10，先升级再继续。

---

- [ ] **Step 0.2: 初始化 Git 仓库**

```bash
cd /d/agent_learning
git init
```

---

- [ ] **Step 0.3: 创建虚拟环境**

```bash
python -m venv venv
```

激活（Windows bash）：
```bash
source venv/Scripts/activate
```

验证：
```bash
which python
# 应输出: .../agent_learning/venv/Scripts/python
```

---

- [ ] **Step 0.4: 创建 .gitignore**

```gitignore
# Python
venv/
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/

# 敏感配置
.env

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

---

- [ ] **Step 0.5: 创建 .env.example（不含密钥的模板）**

```ini
# DeepSeek API 配置
# 获取 Key: https://platform.deepseek.com/api_keys
LLM_API_KEY="your-api-key-here"
LLM_MODEL_ID="deepseek-chat"
LLM_BASE_URL="https://api.deepseek.com/v1"

# 如果后续换本地模型，改下面就行
# LLM_BASE_URL="http://localhost:11434/v1"     # Ollama
# LLM_BASE_URL="http://localhost:8000/v1"      # vLLM
```

---

- [ ] **Step 0.6: 创建 .env（填入你的真实 Key）**

```ini
LLM_API_KEY="sk-你的真实key"
LLM_MODEL_ID="deepseek-chat"
LLM_BASE_URL="https://api.deepseek.com/v1"
```

---

- [ ] **Step 0.7: 创建 requirements.txt**

```
openai>=1.0.0
python-dotenv>=1.0.0
```

---

- [ ] **Step 0.8: 创建 README.md**

```markdown
# Agent 学习项目

从零搭建一个大模型智能体（Agent）平台。

## 学习路线

- [ ] 第 0 步：环境搭建
- [ ] 第 1 步：纯 LLM 对话
- [ ] 第 2 步：Gradio Web 界面
- [ ] 第 3 步：工具系统
- [ ] 第 4 步：ReAct Agent
- [ ] 第 5 步：记忆系统
- [ ] 第 6 步：高级范式（Plan-Solve + Reflection）
- [ ] 第 7 步：部署上线

## 笔记

（每步完成后在这里记录学到的内容和踩过的坑）
```

---

- [ ] **Step 0.9: 安装依赖**

```bash
pip install -r requirements.txt
```

验证：
```bash
pip list | grep openai
# 应显示: openai  x.x.x
```

---

- [ ] **Step 0.10: 首次提交并推送到 GitHub**

```bash
# 1. 在 github.com 新建仓库（比如叫 agent-learning），不要勾选 README/.gitignore

# 2. 关联并推送
git add .
git commit -m "feat: 初始化项目环境"
git branch -M main
git remote add origin git@github.com:你的用户名/agent-learning.git
git push -u origin main
```

---

- [ ] **Step 0.11: 自检**

用设计文档里的自检清单过一遍：

- [ ] `git status` 显示干净的工作区
- [ ] `.env` 在 `.gitignore` 中，不会被提交到 GitHub
- [ ] `source venv/Scripts/activate` 后 `which python` 指向 `venv/`
- [ ] 别人 clone 你的仓库后看到 `.env.example` 知道怎么配置

---

### Task 1: 纯 LLM 对话

**文件：**
- 创建: `step1_chat/config.py`, `step1_chat/llm_client.py`, `step1_chat/main.py`
- 测试: `step1_chat/test_chat.py`

**目标：** 代码能调用 DeepSeek API 进行对话

---

- [ ] **Step 1.1: 写配置加载模块**

```python
# step1_chat/config.py
import os
from dotenv import load_dotenv

load_dotenv()  # 自动读项目根目录的 .env

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
```

---

- [ ] **Step 1.2: 写 LLM 客户端类（你自己实现核心逻辑）**

接口已约定好了：

```python
# step1_chat/llm_client.py
from openai import OpenAI

class LLMClient:
    def __init__(self, api_key: str, model_id: str, base_url: str):
        """初始化 OpenAI 兼容客户端"""
        pass  # 你来写

    def chat(self, messages: list[dict]) -> str:
        """
        发送消息，返回模型回复的文本内容。
        messages 格式: [{"role": "user", "content": "你好"}]
        """
        pass  # 你来写
```

**提示（如果卡住再看）：**
- 用 `OpenAI(api_key=..., base_url=...)` 构造客户端
- 调 `client.chat.completions.create(model=..., messages=...)`
- 从返回对象中取 `response.choices[0].message.content`

---

- [ ] **Step 1.3: 写命令行对话循环**

```python
# step1_chat/main.py
# 框架给你：
from config import LLM_API_KEY, LLM_MODEL_ID, LLM_BASE_URL
from llm_client import LLMClient

def main():
    # 你来写：初始化客户端，循环读 input()，调 client.chat()，print 回复
    pass

if __name__ == "__main__":
    main()
```

---

- [ ] **Step 1.4: 写测试（手动验证）**

```python
# step1_chat/test_chat.py
# 一个简单的自动测试
from config import LLM_API_KEY, LLM_MODEL_ID, LLM_BASE_URL
from llm_client import LLMClient

def test_basic_chat():
    client = LLMClient(LLM_API_KEY, LLM_MODEL_ID, LLM_BASE_URL)
    messages = [{"role": "user", "content": "你好，请用一句话介绍你自己"}]
    reply = client.chat(messages)
    assert reply is not None
    assert len(reply) > 0
    print(f"✅ 测试通过！回复: {reply}")

if __name__ == "__main__":
    test_basic_chat()
```

运行：
```bash
cd step1_chat
python test_chat.py
# 预期: ✅ 测试通过！回复: 你好！我是DeepSeek...
```

---

- [ ] **Step 1.5: 手动跑一遍**

```bash
python main.py
# 输入几句对话，确认能正常回复
```

---

- [ ] **Step 1.6: 提交**

```bash
git add step1_chat/
git commit -m "feat: 第1步 - LLM客户端 + 命令行对话"
```

---

### Task 2: Gradio Web 界面

**文件：**
- 创建: `step2_web/app.py`

**目标：** 在浏览器里和 LLM 聊天

---

- [ ] **Step 2.1: 安装 Gradio**

```bash
pip install gradio
# 更新 requirements.txt，加上 gradio>=4.0.0
```

---

- [ ] **Step 2.2: 写 Gradio 聊天界面（你来写）**

```python
# step2_web/app.py
import sys
sys.path.append("..")  # 方便导入 step1 的代码
import gradio as gr
# 导入你 step1 的 LLMClient...
```

**你要解决的关键问题：**
- Gradio 的 `ChatInterface` 需要一个函数，输入是 `(message, history)`，输出是 bot 的回复
- `history` 的格式是 `[["用户消息1", "bot回复1"], ["用户消息2", "bot回复2"], ...]`
- 你需要把 history 转成 LLMClient 需要的 messages 格式

---

- [ ] **Step 2.3: 手动验证**

```bash
cd step2_web
python app.py
# 浏览器打开 http://localhost:7860
# 发几条消息看能不能回复
```

---

- [ ] **Step 2.4: 提交**

```bash
git add step2_web/ requirements.txt
git commit -m "feat: 第2步 - Gradio Web聊天界面"
```

---

### Task 3: 工具系统

**文件：**
- 创建: `step3_tools/registry.py`, `step3_tools/builtin_tools.py`, `step3_tools/test_tools.py`
- 修改: `step1_chat/llm_client.py` → 复制到 `step3_tools/llm_client.py` 并增强

**目标：** 模型能调用工具（读文件、执行命令、搜索网页）

---

- [ ] **Step 3.1: 实现工具注册表（你来写）**

每个工具的结构：
```python
# 工具定义（给模型看的）
{"name": "...", "description": "...", "parameters": {...}}

# 工具实现（真正执行的）
def tool_func(**kwargs) -> str: ...
```

`ToolRegistry` 需要提供的方法：
- `register(name, description, parameters, func)` — 注册一个工具
- `get_definitions()` — 返回所有工具定义列表（传给 LLM）
- `execute(name, arguments)` — 执行指定工具，返回结果

---

- [ ] **Step 3.2: 实现 3 个内置工具**

```python
# step3_tools/builtin_tools.py
def read_file(path: str) -> str:
    """你来写"""

def execute_command(command: str) -> str:
    """你来写，注意安全"""

def web_search(query: str) -> str:
    """你来写，可用 requests + 搜索引擎API，或先用模拟数据"""
```

---

- [ ] **Step 3.3: 增强 LLMClient 支持 tool_calls**

把 step1 的 `llm_client.py` 复制过来，修改 `chat()` 方法：
- 增加 `tools` 参数
- 返回值从 `str` 改为能同时包含 `content` 和 `tool_calls`

---

- [ ] **Step 3.4: 写测试验证**

```python
# step3_tools/test_tools.py
# 注册一个简单工具 → 获取定义 → 执行 → 验证结果
```

---

- [ ] **Step 3.5: 提交**

```bash
git add step3_tools/
git commit -m "feat: 第3步 - 工具注册表 + 3个内置工具"
```

---

### Task 4: ReAct Agent

**文件：**
- 创建: `step4_react/agent.py`, `step4_react/prompt.py`, `step4_react/app.py`

**目标：** 模型能自主决定"思考→行动→观察"循环

---

- [ ] **Step 4.1: 写 System Prompt（你来推敲措辞）**

```python
# step4_react/prompt.py
SYSTEM_PROMPT = """
你是一个能使用工具的智能助手。请遵循以下模式：

Thought: 分析当前情况，决定下一步做什么
Action: 如果需要使用工具，写出工具名和参数
Observation: （工具返回的结果会出现在这里）
...（重复 Thought → Action → Observation）
Final Answer: 当你有了足够信息，给出最终回答

你可以使用的工具有：{tools_description}
"""
```

**不一定用英文格式**，中文的 Thought/行动/观察/最终回答 也可以。关键是让模型理解这个模式。

---

- [ ] **Step 4.2: 实现 ReAct 循环（你来写）**

```python
# step4_react/agent.py
class ReactAgent:
    def __init__(self, llm_client, tool_registry, system_prompt, max_turns=10):
        pass  # 你来写

    def run(self, user_message: str) -> dict:
        """
        运行 ReAct 循环，返回:
        {
            "answer": "最终回答",
            "steps": [  # 思考过程（给前端展示用）
                {"thought": "...", "action": "...", "observation": "..."},
                ...
            ]
        }
        """
        pass  # 你来写
```

**循环逻辑：**
```
while turns < max_turns:
    调 LLM → 看返回
    如果有 tool_calls → 执行工具 → 结果加入 messages → continue
    如果没有 → 这就是最终回答 → break
```

---

- [ ] **Step 4.3: 对接 Gradio 展示思考过程**

把 step2 的 Gradio 界面复制过来，修改为：
- 调用 ReactAgent 而不是直接调 LLMClient
- 在界面上展示 Thought → Action → Observation 过程

**提示：** Gradio 的回复可以包含 markdown，思考过程用折叠块或代码块展示。

---

- [ ] **Step 4.4: 手动测试**

```bash
# 测试一个需要工具的问题，比如：
"帮我看看当前目录下有什么文件"
"帮我查一下今天天气"
```

观察 Agent 的思考过程是否正确。

---

- [ ] **Step 4.5: 提交**

```bash
git add step4_react/
git commit -m "feat: 第4步 - ReAct Agent循环"
```

---

### Task 5: 记忆系统

**文件：**
- 创建: `step5_memory/short_term.py`, `step5_memory/long_term.py`, `step5_memory/app.py`

**目标：** Agent 能记住对话历史和长期信息

---

- [ ] **Step 5.1: 短期记忆 — 对话窗口管理**

自己实现，不用 LangChain：

```python
# step5_memory/short_term.py
class ConversationMemory:
    def add_message(self, role: str, content: str): ...
    def get_messages(self) -> list[dict]: ...
    def trim(self, max_messages: int): ...  # 历史太长了截断
    def summarize(self, llm_client) -> str: ...  # 对旧消息做摘要
```

---

- [ ] **Step 5.2: 长期记忆 — 持久化存储**

```python
# step5_memory/long_term.py
import json
# 或 import sqlite3

class LongTermMemory:
    def save(self, key: str, value: str): ...
    def get(self, key: str) -> str: ...
    def list_keys(self) -> list[str]: ...
    def delete(self, key: str): ...
```

先最简单的 JSON 文件存，后面再考虑 SQLite。

---

- [ ] **Step 5.3: （可选）向量检索 RAG**

如果你做到这步还有精力，装 chromadb 试试：

```bash
pip install chromadb
```

实现语义搜索——把文本转成 embedding，存到向量库，搜索时返回最相似的内容。

**理解就行，不一定要实现**：RAG 的核心是"把东西变成向量 → 存起来 → 搜索 → 喂给 LLM"。

---

- [ ] **Step 5.4: 整合到 Agent 中**

修改 ReactAgent，在 run() 开头从长期记忆检索相关内容，加到 system prompt 里。

---

- [ ] **Step 5.5: 提交**

```bash
git add step5_memory/
git commit -m "feat: 第5步 - 记忆系统（短期+长期）"
```

---

### Task 6: 高级范式

**文件：**
- 创建: `step6_advanced/planner.py`, `step6_advanced/reflector.py`, `step6_advanced/app.py`

**目标：** 实现 Plan-and-Solve 和 Reflection，Agent 能处理复杂任务

---

- [ ] **Step 6.1: Plan-and-Solve 模式（你来写）**

```python
# step6_advanced/planner.py
class PlanAndSolveAgent:
    def run(self, user_message: str) -> dict:
        # 1. 规划阶段：让 LLM 列出步骤
        # 2. 执行阶段：逐步执行（每步可能又是 ReAct）
        # 3. 汇总阶段：整合结果
        pass
```

---

- [ ] **Step 6.2: Reflection 模式（你来写）**

```python
# step6_advanced/reflector.py
class ReflectionAgent:
    def run(self, user_message: str, max_reflections: int = 3) -> dict:
        # 1. 先执行得到初步回答
        # 2. 让 LLM 反思：回答够好吗？有遗漏吗？
        # 3. 如果需要改进，带着反思结果重新执行
        pass
```

---

- [ ] **Step 6.3: 提交**

```bash
git add step6_advanced/
git commit -m "feat: 第6步 - Plan-Solve + Reflection范式"
```

---

### Task 7: 部署上线

**文件：**
- 创建: `step7_deploy/api.py`, `step7_deploy/Dockerfile`, `step7_deploy/docker-compose.yml`

**目标：** Agent 跑在服务器上，对外提供 API 和 Web 界面

---

- [ ] **Step 7.1: FastAPI 包装（你来写）**

```python
# step7_deploy/api.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@app.post("/api/chat")
async def chat(req: ChatRequest):
    # 调用你的 Agent，返回回复
    pass

@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    # 返回该 session 的对话历史
    pass
```

---

- [ ] **Step 7.2: 会话管理**

不同 session_id 的对话不要串。用一个字典 `{session_id: Agent实例}` 管理。注意并发安全。

---

- [ ] **Step 7.3: 写 Dockerfile（你来写）**

```dockerfile
# 基础: python:3.10-slim
# 安装依赖: COPY requirements.txt + pip install
# 暴露端口: 8000
# 启动命令: uvicorn api:app --host 0.0.0.0 --port 8000
```

---

- [ ] **Step 7.4: 本地验证**

```bash
docker build -t agent-app .
docker run -p 8000:8000 --env-file ../.env agent-app
# 浏览器打开 http://localhost:8000/docs 看 FastAPI 自动生成的文档
# curl 测试 /api/chat
```

---

- [ ] **Step 7.5: 提交**

```bash
git add step7_deploy/
git commit -m "feat: 第7步 - FastAPI + Docker部署"
```

---

## 完成后的下一步

1. 把所有 step 的代码整合到 `agent_framework/` 目录
2. 加更多工具：发邮件、数据库查询、API 调用...
3. 换成本地模型：装 Ollama，改 `.env` 里 `BASE_URL` 就行
4. 尝试多 Agent 协作：多个 Agent 分工完成不同子任务

# Agent 框架设计文档：从零搭建智能体平台

## 概述

**目标**：从零搭建一个可本地运行的 Agent 框架，底层接 DeepSeek API（后续可换本地模型），支持 Web 聊天界面、工具调用和多 Agent 范式。

**学习策略**：每个阶段先理解"要解决什么问题"，再自己动手写代码。框架灵魂在循环设计，不在代码量。

**参考教程**：[Hello-Agents](https://datawhalechina.github.io/hello-agents/) 第四章～第七章 + 第八章（记忆）

---

## 项目目录结构

```
agent_learning/
├── .env                    # API 密钥、模型配置
├── requirements.txt        # Python 依赖
├── README.md               # 你的学习笔记
│
├── step0_setup/            # 第0步：环境搭建
├── step1_chat/             # 第1步：纯LLM对话
│   └── ...
├── step2_web/              # 第2步：Gradio Web界面
│   └── ...
├── step3_tools/            # 第3步：工具系统
│   └── ...
├── step4_react/            # 第4步：ReAct Agent
│   └── ...
├── step5_memory/           # 第5步：记忆系统
│   └── ...
├── step6_advanced/         # 第6步：Plan-Solve + Reflection
│   └── ...
├── step7_deploy/           # 第7步：部署上线
│   └── ...
│
└── agent_framework/        # 最终整合后的框架目录
    ├── core/               # Agent 引擎核心
    ├── tools/              # 工具集
    ├── memory/             # 记忆系统
    ├── web/                # Web 界面
    └── main.py             # 启动入口
```

> 每一步是独立目录，方便你实验和回滚。最后一步整合到 `agent_framework/`。

---

## 第 0 步：环境搭建 & 项目初始化

**要解决的问题**：电脑上什么都没有，先让项目能跑起来，代码能管起来。

**你需要做的**：

1. **安装 Python 3.10+**（如果还没有）
2. **创建项目目录** — `agent_learning/`
3. **初始化 Git 仓库** — `git init`
4. **创建 GitHub 仓库** — 在 github.com 新建仓库，关联本地
5. **创建虚拟环境** — `python -m venv venv`
6. **写 `.gitignore`** — 排除 `venv/`、`.env`、`__pycache__/`
7. **创建 `.env.example`** — 不含真实密钥的模板文件，告诉别人需要什么配置
8. **创建 `requirements.txt`** — 先只需 `openai` 和 `python-dotenv`
9. **创建 `README.md`** — 写项目名、目标、你的学习记录
10. **做第一次 commit** — 推到 GitHub

**自检问题**：

- [ ] `git status` 能看到干净的工作区吗？
- [ ] `.env` 被 `.gitignore` 排除了吗？（绝对不能把密钥传到 GitHub）
- [ ] 虚拟环境激活了吗？`which python` 指向 `venv/` 吗？
- [ ] 别人 clone 你的仓库后，能根据 `.env.example` 知道怎么配置吗？

---

## 第 1 步：纯 LLM 对话

**要解决的问题**：让你的 Python 代码能和 DeepSeek 说话。

**核心概念**：

- **OpenAI 兼容接口**：DeepSeek 的 API 和 OpenAI 的调用方式完全一致，用 `openai` 库直接调
- **`.env` 文件**：密钥和配置与代码分离——这是软件工程的基本习惯，不是框架特有的
- **LLM 客户端封装**：把"怎么调用模型"包成一个类。后续换模型（DeepSeek → 本地 vLLM）只改配置不改代码

**你需要实现的**：

1. 创建 `.env`，配置 DeepSeek API Key / Model / Base URL
2. 写一个 `LLMClient` 类，至少包含 `chat(messages)` 方法
3. 写一个命令行循环：读取用户输入 → 调模型 → 打印回复

**关键接口约定**（所有步骤统一使用）：

```python
# LLM 客户端的核心接口
class LLMClient:
    def chat(self, messages: list[dict], tools: list[dict] = None) -> dict:
        """
        messages: [{"role": "user/system/assistant", "content": "..."}, ...]
        tools: 可选，工具定义列表（OpenAI function calling 格式）
        返回: {"role": "assistant", "content": "...", "tool_calls": [...]}
        """
        pass
```

**自检问题**：

- [ ] 我能解释 `BASE_URL` 是什么，为什么改它就能换模型吗？
- [ ] `messages` 列表里的 `role` 字段有哪几种？各代表什么？
- [ ] 如果 DeepSeek API 挂了，我的代码会怎么表现？需要加什么处理？

---

## 第 2 步：Gradio Web 界面

**要解决的问题**：把命令行对话变成浏览器里的聊天界面。

**核心概念**：

- **Gradio** 的 `ChatInterface`：自带聊天框、历史记录、流式输出，几行代码搞定
- **前后端分离思维**：Gradio 是前端，LLMClient 是后端——虽然跑在同一个进程里，但职责分开

**你需要实现的**：

1. 安装 Gradio，用 `gr.ChatInterface` 做一个聊天页面
2. 把第 1 步的 `LLMClient` 作为 Gradio 的"大脑"
3. （可选）加上流式输出

**参考**：Hello-Agents 第 4 章的环境准备部分

**自检问题**：

- [ ] Gradio 的 `ChatInterface` 帮我自动处理了什么？我自己需要处理的又是什么？
- [ ] 聊天历史存在哪里？刷新页面会丢吗？
- [ ] 如果我想把 Gradio 换成自己写的 HTML 页面，需要改哪些地方？

---

## 第 3 步：工具系统

**要解决的问题**：让模型不止能"说"，还能"做"——执行命令、读文件、搜网页。

**核心概念**：

- **工具的本质**：一个函数 + 一段描述（告诉模型"我长这样，你能这么用我"）
- **OpenAI Function Calling 格式**：业界的通用标准，所有兼容接口都支持
- **工具注册表模式**：用一个字典管理所有工具，添加新工具 = 注册一个函数

**工具的统一接口约定**：

```python
# 每个工具都是一个这样的结构
{
    "name": "read_file",
    "description": "读取指定路径的文件内容",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径"}
        },
        "required": ["path"]
    }
}

# 对应的执行函数
def read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()
```

**你需要实现的**：

1. 一个 `ToolRegistry` 类：注册工具、列出工具定义、按名称执行工具
2. 至少实现 3 个工具：`read_file`、`execute_command`、`web_search`
3. 修改 `LLMClient.chat()` 支持传入 `tools` 参数

**设计要点（你自己想）**：

- 工具执行出错了怎么办？返回什么给模型？
- 命令执行工具很危险——怎么加安全限制？
- 工具的返回结果太长怎么办？截断？总结？

**自检问题**：

- [ ] 为什么工具的定义（description/parameters）这么重要？
- [ ] 如果我想加一个新工具（比如发邮件），需要改哪些地方？
- [ ] 工具注册表模式 vs 硬编码 if-else，优劣各是什么？

---

## 第 4 步：ReAct Agent

**要解决的问题**：模型能自己决定"什么时候用哪个工具"，而不是你告诉它。

**核心概念**：

- **ReAct = Reasoning + Acting**：模型思考 → 选择行动（调用工具）→ 观察结果 → 再思考 → ... → 最终回答
- **Agent 循环**：不是一次调用就结束，是一个 while 循环，直到模型觉得"我回答完了"
- **System Prompt 是灵魂**：Prompt 告诉模型"你是一个能使用工具的 Agent，遵循 ReAct 模式思考"

**ReAct 循环流程**：

```
用户提问
  ↓
[循环开始]
  ↓
调 LLM（带上历史消息 + 工具列表）
  ↓
模型返回：要调工具？──→ 是 → 执行工具 → 把结果加入消息列表 → 回到循环
  ↓ 否
模型返回：最终回答？
  ↓
输出给用户
```

**你需要实现的**：

1. 手写 ReAct 的 System Prompt（这是关键，自己推敲措辞）
2. 实现 Agent 循环：while 循环 + 判断是否有 tool_calls
3. 和 Gradio 对接：展示 Agent 的思考过程（Thought → Action → Observation）

**设计要点（你自己想）**：

- 循环怎么防止无限调用？（max_turns？token 预算？）
- 模型调工具时参数填错了怎么办？
- 思考过程要在界面上展示给用户看吗？

**参考**：Hello-Agents 第 4 章

**自检问题**：

- [ ] 我能画出 ReAct 循环的流程图吗？每一步的输入输出是什么？
- [ ] 模型的"思考"（Thought）和"回答"（Answer）在消息列表里分别是什么 role？
- [ ] 如果我去掉 System Prompt 会怎样？
- [ ] 同样的问题，有 ReAct 和没有 ReAct，模型的回答有什么区别？

---

## 第 5 步：记忆系统

**要解决的问题**：模型本身是无状态的，你需要给它"记忆"——对话上下文、长期知识、重要事实。

**核心概念**：

- **短期记忆** = 对话历史（messages 列表），窗口满了就截断
- **长期记忆** = 存到数据库/文件里的重要信息，跨会话保留
- **工作记忆** = 当前任务的关键上下文，从长期记忆中检索出来的
- **向量检索（RAG）** = 把文本转成向量，语义搜索而非关键词搜索

**三层记忆架构**：

```
┌─────────────┐
│  短期记忆     │ ← 当前对话的 messages，窗口管理
│  (上下文窗口) │
├─────────────┤
│  工作记忆     │ ← 从长期记忆中检索出来的相关内容
│  (RAG 检索)  │
├─────────────┤
│  长期记忆     │ ← SQLite / ChromaDB / JSON 文件持久化
│  (持久存储)  │
└─────────────┘
```

**你需要实现的**：

1. 对话历史持久化（存成 JSON 或 SQLite）
2. 上下文窗口管理：历史太长时怎么截断/总结
3. （进阶）向量检索：用 `chromadb` 或简单的 embedding 做语义搜索

**设计要点（你自己想）**：

- 上下文窗口有限（比如 128K token），满了怎么办？丢弃最早的？做摘要？
- 什么信息值得存到长期记忆？谁来判定？
- 向量检索和关键词检索各适合什么场景？

**参考**：Hello-Agents 第 8 章

**自检问题**：

- [ ] 如果用户说"还记得我昨天说的吗"，你的系统怎么做才能真的"记得"？
- [ ] 短期记忆和长期记忆的边界在哪里？
- [ ] 向量检索的原理是什么？为什么不能只用关键词匹配？

---

## 第 6 步：高级范式 — Plan-Solve + Reflection

**要解决的问题**：简单的 ReAct "想一步做一步" 对复杂任务不够用。需要先规划再执行，执行完还要反思。

**核心概念**：

| 范式 | 适合场景 | 核心思想 |
|------|---------|---------|
| **ReAct** | 简单任务 | 想一步→做一步→看结果→再想 |
| **Plan-and-Solve** | 复杂多步任务 | 先列计划 → 逐步执行 → 汇总 |
| **Reflection** | 需要质量保证 | 执行 → 自我评估 → 不满意就修正 |

**Plan-and-Solve 流程**：

```
用户提问
  ↓
规划阶段：列出步骤 1, 2, 3...
  ↓
执行阶段：逐步执行（每步可能又是 ReAct）
  ↓
汇总阶段：把各步结果整合成最终回答
```

**Reflection 流程**：

```
执行完成，得到回答
  ↓
反思：这个回答满足用户需求吗？有遗漏吗？有错误吗？
  ↓
需要改进？→ 是 → 带着反思结果重新执行
  ↓ 否
输出
```

**你需要实现的**：

1. 在已有 ReAct 基础上加 Plan-and-Solve 模式
2. 实现简单的 Reflection 循环
3. （可选）让 Agent 自己决定用哪种范式

**参考**：Hello-Agents 第 4 章后半部分

**自检问题**：

- [ ] Plan-and-Solve 和 ReAct 的根本区别在哪？（提示：不是步骤多少）
- [ ] Reflection 容易陷入"永远不满意"的死循环，怎么防止？
- [ ] 三种范式能不能组合使用？什么时候该切换？

---

## 第 7 步：部署上线

**要解决的问题**：代码在自己电脑上能跑还不够，要放到服务器上稳定运行，对外提供服务。

**核心概念**：

- **FastAPI**：给 Agent 包一层 REST API，外部可以通过 HTTP 调用
- **无状态 vs 有状态**：API 本身无状态，会话管理要自己处理
- **Docker**：把环境和代码打包，到哪都能跑

**部署架构**：

```
Nginx (可选, 反代)
  ↓
FastAPI (:8000)
  ├── /api/chat        POST - 发送消息
  ├── /api/chat/stream POST - 流式对话
  ├── /api/history     GET  - 获取历史
  └── /                GET  - 聊天 Web 页面
  ↓
Agent 引擎 (ReAct / Plan-Solve / Reflection)
  ↓
LLM (DeepSeek API 或 本地 vLLM)
```

**你需要实现的**：

1. 用 FastAPI 包装 Agent，提供 `/api/chat` 接口
2. 会话管理：不同用户/session_id 的对话不互相干扰
3. 写 Dockerfile，在服务器上用 Docker 跑起来

**自检问题**：

- [ ] 多个用户同时用，Agent 的状态会串吗？怎么解决？
- [ ] 如果 Agent 处理一个复杂任务要 30 秒，HTTP 请求会超时吗？怎么解决？
- [ ] `.env` 文件要跟着进 Docker 镜像吗？不进的话怎么传配置？

---

## 最终框架结构（第 7 步完成后）

```
agent_framework/
├── core/
│   ├── llm_client.py          # LLM 客户端（OpenAI 兼容接口）
│   ├── agent_loop.py          # Agent 循环（ReAct）
│   ├── planner.py             # Plan-and-Solve 模式
│   └── reflector.py           # Reflection 反思模式
│
├── tools/
│   ├── registry.py            # 工具注册表
│   ├── builtin_file.py        # 文件操作工具
│   ├── builtin_shell.py       # Shell 命令工具
│   └── builtin_web.py         # 搜索/抓取工具
│
├── memory/
│   ├── short_term.py          # 短期记忆（对话窗口管理）
│   ├── long_term.py           # 长期记忆（SQLite/JSON 持久化）
│   └── retriever.py           # 向量检索（ChromaDB/简单实现）
│
├── web/
│   ├── app.py                 # Gradio 界面
│   ├── api.py                 # FastAPI 接口
│   └── static/                # （可选）自定义前端页面
│
├── config.py                  # 配置加载（读 .env）
├── main.py                    # 启动入口
└── Dockerfile                 # 容器化部署
```

---

## 技术依赖清单

| 阶段 | 新增依赖 | 用途 |
|------|---------|------|
| 第 1 步 | `openai`, `python-dotenv` | 调 DeepSeek、读配置 |
| 第 2 步 | `gradio` | Web 聊天界面 |
| 第 3 步 | （无新增） | 工具系统纯 Python |
| 第 4 步 | （无新增） | ReAct 循环纯 Python |
| 第 5 步 | `chromadb`（可选）| 向量检索 |
| 第 6 步 | （无新增） | 高级范式纯 Python |
| 第 7 步 | `fastapi`, `uvicorn`, `docker` | API + 容器化 |

---

## 学习建议

1. **每步先理解再动手**：问自己"这步在解决什么问题"，想清楚了再写代码
2. **动手时参考教程，但不要照抄**：Hello-Agents 的代码是参考答案，你的实现可以和它不一样
3. **遇到 bug 先自己调试 15 分钟**：打印中间状态、看 API 返回了什么
4. **每步完成后写笔记**：学到了什么、踩了什么坑、有什么想法——记在 `README.md` 里
5. **框架的每个设计决策都要有自己的理由**：为什么这样做而不是那样做？能说清楚就行

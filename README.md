# Agent Learning — Unified

融合 **agent_learning**（Agent 范式深度）与 **codex_test**（移动端 + Claude Code 集成）的生产级 AI Agent 平台。

## 架构

```
agent_learning/
├── backend/                          # FastAPI 后端（模块化）
│   ├── server.py                     # 统一路由（18 REST + 1 WebSocket）
│   ├── database.py                   # SQLite（6 张表）
│   ├── auth.py                       # JWT + bcrypt
│   ├── models.py                     # Pydantic 模型
│   ├── llm_client.py                 # DeepSeek API 封装
│   ├── context_manager.py            # Token 预算 + 自动压缩
│   ├── claude_config.py              # Claude Code 运行时配置
│   ├── agent/                        # Agent 引擎模块
│   │   ├── engine.py                 # ReAct Agent（增强版）
│   │   ├── plan_solve_engine.py      # Plan-Solve Agent
│   │   ├── reflection_engine.py      # Reflection Agent
│   │   ├── claude_code_engine.py     # Claude Code CLI 引擎
│   │   ├── tools.py                  # 9 工具（通用+代码级）
│   │   └── websocket_manager.py      # WebSocket 实时推送
│   ├── training/                     # 对话蒸馏系统
│   │   ├── dialogue_logger.py        # JSONL 日志记录
│   │   └── rule_extractor.py         # Prompt 优化规则提取
│   └── memory/                       # 记忆系统
│       ├── short_term.py             # 短期对话记忆
│       └── long_term.py              # 长期记忆 (SQLite)
├── mobile/                           # React Native Expo（iOS + Android + Web）
├── web/                              # 纯 HTML5 前端（Apple HIG 设计）
├── agent_framework/                  # [保留] 原始学习代码
├── chat/                             # [保留] 原始学习代码
├── claude_config.json                # Claude Code 配置
├── start-dev.ps1                     # 一键启动
└── requirements.txt
```

## 四引擎对比

| 引擎 | Agent 模式 | 核心能力 | Token 预算 | 自动反思 | 并行执行 |
|------|-----------|---------|-----------|---------|---------|
| **ReAct** | `react` | 思考→行动→观察循环 | ✅ 90K + 智能截断 | ✅ | ✅ asyncio.gather |
| **Plan-Solve** | `plan_solve` | 先规划 → 逐步执行 → 汇总 | ❌ | ✅ | ❌ |
| **Reflection** | `reflection` | 执行 → 自评 → 迭代改进 | ❌ | ✅ (3 轮迭代) | ❌ |
| **Claude Code** | `claude-code` | CLI 子进程 stream-json | ✅ (CLI 自身) | ❌ | ✅ (CLI 自身) |

## 9 个工具

| 工具 | 类型 | 来源 |
|------|------|------|
| `web_search` | DuckDuckGo 搜索 | codex_test |
| `web_fetch` | httpx 异步抓取 | codex_test |
| `calculator` | 安全数学计算 | codex_test |
| `timer_set` | 异步延时 | codex_test |
| `read_file` | 文件/目录读取（路径安全） | agent_learning |
| `execute_command` | 命令白名单 + 超时 | agent_learning |
| `grep_files` | 正则内容搜索（ripgrep 风格） | agent_learning |
| `edit_file` | 精确字符串替换 | agent_learning |
| `glob_files` | glob 文件名匹配 | agent_learning |

## 快速开始

### 一键启动
```powershell
.\start-dev.ps1
```

### 后端
```bash
pip install -r requirements.txt
set DEEPSEEK_API_KEY=sk-xxx
set AGENT_RUNTIME=claude-code    # 或 deepseek
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

### 前端
```bash
# Web H5: 浏览器打开 http://127.0.0.1:8000/app
# React Native (Web):
cd mobile && npm install && npx expo start --web --port 8090
# 手机端:
cd mobile && npx expo start
```

## API 概览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/register | 注册 |
| POST | /api/login | 登录 |
| POST | /api/chat | 非流式对话 |
| POST | /api/chat/stream | 流式对话 (SSE) |
| POST | /api/upload | 文件上传 (含 PDF 解析) |
| GET/POST | /api/conversations | 对话 CRUD |
| POST/GET/DELETE | /api/agent/tasks | Agent 任务管理 |
| GET | /api/agent/modes | 可用模式列表 |
| GET/PUT | /api/claude-config | Claude Code 配置 |
| GET | /api/logs/stats | 日志统计 |
| GET | /api/logs/suggestions | Prompt 优化建议 |
| GET/POST/DELETE | /api/memory | 长期记忆 |
| WS | /ws/agent/{task_id} | Agent 实时进度 |

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | — |
| `AGENT_RUNTIME` | 引擎选择 | `claude-code` |
| `CLAUDE_CODE_BIN` | Claude CLI 路径 | 自动检测 |
| `CLAUDE_CODE_MODEL` | Claude Code 模型 | 使用默认 |
| `CLAUDE_CODE_PERMISSION_MODE` | 权限模式 | `bypassPermissions` |
| `CLAUDE_CODE_ADD_DIRS` | 额外工作目录 | — |
| `JWT_SECRET_KEY` | JWT 签名密钥 | 开发默认值 |
| `EXPO_PUBLIC_API_BASE_URL` | API 地址 | `http://127.0.0.1:8000` |

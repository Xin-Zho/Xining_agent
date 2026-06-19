# Agent Learning

一个生产级 AI Agent 平台，ReAct 引擎 + 18 个工具 + MCP 协议隔离 + ChromaDB RAG 知识库。

## 架构

```
agent_learning/
├── backend/
│   ├── server.py                         # FastAPI 入口 + lifespan
│   ├── dependencies.py                   # 全局单例（DeepSeek、ToolRegistry、MCPManager）
│   ├── database.py                       # SQLite ×3（chat/agent/memory）+ WAL mode
│   ├── auth.py                           # JWT + bcrypt
│   ├── models.py                         # Pydantic 请求模型
│   ├── llm_client.py                     # DeepSeek API 封装
│   │
│   ├── agent/                            # Agent 引擎
│   │   ├── engine.py                     # ReAct 引擎（主引擎，增强版）
│   │   ├── plan_solve_engine.py          # Plan-Solve 引擎（辅助，前端不可选）
│   │   ├── tools.py                      # 本地工具 + 共享安全函数（5 个本地工具）
│   │   ├── websocket_manager.py          # WebSocket 实时推送 + 步骤 DB 写入
│   │   ├── intervention.py              # 用户干预（暂停/反馈/重试）
│   │   └── review_prompt.py             # Plan-Solve 方案审校 prompt
│   │
│   ├── protocols/mcp/                    # MCP 协议层
│   │   ├── registry.py                  # ToolRegistry（全局工具注册表，启动后冻结）
│   │   ├── client_manager.py            # MCPClientManager（stdio 子进程管理，per-session 锁）
│   │   ├── tool_adapter.py              # ToolProtocol + MCPTool（适配 MCP 工具为统一接口）
│   │   └── servers/                     # 5 个独立 MCP Server 子进程
│   │       ├── filesystem_read_server.py  # read_file, read_pdf, grep_files, glob_files
│   │       ├── filesystem_write_server.py # edit_file
│   │       ├── shell_server.py          # execute_command
│   │       ├── document_server.py       # create_excel, create_docx, create_document
│   │       └── memory_server.py         # memory_search, list_downloads, rag_ingest, rag_search
│   │
│   ├── memory/                           # 三层记忆系统
│   │   ├── manager.py                   # MemoryManager（统一入口）
│   │   ├── working.py                   # WorkingMemory（会话级，in-memory）
│   │   ├── episodic.py                  # EpisodicMemory（SQLite + ChromaDB）
│   │   ├── semantic.py                  # SemanticMemory（SQLite + ChromaDB，长期知识）
│   │   ├── embedding.py                 # LocalEmbedding（BGE-small-zh-v1.5，512-dim）
│   │   ├── context.py                   # ContextBuilder（Token 预算 + 时间分窗组装）
│   │   └── tools/                       # MemoryTool + RAGTool 预留
│   │
│   ├── evaluation/                       # 评估系统
│   │   ├── hooks.py                     # 任务完成钩子（定量指标 + LLM Judge 采样）
│   │   ├── judge.py                     # LLMJudge（4 维度评分）
│   │   ├── optimizer.py                 # 优化建议生成器
│   │   ├── metrics.py                   # 定量指标计算
│   │   └── router.py                    # 评估数据 API
│   │
│   ├── training/                         # 对话蒸馏
│   │   ├── dialogue_logger.py           # JSONL 日志记录
│   │   └── rule_extractor.py           # Prompt 优化规则提取
│   │
│   └── routes/                           # HTTP 路由（按领域拆分）
│       ├── auth.py                       # 注册、登录
│       ├── chat.py                       # 对话、文件上传
│       ├── conversations.py             # 对话管理
│       ├── agent.py                     # Agent 任务 CRUD + 确认 + SSE 流式
│       ├── websocket.py                 # WebSocket /ws/agent/{task_id}
│       ├── memory.py                    # 记忆 CRUD
│       ├── downloads.py                 # 文件下载 + 静态资源
│       └── logs.py                      # 日志统计
│
├── web/static/                           # H5 前端（Apple HIG 设计）
├── mobile/                               # React Native Expo
├── docs/                                 # 技术文档
│   ├── tech-roadmap.md                  # 技术路线图（Phase 1/2/3）
│   └── superpowers/specs/               # 设计规格文档
├── data/chroma/                          # ChromaDB 持久化目录
├── .env                                  # 环境变量配置
└── requirements.txt
```

## 工作逻辑

### Agent 引擎 (ReAct)

```
用户输入
  │
  ├── 复杂度判断：短问题/简单问候 → 简单路径（1 轮 LLM + 可选工具）
  │                 复杂任务 → ReAct 循环
  │
  └── ReAct 循环（最多 3 轮）：
       ├── 第 1 轮：LLM 思考 → 并行调用工具（鼓励一次调齐所有需要的工具）
       ├── 第 2 轮：LLM 看结果 → 信息不够补调工具，够了就综合回答
       ├── 第 3 轮：不给工具，LLM 基于已有信息强制总结
       └── 反思：润色文字（不搜新数据），结果投评估系统
```

### 工具执行路径

```
Agent 决定调工具
  │
  ├── 本地工具（web_search, web_fetch, stock_query, calculator, timer_set）
  │     → 进程内 async 调用，毫秒级返回
  │
  ├── MCP 工具（read_file, execute_command, create_excel, ...共 13 个）
  │     → MCPClientManager → stdio → MCP Server 子进程 → 执行 → 返回
  │
  └── 高风险工具（edit_file, execute_command）
        → 确认拦截（主进程弹确认框）→ 用户确认后才进 MCP Server
```

### MCP 协议

高风险工具（文件读写、命令执行、文档生成、数据库操作）从 Agent 进程拆出为独立 MCP Server 子进程，通过 stdio JSON-RPC 通信。

- **权限隔离**：读写分离（filesystem-read / filesystem-write 两个 Server）
- **进程隔离**：工具崩溃不影响 Agent 主进程
- **自动发现**：引擎启动时通过 `tools/list` 自动注册 MCP 工具
- **降级策略**：MCP 工具不可用时自动跳过，不影响剩余工具

### RAG 知识库

| 组件 | 方案 |
|------|------|
| 向量库 | ChromaDB（PersistentClient，`data/chroma/`） |
| 嵌入模型 | BAAI/bge-small-zh-v1.5（512-dim，L2 归一化） |
| 加载方式 | sentence-transformers，惰性加载，首次自动下载 ~100MB |
| Collection | `rag_documents`（与记忆的 `semantic_all`/`episodic_all` 隔离） |
| 分块策略 | 段落优先，~400 字/块，80 字重叠 |
| 入库方式 | 文件上传自动入库 + Agent web_fetch 后 LLM 判断入库 |
| 检索方式 | `rag_search(query)` → embed → ChromaDB.query() → top-k + 相关度 + 来源 |

共享同一 ChromaDB 实例和嵌入模型，三个 collection 互不干扰：

```
ChromaDB (data/chroma/)
├── semantic_all       ← 用户长期偏好/知识
├── episodic_all       ← 任务事件记录
└── rag_documents      ← 外部文档知识库
```

## 工具清单（18 个）

| 工具 | 类型 | 位置 | 说明 |
|------|------|------|------|
| web_search | 本地 | tools.py | 百度→cn.bing→搜狗→DuckDuckGo 多引擎搜索 |
| web_fetch | 本地 | tools.py | httpx 网页抓取 + SSRF 防护 |
| stock_query | 本地 | tools.py | A 股实时行情（新浪 API） |
| calculator | 本地 | tools.py | 安全数学计算 |
| timer_set | 本地 | tools.py | 异步延时 |
| read_file | MCP | filesystem-read | 文件/目录读取（沙盒限定 PROJECT_ROOT） |
| read_pdf | MCP | filesystem-read | PDF 文本提取 |
| grep_files | MCP | filesystem-read | 正则内容搜索 |
| glob_files | MCP | filesystem-read | Glob 文件名匹配 |
| edit_file | MCP | filesystem-write | 精确字符串替换（需确认） |
| execute_command | MCP | shell | 白名单命令执行（需确认） |
| create_excel | MCP | document | 生成 .xlsx（openpyxl） |
| create_docx | MCP | document | Markdown → .docx（python-docx） |
| create_document | MCP | document | 生成 md/csv/html/py 文件 |
| memory_search | MCP | memory | 跨会话记忆 CRUD |
| list_downloads | MCP | memory | 用户文件列表 |
| rag_ingest | MCP | memory | 文档入库（分块+嵌入+ChromaDB） |
| rag_search | MCP | memory | 知识库语义检索 |

## 快速开始

### 环境变量 (.env)

```bash
DEEPSEEK_API_KEY="sk-your-key"
LLM_MODEL_ID="deepseek-v4-pro"
LLM_BASE_URL="https://api.deepseek.com"
LLM_TIMEOUT=300
INVITE_CODE="xin-agent-2026"
ALLOW_REGISTRATION=true
```

### 后端

```bash
pip install -r requirements.txt
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

### 前端

浏览器打开 `http://127.0.0.1:8000/app`，登录后切换到 Agent 模式即可使用。

## API 概览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/register | 注册 |
| POST | /api/login | 登录 |
| POST | /api/chat | 非流式对话 |
| POST | /api/chat/stream | 流式对话 (SSE) |
| POST | /api/upload | 文件上传（自动 RAG 入库） |
| GET/POST | /api/conversations | 对话 CRUD |
| POST/GET/DELETE | /api/agent/tasks | Agent 任务管理 |
| POST | /api/agent/confirm/{id} | 工具执行确认 |
| GET | /api/agent/modes | 可用模式列表 |
| GET/POST/DELETE | /api/memory | 记忆管理 |
| GET | /api/download/{filename} | 文件下载 |
| GET | /api/logs/stats | 日志统计 |
| WS | /ws/agent/{task_id} | Agent 实时进度 |

---

## 待改进问题

### 1. 搜索工具质量不稳定

百度搜索和网页抓取对天气、实时数据等结构化查询返回效果差。`web_fetch` 无法处理 JS 渲染页面，政府/天气类网站多为动态页面。搜索结果依赖 HTML 正则解析，源站改版即失效。

*方向*：引入 Bing API 作为主搜索源，或增加结构化数据 API（天气、股价等直接调 JSON API 而非抓网页）。

### 2. LLM 过度搜索倾向

模型倾向于"再搜一轮确认"，而非基于已有数据总结。已通过 MAX_ITERATIONS=3、去除工具降级指令、反思不给工具三轮收紧，但第 3 轮仍偶有冗余搜索。根本原因是 LLM 天然不擅长"够了就停"的自我判断。

*方向*：引入相邻轮次结果相似度检测，相似度 > 70% 时跳过工具直接总结。

### 3. 缓存命中率可进一步提升

当前 system prompt + 工具参考的前缀 (~1200 tokens) 可缓存，但每轮追加的工具结果各不相同，导致后续前缀断裂。统一流式已保证同 system prompt 的调用共享缓存，但跨不同 prompt 的调用（反思、复杂度判断）仍有隔离。

*方向*：固定工具结果截断长度为 500 字符（替代当前比例截断），减少每轮新增 token 的波动，提高跨轮前缀稳定性。

### 4. Plan-Solve 引擎前端不可选

`plan_solve_engine.py` (400 行) 实现了规划→审校→执行→汇总的工作流，包含副 Agent 方案审校机制。但前端 `modeSelect` 仅暴露 ReAct 和对话模式，Plan-Solve 无法被用户触发，实际为死代码。

*方向*：后续 A2A 多 Agent 架构中将其改造为"员工 Agent"，或直接删除以减少维护负担。

### 5. 对话删除功能缺失

前端和后端均未提供对话删除接口（conversations 路由仅有 GET/POST/GET:id，无 DELETE）。

*方向*：添加 `DELETE /api/conversations/{id}` 端点，前端侧边栏加删除按钮。

### 6. 评估系统单向，无实时反馈

LLM Judge 每 5 个任务采样一次，4 维度评分写入 DB，但结果不反馈给 Agent。Optimizer 每 20 个任务生成建议，同样写 DB 不反馈。反思结果已投评估系统，但聚合分析链路未闭环。

*方向*：将评估聚合数据注入 system prompt 的记忆上下文，形成"上次同类任务评分低 → 本次调整策略"的闭环。

### 7. Python 进程数过多，内存浪费

6 个 MCP Server 子进程 + uvicorn 主进程 + 子进程回收残留 = 12 个 Python 进程，~1.3GB 内存。BGE 模型在主进程和 memory_server 各加载一份 (~400MB ×2)。

*方向*：主进程记忆检索也走 MCP memory_server，省掉主进程的 BGE 加载。或合并低频 MCP Server（filesystem-read + filesystem-write → 单个 filesystem Server）。

### 8. web_search 搜索结果结构不稳定

百度、cn.bing、搜狗的结果页 HTML 结构依赖硬编码正则匹配，搜索引擎改版即失效。当前百度解析已有间歇性返回空的问题。

*方向*：优先使用 Bing API（免费层 1000 次/月），搜索结果以结构化 JSON 返回，不再依赖 HTML 解析。

### 9. 缺少用户反馈闭环

用户无法对 Agent 回答进行点赞/点踩/纠错。对话日志只记录问答对，没有质量标签。评估系统只能靠 LLM Judge 自评，缺少真实用户信号。

*方向*：前端回答下方加 👍👎 按钮，存入对话日志的 feedback 字段，作为评估系统的训练信号。

### 10. RAG 知识库与记忆系统界限模糊

`rag_documents` (外部文档) 和 `semantic_all` (用户偏好) 都是 ChromaDB collection，都通过 embedding 检索，但 Agent 调用时需手动区分。用户常混淆"搜索我的记忆"和"搜索我的文档"。

*方向*：统一入口 `knowledge_search(query)`，同时检索记忆和文档，按相关度混合排序，标注来源类型。

# Calculate Agent

> **Branch:** `calculate_agent` | **Status:** 开发中

面向本科生及研究生的 **化学物理计算 Agent**。基于通用 Agent 底座改造——本地 Ollama 大模型 + sympy 符号计算 + 领域 MCP Server + 科学知识库 + 验证链。

---

## 现有架构（改造起点）

当前 `calculate_agent` 分支继承自主分支的通用 Agent 底座（8.1/10）：

```
ReAct 引擎 + Plan-Solve 双引擎
18 个通用工具（搜索、文件、文档、股票）
MCP 协议隔离（5 个 Server，stdio JSON-RPC）
ChromaDB RAG（BGE-small-zh-v1.5，512-dim）
三层记忆系统（工作/情景/语义）
评估系统（LLM Judge 4 维度评分）
FastAPI + WebSocket 实时推送
DeepSeek API（OpenAI SDK）
```

**通用 Agent 的局限**：计算器只是 `eval()` 的安全封装，RAG 面向中文文档，system prompt 面向搜索/文件操作，没有领域知识。

---

## 预期架构（改造目标）

```
                        ┌── Browser (H5 + KaTeX LaTeX) ──┐
                        └──────────────┬──────────────────┘
                                       │ HTTP/SSE/WS
                        ┌──────────────▼──────────────────┐
                        │   FastAPI (server.py)            │
                        │   ┌─────────────────────────┐    │
                        │   │  ReAct Engine            │    │
                        │   │  + verify chain 验证链    │    │
                        │   └──────────┬──────────────┘    │
                        │              │                   │
                        │   ┌──────────▼──────────────┐    │
                        │   │  ToolRegistry (25+ tools)│    │
                        │   │                          │    │
                        │   │  本地: calculator(sympy)  │    │
                        │   │  本地: timer_set          │    │
                        │   │  本地: web_search/fetch   │    │
                        │   │                          │    │
                        │   │  MCP: chemistry_server    │    │
                        │   │  MCP: physics_server      │    │
                        │   │  MCP: verify_server       │    │
                        │   │  MCP: filesystem ×2       │    │
                        │   │  MCP: shell/document/mem  │    │
                        │   └──────────────────────────┘    │
                        └──────────────┬──────────────────┘
                                       │
                ┌──────────────────────┼──────────────────────┐
                │                      │                      │
        ┌───────▼──────┐    ┌─────────▼─────────┐    ┌───────▼──────┐
        │  Ollama       │    │  ChromaDB          │    │  MCP Servers │
        │  qwen3:14b    │    │  science_kb (新)    │    │  (stdio)     │
        │  localhost    │    │  rag_documents      │    │  chemistry   │
        │  :11434       │    │  semantic_all       │    │  physics     │
        └──────────────┘    │  episodic_all       │    │  verify      │
                            └─────────────────────┘    └──────────────┘
```

### 改动清单

| 文件 | 改动 | 说明 |
|------|------|------|
| `backend/llm_client.py` | **重写** | DeepSeek API → Ollama 本地客户端 |
| `backend/agent/tools.py` | **重写** | calculator: eval() → sympy 符号引擎 |
| `backend/agent/engine.py` | 改动 | 加 verify chain 验证钩子 |
| `backend/memory/embedding.py` | 改动 | 公式感知分块 + 模型可选 |
| `backend/memory/science_kb_ingest.py` | **新建** | 科学知识库摄入管道 |
| `backend/protocols/mcp/servers/chemistry_server.py` | **新建** | 7 个化学工具 |
| `backend/protocols/mcp/servers/physics_server.py` | **新建** | 6 个物理工具 |
| `backend/protocols/mcp/servers/verify_server.py` | **新建** | 5 维验证 |
| `web/static/` | 改动 | KaTeX LaTeX 渲染 |
| `requirements.txt` | 改动 | +sympy, scipy, pint, mendeleev |
| `.env` | 改动 | DeepSeek → Ollama 配置 |

### 新增工具（10 个）

| 工具 | Server | 功能 |
|------|--------|------|
| `balance_equation` | chemistry | 化学方程式配平 |
| `element_lookup` | chemistry | 元素/化合物属性查询 |
| `solution_chem` | chemistry | pH、缓冲溶液 |
| `thermo_calc` | chemistry | 热力学 ΔH/ΔG/ΔS |
| `equilibrium` | chemistry | 化学平衡常数 |
| `kinetics` | chemistry | 反应动力学 |
| `electrochem` | chemistry | 电化学/Nernst 方程 |
| `mechanics` | physics | 运动学、牛顿力学 |
| `electromagnetism` | physics | 库仑力、电磁学 |
| `quantum` | physics | 量子力学（仅解析可解模型） |
| `optics` | physics | 透镜、干涉、衍射 |
| `thermodynamics` | physics | 卡诺循环、理想气体 |
| `error_propagation` | physics | 误差传递 |
| `verify_claim` | verify | 后验证：量纲/数量级/回代 |
| `back_substitute` | verify | 符号解回代验证 |

---

## 架构（通用 Agent 底座）

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

---

## 工作计划（9 Tasks，10 天日历）

> 详细设计：[Scientific Computing Agent — Design Spec](docs/superpowers/specs/2026-06-22-scientific-computing-design.md)
> 实施计划：[Implementation Plan](docs/superpowers/plans/2026-06-22-scientific-computing-plan.md)

### Phase 1：基础设施（3 天）

| Task | 内容 | 工作量 | 测试 gate |
|------|------|--------|-----------|
| 1. Ollama 客户端 | `llm_client.py` 重写，连接本地 Ollama | 1-2d | 流式/非流式正常，超时友好 |
| 2. sympy 计算器 | `tools.py` calculator → sympy 符号引擎 | 2-3d | diff/integrate/solve 正确，代码注入拦截 |
| 2.5 科学 system prompt | `engine.py` prompt 重写：compute → verify → search → output | 0.5d | prompt 不含 stock_query，含 calculator/quantum 引导 |
| 3. LaTeX 前端 | KaTeX CDN，chat.js 自动渲染 `$...$` / `$$...$$` | 0.5d | Chrome/Firefox/Edge 渲染正常 |

```
Phase 1a ──╮
Phase 1b ──┼── 并行
Phase 1c ──╯
```

### Phase 2：领域能力（4 天）

| Task | 内容 | 工作量 | 测试 gate |
|------|------|--------|-----------|
| 4. Chemistry Server | `chemistry_server.py` — 7 工具（配平/热力学/平衡/动力学/电化学） | 2-3d | 配平 Fe+Cl₂→FeCl₃，pH 0.1M HCl=1.0 |
| 5. Physics Server | `physics_server.py` — 6 工具（力学/电磁/量子/光学/误差） | 2-3d | 库仑力=8.99×10⁻³N，He 原子拒绝返回 |
| 6. 科学知识库 | `science_kb_ingest.py` — 公式感知分块 + ChromaDB + 嵌入模型评估 | 3-5d | 500 items 入库，英文召回达标 |

```
Phase 2a ──╮
Phase 2b ──┼── 并行（共享 sympy）
Phase 2c ──╯ 可与 2a/2b 并行
```

### Phase 3：质量保障（3 天）

| Task | 内容 | 工作量 | 测试 gate |
|------|------|--------|-----------|
| 7. RAG 增强 | 自动标注主题标签 + 置信度分级 + 公式索引 | 2d | 标签准确率 ≥85%，peer-reviewed 排在前 |
| 8. Verify Server | `verify_server.py` — 量纲/数量级/回代/交叉验证 | 2d | 单条 ≤5s，总计 ≤15s |
| 9. 验证链 | `engine.py` 加 `_verify_claims()` 钩子 | 1-2d | 提取声明正确，验证报告附加到答案 |

```
Phase 3a ──╮
Phase 3b ──┼── 并行
Phase 3c ──╯（依赖 3a+3b）
```

### 依赖链

```
1a(Ollama) + 1b(sympy) + 1d(prompt) + 1c(LaTeX)  ← 并行
        │
   ┌────┴────┐
  2a(chem)  2b(phys)   2c(KB)        ← 并行
   │         │          │
   └────┬────┘          │
        │               │
        └───────┬───────┘
                │
         3a(RAG增强)  3b(Verify)     ← 并行
                │        │
                └───┬────┘
                    │
                 3c(验证链)
```

### 总计

| | 工作量 | 日历（并行后） |
|--|--------|---------------|
| Phase 1 | 4-6d | 3d |
| Phase 2 | 7-11d | 4d |
| Phase 3 | 5-6d | 3d |
| **合计** | **13-20d** | **10d** |

---

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

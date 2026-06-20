# MCP 集成设计文档

> 日期：2026-06-17
> 状态：设计确认，待实现
> 关联文档：[技术路线图](../tech-roadmap.md)
> 上次 Review：2026-06-17（P0×5 + P1×6 + P2×4，全部修正）+ 二次 Review（P1-NEW×4，全部修正）

---

## 1. 目标

将 agent_learning 项目中的高风险工具从 Agent 进程拆出，封装为独立的 MCP (Model Context Protocol) Server 进程。
实现工具的进程隔离、标准化通信、跨语言复用能力。

## 2. 设计原则

1. **向后兼容** — 现有 REST API（18 端点）、WebSocket、Tool dataclass 全部保留
2. **渐进迁移** — 不一次性重写，通过 ToolRegistry 逐步切换工具调用路径
3. **最小权限** — 每个 MCP Server 只拥有完成职责所需的最小权限集
4. **功能域分组** — 相关工具合并在同一个 Server，避免 14 个独立进程
5. **进程间无共享状态** — 主进程和 MCP Server 不共享内存（ContextVar/全局变量/模块属性），所有上下文通过协议显式传递

## 3. 工具分级

### 3.1 进 MCP Server（高风险，14 个）

| 工具 | 风险来源 | 归属 Server |
|------|---------|-------------|
| `web_search` | 网络请求 | network |
| `web_fetch` | 网络请求 + SSRF | network |
| `stock_query` | 网络请求 | network |
| `read_file` | 文件系统读取 | filesystem-read |
| `read_pdf` | 文件系统读取 | filesystem-read |
| `grep_files` | 文件系统读取 | filesystem-read |
| `glob_files` | 文件系统读取 | filesystem-read |
| `edit_file` | 文件系统写入 | filesystem-write |
| `execute_command` | 系统命令执行 | shell |
| `create_excel` | 文件生成 | document |
| `create_docx` | 文件生成 | document |
| `create_document` | 文件生成 | document |
| `memory_search` | 数据库读写 | memory |
| `list_downloads` | 数据库读取 | memory |

### 3.2 保留进程内（低风险，2 个）

| 工具 | 原因 |
|------|------|
| `calculator` | 纯数学计算，无副作用 |
| `timer_set` | 纯异步延时，无副作用 |

## 4. 核心问题：跨进程上下文传递

MCP Server 是独立子进程，不能访问主进程的内存状态。以下数据必须通过协议或环境变量传递：

| 数据 | 当前访问方式 | 问题 | 解决方案 |
|------|-------------|------|---------|
| `user_id` | `_current_user_id` ContextVar | ContextVar 是进程局部的 | §4.1 — MCPClientManager 在主进程读取 ContextVar，注入 call_tool 参数 |
| `PROJECT_ROOT` | `Path(__file__).resolve().parents[2]` | `__file__` 在子进程中指向不同路径 | §4.2 — 环境变量 `AGENT_PROJECT_ROOT` |
| `downloads_dir` | `os.path.dirname(...)` + `"web/static/downloads"` | 同上，路径层数变了 | §4.2 — 从 `AGENT_PROJECT_ROOT` 推导 |
| `agent.db` 连接 | `get_db("agent")` 共享连接 | 多进程并发写 SQLite | §4.3 — WAL mode |

### 4.1 user_id 传递（通过 MCP `_meta`）

MCP 协议的 `CallToolRequest` 有两个通道：

- `arguments`：必须匹配工具的 `inputSchema`（JSON Schema），SDK 会校验
- `_meta`：元数据通道，不参与 schema 校验，专门用于传递上下文

`_user_id` **不能**放在 `arguments` 中——它不在工具的 `inputSchema` 里，MCP SDK ≥1.9.0 会拒绝未知字段。

```python
# MCPClientManager（主进程侧）
class MCPClientManager:
    async def call_tool(self, server_name: str, tool_name: str,
                        arguments: dict) -> dict:
        # 在主进程读取 ContextVar（这里是有效的）
        user_id = _current_user_id.get()
        session = self._pick_session(server_name, tool_name)
        async with self._session_locks[server_name][session["idx"]]:
            result = await session["session"].call_tool(
                tool_name,
                arguments=arguments,               # 只传 LLM 生成的参数
                _meta={"user_id": user_id},         # 上下文走 _meta
            )
            return result

# MCP Server 侧（子进程）
# handler 通过 _meta 参数接收 user_id
async def handle_memory_search(action: str, key: str = "", value: str = "",
                               limit: int = 10, _meta: dict = None) -> dict:
    user_id = (_meta or {}).get("user_id", 0)
    conn = get_db("agent")
    ...
```

`_user_id` 对 LLM 不可见（不在 `inputSchema` / OpenAPI `parameters` 中）、对 MCP schema 校验不可见、对协议完全合法。

### 4.2 环境变量：PROJECT_ROOT 和路径派生

```python
# MCPServerConfig（主进程侧）
@dataclass
class MCPServerConfig:
    cmd: list[str]
    env: dict  # 传递给子进程的环境变量

# MCPClientManager.__init__
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])  # backend/server.py → 仓库根

self._servers = {
    "network": MCPServerConfig(
        cmd=[sys.executable, "-m", "backend.protocols.mcp.servers.network_server"],
        env={"AGENT_PROJECT_ROOT": PROJECT_ROOT},
    ),
    # ... 其他 Server 同理
}
```

```python
# 每个 MCP Server 入口文件
import os

PROJECT_ROOT = os.environ.get("AGENT_PROJECT_ROOT")
if not PROJECT_ROOT:
    raise RuntimeError("AGENT_PROJECT_ROOT environment variable required")

DOWNLOADS_DIR = os.path.join(PROJECT_ROOT, "web", "static", "downloads")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)
```

所有原来用 `__file__` 推算路径的代码（`_safe_path`、`_read_file`、`_execute_command`、`_grep_files`、`_glob_files`、`create_excel/docx/document`、`_record_download`）统一改用 `PROJECT_ROOT` 环境变量。

### 4.3 SQLite 并发写入：WAL mode

多个 MCP Server + 主进程同时读写 `agent.db`。SQLite 默认 journal mode 是 DELETE，写操作会锁整个数据库。

**解决方案**：在 `database.py` 的 `init_db()` 中启用 WAL mode：

```python
def init_db(db_name: str = "agent"):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")  # 5s 等待而非立即报错
    ...
```

WAL mode 特性：
- 读操作不阻塞写操作
- 写操作不阻塞读操作
- 多个写操作仍然串行化（同一时刻只有一个写入者），但不会直接报 `database is locked`
- 适合本项目的读写混合场景

`database.py` 中 `get_db()` 每次调用都创建新的 sqlite3 连接（当前已经是这样），配合 WAL mode 即可安全跨进程访问。

### 4.4 数据库写操作归主进程（_record_download 问题）

`_record_download` 内部读取 `_current_user_id.get()` 并向 `agent.db` 的 `downloads` 表插入记录。在 MCP Server 子进程中，ContextVar 为默认值 0，记录会错写 user_id。

**解决方案（方案 A）**：DB 写操作归主进程。MCP Server 只做纯工具逻辑，返回结果中携带操作元数据。主进程的 `MCPTool.handler()` 在拿到结果后执行 DB 写入。

```
document_server (子进程)                    MCPTool.handler (主进程)
  │                                           │
  ├── create_excel(filename, data)            │
  ├── 写入文件系统                              │
  ├── return {                                 │
  │     "status": "success",                   │
  │     "file": "stocks.xlsx",                 │
  │     "_download_info": {                     │  ← 元数据，不是工具返回值
  │         "filename": "stocks.xlsx",          │
  │         "size": 12345,                      │
  │         "user_id": 0   ← Server 不设        │
  │     }                                      │
  │   }                                        │
  └───────────────────────────────────────────>│
                                               ├── 拿到 result
                                               ├── if "_download_info" in result:
                                               │     _record_download(user_id, info)
                                               │     ← 主进程 ContextVar 有效 ✅
                                               ├── del result["_download_info"]
                                               └── return result  (干净的，跟原来一样)
```

`_download_info` 是内部门协议字段——MCP Server 返回它，`MCPTool.handler()` 消费它并移除它，LLM 永远看不到。

需要在主进程侧后处理的类似操作：
- `_record_download` → 由 document_server 的 MCPTool handler 处理
- `memory_search(action="save", ...)` → 写到 `agent.db` 的 memory 表
  - 但 memory_search 本来就是 MCP Server 内的工具，需要访问 ChromaDB/SQLite
  - 它们不走 `_record_download`，有自己的 `_user_id` 从 `_meta` 获取——见 §4.1

总结：**DB 读操作**可以在 MCP Server 内完成（带 `_meta.user_id`），**DB 写操作**尽量归主进程后处理。

## 5. 架构

```
backend/
├── protocols/                          # 新增：协议层
│   ├── __init__.py
│   └── mcp/
│       ├── __init__.py
│       ├── client_manager.py           # MCPClientManager — 管理所有 MCP 连接
│       ├── tool_adapter.py             # MCPTool + ToolProtocol — 工具接口协议
│       ├── registry.py                 # ToolRegistry — 统一工具注册表（单例）
│       └── servers/                    # 各 MCP Server 入口（可独立运行）
│           ├── network_server.py       # web_search, web_fetch, stock_query
│           ├── filesystem_read_server.py  # read_file, read_pdf, grep_files, glob_files
│           ├── filesystem_write_server.py # edit_file
│           ├── shell_server.py         # execute_command
│           ├── document_server.py      # create_excel, create_docx, create_document
│           └── memory_server.py        # memory_search, list_downloads
├── agent/
│   ├── tools.py                        # 保留：Tool dataclass + 本地工具 handler
│   ├── engine.py                       # 修改：通过 ToolRegistry 获取工具
│   └── plan_solve_engine.py            # 修改：同上
└── server.py                           # 修改：lifespan 中初始化 ToolRegistry
```

### 5.1 组件职责

#### ToolProtocol (`backend/protocols/mcp/tool_adapter.py`)

```python
from typing import Protocol, Awaitable

class ToolProtocol(Protocol):
    """工具接口协议 — Tool 和 MCPTool 共同满足"""
    name: str
    description: str
    parameters: dict
    require_confirmation: bool

    async def handler(self, **kwargs) -> dict: ...
    def to_openai_schema(self) -> dict: ...
```

`engine.py` 和 `plan_solve_engine.py` 的类型标注从 `list[Tool]` 改为 `list[ToolProtocol]`。
Tool 和 MCPTool 不需要继承关系，只要满足 Protocol 即可。

#### ToolRegistry (`backend/protocols/mcp/registry.py`)

```python
class ToolRegistry:
    """统一工具注册表。
    
    全局单例 — 在 server startup 时通过 lifespan 初始化。
    初始化后工具列表冻结（只读），运行时不变。
    """

    def __init__(self):
        self._local_tools: list[ToolProtocol] = []
        self._mcp_tools: list[MCPTool] = []
        self._frozen = False

    async def initialize(self, mcp_manager: MCPClientManager):
        """启动时初始化：并行连接所有 MCP Server，发现工具"""
        await mcp_manager.connect_all()
        for server_name in mcp_manager.server_names:
            tool_defs = await mcp_manager.list_tools(server_name)
            for td in tool_defs:
                self._mcp_tools.append(MCPTool(
                    name=td["name"],
                    description=td["description"],
                    parameters=td["parameters"],
                    require_confirmation=td.get("require_confirmation", False),
                    server_name=server_name,
                    _client_manager=mcp_manager,
                ))
        self._frozen = True

    def register_local(self, tool: ToolProtocol):
        if self._frozen:
            raise RuntimeError("ToolRegistry is frozen after initialization")
        self._local_tools.append(tool)

    def get_all_tools(self) -> list[ToolProtocol]:
        """返回所有可用工具，自动过滤不可用的 MCP 工具"""
        tools: list[ToolProtocol] = list(self._local_tools)
        for mcp_tool in self._mcp_tools:
            if mcp_tool.available:
                tools.append(mcp_tool)
            else:
                logger.warning(f"MCP tool {mcp_tool.name} unavailable, skipping")
        return tools

    def get_openai_schemas(self) -> list[dict]:
        return [t.to_openai_schema() for t in self.get_all_tools()]

    async def shutdown(self):
        await self._mcp_clients.shutdown()
```

#### MCPClientManager (`backend/protocols/mcp/client_manager.py`)

```python
@dataclass
class MCPServerConfig:
    cmd: list[str]
    env: dict    # 子进程环境变量（AGENT_PROJECT_ROOT 等）

class MCPClientManager:
    """管理 6 个 MCP Server 的 stdio 连接。

    每个 Server 维护 1-N 个 session（支持并发）。
    每个 session 有自己的 Lock——不同 session 可真正并行执行。
    session_count > 1 可实现同 Server 内多个工具并发调用。
    """

    def __init__(self, project_root: str):
        self._server_configs: dict[str, MCPServerConfig] = {
            "network": MCPServerConfig(
                cmd=[sys.executable, "-m", "backend.protocols.mcp.servers.network_server"],
                env={"AGENT_PROJECT_ROOT": project_root},
                session_count=2,  # 网络请求可并发
            ),
            "filesystem-read": MCPServerConfig(
                cmd=[sys.executable, "-m", "backend.protocols.mcp.servers.filesystem_read_server"],
                env={"AGENT_PROJECT_ROOT": project_root},
            ),
            "filesystem-write": MCPServerConfig(
                cmd=[sys.executable, "-m", "backend.protocols.mcp.servers.filesystem_write_server"],
                env={"AGENT_PROJECT_ROOT": project_root},
            ),
            "shell": MCPServerConfig(
                cmd=[sys.executable, "-m", "backend.protocols.mcp.servers.shell_server"],
                env={"AGENT_PROJECT_ROOT": project_root,
                     "AGENT_TOOL_TIMEOUT": "30"},
            ),
            "document": MCPServerConfig(
                cmd=[sys.executable, "-m", "backend.protocols.mcp.servers.document_server"],
                env={"AGENT_PROJECT_ROOT": project_root},
            ),
            "memory": MCPServerConfig(
                cmd=[sys.executable, "-m", "backend.protocols.mcp.servers.memory_server"],
                env={"AGENT_PROJECT_ROOT": project_root},
            ),
        }
        self._sessions: dict[str, list[dict]] = {}      # {server: [{session, idx}]}
        self._session_locks: dict[str, list[asyncio.Lock]] = {}  # 每个 session 一把锁

    async def connect_all(self):
        """并行启动所有 MCP Server 子进程，建立 stdio 连接"""
        tasks = []
        for name, config in self._server_configs.items():
            tasks.append(self._connect_server(name, config))
        await asyncio.gather(*tasks)

    async def _connect_server(self, name: str, config: MCPServerConfig):
        count = getattr(config, 'session_count', 1)
        sessions = []
        locks = []
        for i in range(count):
            session = await create_stdio_session(config.cmd, config.env)
            sessions.append({"session": session, "idx": i})
            locks.append(asyncio.Lock())
        self._sessions[name] = sessions
        self._session_locks[name] = locks

    def _pick_session(self, server_name: str, tool_name: str) -> dict:
        """按 tool_name hash 轮转选择 session，返回 {session, idx}"""
        sessions = self._sessions[server_name]
        idx = hash(tool_name) % len(sessions)
        return sessions[idx]

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> dict:
        """调用指定 Server 的工具。

        自动注入 _user_id（从主进程 ContextVar 读取，走 _meta 通道）。
        同一 session 内的调用串行（用该 session 的 Lock），不同 session 可并发。
        """
        session_entry = self._pick_session(server_name, tool_name)
        session = session_entry["session"]
        session_idx = session_entry["idx"]

        async with self._session_locks[server_name][session_idx]:
            try:
                result = await asyncio.wait_for(
                    session.call_tool(
                        tool_name,
                        arguments=arguments,
                        _meta={"user_id": _current_user_id.get()},
                    ),
                    timeout=STEP_TIMEOUT,  # 60s 总超时
                )
                return result
            except asyncio.TimeoutError:
                raise MCPToolTimeout(f"{server_name}/{tool_name} timed out")
            except Exception as e:
                self._mark_unavailable(server_name, tool_name)
                raise MCPServerUnavailable(f"{server_name}/{tool_name}: {e}")

    async def list_tools(self, server_name: str) -> list[dict]:
        """获取指定 Server 的工具列表（tools/list）"""
        session = self._sessions[server_name][0]
        return await session.list_tools()

    async def shutdown(self):
        """关闭所有子进程连接"""
        for sessions in self._sessions.values():
            for s in sessions:
                await s.close()
```

#### MCPTool (`backend/protocols/mcp/tool_adapter.py`)

```python
@dataclass
class MCPTool:
    """将 MCP 工具适配为 ToolProtocol 接口。
    
    不继承 Tool — 通过 ToolProtocol 保证接口兼容。
    """
    name: str
    description: str
    parameters: dict
    require_confirmation: bool      # 从 MCP Server tools/list 获取
    server_name: str
    _client_manager: MCPClientManager
    available: bool = True           # per-call 健康检查动态更新

    async def handler(self, **kwargs) -> dict:
        """调用 MCP Server 的工具。
        
        确认检查在引擎侧完成（见 §7.1），这里不再重复。
        结果后处理（如 _record_download）在主进程侧完成。
        """
        result = await self._client_manager.call_tool(
            self.server_name, self.name, kwargs
        )
        # 主进程侧后处理：_download_info 消费
        if "_download_info" in result:
            user_id = _current_user_id.get()
            _record_download(user_id, result.pop("_download_info"))
        return result

    def to_openai_schema(self) -> dict:
        """与 Tool.to_openai_schema() 返回结构完全一致"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
```

### 5.2 数据流

```
引擎._call_llm(messages, tool_schemas)
  │
  ├── tool_schemas = ToolRegistry.get_openai_schemas()
  │     ├── [0] calculator       (Tool.to_openai_schema)
  │     ├── [1] timer_set        (Tool.to_openai_schema)
  │     ├── [2] web_search       (MCPTool.to_openai_schema)
  │     ├── [3] execute_command  (MCPTool.to_openai_schema)
  │     └── ...
  │
  ├── LLM 返回 tool_calls
  │
  └── 引擎执行每个 tool_call:
        │
        ├── tool_map[name] → Tool 或 MCPTool
        │
        ├── 如果是 MCPTool 且 require_confirmation=True:
        │     ├── ws_manager.request_confirmation()  ← 主进程拦截
        │     ├── 用户拒绝 → 跳过
        │     └── 用户确认 → 继续
        │
        ├── 本地 Tool:
        │     await handler(**args)  → 直接返回
        │
        └── MCPTool:
              await handler(**args)
                → MCPClientManager.call_tool(server, name, arguments)  # 注入 _meta.user_id
                  → stdio → MCP Server 子进程
                    → 读取 AGENT_PROJECT_ROOT 环境变量
                    → 执行实际逻辑（访问文件/网络/数据库）
                    → 返回结果（含 _download_info 等元数据）
                → 主进程后处理：消费 _download_info → _record_download()
                → 返回干净结果给引擎
```

## 6. 每个 MCP Server 详情

### 6.1 mcp-server-network

| 属性 | 值 |
|------|-----|
| 传输 | stdio |
| session_count | 2（网络请求可并发） |
| 依赖 | duckduckgo_search, httpx |
| 环境变量 | AGENT_PROJECT_ROOT |

**工具列表：**

| 工具 | 描述 | require_confirmation |
|------|------|---------------------|
| `web_search` | DuckDuckGo + Baidu 智能搜索 | False |
| `web_fetch` | 异步 HTTP 抓取，SSRF 防护 | False |
| `stock_query` | A 股实时数据（Sina Finance API） | False |

**stock_query SSRF 防护**：在 handler 中添加 URL 白名单检查——只允许 `vip.stock.finance.sina.com.cn`。

### 6.2 mcp-server-filesystem-read

| 属性 | 值 |
|------|-----|
| 传输 | stdio |
| 权限 | **只读**，沙盒限定 PROJECT_ROOT |
| 依赖 | PyPDF2 |
| 环境变量 | AGENT_PROJECT_ROOT |

**工具列表：**

| 工具 | 描述 | require_confirmation |
|------|------|---------------------|
| `read_file` | 文件读取 / 目录列表 | False |
| `read_pdf` | PDF 文本提取 | False |
| `grep_files` | 正则内容搜索 | False |
| `glob_files` | Glob 文件名匹配 | False |

### 6.3 mcp-server-filesystem-write

| 属性 | 值 |
|------|-----|
| 传输 | stdio |
| 权限 | **写入**，沙盒限定 PROJECT_ROOT |
| 依赖 | 无（标准库） |
| 环境变量 | AGENT_PROJECT_ROOT |

**工具列表：**

| 工具 | 描述 | require_confirmation |
|------|------|---------------------|
| `edit_file` | 精确字符串替换 | **True** |

### 6.4 mcp-server-shell

| 属性 | 值 |
|------|-----|
| 传输 | stdio |
| 权限 | 命令白名单 + shell=False |
| 超时 | AGENT_TOOL_TIMEOUT 环境变量（默认 30s） |
| 依赖 | 无（subprocess 标准库） |
| 环境变量 | AGENT_PROJECT_ROOT, AGENT_TOOL_TIMEOUT |

**工具列表：**

| 工具 | 描述 | require_confirmation |
|------|------|---------------------|
| `execute_command` | 白名单命令执行 | False |

**超时说明**：
- MCP Server 内部 subprocess timeout = 30s（AGENT_TOOL_TIMEOUT）
- 引擎侧 STEP_TIMEOUT = 60s（留给 MCP 通信 + JSON-RPC 序列化 30s 余量）

### 6.5 mcp-server-document

| 属性 | 值 |
|------|-----|
| 传输 | stdio |
| 依赖 | openpyxl, python-docx |
| 环境变量 | AGENT_PROJECT_ROOT（用于推导 downloads_dir） |

**工具列表：**

| 工具 | 描述 | require_confirmation |
|------|------|---------------------|
| `create_excel` | 生成 .xlsx 文件 | False |
| `create_docx` | Markdown → .docx | False |
| `create_document` | 生成 md/csv/html/py 文件 | False |

**_create_docx._table 状态泄漏修复**：将 `_create_docx._table` 函数属性改为局部变量 `table_data: list = []`，在函数内创建并在 finally 块中清理，确保异常退出时状态不残留。

```python
async def handle_create_docx(filename: str, data_md: str, ..., _meta: dict = None) -> dict:
    table_data: list = []  # 局部变量，替代函数属性
    try:
        ...
    finally:
        table_data.clear()  # 确保清理
```

### 6.6 mcp-server-memory

| 属性 | 值 |
|------|-----|
| 传输 | stdio |
| 依赖 | SQLite（agent.db，WAL mode） |
| 环境变量 | AGENT_PROJECT_ROOT |

**工具列表：**

| 工具 | 描述 | require_confirmation |
|------|------|---------------------|
| `memory_search` | 跨会话记忆存储/搜索/列表 | False |
| `list_downloads` | 用户下载记录查询 | False |

## 7. Server 启动配置 (`server.py`)

使用 FastAPI lifespan context manager（替代已弃用的 `@app.on_event`）：

```python
from contextlib import asynccontextmanager
from .protocols.mcp import ToolRegistry, MCPClientManager

PROJECT_ROOT = str(Path(__file__).resolve().parent)  # backend/ 的父目录 = 仓库根

# ToolRegistry 是全局单例
tool_registry = ToolRegistry()
mcp_manager = MCPClientManager(project_root=PROJECT_ROOT)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — 顺序关键：先注册本地工具，再初始化 MCP（initialize 会 freeze registry）
    tool_registry.register_local(Tool("calculator", ...))
    tool_registry.register_local(Tool("timer_set", ...))
    await mcp_manager.connect_all()
    await tool_registry.initialize(mcp_manager)  # 发现 MCP 工具 → _frozen = True
    yield
    # Shutdown
    await tool_registry.shutdown()

app = FastAPI(lifespan=lifespan)

# 引擎工厂
def _get_engine(agent_mode: str):
    tools = tool_registry.get_all_tools()  # 每次从单例读取（只读，线程安全）
    if agent_mode == "plan_solve":
        return PlanSolveEngine(deepseek, tools, ws_manager)
    else:
        return AgentEngine(deepseek, tools, ws_manager, intervention_handler)
```

## 8. 引擎改动点

### 8.1 engine.py

改动最小——只改类型标注和确认检查：

```python
from .protocols.mcp.tool_adapter import ToolProtocol, MCPTool

class AgentEngine:
    def __init__(self, deepseek, tools: list[ToolProtocol], ws_manager,
                 intervention=None):
        self.tools = tools
        self.tool_map: dict[str, ToolProtocol] = {t.name: t for t in tools}
        ...

    # 工具执行路径增加确认拦截（在 handler 调用前）
    async def _execute_tool(self, tool: ToolProtocol, args: dict):
        # 确认检查在主进程执行，不进入 MCP Server
        if tool.require_confirmation:
            approved = await self.ws.request_confirmation(...)
            if not approved:
                return {"status": "rejected", "message": "User denied confirmation"}
        return await tool.handler(**args)

    # run() 逻辑不变
    # _call_llm() 不变
```

### 8.2 tools.py

保留内容：
- `Tool` dataclass 定义（满足 ToolProtocol）
- calculator 和 timer_set 的 handler 函数
- `_current_user_id` ContextVar（主进程侧使用，MCPClientManager 读取）
- 工具相关常量：`ALLOWED_COMMANDS`、`PROJECT_ROOT`、安全函数等（这些被 MCP Server 引用，保留为共享模块）

移除内容：
- 14 个高风险工具的 handler 函数体 → 移到对应 MCP Server 文件
- 模块级 `TOOLS` 列表 → 由 ToolRegistry 替代

注意：`tools.py` 中的安全函数（`_sanitize_url`、`_is_public_url`、`_safe_path`、`_validate_command`、`_record_download` 等）保留，作为 `backend.agent.tools` 的公开 API，被各 MCP Server 文件 import 使用。

## 9. 向后兼容性

| 接口 | 变动 | 说明 |
|------|------|------|
| `POST /api/chat` | 不变 | 直接调 DeepSeek，不涉及工具 |
| `POST /api/chat/stream` | 不变 | 同上 |
| `POST /api/agent/tasks` | 不变 | 引擎实例化改为从 ToolRegistry 取工具 |
| `WS /ws/agent/{task_id}` | 不变 | WebSocket 协议不变 |
| `Tool.to_openai_schema()` | 不变 | MCPTool 实现相同接口 |
| 确认机制 | 不变 | 确认检查在主进程拦截（§8.1），MCPTool 有 require_confirmation 字段 |
| 反思机制 | 不变 | `_reflect()` 不涉及工具调用 |
| 记忆注入 | 不变 | `MemoryManager` 在引擎启动时独立初始化 |
| `server.py` 启动方式 | 改用 lifespan | FastAPI 推荐做法，功能等价 |

### 9.1 降级与健康检查

```python
# MCPTool.available 动态管理
# 策略：per-call check — call_tool 失败时标记 unavailable
#      get_all_tools() 时对 unavailable 工具尝试 reconnect

class MCPClientManager:
    def _mark_unavailable(self, server_name: str, tool_name: str):
        """call_tool 失败时调用"""
        for t in self._mcp_tools:
            if t.server_name == server_name and t.name == tool_name:
                t.available = False

    def _try_reconnect(self, server_name: str) -> bool:
        """get_all_tools() 时尝试重连不可用的 Server"""
        try:
            config = self._server_configs[server_name]
            sessions = await self._connect_server_sync(server_name, config)
            self._sessions[server_name] = sessions
            for t in self._mcp_tools:
                if t.server_name == server_name:
                    t.available = True
            return True
        except Exception:
            return False
```

不自动回退到进程内调用——MCP 工具不可用就是不可用。

## 10. 错误处理

```python
class MCPToolError(Exception):
    """MCP 工具调用错误基类"""
    pass

class MCPServerUnavailable(MCPToolError):
    """Server 进程崩溃或连接断开"""
    pass

class MCPToolTimeout(MCPToolError):
    """工具执行超时（引擎侧 STEP_TIMEOUT=60s 超时）"""
    pass
```

引擎的工具调用路径已有 try/except 和空结果检测。
MCP 错误映射到 `{"error": "..."}` 格式，LLM 的反思机制自动处理。

## 11. 依赖

新增到 `requirements.txt`：

```
mcp~=1.9.0          # Anthropic 官方 MCP Python SDK（锁定 minor 版本）
```

现有依赖不变。所有 MCP Server 共享同一 venv，减少重复安装。

### 11.1 共享 venv + lazy import

6 个 MCP Server 在同一 venv 中运行。为避免无关依赖被加载：

- 每个 Server 文件顶部做 lazy import：只在 `if __name__ == "__main__"` 或 handler 函数内 import 重量级库
- 例如：`document_server.py` 只在 handler 内 `import openpyxl`，不影响其他 Server

## 12. 测试策略

| 层级 | 内容 | 方式 |
|------|------|------|
| 单元测试 | 每个 MCP Server 的工具 handler（进程内直接调 handler 函数） | pytest，参数化测试 |
| 集成测试 | MCPClientManager + 真实子进程 → call_tool → 验证结果 | pytest + subprocess |
| 引擎测试 | 现有引擎 + ToolRegistry（mock MCPClientManager） | pytest |
| 兼容性测试 | 现有 REST API 行为不变 | pytest + httpx |

## 13. 风险与缓解

| 风险 | 缓解 |
|------|------|
| MCP Server 启动慢 | lifespan 中 `asyncio.gather` 并行启动 6 个进程 |
| 子进程崩溃 | per-call 健康检查 + get_all_tools() 时自动重连 |
| stdio 通信开销 | JSON-RPC over stdout/stdin，额外开销 < 1ms |
| 同一 Server 多次调用 | session_count 控制并发度（network=2，其余=1）；未来可按需扩展 |
| SQLite 多进程写冲突 | WAL mode + busy_timeout=5000ms |
| mcp SDK 版本不稳定 | 锁定 `~=1.9.0`，minor 版本内 API 兼容 |
| 跨平台差异（Windows stdio） | mcp SDK 已处理；Windows 是开发环境，充分测试 |
| ContextVar 误用在子进程 | MCPClientManager 在主进程端读取并注入——子进程永远不 import ContextVar |

## 14. 实现注意事项

1. **`_create_docx._table` 函数属性** → 迁移到 MCP Server 时改为局部变量 + finally 清理
2. **`stock_query` SSRF 防护** → MCP Server 版本添加 URL 白名单检查
3. **`tools.py` 保留安全函数** → `_sanitize_url`、`_is_public_url`、`_safe_path`、`_validate_command` 等作为共享模块被 Server import
4. **`_record_download`** → 保留在 `tools.py`，由 `MCPTool.handler()` 主进程侧后处理调用（见 §4.4），不进入 MCP Server 子进程
5. **确认流程** → 引擎侧拦截（`_execute_tool` 方法），不进入 MCP Server
6. **Server 入口** → 每个 Server 文件顶层读取 `AGENT_PROJECT_ROOT` 环境变量，失败时 raise RuntimeError

---

## 版本记录

| 日期 | 内容 | 作者 |
|------|------|------|
| 2026-06-17 | 初始版本 | Xin-Zho + Claude |
| 2026-06-17 | Review v3：修正 P1-NEW×4 — per-session lock、_meta 传 user_id、_record_download 归主进程、lifespan 顺序修复 | Xin-Zho + Claude |

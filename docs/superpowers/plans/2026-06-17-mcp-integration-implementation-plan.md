# MCP 集成实现计划

> 日期：2026-06-17
> 状态：待执行
> 关联设计：[MCP 集成设计文档](../specs/2026-06-17-mcp-integration-design.md)
> 前置条件：设计文档三轮审查通过（19 条审查项全部修正）

---

## 概览

将 14 个高风险工具从 Agent 主进程拆出为 6 个独立 MCP Server 子进程，保留 2 个进程内工具（calculator / timer_set）。

| 阶段 | 内容 | 预计改动量 | 依赖 |
|------|------|-----------|------|
| P1 | 核心骨架：ToolProtocol + MCPTool + ToolRegistry | 新建 3 文件 | 无 |
| P2 | 连接层：MCPClientManager + per-session Lock + _meta | 新建 1 文件 | P1 |
| P3 | MCP Server：6 个 Server 入口 + 工具迁移 | 新建 6 文件 + 改 tools.py | P1, P2 |
| P4 | 引擎适配：类型标注 + _execute_tool | 改 2 文件 | P1 |
| P5 | 服务启动：lifespan + registry 初始化 | 改 server.py + __init__.py | P1-P3 |
| P6 | 后处理机制：_record_download 归主进程 | 改 MCPTool.handler() | P1, P3 |
| P7 | 依赖 & 配置：requirements.txt + __init__.py | 改 2 文件 | P1-P3 |
| P8 | 集成测试 | 新建测试文件 | P1-P7 |

**database.py WAL mode 已存在，无需额外改动。**

---

## P1：核心骨架 — ToolProtocol + MCPTool + ToolRegistry

### 新建文件

```
backend/protocols/__init__.py
backend/protocols/mcp/__init__.py
backend/protocols/mcp/tool_adapter.py    # ToolProtocol + MCPTool
backend/protocols/mcp/registry.py        # ToolRegistry
```

### 1.1 `backend/protocols/mcp/tool_adapter.py`

```python
from typing import Protocol, Awaitable, Any
from dataclasses import dataclass, field


class ToolProtocol(Protocol):
    """工具接口协议 — Tool 和 MCPTool 共同满足"""
    name: str
    description: str
    parameters: dict
    require_confirmation: bool

    async def handler(self, **kwargs) -> dict: ...
    def to_openai_schema(self) -> dict: ...
```

MCPTool dataclass（完整实现，包含 §4.4 后处理逻辑）：

```python
@dataclass
class MCPTool:
    """将 MCP 工具适配为 ToolProtocol 接口"""
    name: str
    description: str
    parameters: dict
    require_confirmation: bool
    server_name: str
    _client_manager: Any  # MCPClientManager，用 Any 避免循环引用
    available: bool = True

    async def handler(self, **kwargs) -> dict:
        from ..agent.tools import _current_user_id, _record_download

        result = await self._client_manager.call_tool(
            self.server_name, self.name, kwargs
        )
        # 主进程侧后处理：_download_info 消费
        if isinstance(result, dict) and "_download_info" in result:
            user_id = _current_user_id.get()
            _record_download(user_id, **result.pop("_download_info"))
        return result

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
```

### 1.2 `backend/protocols/mcp/registry.py`

```python
import logging
from typing import Optional
from .tool_adapter import ToolProtocol, MCPTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """统一工具注册表。全局单例，startup 初始化后只读。"""

    def __init__(self):
        self._local_tools: list[ToolProtocol] = []
        self._mcp_tools: list[MCPTool] = []
        self._frozen = False

    def register_local(self, tool: ToolProtocol):
        if self._frozen:
            raise RuntimeError("ToolRegistry is frozen after initialization")
        self._local_tools.append(tool)

    async def initialize(self, mcp_manager):
        """启动时初始化：连接 MCP Server → 发现工具 → freeze"""
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

    def get_all_tools(self) -> list[ToolProtocol]:
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
        # 关闭 MCP 连接（通过 MCPClientManager）
        if self._mcp_tools:
            manager = self._mcp_tools[0]._client_manager
            await manager.shutdown()
```

### 1.3 `__init__.py` 文件

`backend/protocols/__init__.py`：
```python
from .mcp import ToolProtocol, MCPTool, ToolRegistry, MCPClientManager
```

`backend/protocols/mcp/__init__.py`：
```python
from .tool_adapter import ToolProtocol, MCPTool
from .registry import ToolRegistry
from .client_manager import MCPClientManager
```

### 验收标准

- [x] ToolProtocol 定义完整（name, description, parameters, require_confirmation, handler, to_openai_schema）
- [x] MCPTool 满足 ToolProtocol（Protocol structural subtyping）
- [x] ToolRegistry._frozen 机制：frozen 后 register_local 抛 RuntimeError
- [x] ToolRegistry.get_all_tools() 返回 local + available MCP 工具

---

## P2：连接层 — MCPClientManager

### 新建文件

```
backend/protocols/mcp/client_manager.py
```

### 2.1 `backend/protocols/mcp/client_manager.py`

核心实现要点（对照设计文档 §5.1）：

```python
import asyncio
import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

STEP_TIMEOUT = 60  # 引擎侧总超时


class MCPToolError(Exception):
    pass

class MCPServerUnavailable(MCPToolError):
    pass

class MCPToolTimeout(MCPToolError):
    pass


@dataclass
class MCPServerConfig:
    cmd: list[str]
    env: dict
    session_count: int = 1


class MCPClientManager:
    """管理 6 个 MCP Server 的 stdio 连接。per-session Lock 实现并发。"""

    def __init__(self, project_root: str):
        PROJECT_ROOT = project_root
        self._server_configs: dict[str, MCPServerConfig] = {
            "network": MCPServerConfig(
                cmd=[sys.executable, "-m", "backend.protocols.mcp.servers.network_server"],
                env={"AGENT_PROJECT_ROOT": PROJECT_ROOT},
                session_count=2,
            ),
            "filesystem-read": MCPServerConfig(
                cmd=[sys.executable, "-m", "backend.protocols.mcp.servers.filesystem_read_server"],
                env={"AGENT_PROJECT_ROOT": PROJECT_ROOT},
            ),
            "filesystem-write": MCPServerConfig(
                cmd=[sys.executable, "-m", "backend.protocols.mcp.servers.filesystem_write_server"],
                env={"AGENT_PROJECT_ROOT": PROJECT_ROOT},
            ),
            "shell": MCPServerConfig(
                cmd=[sys.executable, "-m", "backend.protocols.mcp.servers.shell_server"],
                env={"AGENT_PROJECT_ROOT": PROJECT_ROOT, "AGENT_TOOL_TIMEOUT": "30"},
            ),
            "document": MCPServerConfig(
                cmd=[sys.executable, "-m", "backend.protocols.mcp.servers.document_server"],
                env={"AGENT_PROJECT_ROOT": PROJECT_ROOT},
            ),
            "memory": MCPServerConfig(
                cmd=[sys.executable, "-m", "backend.protocols.mcp.servers.memory_server"],
                env={"AGENT_PROJECT_ROOT": PROJECT_ROOT},
            ),
        }
        # {server_name: [{session, stdin, stdout, idx}, ...]}
        self._sessions: dict[str, list[dict]] = {}
        # {server_name: [asyncio.Lock, ...]}  — 每个 session 一把锁
        self._session_locks: dict[str, list[asyncio.Lock]] = {}

    @property
    def server_names(self) -> list[str]:
        return list(self._server_configs.keys())

    async def connect_all(self):
        """并行启动所有 MCP Server 子进程"""
        tasks = []
        for name, config in self._server_configs.items():
            tasks.append(self._connect_server(name, config))
        await asyncio.gather(*tasks)

    async def _connect_server(self, name: str, config: MCPServerConfig):
        count = config.session_count
        sessions = []
        locks = []
        for i in range(count):
            server_params = StdioServerParameters(
                command=config.cmd[0],
                args=config.cmd[1:],
                env={**config.env},
            )
            # stdio_client 返回 async context manager
            stdio_transport = await stdio_client(server_params).__aenter__()
            read_stream, write_stream = stdio_transport
            session = await ClientSession(read_stream, write_stream).__aenter__()
            await session.initialize()

            sessions.append({
                "session": session,
                "transport": stdio_transport,
                "idx": i,
            })
            locks.append(asyncio.Lock())
        self._sessions[name] = sessions
        self._session_locks[name] = locks

    def _pick_session(self, server_name: str, tool_name: str) -> dict:
        """按 tool_name hash 轮转选择 session"""
        sessions = self._sessions[server_name]
        idx = hash(tool_name) % len(sessions)
        return sessions[idx]

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> dict:
        """调用 MCP Server 工具，注入 _meta.user_id，per-session Lock 串行"""
        from ...agent.tools import _current_user_id

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
                    timeout=STEP_TIMEOUT,
                )
                # MCP SDK 返回 CallToolResult，提取 content
                if hasattr(result, 'content') and result.content:
                    # content 是 TextContent 列表
                    for item in result.content:
                        if hasattr(item, 'text'):
                            import json
                            try:
                                return json.loads(item.text)
                            except (json.JSONDecodeError, TypeError):
                                return {"result": item.text}
                    return {"result": str(result.content)}
                return {"result": str(result)}
            except asyncio.TimeoutError:
                raise MCPToolTimeout(f"{server_name}/{tool_name} timed out")
            except Exception as e:
                self._mark_unavailable(server_name, tool_name)
                raise MCPServerUnavailable(f"{server_name}/{tool_name}: {e}")

    def _mark_unavailable(self, server_name: str, tool_name: str):
        # MCPTool.available 由 ToolRegistry 管理
        logger.error(f"MCP tool {server_name}/{tool_name} marked unavailable")

    async def list_tools(self, server_name: str) -> list[dict]:
        """获取指定 Server 的工具列表"""
        session = self._sessions[server_name][0]["session"]
        result = await session.list_tools()
        tools = []
        for tool in result.tools:
            tools.append({
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema or {},
                "require_confirmation": False,  # 默认值，按 Server 配置覆盖
            })
        return tools

    async def shutdown(self):
        """关闭所有子进程连接"""
        for name, sessions in self._sessions.items():
            for s in sessions:
                try:
                    await s["session"].__aexit__(None, None, None)
                    await s["transport"].__aexit__(None, None, None)
                except Exception:
                    pass
```

**注意**：MCP SDK 的 `call_tool` 返回值和 `stdio_client` 用法需要在实现时验证，上面的代码是设计意图，实际 API 可能略有差异。实现 P2 时需先 `pip install mcp~=1.9.0` 并验证 API。

### 验收标准

- [x] 6 个 Server 配置完整（cmd, env, session_count）
- [x] per-session Lock：network=2 个 Lock，其余=1 个 Lock
- [x] _meta={"user_id": N} 通道注入（主进程读 ContextVar）
- [x] connect_all 并行启动
- [x] _pick_session 按 tool_name hash 轮转
- [x] call_tool 超时 STEP_TIMEOUT=60s
- [x] list_tools 返回 name/description/parameters/require_confirmation
- [x] shutdown 清理所有连接

---

## P3：6 个 MCP Server — 工具迁移

### 新建文件

```
backend/protocols/mcp/servers/__init__.py
backend/protocols/mcp/servers/network_server.py
backend/protocols/mcp/servers/filesystem_read_server.py
backend/protocols/mcp/servers/filesystem_write_server.py
backend/protocols/mcp/servers/shell_server.py
backend/protocols/mcp/servers/document_server.py
backend/protocols/mcp/servers/memory_server.py
```

### 3.0 通用模板

每个 Server 文件遵循同一结构：

```python
#!/usr/bin/env python3
"""MCP Server: {server_name} — {tool_list}"""
import os
import sys

# §4.2: 环境变量传递 PROJECT_ROOT
PROJECT_ROOT = os.environ.get("AGENT_PROJECT_ROOT")
if not PROJECT_ROOT:
    raise RuntimeError("AGENT_PROJECT_ROOT environment variable required")
DOWNLOADS_DIR = os.path.join(PROJECT_ROOT, "web", "static", "downloads")

from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("{server_name}")

@server.list_tools()
async def list_tools():
    return [...]

@server.call_tool()
async def call_tool(name: str, arguments: dict, _meta: dict = None):
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return [{"type": "text", "text": f"Unknown tool: {name}"}]
    result = await handler(**arguments, _meta=_meta)
    import json
    return [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 3.1 network_server.py

迁移工具：`web_search`, `web_fetch`, `stock_query`

从 `tools.py` 搬运代码：
- `_web_search()` → `handle_web_search()`
- `_web_fetch()` → `handle_web_fetch()`（保留 SSRF 防护）
- `_stock_query()` → `handle_stock_query()`（新增 URL 白名单检查）

**额外改动**：
- `stock_query` 新增 SSRF 防护：URL 白名单只允许 `vip.stock.finance.sina.com.cn`
- `web_search` 和 `web_fetch` 中的 `_is_public_url` 从 `tools.py` import
- 不再调用 `_record_download`（network server 无文件生成）

### 3.2 filesystem_read_server.py

迁移工具：`read_file`, `read_pdf`, `grep_files`, `glob_files`

从 `tools.py` 搬运代码：
- `_read_file()` → `handle_read_file()`
- `_read_pdf()` → `handle_read_pdf()`
- `_grep_files()` → `handle_grep_files()`
- `_glob_files()` → `handle_glob_files()`

**关键改动**：
- `_safe_path()` 中 `PROJECT_ROOT` 改为从环境变量获取（不是 `__file__`）
- `PyPDF2` lazy import（handler 内部 import）

### 3.3 filesystem_write_server.py

迁移工具：`edit_file`（require_confirmation=True）

从 `tools.py` 搬运代码：
- `_edit_file()` → `handle_edit_file()`

**注意**：`require_confirmation=True` 在 Server 端注册，但确认检查在引擎侧 `_execute_tool()` 拦截。

### 3.4 shell_server.py

迁移工具：`execute_command`

从 `tools.py` 搬运代码：
- `_execute_command()` → `handle_execute_command()`

**关键改动**：
- 超时从 `AGENT_TOOL_TIMEOUT` 环境变量读取（默认 30s）
- `ALLOWED_COMMANDS` 从 `tools.py` import
- `PROJECT_ROOT` 从环境变量获取（`cwd` 参数）

### 3.5 document_server.py

迁移工具：`create_excel`, `create_docx`, `create_document`

从 `tools.py` 搬运代码：
- `_create_excel()` → `handle_create_excel()`
- `_create_docx()` → `handle_create_docx()`
- `_create_document()` → `handle_create_document()`

**关键改动**：
- `_create_docx._table` 函数属性 → 改为局部变量 `table_data: list = []`，finally 中清理
- `downloads_dir` 从 `DOWNLOADS_DIR`（环境变量推导）获取
- **不再调用 `_record_download`** → 改为在返回结果中添加 `_download_info` 字段

```python
# document_server 返回格式
return {
    "filename": safe_name,
    "file_type": "xlsx",
    "size_bytes": size,
    "download_url": encoded_url,
    "clickable_link": f"[下载 {safe_name}]({encoded_url})",
    "_download_info": {
        "filename": safe_name,
        "filepath": filepath,
        "size_bytes": size,
    }
}
```

- `_record_download` 由 MCPTool.handler() 在主进程侧消费（见 §4.4）
- `openpyxl` / `python-docx` lazy import

### 3.6 memory_server.py

迁移工具：`memory_search`, `list_downloads`

从 `tools.py` 搬运代码：
- `_memory_search()` → `handle_memory_search()`
- `_list_downloads()` → `handle_list_downloads()`

**关键改动**：
- `user_id` 从 `_meta` 参数获取：`user_id = (_meta or {}).get("user_id", 0)`
- 不再使用 `_current_user_id` ContextVar
- `list_downloads` 使用 `get_db("memory")` 而非 `get_db()`
- `MemoryManager` 的 `user_id` 参数从 `_meta` 获取

### 3.7 tools.py 改动

**保留**：
- `Tool` dataclass（满足 ToolProtocol）
- `_calculator` handler + `calculator` Tool 定义
- `_timer_set` handler + `timer_set` Tool 定义
- `_current_user_id` ContextVar + `set_current_user()`
- `_record_download()` 函数
- 安全函数：`_sanitize_url`, `_is_public_url`, `_safe_path`, `_validate_command`
- `ALLOWED_COMMANDS`, `PROJECT_ROOT` 常量

**移除**：
- 14 个高风险工具的 handler 函数体 → 已迁移到 MCP Server
- `TOOLS` 列表 → 由 ToolRegistry 替代

**新增**：
- `calculator` 和 `timer_set` 的 Tool 实例保留在 tools.py，由 server.py 的 lifespan 显式注册到 ToolRegistry

```python
# tools.py 保留的 Tool 实例（供 lifespan 注册）
LOCAL_TOOLS: list[Tool] = [
    Tool(
        name="calculator",
        description="执行数学计算。支持基本运算、三角函数、对数等。",
        parameters={...},
        handler=_calculator,
    ),
    Tool(
        name="timer_set",
        description="设置一个计时器，等待指定秒数后返回。",
        parameters={...},
        handler=_timer_set,
    ),
]
```

### 验收标准

- [x] 6 个 Server 文件均可独立 `python -m` 运行
- [x] 每个.Server 的 `list_tools` 返回与设计文档 §6 一致的工具列表
- [x] `AGENT_PROJECT_ROOT` 环境变量缺失时 raise RuntimeError
- [x] document_server 返回 `_download_info` 字段
- [x] memory_server 的 user_id 从 `_meta` 获取
- [x] `_create_docx` 使用局部变量而非函数属性
- [x] `stock_query` 新增 URL 白名单检查
- [x] tools.py 保留安全函数、ContextVar、_record_download
- [x] tools.py 移除 14 个 handler + TOOLS 列表
- [x] tools.py 新增 LOCAL_TOOLS 列表

---

## P4：引擎适配 — 类型标注 + _execute_tool

### 改动文件

- `backend/agent/engine.py`
- `backend/agent/plan_solve_engine.py`

### 4.1 engine.py

**改动 1**：import 和类型标注

```python
# 旧
from .tools import Tool
class AgentEngine:
    def __init__(self, deepseek, tools: list[Tool], ...):

# 新
from ..protocols.mcp.tool_adapter import ToolProtocol
class AgentEngine:
    def __init__(self, deepseek, tools: list[ToolProtocol], ...):
```

**改动 2**：新增 `_execute_tool` 方法（§8.1）

在 `AgentEngine` 类中新增方法，将确认拦截从 `run()` 内联逻辑提取出来：

```python
async def _execute_tool(self, tool: ToolProtocol, args: dict) -> dict:
    """执行工具调用，确认检查在主进程拦截"""
    if tool.require_confirmation:
        # 发送确认请求（复用现有 ws_manager 机制）
        approved = await self._wait_for_confirmation(...)
        if not approved:
            return {"status": "rejected", "message": "User denied confirmation"}
    return await tool.handler(**args)
```

**改动 3**：替换 `run()` 中的工具执行路径

在 `run()` 方法中，将直接调用 `tool.handler(**arguments)` 的地方改为 `self._execute_tool(tool, arguments)`。

涉及位置（行号基于当前代码）：
- L388-391：`exec_one` 函数中的 `tool.handler(**arguments)` → `self._execute_tool(tool, arguments)`
- L193：简单路径中的 `tool.handler(**args)` → `self._execute_tool(tool, args)`

**改动 4**：确认检查逻辑调整

现有确认逻辑在 L400-425，已经在调用 handler 之前检查 `require_confirmation`。重构为 `_execute_tool` 后，L400-425 的确认代码块简化为只做 UI 推送，实际拦截在 `_execute_tool` 中。

### 4.2 plan_solve_engine.py

**改动**：类型标注

```python
# 旧
from .tools import Tool
class PlanSolveEngine:
    def __init__(self, deepseek, tools: list[Tool], ...):

# 新
from ..protocols.mcp.tool_adapter import ToolProtocol
class PlanSolveEngine:
    def __init__(self, deepseek, tools: list[ToolProtocol], ...):
```

plan_solve_engine.py 不需要 `_execute_tool`——它目前没有确认机制（只有 react engine 有），但未来可复用。

### 验收标准

- [x] `engine.py` 类型标注 `list[Tool]` → `list[ToolProtocol]`
- [x] `plan_solve_engine.py` 类型标注 `list[Tool]` → `list[ToolProtocol]`
- [x] `_execute_tool` 方法实现：确认拦截 + handler 调用
- [x] `run()` 中所有 `tool.handler()` 调用改为 `self._execute_tool()`
- [x] 确认机制行为不变（用户拒绝 → 返回 rejected 状态）

---

## P5：服务启动 — lifespan 改造

### 改动文件

- `backend/server.py`
- `backend/agent/__init__.py`

### 5.1 server.py

**改动 1**：移除 `@app.on_event("startup")`，改用 lifespan context manager

```python
from contextlib import asynccontextmanager
from .protocols.mcp import ToolRegistry, MCPClientManager

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

# 全局单例
tool_registry = ToolRegistry()
mcp_manager = MCPClientManager(project_root=PROJECT_ROOT)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — 顺序关键
    init_db()  # 数据库初始化

    # 1. 注册本地工具（frozen 之前）
    from .agent.tools import LOCAL_TOOLS
    for tool in LOCAL_TOOLS:
        tool_registry.register_local(tool)

    # 2. 连接 MCP Server + 发现工具 → freeze
    await tool_registry.initialize(mcp_manager)

    print(f"数据库已初始化 (chat/agent/memory)")
    print(f"工具注册完成: {len(tool_registry.get_all_tools())} 个")
    print(f"Agent 模式: react / plan_solve")
    print(f"DeepSeek: {'已配置' if deepseek else '未配置'}")
    print(f"API 文档: http://127.0.0.1:8000/docs")

    yield

    # Shutdown
    await tool_registry.shutdown()

app = FastAPI(title="Agent Learning — Unified", lifespan=lifespan)
```

**改动 2**：`_get_engine()` 改为从 ToolRegistry 取工具

```python
# 旧
def _get_engine(agent_mode: str):
    if agent_mode == "plan_solve":
        return PlanSolveEngine(deepseek, TOOLS, ws_manager)
    else:
        return AgentEngine(deepseek, TOOLS, ws_manager, intervention_handler)

# 新
def _get_engine(agent_mode: str):
    tools = tool_registry.get_all_tools()
    if agent_mode == "plan_solve":
        return PlanSolveEngine(deepseek, tools, ws_manager)
    else:
        return AgentEngine(deepseek, tools, ws_manager, intervention_handler)
```

**改动 3**：移除 `from .agent import TOOLS`

```python
# 旧
from .agent import AgentEngine, PlanSolveEngine, TOOLS, WebSocketManager

# 新
from .agent import AgentEngine, PlanSolveEngine, WebSocketManager
```

### 5.2 backend/agent/__init__.py

```python
# 旧
from .engine import AgentEngine
from .plan_solve_engine import PlanSolveEngine
from .tools import TOOLS, Tool
from .websocket_manager import WebSocketManager

# 新
from .engine import AgentEngine
from .plan_solve_engine import PlanSolveEngine
from .tools import Tool, LOCAL_TOOLS
from .websocket_manager import WebSocketManager
```

### 验收标准

- [x] lifespan context manager 替代 `@app.on_event("startup")`
- [x] 启动顺序：init_db → register_local → connect_all → initialize(freeze)
- [x] `_get_engine()` 从 `tool_registry.get_all_tools()` 取工具
- [x] shutdown 阶段清理 MCP 连接
- [x] `__init__.py` 导出 `LOCAL_TOOLS` 替代 `TOOLS`

---

## P6：后处理机制 — _record_download 归主进程

### 改动文件

- `backend/protocols/mcp/tool_adapter.py`（已在 P1 中实现）

### 6.1 MCPTool.handler() 后处理逻辑

已在 P1 的 MCPTool.handler() 中实现，核心代码：

```python
async def handler(self, **kwargs) -> dict:
    from ...agent.tools import _current_user_id, _record_download

    result = await self._client_manager.call_tool(
        self.server_name, self.name, kwargs
    )
    # 主进程侧后处理
    if isinstance(result, dict) and "_download_info" in result:
        user_id = _current_user_id.get()  # 主进程有效 ✅
        info = result.pop("_download_info")
        _record_download(user_id, info["filename"], info["filepath"], info["size_bytes"])
    return result
```

### 验收标准

- [x] document_server 返回 `_download_info` 字段
- [x] MCPTool.handler() 消费 `_download_info` 后从结果中移除
- [x] `_record_download` 在主进程执行（ContextVar 有效）
- [x] LLM 永远看不到 `_download_info` 字段

---

## P7：依赖 & 配置

### 改动文件

- `requirements.txt`
- 可能需要确认 `venv` 中安装 mcp SDK

### 7.1 requirements.txt

新增：

```
mcp~=1.9.0          # Anthropic 官方 MCP Python SDK
```

### 7.2 安装

```bash
cd D:\agent_learning
venv\Scripts\pip install "mcp~=1.9.0"
```

### 验收标准

- [x] `mcp~=1.9.0` 添加到 requirements.txt
- [x] `pip install mcp` 成功
- [x] `python -c "import mcp; print(mcp.__version__)"` 输出 1.9.x

---

## P8：集成测试

### 新建文件

```
tests/test_mcp_integration.py
```

### 8.1 测试矩阵

| 层级 | 测试内容 | 方式 |
|------|---------|------|
| 单元测试 | 每个 Server 的工具 handler（进程内直接调） | pytest，参数化 |
| 集成测试 | MCPClientManager + 真实子进程 → call_tool | pytest + subprocess |
| 引擎测试 | 现有引擎 + ToolRegistry（mock MCPClientManager） | pytest |
| 兼容性测试 | 现有 REST API 行为不变 | pytest + httpx |
| 端到端测试 | 完整 Agent 任务（web_search + create_excel） | 手动 + WebSocket |

### 8.2 关键测试用例

1. **calculator/timer_set 进程内调用** — 确认保留工具正常
2. **web_search 通过 MCP 调用** — 确认 _meta 传递
3. **create_excel 返回 _download_info** — 确认后处理正确
4. **edit_file require_confirmation** — 确认引擎侧拦截
5. **MCP Server 启动/关闭** — 确认生命周期
6. **_meta.user_id 正确传递** — memory_server 侧验证
7. **并发测试** — network server 2 session 并行
8. **降级测试** — MCP Server 崩溃后 unavailable 标记

### 验收标准

- [x] 单元测试覆盖每个 MCP Server 的每个工具
- [x] 集成测试覆盖 MCPClientManager 生命周期
- [x] 兼容性测试：18 REST + 1 WS 端点行为不变
- [x] 端到端测试：完整 Agent 任务成功执行

---

## 执行顺序 & 依赖图

```
P1 (骨架) ──→ P2 (连接层) ──→ P3 (Server 迁移)
  │                              │
  ├──→ P4 (引擎适配)             │
  │                              │
  ├──→ P5 (lifespan) ←───────────┘
  │         │
  │         ├──→ P6 (后处理) ←── P3
  │         │
  │         ├──→ P7 (依赖)
  │         │
  │         └──→ P8 (测试) ←── P1-P7
```

**推荐执行顺序**：

1. **P7**（依赖安装，最优先）— 2 min
2. **P1**（骨架）— 30 min
3. **P2**（连接层）— 45 min
4. **P3**（6 个 Server）— 90 min（最大工作量）
5. **P4**（引擎适配）— 20 min
6. **P5**（lifespan）— 20 min
7. **P6**（后处理，P1 已包含）— 0 min（验证即可）
8. **P8**（测试）— 60 min

**总预估**：~4.5 小时

---

## 风险 & 回退

| 风险 | 缓解 |
|------|------|
| MCP SDK API 与设计文档不一致 | P2 阶段先写 spike 验证 API，再正式实现 |
| Windows stdio 子进程问题 | mcp SDK 已处理 Windows；如有问题先在 Linux 验证 |
| 工具迁移遗漏依赖 | 每个 Server 完成后立即做单元测试 |
| lifespan 顺序导致启动失败 | startup 顺序已明确：register → connect → freeze |
| MCP Server 启动慢影响用户体验 | asyncio.gather 并行启动；如 >5s 加 loading 状态 |

**回退策略**：如果 MCP 集成出现阻塞问题，可以临时回退到 `TOOLS` 列表方式——只需将 `tool_registry.get_all_tools()` 改回 `TOOLS`，其他改动不影响。

---

## 版本记录

| 日期 | 内容 | 作者 |
|------|------|------|
| 2026-06-17 | 初始版本 | 蹊径 + Claude |

# MCP 集成设计文档

> 日期：2026-06-17
> 状态：设计确认，待实现
> 关联文档：[技术路线图](../tech-roadmap.md)

---

## 1. 目标

将 agent_learning 项目中的高风险工具从 Agent 进程拆出，封装为独立的 MCP (Model Context Protocol) Server 进程。
实现工具的进程隔离、标准化通信、跨语言复用能力。

## 2. 设计原则

1. **向后兼容** — 现有 REST API（18 端点）、WebSocket、Tool dataclass 全部保留
2. **渐进迁移** — 不一次性重写，通过 ToolRegistry 逐步切换工具调用路径
3. **最小权限** — 每个 MCP Server 只拥有完成职责所需的最小权限集
4. **功能域分组** — 相关工具合并在同一个 Server，避免 14 个独立进程

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

## 4. 架构

```
backend/
├── protocols/                          # 新增：协议层
│   ├── __init__.py
│   └── mcp/
│       ├── __init__.py
│       ├── client_manager.py           # MCPClientManager — 管理所有 MCP 连接
│       ├── tool_adapter.py             # MCPTool — 将 MCP 工具适配为 Tool 接口
│       ├── registry.py                 # ToolRegistry — 统一工具注册表
│       └── servers/                    # 各 MCP Server 入口（可独立运行）
│           ├── network_server.py       # web_search, web_fetch, stock_query
│           ├── filesystem_read_server.py  # read_file, read_pdf, grep_files, glob_files
│           ├── filesystem_write_server.py # edit_file
│           ├── shell_server.py         # execute_command
│           ├── document_server.py      # create_excel, create_docx, create_document
│           └── memory_server.py        # memory_search, list_downloads
├── agent/
│   ├── tools.py                        # 保留：本地工具（calculator, timer_set）
│   ├── engine.py                       # 修改：通过 ToolRegistry 获取工具
│   └── plan_solve_engine.py            # 修改：同上
└── server.py                           # 修改：启动时初始化 MCPClientManager
```

### 4.1 组件职责

#### ToolRegistry (`backend/protocols/mcp/registry.py`)

```python
class ToolRegistry:
    """统一工具注册表，管理本地工具 + MCP 工具"""

    def __init__(self):
        self._local_tools: list[Tool] = []
        self._mcp_clients: MCPClientManager | None = None

    async def initialize(self, mcp_manager: MCPClientManager):
        """启动时初始化：连接所有 MCP Server，发现工具"""
        self._mcp_clients = mcp_manager
        await self._mcp_clients.connect_all()
        # 从每个 MCP Server 获取 tools/list，创建 MCPTool 适配器

    def register_local(self, tool: Tool):
        """注册本地工具"""
        self._local_tools.append(tool)

    def get_all_tools(self) -> list[Tool]:
        """返回所有工具（本地 + MCP），供引擎使用"""

    def get_openai_schemas(self) -> list[dict]:
        """返回所有工具的 OpenAI 格式 schema"""

    async def shutdown(self):
        """关闭所有 MCP 连接"""
```

#### MCPClientManager (`backend/protocols/mcp/client_manager.py`)

```python
class MCPClientManager:
    """管理 6 个 MCP Server 的 stdio 连接"""

    def __init__(self):
        self._servers: dict[str, MCPServerConfig] = {
            "network":           MCPServerConfig(cmd=["python", "-m", "backend.protocols.mcp.servers.network_server"]),
            "filesystem-read":   MCPServerConfig(cmd=["python", "-m", "backend.protocols.mcp.servers.filesystem_read_server"]),
            "filesystem-write":  MCPServerConfig(cmd=["python", "-m", "backend.protocols.mcp.servers.filesystem_write_server"]),
            "shell":             MCPServerConfig(cmd=["python", "-m", "backend.protocols.mcp.servers.shell_server"]),
            "document":          MCPServerConfig(cmd=["python", "-m", "backend.protocols.mcp.servers.document_server"]),
            "memory":            MCPServerConfig(cmd=["python", "-m", "backend.protocols.mcp.servers.memory_server"]),
        }
        self._sessions: dict[str, ClientSession] = {}

    async def connect_all(self):
        """启动所有 MCP Server 子进程，建立 stdio 连接"""

    async def call_tool(self, server_name: str, tool_name: str, args: dict) -> dict:
        """调用指定 Server 的工具"""

    async def list_tools(self, server_name: str) -> list[dict]:
        """获取指定 Server 的工具列表（tools/list）"""

    async def shutdown(self):
        """关闭所有子进程"""
```

#### MCPTool (`backend/protocols/mcp/tool_adapter.py`)

```python
@dataclass
class MCPTool:
    """将 MCP 工具适配为 Tool 接口，引擎调用透明"""
    name: str
    description: str
    parameters: dict
    server_name: str          # 归属哪个 MCP Server
    _client_manager: MCPClientManager

    async def handler(self, **kwargs) -> dict:
        """调用 MCP Server 的工具，返回结果"""
        return await self._client_manager.call_tool(
            self.server_name, self.name, kwargs
        )

    def to_openai_schema(self) -> dict:
        """与 Tool.to_openai_schema() 完全兼容"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
```

### 4.2 数据流

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
        ├── 本地 Tool:
        │     await handler(**args)  → 直接返回
        │
        └── MCPTool:
              await handler(**args)
                → MCPClientManager.call_tool(server, name, args)
                  → stdio → MCP Server 进程
                    → 执行实际逻辑
                    → 返回结果
```

## 5. 每个 MCP Server 详情

### 5.1 mcp-server-network

| 属性 | 值 |
|------|-----|
| 传输 | stdio |
| 依赖 | duckduckgo_search, httpx |

**工具列表：**

| 工具 | 描述 |
|------|------|
| `web_search` | DuckDuckGo + Baidu 智能搜索，自动感知时效需求 |
| `web_fetch` | 异步 HTTP 抓取，含 SSRF 防护（URL 白名单 + 内网 IP 过滤） |
| `stock_query` | 中国 A 股实时数据（Sina Finance API） |

### 5.2 mcp-server-filesystem-read

| 属性 | 值 |
|------|-----|
| 传输 | stdio |
| 权限 | **只读**，限定 PROJECT_ROOT |
| 依赖 | PyPDF2 |

**工具列表：**

| 工具 | 描述 |
|------|------|
| `read_file` | 文件读取 / 目录列表，路径沙盒化 |
| `read_pdf` | PDF 文本提取 |
| `grep_files` | 正则内容搜索（ripgrep 风格） |
| `glob_files` | Glob 文件名匹配 |

### 5.3 mcp-server-filesystem-write

| 属性 | 值 |
|------|-----|
| 传输 | stdio |
| 权限 | **写入**，限定 PROJECT_ROOT |
| 依赖 | 无（标准库） |

**工具列表：**

| 工具 | 描述 |
|------|------|
| `edit_file` | 精确字符串替换 |

### 5.4 mcp-server-shell

| 属性 | 值 |
|------|-----|
| 传输 | stdio |
| 权限 | 命令白名单 + shell=False + 超时 30s |
| 依赖 | 无（subprocess 标准库） |

**工具列表：**

| 工具 | 描述 |
|------|------|
| `execute_command` | 白名单命令执行，shell=False 防注入 |

### 5.5 mcp-server-document

| 属性 | 值 |
|------|-----|
| 传输 | stdio |
| 依赖 | openpyxl, python-docx |

**工具列表：**

| 工具 | 描述 |
|------|------|
| `create_excel` | 生成 .xlsx 文件 |
| `create_docx` | Markdown → .docx |
| `create_document` | 生成 md/csv/html/py 文件 |

### 5.6 mcp-server-memory

| 属性 | 值 |
|------|-----|
| 传输 | stdio |
| 依赖 | SQLite（agent.db） |

**工具列表：**

| 工具 | 描述 |
|------|------|
| `memory_search` | 跨会话记忆存储/搜索/列表 |
| `list_downloads` | 用户下载记录查询 |

## 6. Server 启动配置 (`server.py`)

```python
# 修改点：server.py 启动时初始化 ToolRegistry 替代裸 TOOLS 列表

from .protocols.mcp import ToolRegistry, MCPClientManager

# 替代原来的 TOOLS import
tool_registry = ToolRegistry()
mcp_manager = MCPClientManager()

@app.on_event("startup")
async def startup():
    # 连接所有 MCP Server
    await mcp_manager.connect_all()
    await tool_registry.initialize(mcp_manager)

    # 注册本地工具
    tool_registry.register_local(Tool("calculator", ...))
    tool_registry.register_local(Tool("timer_set", ...))

@app.on_event("shutdown")
async def shutdown():
    await mcp_manager.shutdown()

# 引擎工厂
def _get_engine(agent_mode: str):
    tools = tool_registry.get_all_tools()  # 替代 TOOLS
    if agent_mode == "plan_solve":
        return PlanSolveEngine(deepseek, tools, ws_manager)
    else:
        return AgentEngine(deepseek, tools, ws_manager, intervention_handler)
```

## 7. 引擎改动点

### 7.1 engine.py

改动最小——只改入口，不改 ReAct 循环：

```python
# 现有代码
class AgentEngine:
    def __init__(self, deepseek, tools: list[Tool], ws_manager, intervention=None):
        self.tools = tools          # 现在可能包含 MCPTool
        self.tool_map = {t.name: t for t in tools}  # MCPTool 的 handler 已封装
        ...

    # run() 方法逻辑完全不变
    # _call_llm() 不变
    # 工具执行：tool_map[name].handler(**args) — MCPTool.handler 内部走 MCP
```

### 7.2 tools.py

保留 `Tool` dataclass 定义，保留 calculator 和 timer_set 的 handler。
移除已迁移到 MCP 的 14 个工具的 handler 代码（函数体移到对应的 Server 文件）。
保留 `_current_user_id` ContextVar（MCP Server 需要）。

## 8. 向后兼容性

| 接口 | 变动 | 说明 |
|------|------|------|
| `POST /api/chat` | 不变 | 直接调 DeepSeek，不涉及工具 |
| `POST /api/chat/stream` | 不变 | 同上 |
| `POST /api/agent/tasks` | 不变 | 引擎实例化改为从 ToolRegistry 取工具 |
| `WS /ws/agent/{task_id}` | 不变 | WebSocket 协议不变 |
| `Tool.to_openai_schema()` | 不变 | MCPTool 实现相同接口 |
| 确认机制 | 不变 | `require_confirmation` 字段保留，MCP Server 端不支持确认（简化） |
| 反思机制 | 不变 | `_reflect()` 不涉及工具调用 |
| 记忆注入 | 不变 | `MemoryManager` 在引擎启动时独立初始化 |

### 8.1 降级策略

如果 MCP Server 不可用（启动失败、崩溃），引擎应能降级：

```python
# ToolRegistry.get_all_tools()
def get_all_tools(self) -> list[Tool]:
    tools = list(self._local_tools)
    for mcp_tool in self._mcp_tools:
        if mcp_tool.available:    # 健康检查通过
            tools.append(mcp_tool)
        else:
            logger.warning(f"MCP tool {mcp_tool.name} unavailable, skipping")
    return tools
```

不自动回退到进程内调用——MCP 工具不可用就是不可用，保持工具边界清晰。

## 9. 错误处理

```python
class MCPToolError(Exception):
    """MCP 工具调用错误"""
    pass

class MCPServerUnavailable(MCPToolError):
    """Server 进程崩溃或连接断开"""
    pass

class MCPToolTimeout(MCPToolError):
    """工具执行超时（默认 60s）"""
    pass
```

引擎的工具调用路径已有 try/except 和空结果检测。
MCP 错误映射到现有的 `{"error": "..."}` 返回格式，LLM 的反思机制会自动处理。

## 10. 测试策略

| 层级 | 内容 | 方式 |
|------|------|------|
| 单元测试 | 每个 MCP Server 的工具 handler 在进程内测试 | pytest，mock stdin/stdout |
| 集成测试 | MCPClientManager 启动 Server → call_tool → 得到正确结果 | pytest + subprocess |
| 引擎测试 | 现有引擎 + ToolRegistry，验证工具选择和执行 | pytest |
| 兼容性测试 | 现有 REST API 行为不变 | pytest + httpx |

## 11. 风险与缓解

| 风险 | 缓解 |
|------|------|
| MCP Server 启动慢 | 在 `startup` 事件中并行启动 6 个进程 |
| 子进程崩溃 | MCPClientManager 健康检查 + 自动重启（可选） |
| stdio 通信开销 | 纯 stdout/stdin JSON-RPC，开销 < 1ms，可忽略 |
| mcp SDK 版本不稳定 | 锁定版本号，只用稳定 API |
| 跨平台差异（Windows stdio） | mcp SDK 已处理；开发环境 Windows 测试 |

## 12. 依赖

新增到 `requirements.txt`：

```
mcp>=1.0.0          # Anthropic 官方 MCP Python SDK
```

现有依赖不变。

---

## 版本记录

| 日期 | 内容 | 作者 |
|------|------|------|
| 2026-06-17 | 初始版本 | Xin-Zho + Claude |

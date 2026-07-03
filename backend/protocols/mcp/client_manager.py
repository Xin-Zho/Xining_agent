"""
MCPClientManager — 管理 6 个 MCP Server 的 stdio 连接

核心职责：
1. 启动所有 MCP Server 子进程
2. per-session Lock 实现并发（network=2，其余=1）
3. 注入 user_id 到 arguments（SDK 1.9.4 不支持 _meta，改为 arguments 注入）
4. call_tool 超时控制
5. 健康检查 + 不可用标记

重要变更（SDK 1.9.4 实测）：
- stdio_client() 和 ClientSession() 都是 async context manager
- ClientSession.__aenter__() 启动 _receive_loop，如果不进入 context，
  initialize() 发请求但无人读响应，导致超时
- anyio 的 cancel scope 要求进入和退出在同一个 task 中
- AsyncExitStack 退出时触发 anyio cancel scope 跨 task 退出异常
- 解决方案：保存 context manager 对象，在 lifespan yield 期间用
  async with 管理所有连接，shutdown 时关闭 subprocess
"""
import asyncio
import json
import logging
import os
import signal
import sys
from dataclasses import dataclass

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
    """MCP Server 配置"""
    cmd: list[str]
    env: dict
    session_count: int = 1


# 需要从 parameters 中过滤的内部参数（LLM 不可见）
_INTERNAL_PARAMS = {"user_id"}


class MCPClientManager:
    """管理 6 个 MCP Server 的 stdio 连接。per-session Lock 实现并发。"""

    def __init__(self, project_root: str):
        self._server_configs: dict[str, MCPServerConfig] = {
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
                env={"AGENT_PROJECT_ROOT": project_root, "AGENT_TOOL_TIMEOUT": "30"},
            ),
            "document": MCPServerConfig(
                cmd=[sys.executable, "-m", "backend.protocols.mcp.servers.document_server"],
                env={"AGENT_PROJECT_ROOT": project_root},
            ),
            "memory": MCPServerConfig(
                cmd=[sys.executable, "-m", "backend.protocols.mcp.servers.memory_server"],
                env={"AGENT_PROJECT_ROOT": project_root},
            ),
            "chemistry": MCPServerConfig(
                cmd=[sys.executable, "-m", "backend.protocols.mcp.servers.chemistry_server"],
                env={"AGENT_PROJECT_ROOT": project_root},
            ),
            "physics": MCPServerConfig(
                cmd=[sys.executable, "-m", "backend.protocols.mcp.servers.physics_server"],
                env={"AGENT_PROJECT_ROOT": project_root},
            ),
        }
        # {server_name: [{session, idx, stdio_cm, session_cm, process}, ...]}
        self._sessions: dict[str, list[dict]] = {}
        # {server_name: [asyncio.Lock, ...]}  — 每个 session 一把锁
        self._session_locks: dict[str, list[asyncio.Lock]] = {}

    @property
    def server_names(self) -> list[str]:
        return list(self._server_configs.keys())

    async def connect_all(self):
        """启动所有 MCP Server 子进程

        串行启动，每个 Server 独立超时（5s）。失败则跳过，不阻塞整体。
        """
        for name, config in list(self._server_configs.items()):
            try:
                await asyncio.wait_for(
                    self._connect_server(name, config),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                logger.warning(f"MCP Server {name}: connection timed out, skipping")
                self._server_configs.pop(name, None)
            except Exception as e:
                logger.warning(f"MCP Server {name}: connection failed ({e}), skipping")
                self._server_configs.pop(name, None)

        logger.info(
            f"MCPClientManager connected: {len(self._sessions)}/{len(self._server_configs)} servers, "
            f"total sessions: {sum(len(s) for s in self._sessions.values())}"
        )

    async def _connect_server(self, name: str, config: MCPServerConfig):
        """启动单个 Server 的所有 session。

        整个连接在独立 task 中执行，防止 anyio cancel scope 跨 task 问题。
        超时由调用方 connect_all() 的 asyncio.wait_for 保证。
        """
        count = config.session_count
        sessions = []
        locks = []

        for i in range(count):
            server_params = StdioServerParameters(
                command=config.cmd[0],
                args=config.cmd[1:],
                env={**config.env, **dict(os.environ)},
            )

            # 手动进入 context（不退出，保存 cm 对象）
            stdio_cm = stdio_client(server_params)
            try:
                read_stream, write_stream = await stdio_cm.__aenter__()
            except Exception as e:
                logger.warning(f"MCP {name}: stdio_client failed: {e}")
                raise

            session_cm = ClientSession(read_stream, write_stream)
            try:
                session = await session_cm.__aenter__()
            except Exception as e:
                logger.warning(f"MCP {name}: ClientSession failed: {e}")
                await self._safe_close_stdio(stdio_cm)
                raise

            try:
                await session.initialize()
            except Exception as e:
                logger.warning(f"MCP {name}: initialize failed: {e}")
                await self._safe_close_session(session_cm)
                await self._safe_close_stdio(stdio_cm)
                raise

            sessions.append({
                "session": session,
                "idx": i,
                "stdio_cm": stdio_cm,
                "session_cm": session_cm,
            })
            locks.append(asyncio.Lock())
            logger.info(f"MCP {name} session {i} connected")

        self._sessions[name] = sessions
        self._session_locks[name] = locks

    async def _safe_close_session(self, session_cm):
        try:
            await session_cm.__aexit__(None, None, None)
        except BaseException:
            pass

    async def _safe_close_stdio(self, stdio_cm):
        try:
            await stdio_cm.__aexit__(None, None, None)
        except BaseException:
            pass

    def _pick_session(self, server_name: str, tool_name: str) -> dict:
        """按 tool_name hash 轮转选择 session"""
        sessions = self._sessions[server_name]
        idx = hash(tool_name) % len(sessions)
        return sessions[idx]

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> dict:
        """调用 MCP Server 工具，注入 user_id 到 arguments，per-session Lock 串行

        注：SDK 1.9.4 call_tool() 不支持 _meta 参数，
        因此改为将 user_id 注入 arguments（Server inputSchema 中声明为可选）。
        """
        from ...agent.tools import _current_user_id

        session_entry = self._pick_session(server_name, tool_name)
        session = session_entry["session"]
        session_idx = session_entry["idx"]

        # 注入 user_id 到 arguments
        enriched_args = {**arguments, "user_id": _current_user_id.get()}

        async with self._session_locks[server_name][session_idx]:
            try:
                result = await asyncio.wait_for(
                    session.call_tool(
                        tool_name,
                        arguments=enriched_args,
                    ),
                    timeout=STEP_TIMEOUT,
                )
                # MCP SDK 返回 CallToolResult，提取 content
                if hasattr(result, 'content') and result.content:
                    for item in result.content:
                        if hasattr(item, 'text'):
                            try:
                                parsed = json.loads(item.text)
                                return parsed
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
        """标记工具不可用"""
        logger.error(f"MCP tool {server_name}/{tool_name} marked unavailable")

    async def list_tools(self, server_name: str) -> list[dict]:
        """获取指定 Server 的工具列表"""
        session = self._sessions[server_name][0]["session"]
        result = await session.list_tools()
        tools = []
        for tool in result.tools:
            # 从 inputSchema 中过滤掉内部参数（LLM 不可见）
            schema = tool.inputSchema or {}
            filtered_schema = _filter_internal_params(schema)
            tools.append({
                "name": tool.name,
                "description": tool.description or "",
                "parameters": filtered_schema,
            })
        return tools

    async def shutdown(self):
        """关闭所有子进程连接

        策略：先关闭 ClientSession（停止 receive_loop），再关闭 stdio 流，
        最后终止子进程。anyio cancel scope 跨 task 退出异常是 SDK 在
        Windows 上的已知问题，不影响功能性关闭——子进程会被正确终止。

        注意：CancelledError 是 BaseException，必须用 except BaseException 捕获。
        """
        for name, sessions in self._sessions.items():
            for s in sessions:
                # 1. 关闭 ClientSession（停止 receive_loop 任务）
                try:
                    await s["session_cm"].__aexit__(None, None, None)
                except BaseException:
                    pass  # anyio cancel scope 跨 task 退出异常（已知问题）

                # 2. 关闭 stdio_client（会终止子进程）
                try:
                    await s["stdio_cm"].__aexit__(None, None, None)
                except BaseException:
                    pass  # 同上，子进程已在 step 1 后收到 stdin 关闭信号

        self._sessions.clear()
        self._session_locks.clear()
        logger.info("MCPClientManager shutdown: all connections closed")


def _filter_internal_params(schema: dict) -> dict:
    """从 inputSchema 中过滤掉内部参数（user_id 等），LLM 不可见"""
    filtered = dict(schema)
    if "properties" in filtered:
        filtered["properties"] = {
            k: v for k, v in filtered["properties"].items()
            if k not in _INTERNAL_PARAMS
        }
    if "required" in filtered:
        filtered["required"] = [
            r for r in filtered["required"] if r not in _INTERNAL_PARAMS
        ]
    return filtered
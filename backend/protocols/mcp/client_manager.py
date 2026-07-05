"""
MCPClientManager — 管理 MCP Server 的 stdio 连接

绕过 anyio/MCP SDK transport（WSL2 + Python 3.12 兼容性问题），
直接用 asyncio subprocess + 手动 JSON-RPC。
"""
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)

STEP_TIMEOUT = 60


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


_INTERNAL_PARAMS = {"user_id"}


class MCPClientManager:
    """管理 MCP Server 的 stdio 连接。纯 asyncio subprocess，不依赖 anyio。"""

    def __init__(self, project_root: str):
        # Dynamic import for module names matching server filenames
        import importlib.util
        # MCP servers — re-enabled (anyio 4.14.1 + mcp 1.28.1 fixed Python 3.12 compatibility)
        _mcp_configs: dict[str, MCPServerConfig] = {
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
            # memory 已由本地 tools.py (rag_search / rag_ingest) 实现，不走 MCP
        }
        self._server_configs: dict[str, MCPServerConfig] = _mcp_configs
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._tools: dict[str, list[dict]] = {}
        # Track _rid (request id) per server
        self._next_rid: dict[str, int] = {}

    @property
    def server_names(self) -> list[str]:
        return list(self._server_configs.keys())

    async def connect_all(self):
        """启动所有 MCP Server 子进程，执行 MCP 握手"""
        for name, config in list(self._server_configs.items()):
            try:
                await self._connect_server(name, config)
            except Exception as e:
                logger.warning(f"MCP Server {name}: connection failed ({e}), skipping")
                self._server_configs.pop(name, None)

        logger.info(f"MCPClientManager connected: {len(self._processes)} servers")

    async def _connect_server(self, name: str, config: MCPServerConfig):
        """启动子进程 + MCP initialize + 发现工具"""
        env = {**os.environ, **config.env}
        proc = await asyncio.create_subprocess_exec(
            *config.cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        # MCP initialize handshake (longer timeout — server needs time to load)
        self._next_rid[name] = 0
        init_resp = await self._jsonrpc_call(proc, name, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "agent-learning", "version": "1.0"},
        }, timeout=30.0)

        if "error" in init_resp:
            proc.kill()
            raise Exception(f"initialize failed: {init_resp['error']}")

        # Send initialized notification
        notif = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
        proc.stdin.write((notif + "\n").encode())
        await proc.stdin.drain()

        # Discover tools
        tools_resp = await self._jsonrpc_call(proc, name, "tools/list", {},
                                              timeout=20.0)
        tools = tools_resp.get("result", {}).get("tools", [])

        self._processes[name] = proc
        self._locks[name] = asyncio.Lock()
        self._tools[name] = tools
        logger.info(f"MCP {name}: {len(tools)} tools connected")

    async def _jsonrpc_call(self, proc, name: str, method: str, params: dict,
                            timeout: float = 10.0) -> dict:
        """Send a JSON-RPC request and read the response."""
        rid = self._next_rid.get(name, 0) + 1
        self._next_rid[name] = rid
        request = json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        proc.stdin.write((request + "\n").encode())
        await proc.stdin.drain()

        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout)
            if not line:
                raise Exception("subprocess closed stdout")
            return json.loads(line.decode().strip())
        except asyncio.TimeoutError:
            raise Exception(f"{method} timed out after {timeout}s")

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> dict:
        """调用 MCP 工具"""
        from ...agent.tools import _current_user_id

        proc = self._processes.get(server_name)
        if not proc:
            raise MCPServerUnavailable(f"Server {server_name} not connected")

        enriched_args = {**arguments, "user_id": _current_user_id.get()}
        lock = self._locks.get(server_name)
        if lock:
            async with lock:
                return await self._call_tool_impl(proc, server_name, tool_name, enriched_args)
        return await self._call_tool_impl(proc, server_name, tool_name, enriched_args)

    async def _call_tool_impl(self, proc, server_name: str, tool_name: str,
                              arguments: dict) -> dict:
        try:
            resp = await asyncio.wait_for(
                self._jsonrpc_call(proc, server_name, "tools/call", {
                    "name": tool_name,
                    "arguments": arguments,
                }),
                timeout=STEP_TIMEOUT,
            )
            if "error" in resp:
                return {"error": resp["error"]}
            result = resp.get("result", {})
            # Extract text from content
            content = result.get("content", [])
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    try:
                        return json.loads(item["text"])
                    except (json.JSONDecodeError, TypeError):
                        return {"result": item["text"]}
            return result
        except asyncio.TimeoutError:
            raise MCPToolTimeout(f"{server_name}/{tool_name} timed out")
        except Exception as e:
            raise MCPServerUnavailable(f"{server_name}/{tool_name}: {e}")

    async def list_tools(self, server_name: str) -> list[dict]:
        tools = self._tools.get(server_name, [])
        result = []
        for tool in tools:
            schema = tool.get("inputSchema", {})
            filtered = _filter_internal_params(schema)
            result.append({
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": filtered,
            })
        return result

    async def shutdown(self):
        for name, proc in self._processes.items():
            try:
                proc.stdin.close()
                proc.kill()
                await proc.wait()
            except Exception:
                pass
        self._processes.clear()
        self._locks.clear()
        self._tools.clear()
        logger.info("MCPClientManager shutdown: all connections closed")


def _filter_internal_params(schema: dict) -> dict:
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

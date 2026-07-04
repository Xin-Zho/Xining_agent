"""
ToolRegistry — 统一工具注册表

全局单例，startup 初始化后只读（_frozen 标记）。
注册顺序：先 register_local（本地工具）→ 再 connect_all + initialize（MCP 工具）→ freeze。
"""
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
        """注册本地进程内工具（必须在 freeze 之前）"""
        if self._frozen:
            raise RuntimeError("ToolRegistry is frozen after initialization")
        self._local_tools.append(tool)
        logger.debug(f"Registered local tool: {tool.name}")

    async def initialize(self, mcp_manager):
        """启动时初始化：连接 MCP Server → 发现工具 → freeze

        顺序关键：必须在 register_local 之后调用。
        """
        await mcp_manager.connect_all()

        # 需要确认的工具映射（server → tool → require_confirmation）
        CONFIRMATION_MAP = {
            "filesystem-write": {"edit_file": True},
            "shell": {"execute_command": True},
        }

        for server_name in mcp_manager.server_names:
            tool_defs = await mcp_manager.list_tools(server_name)
            server_confirms = CONFIRMATION_MAP.get(server_name, {})
            for td in tool_defs:
                self._mcp_tools.append(MCPTool(
                    name=td["name"],
                    description=td["description"],
                    parameters=td["parameters"],
                    require_confirmation=server_confirms.get(td["name"], False),
                    server_name=server_name,
                    _client_manager=mcp_manager,
                ))
                logger.debug(f"Registered MCP tool: {server_name}/{td['name']}")

        self._frozen = True
        logger.info(
            f"ToolRegistry frozen: {len(self._local_tools)} local + "
            f"{len(self._mcp_tools)} MCP = {len(self._local_tools) + len(self._mcp_tools)} total"
        )

    def get_all_tools(self) -> list[ToolProtocol]:
        """返回 local + available MCP 工具，本地工具同名时屏蔽 MCP 版本"""
        local_names = {t.name for t in self._local_tools}
        tools: list[ToolProtocol] = list(self._local_tools)
        for mcp_tool in self._mcp_tools:
            if mcp_tool.name in local_names:
                logger.info(f"MCP tool {mcp_tool.name} shadowed by local version, skipping")
                continue
            if mcp_tool.available:
                tools.append(mcp_tool)
            else:
                logger.warning(f"MCP tool {mcp_tool.name} unavailable, skipping")
        return tools

    def get_openai_schemas(self) -> list[dict]:
        """返回所有工具的 OpenAI function calling schema"""
        return [t.to_openai_schema() for t in self.get_all_tools()]

    async def shutdown(self):
        """关闭 MCP 连接（通过 MCPClientManager）"""
        if self._mcp_tools:
            manager = self._mcp_tools[0]._client_manager
            await manager.shutdown()
        self._frozen = False
        self._mcp_tools.clear()
        logger.info("ToolRegistry shutdown: MCP connections closed")
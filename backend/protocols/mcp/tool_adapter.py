"""
ToolProtocol + MCPTool — 工具接口协议与 MCP 工具适配器

ToolProtocol: Protocol class，让 Tool 和 MCPTool 满足同一接口（无需继承）
MCPTool: 将 MCP Server 的远程工具适配为 ToolProtocol 接口
"""
from typing import Protocol, Any, Awaitable
from dataclasses import dataclass


class ToolProtocol(Protocol):
    """工具接口协议 — Tool 和 MCPTool 共同满足（structural subtyping）"""
    name: str
    description: str
    parameters: dict
    require_confirmation: bool

    async def handler(self, **kwargs) -> dict: ...
    def to_openai_schema(self) -> dict: ...


@dataclass
class MCPTool:
    """将 MCP Server 工具适配为 ToolProtocol 接口

    核心职责：
    1. 通过 MCPClientManager 调用远程 MCP Server 工具
    2. 主进程侧后处理：消费 _download_info，调用 _record_download
    3. 移除 _download_info 后返回干净结果（LLM 永远看不到）
    """
    name: str
    description: str
    parameters: dict
    require_confirmation: bool
    server_name: str
    _client_manager: Any  # MCPClientManager，用 Any 避免循环引用
    available: bool = True

    async def handler(self, **kwargs) -> dict:
        from ...agent.tools import _current_user_id, _record_download

        result = await self._client_manager.call_tool(
            self.server_name, self.name, kwargs,
        )

        # §4.4 主进程侧后处理：消费 _download_info
        if isinstance(result, dict) and "_download_info" in result:
            user_id = _current_user_id.get()  # 主进程 ContextVar 有效 ✅
            info = result.pop("_download_info")
            _record_download(
                user_id,
                info["filename"],
                info["filepath"],
                info["size_bytes"],
            )

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
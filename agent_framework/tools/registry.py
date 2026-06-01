"""
工具注册表 — 管理所有可用工具

核心概念：每个工具 = 一个函数 + 一段 OpenAI function calling 格式的描述
"""
from typing import Callable


class ToolRegistry:
    """
    工具注册表：注册工具、列出定义、按名称执行。

    用法：
        registry = ToolRegistry()
        registry.register("read_file", "读取文件", {...}, read_file_func)
        definitions = registry.get_definitions()  # 传给 LLM
        result = registry.execute("read_file", {"path": "/tmp/test.txt"})
    """

    def __init__(self):
        self._tools: dict[str, dict] = {}  # name -> {definition, func}

    def register(self, name: str, description: str, parameters: dict, func: Callable):
        """
        注册一个工具。

        name: 工具名（LLM 用这个名字调用）
        description: 工具用途（LLM 读这个决定要不要用）
        parameters: JSON Schema 格式的参数定义
        func: 实际执行的函数，接收 **kwargs，返回 str
        """
        self._tools[name] = {
            "definition": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": parameters,
                        "required": list(parameters.keys()),
                    }
                }
            },
            "func": func
        }

    def get_definitions(self, names: list[str] = None) -> list[dict]:
        """
        返回工具定义列表（OpenAI function calling 格式）。
        不传 names 则返回全部。
        """
        if names:
            return [self._tools[n]["definition"] for n in names if n in self._tools]
        return [t["definition"] for t in self._tools.values()]

    def execute(self, name: str, arguments: dict) -> str:
        """执行指定工具，返回结果字符串"""
        if name not in self._tools:
            return f"错误：未知工具 '{name}'"
        try:
            func = self._tools[name]["func"]
            result = func(**arguments)
            # 截断过长结果，防止占满上下文窗口
            if len(result) > 8000:
                result = result[:8000] + "\n...(结果过长已截断)"
            return result
        except Exception as e:
            return f"工具执行失败：{type(e).__name__}: {e}"

    def list_tools(self) -> list[str]:
        """列出所有已注册的工具名"""
        return list(self._tools.keys())

"""
MemoryTool — LLM 通过 function calling 调用的记忆工具。
action: add | search | list | forget
"""
from ..manager import MemoryManager


class MemoryTool:
    """挂载到 Agent 工具列表的记忆工具。"""

    def __init__(self, manager: MemoryManager):
        self._mgr = manager

    async def execute(self, action: str, **kwargs) -> dict:
        """
        执行记忆操作，返回结构化结果。

        action:
          - add:     kwargs: content (str), memory_type (str, optional),
                     importance (float, optional)
          - search:  kwargs: query (str), limit (int, optional)
          - list:    kwargs: limit (int, optional)
          - forget:  kwargs: strategy (str, optional)
        """
        try:
            if action == "add":
                content = kwargs.get("content", "")
                if not content:
                    return {"error": "content is required for add action"}
                memory_type = kwargs.get("memory_type", "episodic")
                importance = kwargs.get("importance", None)
                metadata = kwargs.get("metadata", None)
                mid = await self._mgr.add(
                    content=content,
                    memory_type=memory_type,
                    importance=importance,
                    metadata=metadata,
                )
                return {
                    "action": "add",
                    "memory_id": mid,
                    "memory_type": memory_type,
                    "status": "saved",
                }

            elif action == "search":
                query = kwargs.get("query", "")
                if not query:
                    return {"error": "query is required for search action"}
                limit = kwargs.get("limit", 5)
                memory_types = kwargs.get("memory_types", ["episodic", "semantic"])
                items = await self._mgr.search(
                    query=query,
                    memory_types=memory_types,
                    limit=limit,
                )
                return {
                    "action": "search",
                    "query": query,
                    "memories": [item.to_dict() for item in items],
                    "count": len(items),
                }

            elif action == "list":
                limit = kwargs.get("limit", 10)
                all_memories = await self._mgr.search(
                    query="", memory_types=["episodic", "semantic"], limit=limit
                )
                return {
                    "action": "list",
                    "memories": [m.to_dict() for m in all_memories],
                    "count": len(all_memories),
                }

            elif action == "forget":
                strategy = kwargs.get("strategy", "importance_based")
                count = await self._mgr.forget(strategy=strategy)
                return {
                    "action": "forget",
                    "strategy": strategy,
                    "deleted_count": count,
                }

            else:
                return {"error": f"Unknown action: {action}. Supported: add, search, list, forget"}

        except Exception as e:
            return {"error": str(e), "action": action}

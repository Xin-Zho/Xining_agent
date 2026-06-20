"""
WorkingMemory — 会话级短期记忆，纯内存 dict + deque，TTL 自动过期。
不持久化，会话结束即清。v1 限定单进程；多进程需 Redis。
"""
import time
import uuid
from collections import deque
from typing import Optional

from .base import BaseMemory, MemoryItem, MemoryType


class WorkingMemory(BaseMemory):
    """会话工作区：dict[user_id, deque[MemoryItem]]，FIFO + TTL。"""

    def __init__(self, max_items: int = 20, ttl_sec: int = 1800):
        self._store: dict[int, deque[MemoryItem]] = {}
        self._max_items = max_items
        self._ttl_sec = ttl_sec

    async def add(self, item: MemoryItem) -> str:
        """添加一条 working memory，自动设置 TTL 和 FIFO 挤出。"""
        import time
        if not item.id:
            item.id = str(uuid.uuid4())
        if item.memory_type != MemoryType.WORKING:
            item.memory_type = MemoryType.WORKING
        if item.ttl is None:
            item.ttl = time.time() + self._ttl_sec
        item.created_at = time.time()

        if item.user_id not in self._store:
            self._store[item.user_id] = deque()

        self._store[item.user_id].append(item)

        # FIFO 挤出
        while len(self._store[item.user_id]) > self._max_items:
            self._store[item.user_id].popleft()

        return item.id

    async def search(self, query_embedding: list[float], limit: int = 5) -> list[MemoryItem]:
        """Working memory 不做向量检索，返回最近的 limit 条未过期记忆。"""
        import time
        now = time.time()
        results = []
        for uid, items in self._store.items():
            for item in items:
                if not item.is_expired():
                    results.append(item)
        # 按时间倒序
        results.sort(key=lambda x: x.created_at, reverse=True)
        return results[:limit]

    def get_by_user(self, user_id: int, limit: int = 10) -> list[MemoryItem]:
        """获取指定用户的 working memory（非向量检索，直接返回）。"""
        items = self._store.get(user_id, deque())
        results = [item for item in items if not item.is_expired()]
        results.sort(key=lambda x: x.created_at, reverse=True)
        return results[:limit]

    async def delete(self, item_id: str) -> bool:
        """删除指定记忆"""
        for uid, items in self._store.items():
            for item in items:
                if item.id == item_id:
                    items.remove(item)
                    return True
        return False

    async def clear(self, user_id: int) -> int:
        """清空指定用户"""
        if user_id in self._store:
            count = len(self._store[user_id])
            del self._store[user_id]
            return count
        return 0

    def _evict_expired(self, user_id: int):
        """惰性清理过期记忆"""
        if user_id not in self._store:
            return
        self._store[user_id] = deque(
            item for item in self._store[user_id] if not item.is_expired()
        )

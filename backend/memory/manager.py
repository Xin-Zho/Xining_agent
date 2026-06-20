"""
MemoryManager — 统一调度器，Agent Engine 的唯一入口。
协调 Working + Episodic + Semantic 三种记忆，提供 add/search/consolidate/forget。
"""
import time
import uuid

from .base import MemoryItem, MemoryType, MemoryConfig, BaseMemory
from .working import WorkingMemory
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .embedding import LocalEmbedding


class MemoryManager:
    """记忆系统统一入口。每个 Agent 任务创建一个实例（user_id 绑定）。"""

    def __init__(self, user_id: int, config: MemoryConfig = None,
                 embedding: LocalEmbedding = None, working: WorkingMemory = None,
                 episodic: EpisodicMemory = None, semantic: SemanticMemory = None):
        self.user_id = user_id
        self.config = config or MemoryConfig()
        self._emb = embedding or LocalEmbedding()
        self._working = working or WorkingMemory(
            max_items=self.config.working_max_items,
            ttl_sec=self.config.working_ttl_sec,
        )
        self._episodic = episodic or EpisodicMemory(self._emb, self.config)
        self._semantic = semantic or SemanticMemory(self._emb, self.config)

    # ── 写入 ──────────────────────────────────────────────

    async def add(self, content: str, memory_type: str = "episodic",
                  importance: float = None, metadata: dict = None) -> str:
        """
        添加记忆。如果未指定 importance，用启发式默认值。
        如果未指定 memory_type，默认为 episodic。
        metadata: 传入 None 时内部设为 {}。
        """
        if metadata is None:
            metadata = {}

        if importance is None:
            importance = 0.5

        try:
            mtype = MemoryType(memory_type)
        except ValueError:
            mtype = MemoryType.EPISODIC

        item = MemoryItem(
            id=str(uuid.uuid4()),
            user_id=self.user_id,
            content=content,
            memory_type=mtype,
            importance=importance,
            created_at=time.time(),
            last_accessed_at=time.time(),
            metadata=metadata,
        )

        if mtype == MemoryType.WORKING:
            return await self._working.add(item)
        elif mtype == MemoryType.EPISODIC:
            return await self._episodic.add(item)
        elif mtype == MemoryType.SEMANTIC:
            return await self._semantic.add(item)
        else:
            return await self._episodic.add(item)

    # ── 检索 ──────────────────────────────────────────────

    async def search(self, query: str, memory_types: list[str] = None,
                     limit: int = 5) -> list[MemoryItem]:
        """
        混合检索：embed(query) → ChromaDB 多 collection 查询 → 合并排序。
        memory_types: 默认 ["episodic", "semantic"]。
        """
        if memory_types is None:
            memory_types = ["episodic", "semantic"]

        query_vec = self._emb.embed([query])[0]
        all_items = []

        for mt_str in memory_types:
            try:
                mt = MemoryType(mt_str)
            except ValueError:
                continue

            if mt == MemoryType.EPISODIC:
                results = await self._episodic.search(query_vec, limit, self.user_id)
            elif mt == MemoryType.SEMANTIC:
                results = await self._semantic.search(query_vec, limit, self.user_id)
            elif mt == MemoryType.WORKING:
                results = self._working.get_by_user(self.user_id, limit)
            else:
                continue

            for item in results:
                item.touch()
                all_items.append((item.importance, item))

        # 去重 + 排序
        seen = set()
        unique = []
        all_items.sort(key=lambda x: x[0], reverse=True)
        for score, item in all_items:
            if item.id not in seen:
                seen.add(item.id)
                unique.append(item)

        return unique[:limit]

    async def get_working_context(self) -> list[MemoryItem]:
        """获取当前 Working memory（注入 prompt 用）。"""
        return self._working.get_by_user(self.user_id, limit=self.config.max_memory_items)

    # ── 管理 ──────────────────────────────────────────────

    async def consolidate(self, from_type: str = "episodic",
                          to_type: str = "semantic",
                          threshold: float = None) -> int:
        """
        将高重要性的 Episodic 记忆固化为 Semantic。
        返回固化的条数。
        """
        if threshold is None:
            threshold = self.config.episodic_consolidation_threshold

        from ..database import get_db
        conn = get_db("memory")
        rows = conn.execute(
            """SELECT id, user_id, content, importance, created_at, access_count, metadata
               FROM episodic_memory
               WHERE user_id = ? AND importance >= ?
               ORDER BY importance DESC""",
            (self.user_id, threshold),
        ).fetchall()
        conn.close()

        count = 0
        for row in rows:
            # 检查是否已固化过
            conn2 = get_db("memory")
            already = conn2.execute(
                "SELECT 1 FROM semantic_memory WHERE source_episodic_id = ?",
                (row["id"],),
            ).fetchone()
            conn2.close()
            if already:
                continue

            import json
            item = MemoryItem(
                id=str(uuid.uuid4()),
                user_id=row["user_id"],
                content=row["content"],
                memory_type=MemoryType.SEMANTIC,
                importance=row["importance"],
                created_at=row["created_at"],
                last_accessed_at=time.time(),
                access_count=row["access_count"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
            item.metadata["source_episodic_id"] = row["id"]
            await self._semantic.add(item)
            count += 1

        return count

    async def forget(self, strategy: str = "importance_based") -> int:
        """
        执行遗忘策略。
        - importance_based: 遗忘 Episodic 中低价值长期未访问的
        - time_based: 清理 Working 中过期的
        """
        count = 0
        if strategy in ("importance_based", "time_based"):
            count += await self._episodic.forget(
                self.user_id,
                importance_max=self.config.episodic_forget_importance_max,
                days_threshold=self.config.episodic_forget_days,
            )
        return count

    async def get_context_for_prompt(self, query: str = "") -> str:
        """
        把记忆格式化为一段上下文文本，注入 System Prompt。
        优先 Semantic（偏好/知识），其次 Episodic（最近事件）。
        """
        items = await self.search(query=query or "用户偏好 历史任务",
                                  memory_types=["semantic", "episodic"],
                                  limit=self.config.max_memory_items)
        if not items:
            return ""

        lines = ["[Agent 记忆]"]
        for item in items:
            label = "偏好" if item.memory_type == MemoryType.SEMANTIC else "事件"
            lines.append(f"- [{label}] {item.content[:200]}")
        return "\n".join(lines)

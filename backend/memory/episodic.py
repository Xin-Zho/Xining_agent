"""
EpisodicMemory — 任务事件记忆，SQLite(原文) + ChromaDB(向量)。
记录每次 Agent 任务完成的总结，importance > 阈值的可固化为 Semantic。
"""
import json
import time
import uuid

from .base import BaseMemory, MemoryItem, MemoryType, MemoryConfig
from .embedding import LocalEmbedding


class EpisodicMemory(BaseMemory):
    """情景记忆：SQLite source of truth + ChromaDB 向量检索。"""

    def __init__(self, embedding: LocalEmbedding, config: MemoryConfig = None,
                 chroma_client=None):
        self._emb = embedding
        self._config = config or MemoryConfig()
        self._chroma = chroma_client

    def _get_collection(self):
        """懒初始化 ChromaDB collection"""
        if self._chroma is None:
            import chromadb
            import os
            persist_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                self._config.chroma_persist_dir
            )
            os.makedirs(persist_dir, exist_ok=True)
            self._chroma = chromadb.PersistentClient(path=persist_dir)

        return self._chroma.get_or_create_collection(
            name="episodic_all",
            metadata={"hnsw:space": "cosine"},
        )

    def _get_db(self):
        from ..database import get_db
        return get_db("memory")

    async def add(self, item: MemoryItem) -> str:
        """写入 SQLite + ChromaDB。返回 memory_id。"""
        if not item.id:
            item.id = str(uuid.uuid4())
        now = time.time()
        item.memory_type = MemoryType.EPISODIC
        item.created_at = item.created_at or now
        item.last_accessed_at = item.last_accessed_at or now

        # SQLite
        conn = self._get_db()
        conn.execute(
            """INSERT INTO episodic_memory (id, user_id, content, importance,
               created_at, last_accessed_at, access_count, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (item.id, item.user_id, item.content, item.importance,
             item.created_at, item.last_accessed_at, item.access_count,
             json.dumps(item.metadata, ensure_ascii=False)),
        )
        conn.commit()
        conn.close()

        # ChromaDB
        embedding_vec = self._emb.embed([item.content])[0]
        collection = self._get_collection()
        collection.upsert(
            ids=[item.id],
            embeddings=[embedding_vec],
            metadatas=[{
                "user_id": item.user_id,
                "importance": item.importance,
                "created_at": item.created_at,
            }],
        )

        return item.id

    async def search(self, query_embedding: list[float], limit: int = 5,
                     user_id: int = None) -> list[MemoryItem]:
        """向量检索 + 评分公式排序。"""
        collection = self._get_collection()
        where_filter = {"user_id": user_id} if user_id is not None else {}

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit * 3,
            where=where_filter,
            include=["metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        items = []
        now = time.time()
        for i, mid in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            similarity = 1.0 - distance

            importance = metadata.get("importance", 0.5)
            created_at = metadata.get("created_at", now)

            # 时间近因性：30天后衰减到 0
            age_days = (now - created_at) / 86400.0
            recency = max(0.0, 1.0 - age_days / 30.0)

            # 检索公式: (向量相似度 × 0.8 + 时间近因性 × 0.2) × (0.8 + importance × 0.4)
            score = (0.8 * similarity + 0.2 * recency) * (0.8 + importance * 0.4)

            # 从 SQLite 取完整记录
            item = self._get_by_id(mid)
            if item:
                item.last_accessed_at = now
                item.access_count += 1
                items.append((score, item))

        items.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in items[:limit]]

    def _get_by_id(self, item_id: str):
        """从 SQLite 读取单条记忆"""
        conn = self._get_db()
        row = conn.execute(
            "SELECT * FROM episodic_memory WHERE id = ?", (item_id,)
        ).fetchone()
        conn.close()
        if not row:
            return None
        return MemoryItem(
            id=row["id"],
            user_id=row["user_id"],
            content=row["content"],
            memory_type=MemoryType.EPISODIC,
            importance=row["importance"],
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            access_count=row["access_count"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    async def delete(self, item_id: str) -> bool:
        """同时删除 SQLite + ChromaDB"""
        conn = self._get_db()
        conn.execute("DELETE FROM episodic_memory WHERE id = ?", (item_id,))
        conn.commit()
        deleted = conn.total_changes > 0
        conn.close()

        try:
            collection = self._get_collection()
            collection.delete(ids=[item_id])
        except Exception:
            pass

        return deleted

    async def clear(self, user_id: int) -> int:
        """清空指定用户的情景记忆"""
        conn = self._get_db()
        count = conn.execute(
            "SELECT COUNT(*) as c FROM episodic_memory WHERE user_id = ?",
            (user_id,),
        ).fetchone()["c"]
        conn.execute("DELETE FROM episodic_memory WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
        return count

    async def forget(self, user_id: int, importance_max: float = 0.3,
                     days_threshold: int = 7) -> int:
        """遗忘低价值+长期未访问的记忆"""
        now = time.time()
        cutoff = now - days_threshold * 86400.0

        conn = self._get_db()
        rows = conn.execute(
            """SELECT id FROM episodic_memory
               WHERE user_id = ? AND importance < ? AND last_accessed_at < ?""",
            (user_id, importance_max, cutoff),
        ).fetchall()
        conn.close()

        for row in rows:
            await self.delete(row["id"])

        return len(rows)

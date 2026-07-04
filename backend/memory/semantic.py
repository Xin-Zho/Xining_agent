"""
SemanticMemory — 长期知识/偏好记忆，SQLite(原文) + ChromaDB(向量)。
从 Episodic 固化的规则、用户偏好、领域概念。不自动遗忘。
"""
import json
import time
import uuid

from .base import BaseMemory, MemoryItem, MemoryType, MemoryConfig
from .embedding import LocalEmbedding


class SemanticMemory(BaseMemory):
    """语义记忆：不设 TTL，不自动遗忘，仅显式删除。"""

    def __init__(self, embedding: LocalEmbedding, config: MemoryConfig = None,
                 chroma_client=None):
        self._emb = embedding
        self._config = config or MemoryConfig()
        self._chroma = chroma_client

    def _get_collection(self):
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
            name="semantic_all",
            metadata={"hnsw:space": "cosine"},
        )

    def _get_db(self):
        from ..database import get_db
        return get_db("memory")

    async def add(self, item: MemoryItem) -> str:
        """写入 SQLite + ChromaDB"""
        if not item.id:
            item.id = str(uuid.uuid4())
        now = time.time()
        item.memory_type = MemoryType.SEMANTIC
        item.created_at = item.created_at or now
        item.last_accessed_at = item.last_accessed_at or now

        conn = self._get_db()
        source = item.metadata.get("source_episodic_id", None)
        conn.execute(
            """INSERT INTO semantic_memory (id, user_id, content, importance,
               created_at, last_accessed_at, access_count, metadata, source_episodic_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (item.id, item.user_id, item.content, item.importance,
             item.created_at, item.last_accessed_at, item.access_count,
             json.dumps(item.metadata, ensure_ascii=False), source),
        )
        conn.commit()
        conn.close()

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
        """向量检索 + 归一化评分公式。"""
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

            item = self._get_by_id(mid)
            if not item:
                continue

            # 检索公式: 0.7 × sim + 0.1 × freq_norm + 0.2 × importance
            # freq_norm = min(access_count / 10.0, 1.0)
            freq_norm = min(item.access_count / 10.0, 1.0)
            score = 0.7 * similarity + 0.1 * freq_norm + 0.2 * importance

            item.last_accessed_at = now
            item.access_count += 1
            items.append((score, item))

        items.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in items[:limit]]

    def _get_by_id(self, item_id: str):
        conn = self._get_db()
        row = conn.execute(
            "SELECT * FROM semantic_memory WHERE id = ?", (item_id,)
        ).fetchone()
        conn.close()
        if not row:
            return None
        return MemoryItem(
            id=row["id"],
            user_id=row["user_id"],
            content=row["content"],
            memory_type=MemoryType.SEMANTIC,
            importance=row["importance"],
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            access_count=row["access_count"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    async def delete(self, item_id: str) -> bool:
        conn = self._get_db()
        conn.execute("DELETE FROM semantic_memory WHERE id = ?", (item_id,))
        conn.commit()
        deleted = conn.total_changes > 0
        conn.close()
        try:
            self._get_collection().delete(ids=[item_id])
        except Exception:
            pass
        return deleted

    async def clear(self, user_id: int) -> int:
        conn = self._get_db()
        count = conn.execute(
            "SELECT COUNT(*) as c FROM semantic_memory WHERE user_id = ?",
            (user_id,),
        ).fetchone()["c"]
        conn.execute("DELETE FROM semantic_memory WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
        return count

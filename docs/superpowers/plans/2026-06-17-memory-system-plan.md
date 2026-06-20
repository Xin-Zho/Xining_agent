# Memory System Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the project memory system from single key-value SQLite storage to a 3-layer memory architecture (Working + Episodic + Semantic) with ChromaDB vector search and local BGE embeddings.

**Architecture:** Build from bottom up — embedding layer first, then data structures, then individual memory types, then the orchestrating MemoryManager, then ContextBuilder, then tool/integration layer. Each memory type is independently testable before wiring together.

**Tech Stack:** Python dataclasses, ChromaDB (embedded), sentence-transformers (BGE-small-zh), SQLite (existing), asyncio

**Dependency chain:**
```
embedding.py → base.py → database.py → working.py → episodic.py → semantic.py → manager.py → context.py → memory_tool.py → engine.py / tools.py / server.py
```

---

### Task 1: Install dependencies and create LocalEmbedding

**Files:**
- Create: `backend/memory/embedding.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add new dependencies to requirements.txt**

```bash
echo "" >> requirements.txt
echo "# Vector DB + Embeddings" >> requirements.txt
echo "chromadb>=0.5.0" >> requirements.txt
echo "sentence-transformers>=3.0.0" >> requirements.txt
```

- [ ] **Step 2: Install dependencies**

```bash
pip install chromadb sentence-transformers
```

Expected: Both packages install successfully. Verify with `pip show chromadb sentence-transformers`.

- [ ] **Step 3: Create backend/memory/embedding.py**

Write the file:

```python
"""
LocalEmbedding — BGE 模型本地加载，text → 512-dim 归一化向量
零外部 API 依赖，首次使用自动下载模型 (~100MB)，后续从缓存加载。
"""
import numpy as np


class LocalEmbedding:
    """BGE-small-zh 嵌入模型封装，返回归一化向量供 ChromaDB 使用。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        from sentence_transformers import SentenceTransformer
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """将文本列表转为向量列表，L2 归一化。返回 list[list[float]] 兼容 ChromaDB。"""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    @property
    def dim(self) -> int:
        """向量维度"""
        return self._model.get_sentence_embedding_dimension()
```

- [ ] **Step 4: Verify embedding works**

Create and run a quick smoke test (delete after):

```bash
cd D:\agent_learning && python -c "
from backend.memory.embedding import LocalEmbedding
e = LocalEmbedding()
vec = e.embed(['你好世界', 'Hello World'])
print(f'dim={e.dim}, len={len(vec)}, vec[0][:3]={vec[0][:3]}')
assert e.dim == 512, f'Expected 512, got {e.dim}'
assert len(vec) == 2
assert len(vec[0]) == 512
print('OK')
"
```

Expected: `dim=512, len=2, vec[0][:3]=[...], OK`

- [ ] **Step 5: Commit**

```bash
git add requirements.txt backend/memory/embedding.py
git commit -m "feat: add LocalEmbedding — BGE-small-zh, text→512-dim vector"
```

---

### Task 2: Create MemoryItem, MemoryConfig, MemoryType, BaseMemory

**Files:**
- Create: `backend/memory/base.py`

- [ ] **Step 1: Create backend/memory/base.py**

Write the file:

```python
"""
Base classes for the memory system.
MemoryItem — standardized data structure
MemoryType — enum for working/episodic/semantic
MemoryConfig — all tunable parameters
BaseMemory — abstract interface for all memory types
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MemoryType(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass
class MemoryItem:
    """标准化记忆数据结构，所有记忆类型共用。"""
    id: str                          # uuid
    user_id: int                     # 多用户隔离
    content: str                     # 记忆文本
    memory_type: MemoryType          # WORKING | EPISODIC | SEMANTIC
    importance: float = 0.5          # 0.0~1.0，LLM 打分
    created_at: float = 0.0          # unix timestamp（填入时设为 time.time()）
    last_accessed_at: float = 0.0    # 最近检索时间
    access_count: int = 0            # 检索次数
    ttl: Optional[float] = None      # Working 过期时间（unix timestamp）
    metadata: dict = field(default_factory=dict)  # 扩展字段
    embedding: Optional[list[float]] = None       # 向量缓存（仅 ChromaDB）

    def is_expired(self) -> bool:
        """检查 Working memory TTL 是否过期"""
        import time
        if self.ttl is None:
            return False
        return time.time() > self.ttl

    def touch(self):
        """更新访问记录"""
        import time
        self.last_accessed_at = time.time()
        self.access_count += 1

    def to_dict(self) -> dict:
        """转为字典，排除 embedding"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "importance": self.importance,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "metadata": self.metadata,
        }


@dataclass
class MemoryConfig:
    """全局配置，所有可调参数集中管理。"""
    # Working memory
    working_max_items: int = 20
    working_ttl_sec: int = 1800       # 30 分钟

    # Episodic memory
    episodic_consolidation_threshold: float = 0.7
    episodic_forget_days: int = 7
    episodic_forget_importance_max: float = 0.3

    # Semantic memory
    semantic_no_auto_forget: bool = True

    # Context builder
    hot_window_sec: int = 180          # 3 分钟
    history_budget_tokens: int = 90_000
    memory_context_max_ratio: float = 0.15
    keep_min_hot_rounds: int = 2
    max_memory_items: int = 5

    # ChromaDB
    chroma_persist_dir: str = "data/chroma"


class BaseMemory(ABC):
    """所有记忆类型的抽象基类。子类必须实现 add / search / delete / clear。"""

    @abstractmethod
    async def add(self, item: MemoryItem) -> str:
        """添加一条记忆，返回 memory_id"""
        ...

    @abstractmethod
    async def search(self, query_embedding: list[float], limit: int = 5) -> list[MemoryItem]:
        """向量检索，返回 top-N MemoryItem"""
        ...

    @abstractmethod
    async def delete(self, item_id: str) -> bool:
        """删除一条记忆，返回是否成功"""
        ...

    @abstractmethod
    async def clear(self, user_id: int) -> int:
        """清空指定用户的所有此类记忆，返回删除条数"""
        ...
```

- [ ] **Step 2: Verify dataclass and enum**

```bash
cd D:\agent_learning && python -c "
from backend.memory.base import MemoryItem, MemoryConfig, MemoryType, BaseMemory
import uuid, time

# Test MemoryItem
item = MemoryItem(
    id=str(uuid.uuid4()), user_id=1, content='test',
    memory_type=MemoryType.EPISODIC, importance=0.8,
    created_at=time.time(), last_accessed_at=time.time(),
)
assert item.memory_type == MemoryType.EPISODIC
assert item.importance == 0.8
assert item.access_count == 0
assert item.metadata == {}

# Test mutable default — each instance gets its own dict
item2 = MemoryItem(id='x', user_id=1, content='x', memory_type=MemoryType.WORKING)
item2.metadata['key'] = 'val'
item3 = MemoryItem(id='y', user_id=1, content='y', memory_type=MemoryType.WORKING)
assert item3.metadata == {}  # not {'key': 'val'}!

# Test is_expired
item.ttl = time.time() - 1   # 1 second ago
assert item.is_expired()
item.ttl = time.time() + 999
assert not item.is_expired()

# Test touch
item2.touch()
assert item2.access_count == 1

# Test to_dict
d = item.to_dict()
assert 'embedding' not in d  # excluded from dict
assert d['memory_type'] == 'episodic'

# Test MemoryConfig defaults
cfg = MemoryConfig()
assert cfg.working_max_items == 20
assert cfg.hot_window_sec == 180

print('OK')
"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/memory/base.py
git commit -m "feat: add MemoryItem, MemoryConfig, MemoryType, BaseMemory base classes"
```

---

### Task 3: Add episodic_memory and semantic_memory tables to database.py

**Files:**
- Modify: `backend/database.py`

- [ ] **Step 1: Add table creation to init_db() in database.py**

Read the current `init_db()` function. Add the following block after the existing `memory.db` section (after line 112, where `m.commit(); m.close()` appears):

```python
    # memory.db — episodic + semantic tables (NEW)
    m2 = get_db("memory")
    m2.executescript("""
        CREATE TABLE IF NOT EXISTS episodic_memory (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            importance REAL DEFAULT 0.5,
            created_at REAL NOT NULL DEFAULT (unixepoch()),
            last_accessed_at REAL NOT NULL DEFAULT (unixepoch()),
            access_count INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_episodic_user ON episodic_memory(user_id, created_at);

        CREATE TABLE IF NOT EXISTS semantic_memory (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            importance REAL DEFAULT 0.5,
            created_at REAL NOT NULL DEFAULT (unixepoch()),
            last_accessed_at REAL NOT NULL DEFAULT (unixepoch()),
            access_count INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}',
            source_episodic_id TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_semantic_user ON semantic_memory(user_id, created_at);
    """)
    m2.commit(); m2.close()
```

- [ ] **Step 2: Verify tables exist**

```bash
cd D:\agent_learning && python -c "
from backend.database import init_db, get_db
init_db()
conn = get_db('memory')
tables = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()
conn.close()
table_names = [t['name'] for t in tables]
assert 'episodic_memory' in table_names, f'Missing episodic_memory, got {table_names}'
assert 'semantic_memory' in table_names, f'Missing semantic_memory, got {table_names}'
print(f'Tables OK: {table_names}')
"
```

Expected: `Tables OK: ['agent_memory', 'downloads', 'episodic_memory', 'semantic_memory']`

- [ ] **Step 3: Commit**

```bash
git add backend/database.py
git commit -m "feat: add episodic_memory and semantic_memory tables to memory.db"
```

---

### Task 4: Create WorkingMemory

**Files:**
- Create: `backend/memory/working.py`

- [ ] **Step 1: Create backend/memory/working.py**

Write the file:

```python
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
        user_items = self._store.get(0, deque())  # user_id 由调用方在 embedding 层处理
        if not user_items:
            # 兼容：直接遍历所有用户找（单用户场景）
            results = []
            for uid, items in self._store.items():
                for item in items:
                    if not item.is_expired():
                        results.append(item)
            # 按时间倒序
            results.sort(key=lambda x: x.created_at, reverse=True)
            return results[:limit]

        results = [item for item in user_items if not item.is_expired()]
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
```

- [ ] **Step 2: Test WorkingMemory**

```bash
cd D:\agent_learning && python -c "
import asyncio, uuid, time
from backend.memory.working import WorkingMemory
from backend.memory.base import MemoryItem, MemoryType

async def test():
    wm = WorkingMemory(max_items=5, ttl_sec=2)

    # Test add
    item = MemoryItem(id=str(uuid.uuid4()), user_id=1, content='test1',
                      memory_type=MemoryType.WORKING)
    mid = await wm.add(item)
    assert mid == item.id

    # Test get_by_user
    items = wm.get_by_user(1)
    assert len(items) == 1
    assert items[0].content == 'test1'

    # Test TTL expiry
    await asyncio.sleep(2.1)
    items = wm.get_by_user(1)
    assert len(items) == 0, f'Expected 0 after TTL, got {len(items)}'

    # Test FIFO overflow
    for i in range(7):
        it = MemoryItem(id=str(uuid.uuid4()), user_id=1, content=f'm{i}',
                        memory_type=MemoryType.WORKING, ttl=time.time()+999)
        await wm.add(it)
    items = wm.get_by_user(1)
    assert len(items) == 5, f'Expected 5 after FIFO, got {len(items)}'
    # First 2 should be evicted
    contents = [i.content for i in items]
    assert 'm0' not in contents
    assert 'm1' not in contents
    assert 'm6' in contents

    # Test clear
    count = await wm.clear(1)
    assert count == 5
    assert len(wm.get_by_user(1)) == 0

    print('OK')

asyncio.run(test())
"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/memory/working.py
git commit -m "feat: add WorkingMemory — session-level, in-memory dict + TTL + FIFO"
```

---

### Task 5: Create EpisodicMemory

**Files:**
- Create: `backend/memory/episodic.py`

- [ ] **Step 1: Create backend/memory/episodic.py**

Write the file:

```python
"""
EpisodicMemory — 任务事件记忆，SQLite(原文) + ChromaDB(向量)。
记录每次 Agent 任务完成的总结，importance > 阈值的可固化为 Semantic。
"""
import json
import time
import uuid
import math

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
        import time
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
            n_results=limit * 3,  # 多取一些，排序后截断
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
            similarity = 1.0 - distance  # cosine distance → similarity

            importance = metadata.get("importance", 0.5)
            created_at = metadata.get("created_at", now)

            # 时间近因性：7天内的加分
            age_days = (now - created_at) / 86400.0
            recency = max(0.0, 1.0 - age_days / 30.0)  # 30天后衰减到 0

            # 检索公式: (向量相似度 × 0.8 + 时间近因性 × 0.2) × (0.8 + importance × 0.4)
            score = (0.8 * similarity + 0.2 * recency) * (0.8 + importance * 0.4)

            # 从 SQLite 取完整记录
            item = self._get_by_id(mid)
            if item:
                item.embedding = query_embedding  # not stored, but set for upstream
                item.last_accessed_at = now
                item.access_count += 1
                items.append((score, item))

        items.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in items[:limit]]

    def _get_by_id(self, item_id: str) -> MemoryItem | None:
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
            pass  # ChromaDB 删除失败不阻断

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

        # ChromaDB 不支持按 metadata 批量删除，做懒处理：标记后下次搜索过滤
        return count

    async def forget(self, user_id: int, importance_max: float = 0.3,
                     days_threshold: int = 7) -> int:
        """遗忘低价值+长期未访问的记忆"""
        import time
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
```

- [ ] **Step 2: Smoke test EpisodicMemory (SQLite only, ChromaDB requires init_db)**

```bash
cd D:\agent_learning && python -c "
import asyncio, uuid
from backend.database import init_db
from backend.memory.base import MemoryItem, MemoryType
from backend.memory.embedding import LocalEmbedding
from backend.memory.episodic import EpisodicMemory

async def test():
    init_db()
    emb = LocalEmbedding()
    ep = EpisodicMemory(emb)

    # Add
    item = MemoryItem(id=str(uuid.uuid4()), user_id=1, content='用户问了股票',
                      memory_type=MemoryType.EPISODIC, importance=0.6)
    mid = await ep.add(item)
    assert mid == item.id

    # Search
    q_emb = emb.embed(['股票'])[0]
    results = await ep.search(q_emb, limit=5, user_id=1)
    assert len(results) >= 1
    assert results[0].user_id == 1

    # Delete
    ok = await ep.delete(item.id)
    assert ok

    print('OK')

asyncio.run(test())
"
```

Expected: `OK` (first run downloads BGE model if not cached)

- [ ] **Step 3: Commit**

```bash
git add backend/memory/episodic.py
git commit -m "feat: add EpisodicMemory — SQLite + ChromaDB with scoring formula"
```

---

### Task 6: Create SemanticMemory

**Files:**
- Create: `backend/memory/semantic.py`

- [ ] **Step 1: Create backend/memory/semantic.py**

Write the file:

```python
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
        import time
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

            # 从 SQLite 取 access_count
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

    def _get_by_id(self, item_id: str) -> MemoryItem | None:
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
```

- [ ] **Step 2: Smoke test SemanticMemory**

```bash
cd D:\agent_learning && python -c "
import asyncio, uuid
from backend.database import init_db
from backend.memory.base import MemoryItem, MemoryType
from backend.memory.embedding import LocalEmbedding
from backend.memory.semantic import SemanticMemory

async def test():
    init_db()
    emb = LocalEmbedding()
    sm = SemanticMemory(emb)

    # Add
    item = MemoryItem(id=str(uuid.uuid4()), user_id=1, content='用户偏好 Python',
                      memory_type=MemoryType.SEMANTIC, importance=0.9)
    mid = await sm.add(item)
    assert mid == item.id

    # Search
    q_emb = emb.embed(['编程语言偏好'])[0]
    results = await sm.search(q_emb, limit=5, user_id=1)
    assert len(results) >= 1, f'Expected >=1 results, got {len(results)}'

    # Verify scoring formula produces valid scores
    # Just ensure search doesn't crash

    # Delete
    ok = await sm.delete(item.id)
    assert ok
    print('OK')

asyncio.run(test())
"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/memory/semantic.py
git commit -m "feat: add SemanticMemory — long-term knowledge, normalized scoring, no auto-forget"
```

---

### Task 7: Create MemoryManager orchestrator

**Files:**
- Create: `backend/memory/manager.py`

- [ ] **Step 1: Create backend/memory/manager.py**

Write the file:

```python
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
        all_items: list[tuple[float, MemoryItem]] = []

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
                # 用各类型的评分公式重新算（已内嵌在各自 search 中，这里直接 append）
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
        从 SQLite 读取 importance > threshold 且未固化过的记录。
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
            # 检查是否已固化过（避免重复）
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
        - capacity_based: Working FIFO 已自动处理，这里返回 0
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
```

- [ ] **Step 2: Smoke test MemoryManager**

```bash
cd D:\agent_learning && python -c "
import asyncio
from backend.database import init_db
from backend.memory.manager import MemoryManager

async def test():
    init_db()
    mm = MemoryManager(user_id=1)

    # Add episodic
    mid1 = await mm.add('用户查询了股票行情', memory_type='episodic', importance=0.5)
    assert mid1

    # Add semantic
    mid2 = await mm.add('用户偏好简短回答', memory_type='semantic', importance=0.8)
    assert mid2

    # Search
    results = await mm.search('股票', memory_types=['episodic'], limit=3)
    assert len(results) >= 1

    results2 = await mm.search('偏好', memory_types=['semantic'], limit=3)
    assert len(results2) >= 1

    # get_context_for_prompt
    ctx = await mm.get_context_for_prompt('test')
    assert '[Agent 记忆]' in ctx
    assert '偏好' in ctx
    print(ctx)

    # consolidate
    # Add a high-importance episodic
    await mm.add('用户发现Python库X很好用', memory_type='episodic', importance=0.9)
    n = await mm.consolidate(threshold=0.7)
    print(f'Consolidated: {n}')

    # forget (low importance, should delete nothing new)
    n2 = await mm.forget(strategy='importance_based')
    print(f'Forgot: {n2}')

    print('OK')

asyncio.run(test())
"
```

Expected: `OK` with context output and consolidation count.

- [ ] **Step 3: Commit**

```bash
git add backend/memory/manager.py
git commit -m "feat: add MemoryManager — unified orchestrator for Working/Episodic/Semantic"
```

---

### Task 8: Create ContextBuilder

**Files:**
- Create: `backend/memory/context.py`

- [ ] **Step 1: Create backend/memory/context.py**

Write the file:

```python
"""
ContextBuilder — 上下文工程层。
将记忆 + 对话历史 + 任务 按时间分窗策略组装进 LLM 上下文窗口，
不超出 Token 预算。
"""
import time

from .base import MemoryConfig, MemoryItem
from ..llm_client import estimate_tokens


class ContextBuilder:
    """记忆与 LLM 之间的最后一公里。"""

    def __init__(self, config: MemoryConfig = None, llm_client=None):
        self.config = config or MemoryConfig()
        self._llm = llm_client  # 用于温窗口摘要压缩

    def build(self, system_prompt: str, memories: list[MemoryItem],
              history: list[dict], task: str,
              hot_window_sec: int = None) -> list[dict]:
        """
        组装最终 messages 列表。

        history: 带 timestamp 的消息列表，格式:
            [{"role": "user", "content": "...", "ts": 1700000000.0}, ...]
        """
        if hot_window_sec is None:
            hot_window_sec = self.config.hot_window_sec

        budget = self.config.history_budget_tokens
        messages = []

        # ① System Prompt
        system_tokens = estimate_tokens(system_prompt)
        messages.append({"role": "system", "content": system_prompt})
        used = system_tokens

        # ② 记忆上下文
        if memories:
            memory_block = self._format_memories(memories)
            mem_tokens = estimate_tokens(memory_block)
            max_mem = int(budget * self.config.memory_context_max_ratio)
            if mem_tokens > max_mem:
                memory_block = self._truncate_to_tokens(memory_block, max_mem)
            messages.append({"role": "system", "content": memory_block})
            used += estimate_tokens(memory_block)

        # ③ 切分对话历史
        now = time.time()
        hot_msgs = []
        warm_msgs = []
        for msg in history:
            ts = msg.get("ts", 0)
            if ts and (now - ts) < hot_window_sec:
                hot_msgs.append(msg)
            else:
                warm_msgs.append(msg)

        # ④ 压缩温窗口
        if warm_msgs:
            compressed_block = self._compress_warm(warm_msgs)
            warm_tokens = estimate_tokens(compressed_block)
            messages.append({"role": "system", "content": compressed_block})
            used += warm_tokens

        # ⑤ Token 总量兜底检查
        hot_tokens = sum(estimate_tokens(m.get("content", "")) for m in hot_msgs)
        if used + hot_tokens + estimate_tokens(task) > budget:
            # 全量二次压缩：把 compressed_block + 大部分 hot_msgs 压掉
            all_to_compress = warm_msgs + hot_msgs[:max(0, len(hot_msgs) - self.config.keep_min_hot_rounds)]
            if all_to_compress:
                mega_summary = self._compress_warm(all_to_compress)
                # Replace the warm block in messages
                messages = [m for m in messages if "[会话早期摘要]" not in str(m.get("content", ""))]
                messages.append({"role": "system", "content": mega_summary})
                hot_msgs = hot_msgs[-self.config.keep_min_hot_rounds:]

        # ⑥ 热窗口 + 任务
        for msg in hot_msgs:
            messages.append({"role": msg.get("role", "user"),
                             "content": msg.get("content", "")})
        messages.append({"role": "user", "content": task})

        return messages

    def _format_memories(self, memories: list[MemoryItem]) -> str:
        """格式化记忆为文本块。"""
        lines = ["[Agent 记忆]"]
        for item in memories:
            label = "偏好" if item.memory_type.value == "semantic" else "事件"
            lines.append(f"- [{label}] {item.content[:200]}")
        return "\n".join(lines)

    def _compress_warm(self, warm_msgs: list[dict]) -> str:
        """将温窗口消息压缩为摘要。如果有 LLM client 就调 API，否则做简单截断。"""
        if not warm_msgs:
            return ""

        if self._llm:
            try:
                text = "\n".join(
                    f"[{m.get('role', '?')}]: {str(m.get('content', ''))[:300]}"
                    for m in warm_msgs[-20:]  # max 20 messages to summarize
                )
                prompt = (
                    f"将以下对话历史总结为一段简洁文字（中文，200字以内），"
                    f"保留关键信息和结论：\n\n{text[:6000]}"
                )
                summary = self._llm.chat([{"role": "user", "content": prompt}])
                return f"[会话早期摘要]\n{summary}"
            except Exception:
                pass

        # Fallback: 简单截断拼接
        lines = []
        for m in warm_msgs[-5:]:
            content = str(m.get("content", ""))[:100]
            if content.strip():
                lines.append(f"[{m.get('role', '?')}]: {content}")
        return "[会话早期摘要]\n" + "\n".join(lines) if lines else ""

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """截断文本到指定 token 数，保留前 70% + 尾 10%。"""
        current = estimate_tokens(text)
        if current <= max_tokens:
            return text
        head_size = int(len(text) * 0.7)
        tail_size = int(len(text) * 0.1)
        return (
            text[:head_size]
            + "\n...(记忆已截断)...\n"
            + (text[-tail_size:] if tail_size > 0 else "")
        )
```

- [ ] **Step 2: Smoke test ContextBuilder**

```bash
cd D:\agent_learning && python -c "
import time
from backend.memory.base import MemoryConfig, MemoryItem, MemoryType
from backend.memory.context import ContextBuilder

cb = ContextBuilder()

# Create fake memories
memories = [
    MemoryItem(id='1', user_id=1, content='用户偏好简短回答', memory_type=MemoryType.SEMANTIC, importance=0.8),
    MemoryItem(id='2', user_id=1, content='上次任务是股票查询', memory_type=MemoryType.EPISODIC, importance=0.5),
]

# Create fake history with timestamps
now = time.time()
history = [
    {'role': 'user', 'content': '你好', 'ts': now - 600},                   # 10min ago → warm
    {'role': 'assistant', 'content': '你好！有什么可以帮你？', 'ts': now - 590},
    {'role': 'user', 'content': '帮我查股票', 'ts': now - 60},               # 1min ago → hot
    {'role': 'assistant', 'content': '正在查询...', 'ts': now - 55},
]

msgs = cb.build(
    system_prompt='You are a helpful assistant.',
    memories=memories,
    history=history,
    task='再查一下腾讯的',
    hot_window_sec=120,  # 2min for testing
)

print(f'Total messages: {len(msgs)}')
for i, m in enumerate(msgs):
    print(f'  [{i}] {m[\"role\"]}: {str(m[\"content\"])[:80]}...')

# Verify: first message is system prompt
assert msgs[0]['role'] == 'system' and 'helpful' in msgs[0]['content']
# Verify: hot messages are preserved
hot_contents = [m['content'] for m in msgs if m['role'] == 'user']
assert '再查一下腾讯的' in hot_contents[-1]  # task is last
assert '帮我查股票' in hot_contents[-2]       # hot window preserved
# Verify: warm messages are compressed
compressed = [m for m in msgs if '[会话早期摘要]' in m.get('content', '') or '早期' in m.get('content', '')]
print(f'Compressed blocks: {len(compressed)}')

print('OK')
"
```

Expected: `OK` with 5+ messages and warm compression.

- [ ] **Step 3: Commit**

```bash
git add backend/memory/context.py
git commit -m "feat: add ContextBuilder — time-windowed context assembly with token budget"
```

---

### Task 9: Create tool stubs and __init__.py exports

**Files:**
- Create: `backend/memory/tools/__init__.py`
- Create: `backend/memory/tools/memory_tool.py`
- Create: `backend/memory/tools/rag_tool.py`
- Modify: `backend/memory/__init__.py`

- [ ] **Step 1: Create backend/memory/tools/__init__.py**

```python
"""Memory system tools for LLM function calling."""
from .memory_tool import MemoryTool

__all__ = ["MemoryTool"]
```

- [ ] **Step 2: Create backend/memory/tools/rag_tool.py (placeholder)**

```python
"""
RAGTool — 第二期实现。
文档解析 → 分块 → 嵌入 → ChromaDB → 问答。
当前为空壳占位。
"""

class RAGTool:
    """TODO: 第二期实现 RAG 管道。"""
    pass
```

- [ ] **Step 3: Create backend/memory/tools/memory_tool.py**

```python
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
                memories = await self._mgr.get_working_context()
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
```

- [ ] **Step 4: Update backend/memory/__init__.py**

Replace the current content:

```python
"""Memory system — 3-layer architecture with vector search."""
from .base import MemoryItem, MemoryConfig, MemoryType, BaseMemory
from .manager import MemoryManager
from .context import ContextBuilder
from .embedding import LocalEmbedding
from .long_term import LongTermMemory  # deprecated, kept for compatibility

__all__ = [
    "MemoryManager",
    "ContextBuilder",
    "LocalEmbedding",
    "MemoryItem",
    "MemoryConfig",
    "MemoryType",
    "BaseMemory",
    "LongTermMemory",  # deprecated
]
```

- [ ] **Step 5: Verify imports**

```bash
cd D:\agent_learning && python -c "
from backend.memory import MemoryManager, ContextBuilder, MemoryItem, MemoryConfig, MemoryType
from backend.memory.tools import MemoryTool
from backend.memory.tools.rag_tool import RAGTool
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 6: Commit**

```bash
git add backend/memory/tools/__init__.py backend/memory/tools/memory_tool.py backend/memory/tools/rag_tool.py backend/memory/__init__.py
git commit -m "feat: add MemoryTool, RAGTool placeholder, and memory/__init__.py exports"
```

---

### Task 10: Migrate engine.py to use MemoryManager + ContextBuilder

**Files:**
- Modify: `backend/agent/engine.py`

- [ ] **Step 1: Replace memory integration in AgentEngine.run()**

In `engine.py`, find the block (approximately lines 130-136):

```python
        # 注入跨会话记忆上下文
        from ..memory.long_term import LongTermMemory
        ltm = LongTermMemory(user_id)
        memory_context = ltm.get_context_for_prompt(max_items=5)
        system_prompt = AGENT_SYSTEM_PROMPT
        if memory_context:
            system_prompt = AGENT_SYSTEM_PROMPT + "\n\n" + memory_context
```

Replace with:

```python
        # 注入记忆上下文（三层记忆系统）
        from ..memory.manager import MemoryManager
        from ..memory.context import ContextBuilder
        mem_mgr = MemoryManager(user_id)
        self._mem_mgr = mem_mgr  # 保留引用，供任务结束时保存
        memory_context = await mem_mgr.get_context_for_prompt(task_description[:100])
        system_prompt = AGENT_SYSTEM_PROMPT
        if memory_context:
            system_prompt = AGENT_SYSTEM_PROMPT + "\n\n" + memory_context
```

- [ ] **Step 2: Replace the simple-path memory save (around line 206-213)**

Find:

```python
            if len(final_answer) > 30 and not any(w == raw_question for w in ["你好", "谢谢", "再见", "OK", "Hi", "hi"]):
                try:
                    ltm = LongTermMemory(user_id)
                    mem_key = raw_question[:50].replace("\n", " ").strip()
                    mem_value = final_answer[:800]
                    ltm.save(mem_key, mem_value)
                except Exception:
                    pass
```

Replace with:

```python
            if len(final_answer) > 30 and not any(w == raw_question for w in ["你好", "谢谢", "再见", "OK", "Hi", "hi"]):
                try:
                    await self._mem_mgr.add(
                        content=f"Q: {raw_question[:80]}\nA: {final_answer[:200]}",
                        memory_type="episodic",
                        importance=0.4,
                    )
                except Exception:
                    pass
```

- [ ] **Step 3: Replace the complex-path memory save (around lines 310-318)**

Find:

```python
                    # 自动保存关键发现到长期记忆
                    try:
                        from ..memory.long_term import LongTermMemory
                        ltm = LongTermMemory(user_id)
                        # 提取任务关键词作为记忆标题
                        mem_key = task_description[:50].replace("\n", " ").strip()
                        mem_value = final_answer[:800]
                        ltm.save(mem_key, mem_value)
                    except Exception:
                        pass  # 记忆保存失败不阻塞
```

Replace with:

```python
                    # 自动保存任务总结到情景记忆
                    try:
                        await self._mem_mgr.add(
                            content=f"任务: {task_description[:80]}\n结论: {final_answer[:300]}",
                            memory_type="episodic",
                            importance=0.5,
                            metadata={"task_id": task_id},
                        )
                    except Exception:
                        pass  # 记忆保存失败不阻塞
```

- [ ] **Step 4: Replace max-iterations path memory save (around lines 544-552)**

Find:

```python
            if len(final_answer) > 30:
                try:
                    ltm = LongTermMemory(user_id)
                    mem_key = task_description[:50].replace("\n", " ").strip()
                    mem_value = final_answer[:800]
                    ltm.save(mem_key, mem_value)
                except Exception:
                    pass
```

Replace with:

```python
            if len(final_answer) > 30:
                try:
                    await self._mem_mgr.add(
                        content=f"任务(达最大轮次): {task_description[:80]}\n结论: {final_answer[:300]}",
                        memory_type="episodic",
                        importance=0.5,
                        metadata={"task_id": task_id},
                    )
                    # 定期触发记忆管理
                    await self._mem_mgr.consolidate()
                    await self._mem_mgr.forget()
                except Exception:
                    pass
```

- [ ] **Step 5: Verify engine still imports**

```bash
cd D:\agent_learning && python -c "
from backend.agent.engine import AgentEngine
print('Engine import OK')
"
```

Expected: `Engine import OK`

- [ ] **Step 6: Commit**

```bash
git add backend/agent/engine.py
git commit -m "refactor: migrate engine.py to MemoryManager — 3-layer memory replaces LongTermMemory"
```

---

### Task 11: Migrate tools.py memory_search to MemoryManager

**Files:**
- Modify: `backend/agent/tools.py`

- [ ] **Step 1: Replace _memory_search function in tools.py**

Find `_memory_search` (around line 782). Replace the entire function:

```python
async def _memory_search(query: str = "", action: str = "search", key: str = "", value: str = "") -> dict:
    """搜索/保存/列出跨会话记忆。支持三层记忆（working/episodic/semantic）。"""
    from ..memory.manager import MemoryManager
    mgr = MemoryManager(user_id=_current_user_id.get())

    if action == "save" and key and value:
        mid = await mgr.add(content=f"{key}: {value}", memory_type="semantic", importance=0.7)
        return {"action": "save", "key": key, "value": value[:200], "memory_id": mid, "status": "saved"}

    elif action == "list":
        items = await mgr.search(query="", memory_types=["episodic", "semantic"], limit=10)
        return {"action": "list", "memories": [i.to_dict() for i in items], "count": len(items)}

    else:  # search
        if not query:
            items = await mgr.search(query="", memory_types=["episodic", "semantic"], limit=5)
        else:
            items = await mgr.search(query=query, memory_types=["episodic", "semantic"], limit=5)
        return {"action": "search", "query": query, "memories": [i.to_dict() for i in items], "count": len(items)}
```

- [ ] **Step 2: Verify tool still works**

```bash
cd D:\agent_learning && python -c "
import asyncio
from backend.database import init_db
from backend.agent.tools import _current_user_id, set_current_user, _memory_search
init_db()
set_current_user(1)

async def test():
    # Save
    r = await _memory_search(action='save', key='测试', value='这是一条测试记忆')
    assert r['status'] == 'saved'
    print('Save:', r)

    # Search
    r2 = await _memory_search(query='测试', action='search')
    assert r2['count'] >= 1
    print('Search:', r2)

    # List
    r3 = await _memory_search(action='list')
    assert r3['count'] >= 1
    print('List:', r3)

    print('OK')

asyncio.run(test())
"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/agent/tools.py
git commit -m "refactor: migrate memory_search tool to MemoryManager — 3-layer memory support"
```

---

### Task 12: Migrate server.py /api/memory routes

**Files:**
- Modify: `backend/server.py`

- [ ] **Step 1: Replace memory route handlers in server.py**

Find the three memory routes (around lines 708-729). Replace:

```python
@app.get("/api/memory")
def list_memory(user: dict = Depends(get_current_user)):
    ltm = LongTermMemory(user["id"])
    return {"memories": ltm.list_all()}


@app.post("/api/memory")
def save_memory(body: dict, user: dict = Depends(get_current_user)):
    key = body.get("key", "").strip()
    value = body.get("value", "").strip()
    if not key or not value:
        raise HTTPException(400, "key 和 value 不能为空")
    ltm = LongTermMemory(user["id"])
    ltm.save(key, value)
    return {"ok": True, "key": key}


@app.delete("/api/memory/{key}")
def delete_memory(key: str, user: dict = Depends(get_current_user)):
    ltm = LongTermMemory(user["id"])
    ltm.delete(key)
    return {"ok": True}
```

Replace with:

```python
@app.get("/api/memory")
async def list_memory(user: dict = Depends(get_current_user)):
    mgr = MemoryManager(user["id"])
    items = await mgr.search(query="", memory_types=["episodic", "semantic"], limit=50)
    return {"memories": [i.to_dict() for i in items]}


@app.post("/api/memory")
async def save_memory(body: dict, user: dict = Depends(get_current_user)):
    content = body.get("content", body.get("value", "")).strip()
    if not content:
        raise HTTPException(400, "content 不能为空")
    memory_type = body.get("memory_type", "semantic")
    importance = body.get("importance", 0.7)
    mgr = MemoryManager(user["id"])
    mid = await mgr.add(content=content, memory_type=memory_type, importance=importance)
    return {"ok": True, "memory_id": mid}


@app.delete("/api/memory/{memory_id}")
async def delete_memory(memory_id: str, user: dict = Depends(get_current_user)):
    # Try both episodic and semantic
    mgr = MemoryManager(user["id"])
    ok = await mgr._episodic.delete(memory_id) or await mgr._semantic.delete(memory_id)
    if not ok:
        raise HTTPException(404, "记忆不存在")
    return {"ok": True}
```

- [ ] **Step 2: Update import in server.py**

The import line already has `from .memory import LongTermMemory`. Change it to:

```python
from .memory import LongTermMemory, MemoryManager
```

(or just use `from .memory.manager import MemoryManager` inline in the route functions — either approach works.)

- [ ] **Step 3: Verify server imports**

```bash
cd D:\agent_learning && python -c "
from backend.server import app
print('Server import OK')
"
```

Expected: `Server import OK`

- [ ] **Step 4: Commit**

```bash
git add backend/server.py
git commit -m "refactor: migrate /api/memory routes to MemoryManager, keep API signature compatible"
```

---

### Task 13: Deprecate LongTermMemory and ContextManager

**Files:**
- Modify: `backend/memory/long_term.py` (add deprecation notice)
- Create or modify: `backend/context_manager.py` (add deprecation notice)

- [ ] **Step 1: Add deprecation warning to long_term.py**

Add at the top of `backend/memory/long_term.py`, after the docstring:

```python
import warnings
warnings.warn(
    "LongTermMemory is deprecated. Use MemoryManager with SemanticMemory instead.",
    DeprecationWarning,
    stacklevel=2,
)
```

And add a module-level comment:

```python
"""
[DEPRECATED] 长期记忆 — 已迁移到 SemanticMemory。
保留此文件仅为向后兼容。新代码请使用:
    from backend.memory import MemoryManager
    mgr = MemoryManager(user_id)
"""
```

- [ ] **Step 2: Add deprecation notice to context_manager.py**

Add at the top of `backend/context_manager.py`:

```python
"""
[DEPRECATED] Context 窗口管理 — 已迁移到 ContextBuilder。
保留此文件仅为向后兼容。新代码请使用:
    from backend.memory import ContextBuilder
    cb = ContextBuilder()
    msgs = cb.build(system, memories, history, task)
"""
```

- [ ] **Step 3: Verify backwards compatibility**

```bash
cd D:\agent_learning && python -c "
import warnings
warnings.simplefilter('always')
# Both should still import (with deprecation warning)
from backend.memory.long_term import LongTermMemory
from backend.context_manager import ContextManager
print('Backward compatibility OK')
"
```

Expected: `Backward compatibility OK` (with deprecation warnings)

- [ ] **Step 4: Commit**

```bash
git add backend/memory/long_term.py backend/context_manager.py
git commit -m "refactor: deprecate LongTermMemory and ContextManager, migrated to MemoryManager + ContextBuilder"
```

---

### Task 14: Add PlanSolveEngine memory support and final integration test

**Files:**
- Modify: `backend/agent/plan_solve_engine.py`

- [ ] **Step 1: Add memory context injection to PlanSolveEngine.run()**

In `plan_solve_engine.py`, after line 61 (`_update_task(task_id, status="planning")`), add:

```python
        # 注入记忆上下文
        from ..memory.manager import MemoryManager
        mem_mgr = MemoryManager(user_id)
        self._mem_mgr = mem_mgr
        memory_context = await mem_mgr.get_context_for_prompt(task_description[:100])
```

And after task completion (around line 342, before `duration_ms = int(...)`), add:

```python
            # 保存任务总结到情景记忆
            try:
                await self._mem_mgr.add(
                    content=f"Plan-Solve任务: {task_description[:80]}\n结论: {final_answer[:300]}",
                    memory_type="episodic",
                    importance=0.5,
                    metadata={"task_id": task_id, "mode": "plan_solve"},
                )
            except Exception:
                pass
```

- [ ] **Step 2: Run final integration smoke test**

```bash
cd D:\agent_learning && python -c "
import asyncio
from backend.database import init_db
from backend.memory.manager import MemoryManager
from backend.memory.context import ContextBuilder

async def full_integration_test():
    init_db()
    mm = MemoryManager(user_id=1)

    # 1. Add memories of different types
    await mm.add('用户偏好Python', memory_type='semantic', importance=0.9)
    await mm.add('用户昨天查询了茅台股票', memory_type='episodic', importance=0.5)
    await mm.add('当前正在编写报告', memory_type='working', importance=0.3)

    # 2. Search
    results = await mm.search('编程语言', limit=3)
    assert len(results) >= 1
    print(f'Search results: {len(results)}')

    # 3. Context for prompt
    ctx = await mm.get_context_for_prompt('编程')
    assert 'Python' in ctx
    print(f'Context:\n{ctx}')

    # 4. ContextBuilder
    cb = ContextBuilder()
    msgs = cb.build(
        system_prompt='You are helpful.',
        memories=results,
        history=[],
        task='帮我写Python代码',
    )
    assert len(msgs) >= 2

    # 5. Consolidate
    await mm.add('用户发现pandas很好用', memory_type='episodic', importance=0.85)
    n = await mm.consolidate(threshold=0.7)
    print(f'Consolidated: {n} items')

    # 6. Forget
    n2 = await mm.forget()
    print(f'Forgot: {n2} items')

    print('=== FULL INTEGRATION TEST PASSED ===')

asyncio.run(full_integration_test())
"
```

Expected: `=== FULL INTEGRATION TEST PASSED ===`

- [ ] **Step 3: Commit**

```bash
git add backend/agent/plan_solve_engine.py
git commit -m "feat: add memory support to PlanSolveEngine, final integration test passed"
```

---

## Completion Checklist

After all tasks, verify:

- [ ] `pip check` — no dependency conflicts
- [ ] `python -c "from backend.memory import *"` — all exports work
- [ ] `python -c "from backend.server import app"` — server starts
- [ ] ChromaDB data directory created at `data/chroma/` after first add
- [ ] `grep -r "LongTermMemory" backend/ --include="*.py" | grep -v deprecated | grep -v "from .memory.long_term"` — no non-deprecated usage remains

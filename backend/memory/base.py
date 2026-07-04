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

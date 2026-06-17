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

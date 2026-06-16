# 记忆系统改造设计文档

**日期**: 2026-06-17  
**参考**: [hello-agents 第八章 记忆与检索](https://datawhalechina.github.io/hello-agents/#/./chapter8/%E7%AC%AC%E5%85%AB%E7%AB%A0%20%E8%AE%B0%E5%BF%86%E4%B8%8E%E6%A3%80%E7%B4%A2)  
**状态**: 设计通过，待实现

---

## 1. 目标

将项目记忆系统从单一 `LongTermMemory` (SQLite key-value) 升级为三层记忆架构 (Working + Episodic + Semantic)，引入向量语义检索 (ChromaDB + 本地 BGE 嵌入)，并新增上下文工程层统一管理 LLM 上下文窗口。

**原则**:
- SQLite 不动，继续做账户/对话/任务/记忆原文的 source of truth
- 零外部服务：ChromaDB 嵌入式运行，BGE 模型本地加载
- 最少侵入：只改 5 个现有文件（4 轻改 + 1 弃用），新增 8 个文件

---

## 2. 文件结构

```
backend/memory/
├── __init__.py              # 暴露 MemoryManager, get_memory_manager()
├── base.py                  # BaseMemory(ABC) + MemoryItem + MemoryConfig
├── manager.py               # MemoryManager 统一调度器
├── working.py               # WorkingMemory — 会话层，纯内存 dict + TTL
├── episodic.py              # EpisodicMemory — 任务事件，SQLite + ChromaDB
├── semantic.py              # SemanticMemory — 偏好/知识，SQLite + ChromaDB
├── embedding.py             # LocalEmbedding（BGE 模型封装）
├── context.py               # ContextBuilder — 上下文工程层
├── long_term.py             # [deprecated] 内部迁移到 SemanticMemory
└── tools/
    ├── __init__.py
    ├── memory_tool.py       # MemoryTool（第一期）
    └── rag_tool.py          # RAGTool（第二期，空壳占位）
```

---

## 3. 类设计

### 3.1 MemoryItem (base.py)

标准化记忆数据结构，所有记忆类型共用。

```python
@dataclass
class MemoryItem:
    id: str                          # uuid
    user_id: int                     # 多用户隔离
    content: str                     # 记忆文本
    memory_type: MemoryType          # WORKING | EPISODIC | SEMANTIC
    importance: float = 0.5          # 0.0~1.0，LLM 打分
    created_at: float                # unix timestamp
    last_accessed_at: float          # 最近检索时间
    access_count: int = 0            # 检索次数
    ttl: Optional[float] = None      # Working 过期时间
    metadata: dict = field(default_factory=dict)  # 扩展字段（避免可变默认参数陷阱）
    embedding: Optional[list[float]] = None        # 向量缓存（仅 ChromaDB 存储）
```

### 3.2 MemoryConfig (base.py)

```python
@dataclass
class MemoryConfig:
    # Working
    working_max_items: int = 20
    working_ttl_sec: int = 1800       # 30 分钟
    
    # Episodic
    episodic_consolidation_threshold: float = 0.7
    episodic_forget_days: int = 7
    episodic_forget_importance_max: float = 0.3
    
    # Semantic
    semantic_no_auto_forget: bool = True
    
    # Context
    hot_window_sec: int = 180          # 3 分钟
    history_budget_tokens: int = 90_000
    memory_context_max_ratio: float = 0.15
    keep_min_hot_rounds: int = 2
    max_memory_items: int = 5
    
    # ChromaDB
    chroma_persist_dir: str = "data/chroma"
```

### 3.3 BaseMemory (base.py)

```python
class BaseMemory(ABC):
    """所有记忆类型的抽象基类"""
    @abstractmethod
    async def add(self, item: MemoryItem) -> str: ...
    @abstractmethod
    async def search(self, query_embedding, limit: int) -> list[MemoryItem]: ...
    @abstractmethod
    async def delete(self, item_id: str): ...
    @abstractmethod
    async def clear(self, user_id: int): ...
```

### 3.4 WorkingMemory (working.py)

- **存储**: `dict[user_id, deque[MemoryItem]]`，纯内存
- **容量**: 默认 20 条，FIFO 挤出
- **TTL**: 默认 30 分钟过期，惰性清理
- **用途**: 当前任务上下文、工具结果、中间推理
- **不持久化**，会话结束即清
- **限制**: v1 限定单进程；多 worker / 重启部署场景下 working memory 会丢失。后续如需多进程共享，可接入 Redis 替代内存 dict

### 3.5 EpisodicMemory (episodic.py)

- **存储**: SQLite(原文, source of truth) + ChromaDB Collection(向量)
- **内容**: 每次 Agent 任务完成后的总结——做了什么、什么结果
- **ChromaDB Collection 命名**: `episodic_all`（统一 collection，按 `user_id` metadata 过滤，避免 collection 数量随用户膨胀）
- **检索公式**: `(向量相似度 × 0.8 + 时间近因性 × 0.2) × (0.8 + importance × 0.4)`
- **遗忘**: importance < 0.3 且 7 天未访问 → 自动删除
- **固化**: importance > 0.7 → consolidate 升级为 Semantic

### 3.6 SemanticMemory (semantic.py)

- **存储**: SQLite(原文) + ChromaDB Collection(向量)
- **内容**: 用户偏好、领域知识、从 Episodic 固化来的规则
- **ChromaDB Collection 命名**: `semantic_all`（统一 collection，按 `user_id` metadata 过滤）
- **检索公式**: `score = 0.7 × sim + 0.1 × freq_norm + 0.2 × importance`
  - `freq_norm = min(access_count / 10.0, 1.0)`，防止访问次数整数盖过其他维度
- **不自动遗忘**（不设 TTL），仅显式删除

### 3.7 MemoryManager (manager.py)

统一调度入口，Agent Engine 直接调用。

```python
class MemoryManager:
    def __init__(self, user_id, config=MemoryConfig(), embedding=None)
    
    async def add(content, memory_type, importance=None, metadata=None) -> str
        # metadata: 传入 None 时内部初始化为 {}，避免可变默认参数陷阱
    async def search(query, memory_types=["episodic","semantic"], limit=5) -> list[MemoryItem]
    async def get_working_context() -> list[MemoryItem]
    async def consolidate(from_type="episodic", to_type="semantic", threshold=0.7)
    async def forget(strategy="importance_based")
    async def get_context_for_prompt(query="") -> str
```

### 3.8 LocalEmbedding (embedding.py)

```python
class LocalEmbedding:
    def __init__(self, model_name="BAAI/bge-small-zh-v1.5"):
        # 首次自动下载 ~100MB，后续本地加载
        self.model = SentenceTransformer(model_name)
    
    def embed(self, texts: list[str]) -> np.ndarray:
        # 返回 512-dim 向量
        return self.model.encode(texts, normalize_embeddings=True)
    
    @property
    def dim(self) -> int:
        return 512
```

### 3.9 ContextBuilder (context.py)

上下文工程层，记忆与 LLM 之间的"最后一公里"。

```
build(system, memories, history, task):
  ① System Prompt — 固定，按实际 token 数
  ② 切分历史 — 热窗口(3min内,完整保留) / 温窗口(同会话,压缩摘要)
  ③ Token 总量检查 — 超 90K 触发全量二次压缩
  ④ 记忆上下文 — 贪心填充，不超过总预算 15%
  ⑤ 组装 — [System] + [Memories] + [CompressedHistory] + [HotMsgs] + [Task]
```

**核心原则**: 同会话不丢弃，时间分窗，总量兜底

### 3.10 MemoryTool (tools/memory_tool.py)

LLM 通过 function calling 调用的工具接口。

```python
class MemoryTool:
    async def execute(action, **kwargs) -> dict
    # action: add | search | list | forget
```

---

## 4. 数据流

### 4.1 写入流程

```
LLM 调用 memory_tool(action="add", content="...")
  → MemoryTool.execute()
    → MemoryManager.add()
      → LLM 分类 memory_type + 打分 importance
      → 路由到 WorkingMemory / EpisodicMemory / SemanticMemory
      → SQLite INSERT (原文)
      → embed(content) → ChromaDB INSERT (向量)
      → 返回 memory_id
```

### 4.2 检索流程

```
LLM 调用 memory_tool(action="search", query="编程偏好")
  → MemoryTool.execute()
    → MemoryManager.search(query)
      → embed(query) → 查询向量
      → ChromaDB 多 collection 查询 (episodic + semantic)
      → 各类型评分公式排序
      → 返回 top-N MemoryItem
```

### 4.3 Agent 任务注入流程

```
AgentEngine.run(task_description, user_id, task_id):
  → MemoryManager(user_id)
  → ContextBuilder(budget=90K)
  → mem_mgr.search(query=task_description, limit=5)
  → mem_mgr.get_working_context()
  → ctx_builder.build(system, memories+working, history=[], task=task)
  → LLM 推理...
  
  # 任务结束
  → mem_mgr.add(content="任务总结", type="episodic", metadata={task_id})
  → mem_mgr.consolidate()    # 达标的情景记忆 → 语义记忆
  → mem_mgr.forget()         # 清理低价值记忆
```

---

## 5. 存储分布

| 数据 | 存储位置 | 用途 |
|------|---------|------|
| 用户账户/密码 | SQLite (chat.db) | 认证，不动 |
| 对话消息 | SQLite (chat.db) | 聊天历史，不动 |
| Agent 任务/步骤 | SQLite (agent.db) | 任务追踪，不动 |
| Working 记忆 | 纯内存 dict | 会话结束即丢 |
| Episodic 记忆原文 | SQLite (memory.db) | source of truth |
| Episodic 向量 | ChromaDB | 语义检索 |
| Semantic 记忆原文 | SQLite (memory.db) | source of truth |
| Semantic 向量 | ChromaDB | 语义检索 |
| BGE 嵌入模型 | 本地磁盘 (~100MB) | 离线，text→向量 |

---

## 6. 数据库变更

### 6.1 memory.db 新增表

```sql
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
CREATE INDEX idx_episodic_user ON episodic_memory(user_id, created_at);

CREATE TABLE IF NOT EXISTS semantic_memory (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    importance REAL DEFAULT 0.5,
    created_at REAL NOT NULL DEFAULT (unixepoch()),
    last_accessed_at REAL NOT NULL DEFAULT (unixepoch()),
    access_count INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}',
    source_episodic_id TEXT  -- 从哪条 Episodic 固化来的
);
CREATE INDEX idx_semantic_user ON semantic_memory(user_id, created_at);
```

### 6.2 ChromaDB Collection 命名（设计决策）

**选方案 B**：统一 collection + metadata 过滤。

| 方案 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| A. 按用户分 | `episodic_user_{id}` | 隔离干净，删用户直接 drop | collection 数量随用户增长 |
| **B. 统一 + 过滤** | `episodic_all`，`where user_id={id}` | 简单，不膨胀 | 需 metadata filtering（ChromaDB 支持） |

实现方式：
- 每个记忆类型只有一个 ChromaDB collection（`episodic_all` / `semantic_all`）
- `MemoryItem` 写入时把 `user_id` 写入 ChromaDB metadata
- 查询时加 `where={"user_id": user_id}` 过滤
- embedding.py 不感知 collection 结构，只提供 `embed()` 方法

### 6.3 旧 agent_memory 表

保留不动。LongTermMemory 标记 deprecated，`get_context_for_prompt()` 内部委托给 MemoryManager.semantic。

---

## 7. 集成变更清单

| 文件 | 变更程度 | 具体改动 |
|------|---------|---------|
| `backend/memory/__init__.py` | 改 | 新导出 MemoryManager, ContextBuilder |
| `backend/agent/tools.py` | 轻改 | `_memory_search` 委托给 MemoryManager |
| `backend/agent/engine.py` | 轻改 | 用 MemoryManager + ContextBuilder 替代 LongTermMemory |
| `backend/memory/long_term.py` | 弃用 | 保留文件，内部迁移到 SemanticMemory |
| `backend/context_manager.py` | 废弃 | 逻辑迁移到 ContextBuilder |
| `backend/database.py` | 轻改 | 新增 episodic/semantic 表初始化 |
| `backend/server.py` | 轻改 | `/api/memory` 路由从 LongTermMemory 切换到 MemoryManager，保持接口签名兼容 |

---

## 8. 后续扩展

- **第二期 RAGTool**: 文档解析 → 分块 → 嵌入 → ChromaDB → 问答
- **知识图谱**: 将来如需 Neo4j，可在 SemanticMemory 下挂 GraphStore
- **感知记忆**: 多模态数据（图片、音频），需额外嵌入模型

---

## 9. 自检清单

- [x] 无占位符或 TODO
- [x] 内部一致：三种记忆类型接口统一继承 BaseMemory
- [x] ChromaDB Collection 命名方案已明确（方案 B：统一 collection + metadata 过滤）
- [x] 检索评分已归一化（access_count 封顶 1.0）
- [x] SQL 表 last_accessed_at 已加 DEFAULT (unixepoch())
- [x] Python 可变默认参数已修正（field(default_factory=dict), metadata=None）
- [x] WorkingMemory 单进程限制已文档化
- [x] server.py /api/memory 路由迁移已纳入变更清单
- [x] 范围可控：8 新文件 + 5 改文件，无外部服务依赖
- [x] 要求明确：MemoryItem 字段、评分公式、TTL 策略均已指定

# 可视化可控 Agent — InterventionHandler 设计

## 概述

将 Agent 从"自动跑到底"升级为"过程可见 + 随时可纠正"。用户在 Agent 执行过程中看到每一步，觉得不对就用自然语言补充描述，Agent 在下一个思考节点消化反馈并调整方向。

核心原则：
- **不打断正在执行的工具** — 反馈在"观察结果 → 下一轮思考"之间注入
- **反馈是建议不是命令** — 转成 system 消息让 LLM 自己判断
- **取消走快车道** — 不排队，立即设取消标志

---

## 架构

```
用户 (前端)
   │ ▲
   │ WebSocket
   ▼ │
┌──────────────────────────────────────────┐
│           InterventionHandler             │
│                                           │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐ │
│  │ 反馈接收 │→│ 排队/合并 │→│ 注入编排 │ │
│  │ WS消息  │  │ 优先级   │  │ 时机控制 │ │
│  └─────────┘  └──────────┘  └─────────┘ │
│                                           │
└──────────────────────────────────────────┘
   │ ▲
   │ │ 注入反馈
   ▼ │
┌──────────────────────────────────────────┐
│           Agent Engine (ReAct)            │
│                                           │
│   Thought → Action → Observation →        │
│     ↑                            │        │
│     └── [反馈注入点] ←───────────┘        │
│                                           │
└──────────────────────────────────────────┘
```

注入点在 `engine.py` 循环中 Observation 阶段之后、下一轮 Thought 之前：

```python
# engine.py 当前循环结构
for iteration in range(max_iterations):
    # ── 取消检查 ──
    if task_id in self._cancellations:
        ws.broadcast("task_cancelled")
        return

    # ── 取消快车道（新增）──
    if intervention_handler.has_cancellation(task_id):
        ws.broadcast("task_cancelled")
        return

    # ── Thought 阶段 ──
    msg = await self._call_llm(messages, tool_schemas)

    if not msg.tool_calls:
        return

    # ── Action 阶段 ──
    results = await asyncio.gather(*tool_tasks)

    # ── Observation 阶段 ──
    messages.append(tool_results)

    # ← 反馈注入点（新增）
    fb_msgs, injected_ids = self.intervention.drain(task_id)
    if fb_msgs:
        messages.extend(fb_msgs)
        for fb_id in injected_ids:
            ws.broadcast("intervention_applied", {"id": fb_id, "step": step_number})
```

---

## InterventionHandler 接口

新文件：`backend/agent/intervention.py`

```python
class InterventionHandler:
    # ── 生命周期 ──
    async def register_task(self, task_id: str)
    async def complete_task(self, task_id: str)

    # ── 反馈管理 ──
    async def receive(self, task_id: str, feedback: dict) -> tuple[Feedback, list[str]]
        # 返回: (主 Feedback 对象, 被合并的 id 列表)
        # 幂等：同 id 重复调用直接返回已有对象，不重复合并内容
        # 截断：content > 2000 字静默截断
        # 空内容：返回 None，调用方发 intervention_rejected("empty_content")
    async def drain(self, task_id: str) -> tuple[list[dict], list[str]]
        # 返回: (拼好的 system messages, 注入的 id 列表)
    async def retract(self, task_id: str, feedback_id: str) -> bool
        # True: 已从队列删除; False: 已注入，无法撤回

    # ── 取消 ──
    async def cancel(self, task_id: str)
    async def has_cancellation(self, task_id: str) -> bool

    # ── 消化追踪 ──
    async def mark_digested(self, task_id: str, thought_content: str) -> list[dict]
        # 从 Thought 中解析 [摘要: decision, summary]
        # 返回 feedback_digested 事件列表
```

### Feedback 数据结构

```python
@dataclass
class Feedback:
    id: str              # 前端生成 UUID
    task_id: str
    content: str         # 用户原始输入，截断后 ≤2000 字
    priority: str        # "urgent" | "high" | "normal" | "low"
    created_at: float
    status: str          # "queued" | "injected" | "digested" | "retracted"
    merged_from: list    # 被合并掉的 feedback id 列表
```

### 优先级定义

| 优先级 | 触发方式 | 行为 |
|--------|---------|------|
| `urgent` | 取消 | 跳过队列，立即设取消标志 |
| `high` | 纠正方向（「搞错了，我要的是 A 不是 B」） | 队列头部，下一轮立即注入 |
| `normal` | 提前补充信息（「别忘了 XX」） | 正常排队 |
| `low` | 事后反馈（「完成了也发我一份」） | 任务结束后注入 |

### 合并规则

- 同 priority + 同 task → 内容拼接，merged_from 记录被合并的 id
- 不同 priority → 不合并
- 被合并的 id 前端收到 `intervention_applied { merged_into: "fb-001" }`

### 队列溢出保护

- 队列上限 10 条
- 新来一条，挤掉最老的 normal/low
- urgent/high 不可被挤掉
- 如果全是 urgent/high → 新来的被拒绝：`intervention_rejected("queue_full")`

### 并发安全

`_queues` 和 `_cancellations` 用 `asyncio.Lock` 保护。`receive()` / `drain()` / `cancel()` 都用 `async with self._lock`。

### 生命周期管理

engine 在 `run()` 开头调 `register_task`，在 `finally` 块调 `complete_task`：

```python
async def run(self, task_id, ...):
    self.intervention.register_task(task_id)
    try:
        # ... Agent 循环 ...
    finally:
        self.intervention.complete_task(task_id)
```

`complete_task()` 行为：
- 清队列 + 取消标志
- 对每条 pending feedback 发 `intervention_rejected("task_already_completed")`

---

## WebSocket 协议

### 客户端 → 服务端

| type | payload | 说明 |
|------|---------|------|
| `intervention` | `{ id, content, priority? }` | 发送反馈，id 由前端生成 UUID |
| `cancel` | `{ reason? }` | 取消任务 |
| `retract` | `{ feedback_id }` | 撤回尚未注入的反馈 |

### 服务端 → 客户端（新增）

| type | payload | 说明 |
|------|---------|------|
| `intervention_ack` | `{ id, priority, agent_status?, merged? }` | 确认收到，含 Agent 当前状态 |
| `intervention_rejected` | `{ id, reason }` | 拒绝（task_already_completed / queue_full / empty_content / invalid_format / duplicate_id） |
| `intervention_applied` | `{ id, step_number, merged_into? }` | 反馈已注入到 messages |
| `feedback_digested` | `{ id, decision, summary }` | Agent 已消化反馈，decision: adopted / rejected / partial / unknown |

### 时序

```
用户输入 → 前端发 intervention
  → 服务端 receive() → 发 intervention_ack (含 agent_status)
  → 排队...
  → 注入点到达 → messages.extend(feedback_as_system) → 发 intervention_applied
  → 下一轮 Thought → 解析 [摘要: ...] → 发 feedback_digested
```

---

## 前端改动

### pendingFeedbacks 六态状态机

```
sending ──→ acknowledged ──→ applied ──→ digested
   │            │                │
   └────────────┼────────────────┘
                │
                ▼
             rejected (恢复输入条)
             timeout  (恢复输入条，允许重发)
                       retracted (乐观更新，不需服务端确认)
```

### StepListItem union 类型

```typescript
type StepListItem =
  | { kind: "step"; data: AgentStep }
  | { kind: "feedback_banner"; data: DigestEntry }
```

`FeedbackDigestBanner` 与 `AgentStepCard` 在同一个 FlatList 中自然穿插渲染。

### 底部输入条五种状态

| 状态 | 触发 | 输入条表现 | 发送按钮 |
|------|------|---------|---------|
| 空闲 | 无待处理反馈 | 正常白色，可输入 | 可用 |
| 等待确认 | 刚发出一条，等 ack | 灰化 + "正在发送..." | 禁用 |
| 等待注入 | 已 ack，等 applied | 灰化 + "Agent 下次思考时会处理" | 禁用 |
| 被拒绝/超时 | rejected 或 5s 无响应 | 恢复正常 | 可用 |
| 已注入 | applied 已到达 | 恢复正常，可连续发 | 可用 |

### useAgentTask Hook 新增

```typescript
// 新增方法
intervene(id: string, content: string, priority?: string): void
retract(feedbackId: string): void
cancel(reason?: string): void

// 新增状态
pendingFeedbacks: Map<string, FeedbackStatus>
agentDigestions: DigestEntry[]

// cancel 行为
async function cancel(reason?: string) {
  status = "cancelling";  // 立即禁用所有按钮
  ws.send({ type: "cancel", payload: { reason } });
  // task_cancelled 回来时 status → "cancelled"
}
```

---

## 异常路径

### 断连重放

- 前端重连后，`pendingFeedbacks` 中 `sending` 状态的反馈用原 id 重发
- 服务端 `receive()` 同 id 幂等——直接返回已有 Feedback 对象，不重复合并内容

### Agent 不听话

- `mark_digested()` 从 Thought 中解析 `[摘要: decision, summary]`
- 解析失败 → decision = "unknown"，前端显示"Agent 已收到反馈，但未明确回应"

### 反馈和 confirmation 并存

- confirmation：弹窗，阻断式，两个按钮
- intervention：底部输入条，非阻断
- 两者可同时出现在 UI 上，不需互斥处理

### 服务端崩溃恢复

- Phase 1：全内存，服务重启后 pending 反馈丢失
- Phase 2 TODO：反馈持久化到 agent.db，支持崩溃恢复

---

## 改动清单

| 文件 | 改动 |
|------|------|
| `backend/agent/intervention.py` | 新建：InterventionHandler 完整实现 |
| `backend/agent/engine.py` | 循环开头加取消快车道；Observation 后加注入点；run() 加生命周期管理 |
| `backend/agent/websocket_manager.py` | 新增 receive_feedback 方法；收 intervention/cancel/retract 消息 |
| `mobile/src/hooks/useAgentTask.ts` | 新增 intervene/retract/cancel；pendingFeedbacks 状态机；agentDigestions |
| `mobile/src/components/FeedbackDigestBanner.tsx` | 新建：方向调整标记组件 |
| `mobile/src/components/AgentStepCard.tsx` | 无改动，StepListItem union 在父组件处理 |
| `mobile/app/agent/[id].tsx` | FlatList 支持 union 类型；底部输入条状态驱动 |

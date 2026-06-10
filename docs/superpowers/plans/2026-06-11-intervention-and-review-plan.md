# 可视化可控 Agent + 专家模式审校 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Agent 平台增加两大能力：(1) 用户可在 Agent 执行过程中随时发送反馈纠正方向；(2) 专家模式引入副 Agent 审校方案。

**Architecture:** InterventionHandler 插入 engine 循环的 Observation 阶段之后，反馈通过 WebSocket 双向通信；PlanSolveEngine 在规划阶段后创建独立副 Agent 审校方案，结构化结果注入主 Agent 继续执行。

**Tech Stack:** Python asyncio / FastAPI / WebSocket / React Native Expo / TypeScript

---

## 文件结构

```
backend/agent/
├── intervention.py          # 新建：InterventionHandler 完整实现
├── review_prompt.py          # 新建：副 Agent 审校 System Prompt
├── engine.py                 # 修改：集成 InterventionHandler（3 处注入）
├── plan_solve_engine.py      # 修改：规划后加审校阶段
└── websocket_manager.py      # 修改：新增反馈消息接收方法

backend/
└── server.py                 # 修改：WebSocket handler 路由干预消息

mobile/src/
├── hooks/useAgentTask.ts     # 修改：新增 intervene/retract/cancel + 状态机
└── components/
    ├── FeedbackDigestBanner.tsx  # 新建：方向调整标记
    └── ReviewBanner.tsx          # 新建：方案审校标记

mobile/app/agent/
└── [id].tsx                  # 修改：StepListItem union + 底部输入条 + ReviewBanner
```

---

### Task 1: 创建 InterventionHandler 类

**Files:**
- Create: `backend/agent/intervention.py`

- [ ] **Step 1: 创建 intervention.py 完整实现**

```python
"""
InterventionHandler — 用户反馈队列管理 + 注入编排

用户在 Agent 执行过程中发送反馈，Handler 负责：
  - 接收/排队/合并/优先级排序
  - 在注入点将反馈转为 system 消息
  - 取消快车道（不排队，直接设标志）
  - 反馈消化追踪
"""
import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Feedback:
    id: str
    task_id: str
    content: str
    priority: str  # "urgent" | "high" | "normal" | "low"
    created_at: float = field(default_factory=time.time)
    status: str = "queued"  # "queued" | "injected" | "digested" | "retracted"
    merged_from: list = field(default_factory=list)

    def merge(self, other: "Feedback"):
        self.content = f"{self.content}\n[补充] {other.content}"
        self.merged_from.append(other.id)
        other.status = "retracted"


_PRIORITY_ORDER = {"urgent": 0, "high": 1, "normal": 2, "low": 3}
_MAX_QUEUE_SIZE = 10
_MAX_CONTENT_LENGTH = 2000


class InterventionHandler:
    """用户干预管理器 — 每个 Agent 引擎共享一个全局实例"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._queues: dict[str, list[Feedback]] = {}  # task_id → [Feedback, ...]
        self._cancellations: set[str] = set()
        self._task_states: dict[str, str] = {}  # "running" | "completed"

    # ── 生命周期 ──────────────────────────────────────────

    async def register_task(self, task_id: str):
        async with self._lock:
            self._task_states[task_id] = "running"
            if task_id not in self._queues:
                self._queues[task_id] = []

    async def complete_task(self, task_id: str):
        """任务结束（正常/取消/失败），清理资源并通知未处理反馈"""
        async with self._lock:
            self._task_states[task_id] = "completed"
            pending = self._queues.pop(task_id, [])
            self._cancellations.discard(task_id)

        # 通知每条 pending 反馈：任务已结束
        for fb in pending:
            if fb.status == "queued":
                yield fb.id, "task_already_completed"

    # ── 反馈接收 ──────────────────────────────────────────

    async def receive(self, task_id: str, feedback_item: dict):
        """
        接收一条用户反馈。幂等：同 id 重复调用直接返回已有对象。

        Args:
            task_id: 任务 ID
            feedback_item: { id, content, priority? }

        Returns:
            (Feedback, merged_ids) — 主反馈对象 + 被合并的 id 列表
            或 (None, []) — 内容为空，调用方应发 intervention_rejected
        """
        content = (feedback_item.get("content") or "").strip()
        if not content:
            return None, []

        if len(content) > _MAX_CONTENT_LENGTH:
            content = content[:_MAX_CONTENT_LENGTH - 3] + "..."

        priority = feedback_item.get("priority", "normal")
        if priority not in _PRIORITY_ORDER:
            priority = "normal"
        fb_id = feedback_item["id"]

        async with self._lock:
            # 幂等检查
            existing = self._find_by_id(task_id, fb_id)
            if existing:
                return existing, []

            # 同优先级合并
            merged = self._find_same_priority(task_id, priority)
            if merged:
                new_fb = Feedback(id=fb_id, task_id=task_id, content=content, priority=priority)
                merged.merge(new_fb)
                return merged, [fb_id]

            # 队列溢出保护
            queue = self._queues.setdefault(task_id, [])
            if len(queue) >= _MAX_QUEUE_SIZE:
                # 尝试挤掉最老的 normal/low
                evicted = self._evict_low_priority(queue)
                if evicted is None:
                    # 全是 urgent/high，拒绝新反馈
                    return None, []  # 调用方发 intervention_rejected("queue_full")

            fb = Feedback(id=fb_id, task_id=task_id, content=content, priority=priority)
            queue.append(fb)
            return fb, []

    def _find_by_id(self, task_id: str, fb_id: str) -> Optional[Feedback]:
        for fb in self._queues.get(task_id, []):
            if fb.id == fb_id:
                return fb
        return None

    def _find_same_priority(self, task_id: str, priority: str) -> Optional[Feedback]:
        for fb in self._queues.get(task_id, []):
            if fb.priority == priority and fb.status == "queued":
                return fb
        return None

    def _evict_low_priority(self, queue: list[Feedback]) -> Optional[str]:
        """挤掉最老的 normal/low，返回被挤掉的 id。无低优先级返回 None。"""
        for i, fb in enumerate(queue):
            if fb.priority in ("normal", "low"):
                evicted = queue.pop(i)
                return evicted.id
        return None

    # ── 注入 ────────────────────────────────────────────

    async def drain(self, task_id: str) -> tuple[list[dict], list[str]]:
        """
        取出队列中所有待注入反馈，转为 system 消息。

        Returns:
            (messages, injected_ids)
        """
        async with self._lock:
            queue = self._queues.get(task_id, [])
            if not queue:
                return [], []

            # 按优先级排序
            pending = [fb for fb in queue if fb.status == "queued"]
            pending.sort(key=lambda fb: _PRIORITY_ORDER.get(fb.priority, 2))

            messages = []
            injected_ids = []

            for fb in pending:
                fb.status = "injected"
                injected_ids.append(fb.id)
                msg_content = (
                    f"[用户反馈] {fb.content}\n"
                    f"请结合当前任务判断是否需要调整方向。"
                )
                messages.append({"role": "system", "content": msg_content})

            return messages, injected_ids

    # ── 撤回 ────────────────────────────────────────────

    async def retract(self, task_id: str, feedback_id: str) -> bool:
        """
        从队列中删除尚未注入的反馈。

        Returns:
            True: 已从队列删除
            False: 已注入或不存在，无法撤回
        """
        async with self._lock:
            queue = self._queues.get(task_id, [])
            for i, fb in enumerate(queue):
                if fb.id == feedback_id and fb.status == "queued":
                    fb.status = "retracted"
                    return True
            return False

    # ── 取消 ────────────────────────────────────────────

    async def cancel(self, task_id: str):
        async with self._lock:
            self._cancellations.add(task_id)

    async def has_cancellation(self, task_id: str) -> bool:
        async with self._lock:
            return task_id in self._cancellations

    # ── 消化追踪 ─────────────────────────────────────────

    async def mark_digested(self, task_id: str, thought_content: str):
        """
        从 LLM Thought 内容中解析反馈消化摘要。

        期望格式: [摘要: adopted|rejected|partial|unknown, 说明文字]

        Returns:
            feedback_digested 事件列表
        """
        pattern = r'\[摘要:\s*(adopted|rejected|partial|unknown)\s*[,，]\s*(.+?)\]'
        matches = re.findall(pattern, thought_content)

        if not matches:
            # 尝试找第一条待消化的反馈，标记为 unknown
            async with self._lock:
                queue = self._queues.get(task_id, [])
                undigested = [fb for fb in queue if fb.status == "injected" and fb.id]
            if undigested:
                fb = undigested[0]
                fb.status = "digested"
                return [{
                    "type": "feedback_digested",
                    "id": fb.id,
                    "decision": "unknown",
                    "summary": "Agent 已收到反馈，但未明确结构化回应"
                }]
            return []

        events = []
        async with self._lock:
            queue = self._queues.get(task_id, [])
            for i, (decision, summary) in enumerate(matches):
                if i < len(queue):
                    fb = queue[i]
                    if fb.status == "injected":
                        fb.status = "digested"
                        events.append({
                            "type": "feedback_digested",
                            "id": fb.id,
                            "decision": decision.strip(),
                            "summary": summary.strip()
                        })

        return events

    # ── agent_status 查询 ────────────────────────────────

    def get_agent_status(self, task_id: str) -> Optional[str]:
        """返回 Agent 当前状态用于 ack 消息 (thinking/executing_tool/idle)"""
        return "executing_tool" if task_id in self._task_states else None
```

- [ ] **Step 2: 验证文件可以导入**

```bash
cd /d/agent_learning && python -c "from backend.agent.intervention import InterventionHandler, Feedback; h = InterventionHandler(); print('Import OK')"
```

Expected: `Import OK`

---

### Task 2: 集成 InterventionHandler 到 AgentEngine

**Files:**
- Modify: `backend/agent/engine.py:18-19` (import)
- Modify: `backend/agent/engine.py:88-93` (__init__)
- Modify: `backend/agent/engine.py:102-107` (run 方法签名和开头)
- Modify: `backend/agent/engine.py:133-138` (取消检查加快车道)
- Modify: `backend/agent/engine.py:378` (注入点)
- Modify: `backend/agent/engine.py:170-216` (无工具调用时调 mark_digested)
- Modify: `backend/agent/engine.py:399-408` (异常处理加 complete_task)

- [ ] **Step 1: 在 engine.py 顶部增加 import**

修改 `backend/agent/engine.py:18-19`：

```python
# 改为（在现有 import 后添加）：
from .tools import Tool
from .websocket_manager import WebSocketManager, _save_step, _update_step, _update_task
from .intervention import InterventionHandler  # 新增
from ..llm_client import estimate_tokens
```

- [ ] **Step 2: 修改 AgentEngine.__init__ 接受 InterventionHandler**

修改 `backend/agent/engine.py:88-93`：

```python
# 改为：
class AgentEngine:
    """增强版 ReAct Agent — Token 预算 + 并行执行 + 自动反思 + 用户干预"""

    def __init__(self, deepseek, tools: list[Tool], ws_manager: WebSocketManager,
                 intervention: InterventionHandler):
        self.deepseek = deepseek
        self.tools = tools
        self.tool_map = {t.name: t for t in tools}
        self.ws = ws_manager
        self.intervention = intervention  # 新增
        self._cancellations: set[int] = set()
        self._called_history: list[str] = []
        self._tool_desc = ...
```

- [ ] **Step 3: run() 方法开头加生命周期注册**

修改 `backend/agent/engine.py:102-107`，在 `start_time = time.time()` 后添加：

```python
async def run(self, task_description: str, user_id: int, task_id: int,
              max_iterations: int = MAX_ITERATIONS):
    start_time = time.time()
    self._called_history = []
    task_id_str = str(task_id)

    # 生命周期：注册
    await self.intervention.register_task(task_id_str)

    _update_task(task_id, status="executing")
```

- [ ] **Step 4: 用 try/finally 包住循环体，finally 中调 complete_task**

修改 `backend/agent/engine.py:131-138`，在 `try:` 开始处保持不变，将 `except Exception` 改为 `finally` 模式：

```python
    try:
        for iteration in range(max_iterations):
            # ── 取消检查（已有）──────────────────────
            if task_id in self._cancellations:
                _update_task(task_id, status="cancelled")
                await self.ws.broadcast(task_id, "task_cancelled", {"task_id": task_id})
                self._cancellations.discard(task_id)
                return

            # ── 取消快车道（新增）─────────────────────
            if await self.intervention.has_cancellation(task_id_str):
                _update_task(task_id, status="cancelled")
                await self.ws.broadcast(task_id, "task_cancelled", {"task_id": task_id})
                self._cancellations.discard(task_id)
                return
```

- [ ] **Step 5: 在 Observation 之后插入反馈注入点**

修改 `backend/agent/engine.py:378`（`step_number += len(call_tasks)` 之后，下一轮循环之前）：

```python
                step_number += len(call_tasks)

                # ── 反馈注入点（新增）─────────────────
                fb_msgs, injected_ids = await self.intervention.drain(task_id_str)
                if fb_msgs:
                    messages.extend(fb_msgs)
                    for fb_id in injected_ids:
                        await self.ws.broadcast(task_id, "intervention_applied", {
                            "id": fb_id,
                            "step_number": step_number,
                        })
```

- [ ] **Step 6: 无工具调用（任务完成）时调 mark_digested**

修改 `backend/agent/engine.py:170-216`，在 `if not msg.tool_calls:` 代码块中，`final_answer` 赋值后、`_update_task` 之前加入：

```python
                if not msg.tool_calls:
                    final_answer = msg.content or "任务已完成。"

                    # 消化追踪（新增）
                    digest_events = await self.intervention.mark_digested(
                        task_id_str, msg.content or ""
                    )
                    for event in digest_events:
                        await self.ws.broadcast(task_id, event["type"], {
                            "id": event["id"],
                            "decision": event["decision"],
                            "summary": event["summary"],
                        })
```

- [ ] **Step 7: 异常处理和 finally 清理**

修改 `backend/agent/engine.py:399-408`，在 `except Exception` 后、函数结束前加入 `finally`：

```python
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            error_msg = f"任务执行失败: {str(e)}"
            _update_task(task_id, status="failed", final_answer=error_msg,
                         total_tokens=total_tokens, duration_ms=duration_ms)
            await self.ws.broadcast(task_id, "task_error", {
                "error": str(e),
                "last_step": step_number,
            })

        finally:
            # 生命周期：清理（正常/取消/异常全部覆盖）
            async for fb_id, reason in await self._complete_and_notify(task_id_str):
                await self.ws.broadcast(task_id, "intervention_rejected", {
                    "id": fb_id, "reason": reason,
                })
```

- [ ] **Step 8: 添加 _complete_and_notify 辅助方法**

在 `AgentEngine` 类的末尾（`cancel` 方法后面）添加：

```python
    async def _complete_and_notify(self, task_id_str: str):
        """将 complete_task 的 generator 转为异步可迭代"""
        gen = self.intervention.complete_task(task_id_str)
        async for item in gen:
            yield item
```

修改 `complete_task` 签名使其成为 async generator。实际上，`complete_task` 当前返回 generator，不支持 async for。改为：

```python
# 在 intervention.py 的 complete_task 中：
async def complete_task(self, task_id: str):
    async with self._lock:
        self._task_states[task_id] = "completed"
        pending = self._queues.pop(task_id, [])
        self._cancellations.discard(task_id)

    for fb in pending:
        if fb.status == "queued":
            await self._broadcast_rejected(fb.id, "task_already_completed")
```

不，InterventionHandler 本身没有 broadcast 能力。cleaner 的方案是 `complete_task` 返回被清理的 feedback id 列表，由 engine 负责发 rejected：

修改 `backend/agent/intervention.py` 中的 `complete_task`：

```python
async def complete_task(self, task_id: str) -> list[tuple[str, str]]:
    """返回 [(feedback_id, reason), ...]"""
    async with self._lock:
        self._task_states[task_id] = "completed"
        pending = self._queues.pop(task_id, [])
        self._cancellations.discard(task_id)

    rejected = []
    for fb in pending:
        if fb.status == "queued":
            rejected.append((fb.id, "task_already_completed"))
    return rejected
```

在 engine.py 的 finally 中：

```python
        finally:
            pending_rejections = await self.intervention.complete_task(task_id_str)
            for fb_id, reason in pending_rejections:
                await self.ws.broadcast(task_id, "intervention_rejected", {
                    "id": fb_id, "reason": reason,
                })
```

替代 `_complete_and_notify` 方法。

- [ ] **Step 9: 验证 import 链**

```bash
cd /d/agent_learning && python -c "from backend.agent.engine import AgentEngine; print('Engine import OK')"
```

Expected: `Engine import OK`

---

### Task 3: 修改 WebSocketManager 支持反馈消息

**Files:**
- Modify: `backend/agent/websocket_manager.py`

- [ ] **Step 1: 在 WebSocketManager 中添加 receive_feedback 方法**

修改 `backend/agent/websocket_manager.py`，在 `disconnect` 方法后添加：

```python
    def __init__(self):
        self._connections: dict[int, WebSocket] = {}
        self.confirmations: dict[int, dict] = {}
        self._agent_status_cache: dict[int, str] = {}  # 新增：Agent 状态缓存

    def update_agent_status(self, task_id: int, status: str):
        """engine 在状态变化时调用，供 ack 消息使用"""
        self._agent_status_cache[task_id] = status

    def get_agent_status(self, task_id: int) -> str:
        return self._agent_status_cache.get(task_id, "executing_tool")
```

- [ ] **Step 2: 修改 disconnect 清理 agent_status_cache**

```python
    def disconnect(self, task_id: int):
        self._connections.pop(task_id, None)
        self.confirmations.pop(task_id, None)
        self._agent_status_cache.pop(task_id, None)
```

---

### Task 4: 修改 server.py WebSocket handler 路由干预消息

**Files:**
- Modify: `backend/server.py:618-628`

- [ ] **Step 1: 创建全局 InterventionHandler 实例**

修改 `backend/server.py`，在 `ws_manager = WebSocketManager()` 行下方添加：

```python
# 读取实际代码中 ws_manager 的位置并在此行后添加
from backend.agent.intervention import InterventionHandler

# 在 ws_manager = WebSocketManager() 后：
intervention_handler = InterventionHandler()
```

- [ ] **Step 2: 修改 _get_engine 传入 intervention_handler**

修改 `backend/server.py:100-105`：

```python
def _get_engine(agent_mode: str):
    """为每个请求创建独立的引擎实例，保证会话隔离"""
    if agent_mode == "plan_solve":
        return PlanSolveEngine(deepseek, TOOLS, ws_manager)
    else:
        return AgentEngine(deepseek, TOOLS, ws_manager, intervention_handler)
```

- [ ] **Step 3: 修改 WebSocket handler 消息循环**

修改 `backend/server.py:618-628`，将 while 循环体改为：

```python
    try:
        while True:
            msg = await websocket.receive_json()
            action = msg.get("action")
            msg_type = msg.get("type")

            if action == "cancel" or msg_type == "cancel":
                await intervention_handler.cancel(str(task_id))
                # 同时保留旧的取消机制
                engine = _get_engine(task["agent_mode"])
                await engine.cancel(task_id)

            elif action == "confirm":
                ws_manager.confirmations[task_id] = msg

            elif msg_type == "intervention":
                # 用户反馈
                fb_id = msg.get("id") or msg.get("payload", {}).get("id", "")
                content = msg.get("content") or msg.get("payload", {}).get("content", "")
                priority = msg.get("priority") or msg.get("payload", {}).get("priority", "normal")

                feedback_item = {"id": fb_id, "content": content, "priority": priority}
                fb, merged_ids = await intervention_handler.receive(str(task_id), feedback_item)

                if fb is None:
                    # 队列满或空内容
                    reason = "queue_full"  # 或 empty_content
                    await websocket.send_json({
                        "type": "intervention_rejected",
                        "id": fb_id,
                        "reason": reason,
                    })
                else:
                    agent_status = ws_manager.get_agent_status(task_id)
                    ack_msg = {
                        "type": "intervention_ack",
                        "id": fb.id,
                        "priority": fb.priority,
                        "agent_status": agent_status,
                    }
                    if merged_ids:
                        ack_msg["merged"] = True
                    await websocket.send_json(ack_msg)

                    # 对被合并的 id 发 applied 通知
                    for mid in merged_ids:
                        await websocket.send_json({
                            "type": "intervention_applied",
                            "id": mid,
                            "merged_into": fb.id,
                        })

            elif msg_type == "retract":
                fb_id = msg.get("feedback_id") or msg.get("payload", {}).get("feedback_id", "")
                ok = await intervention_handler.retract(str(task_id), fb_id)
                if not ok:
                    await websocket.send_json({
                        "type": "intervention_rejected",
                        "id": fb_id,
                        "reason": "already_injected",
                    })

    except WebSocketDisconnect:
        ws_manager.disconnect(task_id)
```

- [ ] **Step 4: 验证 server 启动无语法错误**

```bash
cd /d/agent_learning && python -c "import backend.server; print('Server import OK')"
```

Expected: `Server import OK`

---

### Task 5: 创建 FeedbackDigestBanner 组件

**Files:**
- Create: `mobile/src/components/FeedbackDigestBanner.tsx`

- [ ] **Step 1: 创建组件**

```tsx
import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';

export interface DigestEntry {
  id: string;
  decision: 'adopted' | 'rejected' | 'partial' | 'unknown';
  summary: string;
  userContent?: string;  // 用户原始反馈文本
}

interface Props {
  entry: DigestEntry;
}

const decisionConfig: Record<string, { label: string; color: string; icon: string }> = {
  adopted: { label: '已采纳', color: '#4CAF50', icon: '✓' },
  rejected: { label: '未采纳', color: '#F44336', icon: '✗' },
  partial: { label: '部分采纳', color: '#FF9800', icon: '△' },
  unknown: { label: '已收到', color: '#9E9E9E', icon: '?' },
};

export default function FeedbackDigestBanner({ entry }: Props) {
  const [expanded, setExpanded] = useState(false);
  const config = decisionConfig[entry.decision] || decisionConfig.unknown;

  return (
    <TouchableOpacity
      style={styles.container}
      onPress={() => setExpanded(!expanded)}
      activeOpacity={0.7}
    >
      <View style={styles.header}>
        <Text style={styles.icon}>↗</Text>
        <View style={styles.headerText}>
          <Text style={styles.label}>方向调整</Text>
          {entry.userContent && (
            <Text style={styles.userText} numberOfLines={expanded ? 0 : 1}>
              "{entry.userContent}"
            </Text>
          )}
        </View>
        <View style={[styles.badge, { backgroundColor: config.color }]}>
          <Text style={styles.badgeText}>{config.icon} {config.label}</Text>
        </View>
      </View>

      {expanded && (
        <View style={styles.body}>
          <Text style={styles.summaryLabel}>Agent 回应：</Text>
          <Text style={styles.summary}>{entry.summary}</Text>
        </View>
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  container: {
    marginHorizontal: 16,
    marginVertical: 4,
    borderRadius: 10,
    backgroundColor: '#F0F4FF',
    borderLeftWidth: 3,
    borderLeftColor: '#4A90D9',
    padding: 12,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  icon: { fontSize: 18, marginRight: 8, color: '#4A90D9' },
  headerText: { flex: 1 },
  label: { fontSize: 13, fontWeight: '600', color: '#4A90D9' },
  userText: { fontSize: 12, color: '#666', marginTop: 2, fontStyle: 'italic' },
  badge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
  },
  badgeText: { fontSize: 11, color: '#fff', fontWeight: '600' },
  body: {
    marginTop: 10,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: '#DCE4F5',
  },
  summaryLabel: { fontSize: 12, fontWeight: '600', color: '#333', marginBottom: 4 },
  summary: { fontSize: 13, color: '#555', lineHeight: 18 },
});
```

---

### Task 6: 修改 useAgentTask Hook

**Files:**
- Modify: `mobile/src/hooks/useAgentTask.ts`

- [ ] **Step 1: 新增类型定义**

在文件顶部，现有 interface 下方添加：

```typescript
// 反馈状态机
export type FeedbackStatus = 'sending' | 'acknowledged' | 'rejected' | 'timeout' | 'applied' | 'digested';

export interface PendingFeedback {
  id: string;
  content: string;
  priority: string;
  status: FeedbackStatus;
  mergedInto?: string;
}

export interface DigestEntry {
  id: string;
  decision: 'adopted' | 'rejected' | 'partial' | 'unknown';
  summary: string;
  userContent?: string;
}

// StepListItem union 类型
export type StepListItem =
  | { kind: 'step'; data: AgentStep }
  | { kind: 'feedback_banner'; data: DigestEntry };
```

- [ ] **Step 2: 修改 UseAgentTaskReturn 接口**

```typescript
interface UseAgentTaskReturn {
  status: TaskStatus;
  taskStatus: string;
  steps: StepListItem[];           // 从 AgentStep[] 改为 StepListItem[]
  finalAnswer: string | null;
  error: string | null;
  totalSteps: number;
  totalTokens: number;
  durationMs: number;
  cancel: (reason?: string) => void;
  // 新增
  intervene: (id: string, content: string, priority?: string) => void;
  retract: (feedbackId: string) => void;
  pendingFeedbacks: Map<string, PendingFeedback>;
  agentDigestions: DigestEntry[];
  inputBarState: 'idle' | 'waiting_ack' | 'waiting_injection' | 'injected';
}
```

- [ ] **Step 3: 在 hook 函数体内新增状态**

```typescript
export function useAgentTask(taskId: number | null): UseAgentTaskReturn {
  // ... 现有状态 ...
  const [steps, setSteps] = useState<StepListItem[]>([]);  // 修改类型
  
  // 新增状态
  const [pendingFeedbacks, setPendingFeedbacks] = useState<Map<string, PendingFeedback>>(new Map());
  const [digestions, setAgentDigestions] = useState<DigestEntry[]>([]);
  const [inputBarState, setInputBarState] = useState<'idle' | 'waiting_ack' | 'waiting_injection' | 'injected'>('idle');
  const ackTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());
```

- [ ] **Step 4: 新增 intervene 方法**

```typescript
  const intervene = useCallback((id: string, content: string, priority: string = 'normal') => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    // 本地状态：sending
    setPendingFeedbacks(prev => {
      const next = new Map(prev);
      next.set(id, { id, content, priority, status: 'sending' });
      return next;
    });
    setInputBarState('waiting_ack');

    // 发送
    wsRef.current.send(JSON.stringify({
      type: 'intervention',
      id,
      content,
      priority,
    }));

    // 5 秒超时定时器
    const timer = setTimeout(() => {
      setPendingFeedbacks(prev => {
        const next = new Map(prev);
        const fb = next.get(id);
        if (fb && fb.status === 'sending') {
          next.set(id, { ...fb, status: 'timeout' });
        }
        return next;
      });
      setInputBarState('idle');
      ackTimersRef.current.delete(id);
    }, 5000);
    ackTimersRef.current.set(id, timer);
  }, []);
```

- [ ] **Step 5: 新增 retract 方法**

```typescript
  const retract = useCallback((feedbackId: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    // 乐观更新
    setPendingFeedbacks(prev => {
      const next = new Map(prev);
      const fb = next.get(feedbackId);
      if (fb && (fb.status === 'sending' || fb.status === 'acknowledged')) {
        next.set(feedbackId, { ...fb, status: 'digested' }); // terminal: retracted
      }
      return next;
    });

    wsRef.current.send(JSON.stringify({
      type: 'retract',
      feedback_id: feedbackId,
    }));
  }, []);
```

- [ ] **Step 6: 修改 cancel 方法**

```typescript
  const cancel = useCallback((reason?: string) => {
    setConnStatus('cancelling' as TaskStatus);  // 扩展 TaskStatus
    wsRef.current?.send(JSON.stringify({ type: 'cancel', reason }));
  }, []);
```

扩展 TaskStatus 类型：

```typescript
export type TaskStatus = 'connecting' | 'connected' | 'disconnected' | 'completed' | 'failed' | 'cancelled' | 'cancelling';
```

- [ ] **Step 7: 在 onmessage 中添加新消息类型处理**

在 `connect` 中的 `ws.onmessage` 回调里，在现有 `switch` 语句后添加：

```typescript
          // 新增消息类型
          case 'intervention_ack':
            setPendingFeedbacks(prev => {
              const next = new Map(prev);
              const fb = next.get(msg.id);
              if (fb) {
                next.set(msg.id, { ...fb, status: 'acknowledged' });
              }
              return next;
            });
            setInputBarState('waiting_injection');
            // 清除超时定时器
            const timer = ackTimersRef.current.get(msg.id);
            if (timer) { clearTimeout(timer); ackTimersRef.current.delete(msg.id); }
            break;

          case 'intervention_rejected':
            setPendingFeedbacks(prev => {
              const next = new Map(prev);
              const fb = next.get(msg.id);
              if (fb) {
                next.set(msg.id, { ...fb, status: 'rejected' });
              }
              return next;
            });
            setInputBarState('idle');
            break;

          case 'intervention_applied':
            if (msg.merged_into) {
              // 合并通知：标记被合并的反馈
              setPendingFeedbacks(prev => {
                const next = new Map(prev);
                const fb = next.get(msg.id);
                if (fb) {
                  next.set(msg.id, { ...fb, status: 'digested', mergedInto: msg.merged_into });
                }
                return next;
              });
            } else {
              setPendingFeedbacks(prev => {
                const next = new Map(prev);
                const fb = next.get(msg.id);
                if (fb) {
                  next.set(msg.id, { ...fb, status: 'applied' });
                }
                return next;
              });
              setInputBarState('injected');
            }
            break;

          case 'feedback_digested':
            setPendingFeedbacks(prev => {
              const next = new Map(prev);
              const fb = next.get(msg.id);
              if (fb) {
                next.set(msg.id, { ...fb, status: 'digested' });
              }
              return next;
            });
            // 插入 DigestBanner
            const entry: DigestEntry = {
              id: msg.id,
              decision: msg.decision as DigestEntry['decision'],
              summary: msg.summary,
              userContent: pendingFeedbacks.get(msg.id)?.content,
            };
            setAgentDigestions(prev => [...prev, entry]);
            setSteps(prev => [
              ...prev,
              { kind: 'feedback_banner', data: entry },
            ]);
            break;
```

### Task 7: 修改 agent/[id].tsx 集成干预 UI

**Files:**
- Modify: `mobile/app/agent/[id].tsx`

- [ ] **Step 1: 新增 import**

```tsx
import React, { useEffect, useState, useRef } from 'react';  // 加 useRef
import { View, Text, FlatList, TouchableOpacity, StyleSheet, ActivityIndicator, TextInput } from 'react-native';  // 加 TextInput
import FeedbackDigestBanner, { DigestEntry } from '../../src/components/FeedbackDigestBanner';
import { useAgentTask, StepListItem, TaskStatus } from '../../src/hooks/useAgentTask';
```

- [ ] **Step 2: 从 hook 获取新增返回值**

```tsx
  const {
    status: connStatus,
    taskStatus,
    steps,
    finalAnswer,
    error,
    totalTokens,
    durationMs,
    cancel,
    intervene,            // 新增
    pendingFeedbacks,     // 新增
    agentDigestions,      // 新增
    inputBarState,        // 新增
  } = useAgentTask(/* ... */);
```

- [ ] **Step 3: 新增底部输入条状态变量**

在组件中添加：

```tsx
  const [feedbackInput, setFeedbackInput] = useState('');
  const isRunning = connStatus === 'connected' || connStatus === 'connecting';
```

- [ ] **Step 4: 修改 FlatList renderItem 支持 union 类型**

```tsx
      <FlatList
        data={steps}
        keyExtractor={(item) => item.kind === 'feedback_banner' ? `digest-${item.data.id}` : `step-${item.data.step_num}`}
        renderItem={({ item }) => {
          if (item.kind === 'feedback_banner') {
            return <FeedbackDigestBanner entry={item.data} />;
          }
          return <AgentStepCard step={item.data} />;
        }}
        style={styles.list}
        // ...
      />
```

- [ ] **Step 5: 新增底部输入条**

在取消按钮之前添加：

```tsx
      {/* Intervention input bar */}
      {isRunning && (
        <View style={styles.interventionBar}>
          <TextInput
            style={[
              styles.interventionInput,
              inputBarState !== 'idle' && inputBarState !== 'injected' && styles.interventionInputDisabled,
            ]}
            value={feedbackInput}
            onChangeText={setFeedbackInput}
            placeholder={
              inputBarState === 'waiting_ack' ? '正在发送...'
              : inputBarState === 'waiting_injection' ? 'Agent 下次思考时会处理'
              : '告诉 Agent 调整方向...'
            }
            placeholderTextColor={inputBarState === 'idle' || inputBarState === 'injected' ? '#999' : '#ccc'}
            editable={inputBarState === 'idle' || inputBarState === 'injected'}
            multiline={false}
            returnKeyType="send"
            onSubmitEditing={() => {
              const content = feedbackInput.trim();
              if (!content) return;
              const id = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
              intervene(id, content, 'normal');
              setFeedbackInput('');
            }}
          />
          <TouchableOpacity
            style={[
              styles.interventionSendBtn,
              (inputBarState !== 'idle' && inputBarState !== 'injected') && styles.sendBtnDisabled,
            ]}
            onPress={() => {
              const content = feedbackInput.trim();
              if (!content) return;
              const id = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
              intervene(id, content, 'normal');
              setFeedbackInput('');
            }}
            disabled={inputBarState !== 'idle' && inputBarState !== 'injected'}
          >
            <Text style={styles.sendBtnText}>发送</Text>
          </TouchableOpacity>
        </View>
      )}
```

- [ ] **Step 6: 新增样式**

```tsx
  interventionBar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  interventionInput: {
    flex: 1,
    height: 40,
    backgroundColor: '#f5f5f5',
    borderRadius: 20,
    paddingHorizontal: 16,
    fontSize: 14,
    color: '#333',
  },
  interventionInputDisabled: {
    backgroundColor: '#efefef',
    color: '#aaa',
  },
  interventionSendBtn: {
    marginLeft: 10,
    backgroundColor: '#4A90D9',
    borderRadius: 20,
    paddingHorizontal: 18,
    paddingVertical: 8,
  },
  sendBtnDisabled: {
    backgroundColor: '#ccc',
  },
  sendBtnText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
```

---

### Task 8: 创建审校 Prompt

**Files:**
- Create: `backend/agent/review_prompt.py`

- [ ] **Step 1: 创建文件**

```python
"""
副 Agent 审校 System Prompt — 三步方法论
"""

REVIEW_SYSTEM_PROMPT = """You are a plan reviewer. Your job is to find structural flaws in execution plans — NOT to execute, NOT to guess what the user wants.

You will receive a plan (a list of steps). Follow this methodology:

## Step 1: Map Each Step to a Concrete Goal
- Identify vague descriptions (e.g., "analyze data" — analyze what? how?)
- Each step should have a clear, measurable output
- Flag steps that cannot be mapped

## Step 2: Run State Machine Enumeration
For each entity mentioned in the plan, enumerate:
- What states can it be in? (success / failure / timeout / empty result / permission denied)
- Are failure branches explicitly covered?
- Is there a fallback path for each failure mode?

## Step 3: Check Timing and Dependencies
- Is the step order correct?
- Are implicit dependencies respected (step B needs step A's result)?
- Are parallel steps free of data races?

## Output Format

You MUST respond in this exact JSON structure. No other text:

```json
{
  "missing_steps": [
    {
      "description": "...",
      "why_important": "...",
      "suggested_insert_after": "..."
    }
  ],
  "flawed_logic": [
    {
      "step_reference": "...",
      "issue": "...",
      "suggested_fix": "..."
    }
  ],
  "boundary_gaps": [
    {
      "scenario": "...",
      "impact": "...",
      "suggested_handling": "..."
    }
  ],
  "suggestions": [
    {
      "aspect": "performance|readability|robustness|user_experience",
      "current_approach": "...",
      "alternative": "..."
    }
  ]
}
```

All four arrays can be empty. If all are empty, the plan has no structural issues.

Be ruthless: false positives are better than missed problems. But do NOT fabricate issues — only flag what you can articulate a reason for."""
```

---

### Task 9: 修改 PlanSolveEngine 加入审校阶段

**Files:**
- Modify: `backend/agent/plan_solve_engine.py`

- [ ] **Step 1: 添加 import**

在文件顶部添加：

```python
from .review_prompt import REVIEW_SYSTEM_PROMPT
```

- [ ] **Step 2: 修改 __init__ 接受 llm_client 以便创建副 Agent**

```python
class PlanSolveEngine:
    def __init__(self, deepseek, tools: list[Tool], ws_manager: WebSocketManager):
        self.deepseek = deepseek
        self.tools = tools
        self.tool_map = {t.name: t for t in tools}
        self.ws = ws_manager
        self._cancellations: set[int] = set()
        self._tool_desc = ...
```

不需要修改 `__init__`——`self.deepseek` 已经可以用来创建副 Agent 的 LLM 调用。

- [ ] **Step 3: 在规划阶段后插入审校调用**

修改 `backend/agent/plan_solve_engine.py`，在阶段 1（制定计划）完成后、阶段 2（执行）开始前插入：

在 `_update_step(task_id, step_number, "completed", tool_result=...plan...)` 和 `await self.ws.broadcast(task_id, "step_complete", ...)` 之后，`results = []` 之前：

```python
            # ── 阶段 1.5：副 Agent 审校方案 ──────────────
            step_number += 1
            _save_step(task_id, step_number, "review", status="running",
                       thought="副 Agent 审校方案中...")

            await self.ws.broadcast(task_id, "step_start", {
                "step_num": step_number,
                "type": "review",
                "message": "Reviewing plan for gaps...",
            })

            # 构建审校消息（独立上下文）
            plan_for_review = json.dumps({
                "task": task_description,
                "plan": plan,
            }, ensure_ascii=False)

            review_messages = [
                {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": f"Review this plan:\n\n{plan_for_review}"},
            ]

            review_resp = await self._call_llm(review_messages)  # 无工具
            total_tokens += review_resp.usage.total_tokens if review_resp.usage else 0
            review_text = review_resp.choices[0].message.content or "{}"

            # 解析审校结果 JSON
            try:
                # 提取 JSON 块（可能被 markdown 代码块包裹）
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', review_text)
                if json_match:
                    review_text = json_match.group(1).strip()
                review_data = json.loads(review_text)
            except json.JSONDecodeError:
                review_data = {
                    "missing_steps": [],
                    "flawed_logic": [],
                    "boundary_gaps": [],
                    "suggestions": [],
                    "_parse_error": review_text[:200]
                }

            # 检查是否有实际发现
            has_findings = any(
                review_data.get(k)
                for k in ["missing_steps", "flawed_logic", "boundary_gaps", "suggestions"]
            )

            _update_step(task_id, step_number, "completed",
                         tool_result={"review": review_data}, duration_ms=0)

            await self.ws.broadcast(task_id, "step_complete", {
                "step_num": step_number,
                "type": "review",
                "content": "方案审校完成" if not has_findings else f"发现 {sum(len(review_data.get(k, [])) for k in ['missing_steps','flawed_logic','boundary_gaps','suggestions'])} 条改进建议",
                "review": review_data,
            })

            # 将审校结果注入主 Agent 上下文
            if has_findings:
                review_summary = f"""[方案审校结果]
副 Agent 对方案进行了审校，发现以下改进点：

遗漏步骤：{json.dumps(review_data.get('missing_steps', []), ensure_ascii=False, indent=2)}
逻辑缺陷：{json.dumps(review_data.get('flawed_logic', []), ensure_ascii=False, indent=2)}
边界缺口：{json.dumps(review_data.get('boundary_gaps', []), ensure_ascii=False, indent=2)}
优化建议：{json.dumps(review_data.get('suggestions', []), ensure_ascii=False, indent=2)}

请逐条判断是否采纳，修正方案后继续执行。"""

                # 将审校结果追加到执行上下文
                context += f"\n\n{review_summary}"
```

- [ ] **Step 4: 验证 import**

```bash
cd /d/agent_learning && python -c "from backend.agent.plan_solve_engine import PlanSolveEngine; from backend.agent.review_prompt import REVIEW_SYSTEM_PROMPT; print('PlanSolve+Review import OK')"
```

Expected: `PlanSolve+Review import OK`

---

### Task 10: 创建 ReviewBanner 组件

**Files:**
- Create: `mobile/src/components/ReviewBanner.tsx`

- [ ] **Step 1: 创建组件**

```tsx
import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import MarkdownRenderer from './MarkdownRenderer';

export interface ReviewEntry {
  missing_steps?: Array<{ description: string; why_important: string; suggested_insert_after: string }>;
  flawed_logic?: Array<{ step_reference: string; issue: string; suggested_fix: string }>;
  boundary_gaps?: Array<{ scenario: string; impact: string; suggested_handling: string }>;
  suggestions?: Array<{ aspect: string; current_approach: string; alternative: string }>;
}

interface Props {
  entry: ReviewEntry;
}

export default function ReviewBanner({ entry }: Props) {
  const [expanded, setExpanded] = useState(false);

  const totalFindings =
    (entry.missing_steps?.length || 0) +
    (entry.flawed_logic?.length || 0) +
    (entry.boundary_gaps?.length || 0) +
    (entry.suggestions?.length || 0);

  if (totalFindings === 0) {
    return (
      <View style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.icon}>✓</Text>
          <Text style={styles.label}>方案已优化 — 未发现结构性问题</Text>
        </View>
      </View>
    );
  }

  return (
    <TouchableOpacity
      style={styles.container}
      onPress={() => setExpanded(!expanded)}
      activeOpacity={0.7}
    >
      <View style={styles.header}>
        <Text style={styles.icon}>↗</Text>
        <Text style={styles.label}>方案已优化 — {totalFindings} 条改进建议</Text>
        <Text style={styles.expandHint}>{expanded ? '收起' : '展开'}</Text>
      </View>

      {expanded && (
        <View style={styles.body}>
          {entry.missing_steps?.map((s, i) => (
            <View key={`ms-${i}`} style={styles.findingItem}>
              <Text style={styles.findingType}>[遗漏] {s.description}</Text>
              <Text style={styles.findingDetail}>重要性：{s.why_important}</Text>
              <Text style={styles.findingDetail}>建议位置：{s.suggested_insert_after}</Text>
            </View>
          ))}
          {entry.flawed_logic?.map((f, i) => (
            <View key={`fl-${i}`} style={styles.findingItem}>
              <Text style={styles.findingType}>[逻辑] {f.step_reference}</Text>
              <Text style={styles.findingDetail}>问题：{f.issue}</Text>
              <Text style={styles.findingDetail}>建议：{f.suggested_fix}</Text>
            </View>
          ))}
          {entry.boundary_gaps?.map((b, i) => (
            <View key={`bg-${i}`} style={styles.findingItem}>
              <Text style={styles.findingType}>[边界] {b.scenario}</Text>
              <Text style={styles.findingDetail}>影响：{b.impact}</Text>
              <Text style={styles.findingDetail}>处理：{b.suggested_handling}</Text>
            </View>
          ))}
          {entry.suggestions?.map((s, i) => (
            <View key={`sg-${i}`} style={styles.findingItem}>
              <Text style={styles.findingType}>[{s.aspect}] {s.current_approach}</Text>
              <Text style={styles.findingDetail}>替代：{s.alternative}</Text>
            </View>
          ))}
        </View>
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  container: {
    marginHorizontal: 16,
    marginVertical: 4,
    borderRadius: 10,
    backgroundColor: '#F0FFF0',
    borderLeftWidth: 3,
    borderLeftColor: '#4CAF50',
    padding: 12,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  icon: { fontSize: 18, marginRight: 8, color: '#4CAF50' },
  label: { flex: 1, fontSize: 13, fontWeight: '600', color: '#4CAF50' },
  expandHint: { fontSize: 12, color: '#888' },
  body: {
    marginTop: 10,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: '#DCEFD5',
  },
  findingItem: {
    marginBottom: 10,
    paddingLeft: 8,
    borderLeftWidth: 2,
    borderLeftColor: '#E0E0E0',
  },
  findingType: { fontSize: 13, fontWeight: '600', color: '#333', marginBottom: 2 },
  findingDetail: { fontSize: 12, color: '#666', marginBottom: 1 },
});
```

---

### Task 11: 修改 agent/[id].tsx 支持 ReviewBanner

**Files:**
- Modify: `mobile/app/agent/[id].tsx`

- [ ] **Step 1: 导入 ReviewBanner 和扩展 StepListItem**

```tsx
import ReviewBanner, { ReviewEntry } from '../../src/components/ReviewBanner';
```

- [ ] **Step 2: 扩展 StepListItem union 类型**

在 `useAgentTask` 返回的 steps 中，review 步骤通过 `step_start` / `step_complete` 的 `type: "review"` 来识别。`agent/[id].tsx` 的 renderItem 添加：

```tsx
        renderItem={({ item }) => {
          if (item.kind === 'feedback_banner') {
            return <FeedbackDigestBanner entry={item.data} />;
          }
          if (item.kind === 'review_banner') {
            return <ReviewBanner entry={item.data} />;
          }
          // 检查是否是 review 类型的 step
          if (item.data.type === 'review' && item.data.result?.review) {
            return <ReviewBanner entry={item.data.result.review as ReviewEntry} />;
          }
          return <AgentStepCard step={item.data} />;
        }}
```

- [ ] **Step 3: 在 useAgentTask 中处理 review 类型的 step_complete**

修改 `mobile/src/hooks/useAgentTask.ts` 的 `onmessage`，在 `step_complete` 处理中添加 review 类型的特殊处理：

在 `case 'step_complete':` 中，当 `msg.type === 'review' && msg.review` 时，将步骤转为 review_banner：

```typescript
          case 'step_complete':
            setSteps((prev) => {
              const updated = prev.map((s) => {
                if (s.kind === 'step' && s.data.step_num === msg.step_num) {
                  if (msg.type === 'review' && msg.review) {
                    // 转为 review_banner
                    return {
                      kind: 'review_banner' as const,
                      data: msg.review,
                    };
                  }
                  return {
                    kind: 'step' as const,
                    data: {
                      ...s.data,
                      status: 'completed' as const,
                      result: msg.result,
                      content: msg.content,
                      duration_ms: msg.duration_ms,
                    },
                  };
                }
                return s;
              });
              return updated;
            });
            break;
```

但这太复杂。更简单的方案：在 `agent/[id].tsx` 的 `displaySteps` 构建中处理：

```tsx
  const displaySteps: StepListItem[] = steps.length > 0 ? steps : (taskDetail?.steps?.map((s) => ({
    kind: 'step' as const,
    data: {
      step_num: s.step_number,
      type: s.step_type,
      status: s.status as AgentStep['status'],
      tool_name: s.tool_name || undefined,
      args: parseJsonField(s.tool_args),
      result: parseJsonField(s.tool_result),
      content: s.thought || undefined,
      duration_ms: s.duration_ms || undefined,
    },
  })) || []);

  // 将 review 类型的步骤转换为 review_banner
  const displayItems: StepListItem[] = displaySteps.map(item => {
    if (item.kind === 'step' && item.data.type === 'review' && item.data.result?.review) {
      return { kind: 'review_banner' as const, data: item.data.result.review as ReviewEntry };
    }
    return item;
  });
```

FlatList data 使用 `displayItems`。

---

### Task 12: 端到端验证

- [ ] **Step 1: 启动后端验证无导入错误**

```bash
cd /d/agent_learning && python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000 &
sleep 3
curl -s http://127.0.0.1:8000/ | head -5
```

Expected: 正常启动，返回 JSON

- [ ] **Step 2: 验证 InterventionHandler 独立逻辑**

```bash
cd /d/agent_learning && python -c "
import asyncio
from backend.agent.intervention import InterventionHandler

async def test():
    h = InterventionHandler()
    await h.register_task('test-1')

    # 正常 feedback
    fb, merged = await h.receive('test-1', {'id': 'fb1', 'content': '不要查华北区', 'priority': 'high'})
    assert fb.id == 'fb1'
    assert fb.status == 'queued'
    print('PASS: receive')

    # 幂等
    fb2, merged2 = await h.receive('test-1', {'id': 'fb1', 'content': 'duplicate', 'priority': 'high'})
    assert fb2.id == 'fb1'
    assert len(merged2) == 0
    print('PASS: idempotent')

    # 同优先级合并
    fb3, merged3 = await h.receive('test-1', {'id': 'fb2', 'content': '只看华南区', 'priority': 'high'})
    assert len(merged3) == 1
    assert '华南区' in fb3.content and '华北区' in fb3.content
    print('PASS: merge')

    # drain
    msgs, ids = await h.drain('test-1')
    assert len(msgs) >= 1
    assert 'fb1' in ids
    print('PASS: drain')

    # retract (已注入, 应失败)
    ok = await h.retract('test-1', 'fb1')
    assert not ok
    print('PASS: retract after drain')

    # 取消
    await h.cancel('test-1')
    assert await h.has_cancellation('test-1')
    print('PASS: cancel')

    # complete
    rej = await h.complete_task('test-1')
    print(f'PASS: complete_task, rejected {len(rej)} pending')
    print('ALL TESTS PASSED')

asyncio.run(test())
"
```

Expected: ALL TESTS PASSED

- [ ] **Step 3: 验证前端 TypeScript 编译**

```bash
cd /d/agent_learning/mobile && npx tsc --noEmit 2>&1 | head -20
```

Expected: 无新增 TypeScript 错误

- [ ] **Step 4: 提交**

```bash
git add backend/agent/intervention.py backend/agent/engine.py backend/agent/websocket_manager.py backend/server.py
git add backend/agent/review_prompt.py backend/agent/plan_solve_engine.py
git add mobile/src/components/FeedbackDigestBanner.tsx mobile/src/components/ReviewBanner.tsx
git add mobile/src/hooks/useAgentTask.ts mobile/app/agent/[id].tsx
git commit -m "feat: InterventionHandler + ExpertMode review — visual controllable agent"
```

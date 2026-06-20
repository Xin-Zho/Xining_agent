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
        self._events: dict[str, list[dict]] = {}  # task_id → [SSE events, ...]
        self._events_lock = asyncio.Lock()  # 独立锁，避免与主锁竞争

    # ── 生命周期 ──────────────────────────────────────────

    async def register_task(self, task_id: str):
        async with self._lock:
            self._task_states[task_id] = "running"
            if task_id not in self._queues:
                self._queues[task_id] = []

    async def complete_task(self, task_id: str) -> list[tuple[str, str]]:
        """
        任务结束（正常/取消/失败），清理资源。

        Returns:
            [(feedback_id, reason), ...] — 待通知的未注入反馈
        """
        async with self._lock:
            self._task_states[task_id] = "completed"
            pending = self._queues.pop(task_id, [])
            self._cancellations.discard(task_id)

        rejected = []
        for fb in pending:
            if fb.status == "queued":
                rejected.append((fb.id, "task_already_completed"))

        # 清理事件队列
        async with self._events_lock:
            self._events.pop(task_id, None)

        return rejected

    # ── SSE 事件轮询 ──────────────────────────────────────

    async def _push_event(self, task_id: str, event: dict):
        """供 engine 广播事件时同步推送到 SSE 事件队列"""
        async with self._events_lock:
            if task_id not in self._events:
                self._events[task_id] = []
            self._events[task_id].append(event)

    async def poll_events(self, task_id: str, after_index: int) -> tuple[list[dict], int]:
        """
        SSE 轮询：返回 after_index 之后的新事件。

        Returns:
            (events, new_index)
        """
        async with self._events_lock:
            events = self._events.get(task_id, [])
            if after_index >= len(events):
                return [], after_index
            new_events = events[after_index:]
            return new_events, len(events)

    # ── 反馈接收 ──────────────────────────────────────────

    async def receive(self, task_id: str, feedback_item: dict):
        """
        接收一条用户反馈。幂等：同 id 重复调用直接返回已有对象。

        Args:
            task_id: 任务 ID
            feedback_item: { id, content, priority? }

        Returns:
            (Feedback, merged_ids) — 主反馈对象 + 被合并的 id 列表
            或 (None, []) — 队列满或内容为空，调用方应发 intervention_rejected
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
                evicted = self._evict_low_priority(queue)
                if evicted is None:
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
        low 优先级跳过，留在队列中由 complete_task 处理。

        Returns:
            (messages, injected_ids)
        """
        async with self._lock:
            queue = self._queues.get(task_id, [])
            if not queue:
                return [], []

            # low 优先级不在此注入，留给 complete_task 处理
            pending = [fb for fb in queue if fb.status == "queued" and fb.priority != "low"]
            pending.sort(key=lambda fb: _PRIORITY_ORDER.get(fb.priority, 2))

            messages = []
            injected_ids = []

            for fb in pending:
                fb.status = "injected"
                injected_ids.append(fb.id)
                msg_content = (
                    f"[用户反馈 (id={fb.id})] {fb.content}\n"
                    f"请结合当前任务判断是否需要调整方向。"
                    f"如调整，请在回复末尾标注 [摘要: adopted|rejected|partial, {fb.id}, 说明文字]。"
                )
                messages.append({"role": "system", "content": msg_content})

            # 同步推送到 SSE 事件队列
            for fb_id in injected_ids:
                await self._push_event(task_id, {
                    "type": "intervention_applied",
                    "id": fb_id,
                })

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

    async def mark_digested(self, task_id: str, thought_content: str) -> list[dict]:
        """
        从 LLM Thought 内容中解析反馈消化摘要。

        期望格式: [摘要: adopted|rejected|partial|unknown, feedback_id, 说明文字]

        Returns:
            feedback_digested 事件列表
        """
        # 摘要带 feedback_id，按 id 精确匹配
        pattern = r'\[摘要:\s*(adopted|rejected|partial|unknown)\s*[,，]\s*(\S+)\s*[,，]\s*(.+?)\]'
        matches = re.findall(pattern, thought_content)

        async with self._lock:
            queue = self._queues.get(task_id, [])

            if not matches:
                # 无结构化摘要 → 第一条 injected 标记为 unknown
                undigested = [fb for fb in queue if fb.status == "injected"]
                if undigested:
                    fb = undigested[0]
                    fb.status = "digested"
                    events = [{
                        "type": "feedback_digested",
                        "id": fb.id,
                        "decision": "unknown",
                        "summary": "Agent 已收到反馈，但未明确结构化回应"
                    }]
                    for evt in events:
                        await self._push_event(task_id, evt)
                    return events
                return []

            # 按 feedback_id 精确匹配
            fb_map = {fb.id: fb for fb in queue if fb.status == "injected"}
            events = []
            for decision, fb_id, summary in matches:
                fb = fb_map.get(fb_id.strip())
                if fb:
                    fb.status = "digested"
                    events.append({
                        "type": "feedback_digested",
                        "id": fb.id,
                        "decision": decision.strip(),
                        "summary": summary.strip()
                    })

            for evt in events:
                await self._push_event(task_id, evt)
            return events

    # ── agent_status 查询 ────────────────────────────────

    def get_agent_status(self, task_id: str) -> Optional[str]:
        """返回 Agent 当前状态用于 ack 消息 (thinking/executing_tool/idle)"""
        return "executing_tool" if task_id in self._task_states else None

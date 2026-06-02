import json
from datetime import datetime, timezone

from fastapi import WebSocket

from ..database import get_db


class WebSocketManager:
    def __init__(self):
        self._connections: dict[int, WebSocket] = {}
        self.confirmations: dict[int, dict] = {}

    async def connect(self, task_id: int, ws: WebSocket, user_id: int):
        await ws.accept()
        self._connections[task_id] = ws

    def disconnect(self, task_id: int):
        self._connections.pop(task_id, None)
        self.confirmations.pop(task_id, None)

    async def broadcast(self, task_id: int, event_type: str, payload: dict):
        ws = self._connections.get(task_id)
        if ws is None:
            return
        try:
            await ws.send_json({"type": event_type, **payload})
        except Exception:
            self.disconnect(task_id)

    async def request_confirmation(self, task_id: int, step_num: int,
                                   tool_name: str, args: dict) -> bool:
        await self.broadcast(task_id, "confirmation_required", {
            "step_num": step_num,
            "tool_name": tool_name,
            "args": args,
        })
        import asyncio
        for _ in range(120):
            confirm = self.confirmations.pop(task_id, None)
            if confirm is not None:
                return confirm.get("approved", False)
            await asyncio.sleep(0.5)
        return False


def _save_step(task_id: int, step_number: int, step_type: str, status: str = "running",
               tool_name: str = None, tool_args: dict = None, thought: str = None):
    conn = get_db()
    conn.execute(
        """INSERT INTO agent_steps (task_id, step_number, status, step_type, tool_name, tool_args, thought)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (task_id, step_number, status, step_type, tool_name,
         json.dumps(tool_args, ensure_ascii=False) if tool_args else None,
         thought),
    )
    conn.commit()
    conn.close()


def _update_step(
    task_id: int,
    step_number: int,
    status: str,
    tool_result: dict = None,
    duration_ms: int = None,
    tool_args: dict = None,
):
    conn = get_db()
    if tool_result is not None and tool_args is not None:
        conn.execute(
            """UPDATE agent_steps
               SET status = ?, tool_result = ?, duration_ms = ?, tool_args = ?
               WHERE task_id = ? AND step_number = ?""",
            (status, json.dumps(tool_result, ensure_ascii=False), duration_ms,
             json.dumps(tool_args, ensure_ascii=False), task_id, step_number),
        )
    elif tool_result is not None:
        conn.execute(
            "UPDATE agent_steps SET status = ?, tool_result = ?, duration_ms = ? WHERE task_id = ? AND step_number = ?",
            (status, json.dumps(tool_result, ensure_ascii=False), duration_ms, task_id, step_number),
        )
    elif tool_args is not None:
        conn.execute(
            "UPDATE agent_steps SET status = ?, duration_ms = ?, tool_args = ? WHERE task_id = ? AND step_number = ?",
            (status, duration_ms, json.dumps(tool_args, ensure_ascii=False), task_id, step_number),
        )
    else:
        conn.execute(
            "UPDATE agent_steps SET status = ?, duration_ms = ? WHERE task_id = ? AND step_number = ?",
            (status, duration_ms, task_id, step_number),
        )
    conn.commit()
    conn.close()


def _update_task(task_id: int, **kwargs):
    conn = get_db()
    set_clause = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [task_id]
    conn.execute(
        f"UPDATE agent_tasks SET {set_clause}, updated_at = datetime('now') WHERE id = ?",
        values,
    )
    conn.commit()
    conn.close()

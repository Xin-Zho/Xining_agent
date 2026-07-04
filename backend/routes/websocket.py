"""WebSocket 路由 — Agent 实时进度推送"""
import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.exceptions import HTTPException

from ..database import get_db
from ..auth import verify_token
from ..dependencies import ws_manager, intervention_handler, get_engine

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/agent/{task_id}")
async def agent_websocket(websocket: WebSocket, task_id: int):
    """WebSocket 连接 — Token 通过首条消息传递，不放 URL 避免日志泄露"""
    await websocket.accept()

    try:
        auth_msg = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
    except asyncio.TimeoutError:
        await websocket.close(code=4001, reason="Auth timeout")
        return
    except WebSocketDisconnect:
        return

    token = auth_msg.get("token") if isinstance(auth_msg, dict) else None
    if not token:
        await websocket.close(code=4001, reason="Missing token")
        return

    try:
        user = verify_token(token)
    except HTTPException:
        await websocket.close(code=4002, reason="Invalid token")
        return

    conn = get_db("agent")
    task = conn.execute(
        "SELECT id, agent_mode FROM agent_tasks WHERE id = ? AND user_id = ?",
        (task_id, user["id"]),
    ).fetchone()
    conn.close()
    if not task:
        await websocket.close(code=4004, reason="Task not found")
        return

    await websocket.send_json({"type": "auth_ok"})
    await ws_manager.connect(task_id, websocket, user["id"])

    try:
        while True:
            msg = await websocket.receive_json()
            action = msg.get("action")
            msg_type = msg.get("type")

            if action == "cancel" or msg_type == "cancel":
                await intervention_handler.cancel(str(task_id))
                engine = get_engine(task["agent_mode"])
                await engine.cancel(task_id)

            elif action == "confirm":
                ws_manager.confirmations[task_id] = msg

            elif msg_type == "intervention":
                if task["agent_mode"] != "react":
                    await websocket.send_json({
                        "type": "intervention_rejected",
                        "id": msg.get("id", ""), "reason": "unsupported_mode",
                    })
                    continue

                fb_id = msg.get("id") or msg.get("payload", {}).get("id", "")
                content = msg.get("content") or msg.get("payload", {}).get("content", "")
                priority = msg.get("priority") or msg.get("payload", {}).get("priority", "normal")

                feedback_item = {"id": fb_id, "content": content, "priority": priority}
                fb, merged_ids = await intervention_handler.receive(str(task_id), feedback_item)

                if fb is None:
                    await websocket.send_json({
                        "type": "intervention_rejected", "id": fb_id,
                        "reason": "queue_full_or_empty",
                    })
                else:
                    agent_status = ws_manager.get_agent_status(task_id)
                    ack_msg = {
                        "type": "intervention_ack", "id": fb.id,
                        "priority": fb.priority, "agent_status": agent_status,
                    }
                    if merged_ids:
                        ack_msg["merged"] = True
                    await websocket.send_json(ack_msg)

                    for mid in merged_ids:
                        await websocket.send_json({
                            "type": "intervention_applied", "id": mid,
                            "merged_into": fb.id,
                        })

            elif msg_type == "retract":
                fb_id = msg.get("feedback_id") or msg.get("payload", {}).get("feedback_id", "")
                ok = await intervention_handler.retract(str(task_id), fb_id)
                if not ok:
                    await websocket.send_json({
                        "type": "intervention_rejected", "id": fb_id,
                        "reason": "already_injected",
                    })
    except WebSocketDisconnect:
        ws_manager.disconnect(task_id)

"""Agent 任务路由 — CRUD、确认、SSE 兼容流式、用户干预"""
import asyncio
import json

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..models import CreateAgentTaskRequest
from ..database import get_db
from ..auth import get_current_user
from ..dependencies import (
    deepseek, ws_manager, intervention_handler, dialogue_logger, get_engine,
    verify_token_from_header,
)

router = APIRouter(prefix="/api", tags=["agent"])


# ── Legacy request model ──────────────────────────────────────────────

class LegacyAgentRequest(BaseModel):
    messages: list[dict]
    model: str = "chat"
    mode: str = "react"


# ── Task CRUD ─────────────────────────────────────────────────────────

@router.post("/agent/tasks")
async def create_agent_task(
    body: CreateAgentTaskRequest, user: dict = Depends(get_current_user)
):
    agent_mode = body.agent_mode or "react"
    conn = get_db("agent")
    cur = conn.execute(
        """INSERT INTO agent_tasks (user_id, title, description, status, agent_mode, conversation_id)
           VALUES (?, ?, ?, 'pending', ?, ?)""",
        (user["id"], body.description[:50], body.description, agent_mode, body.conversation_id),
    )
    conn.commit()
    task_id = cur.lastrowid
    conn.close()

    engine = get_engine(agent_mode)

    asyncio.create_task(engine.run(
        task_description=body.description,
        user_id=user["id"],
        task_id=task_id,
    ))

    return {"task_id": task_id, "status": "pending", "agent_mode": agent_mode}


@router.get("/agent/tasks")
def list_agent_tasks(user: dict = Depends(get_current_user)):
    conn = get_db("agent")
    rows = conn.execute(
        """SELECT id, title, status, agent_mode, total_tokens, duration_ms, created_at
           FROM agent_tasks WHERE user_id = ? ORDER BY created_at DESC""",
        (user["id"],),
    ).fetchall()
    conn.close()
    return [
        {
            "id": r["id"], "title": r["title"], "status": r["status"],
            "agent_mode": r["agent_mode"],
            "total_tokens": r["total_tokens"], "duration_ms": r["duration_ms"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


@router.get("/agent/tasks/{task_id}")
def get_agent_task(task_id: int, user: dict = Depends(get_current_user)):
    conn = get_db("agent")
    task = conn.execute(
        "SELECT * FROM agent_tasks WHERE id = ? AND user_id = ?",
        (task_id, user["id"]),
    ).fetchone()
    if not task:
        conn.close()
        raise HTTPException(status_code=404, detail="任务不存在")

    steps = conn.execute(
        """SELECT step_number, status, step_type, tool_name, tool_args,
                  tool_result, thought, duration_ms, created_at
           FROM agent_steps WHERE task_id = ? ORDER BY step_number""",
        (task_id,),
    ).fetchall()
    conn.close()

    return {
        "id": task["id"],
        "title": task["title"],
        "description": task["description"],
        "status": task["status"],
        "agent_mode": task["agent_mode"],
        "plan_json": task["plan_json"],
        "final_answer": task["final_answer"],
        "conversation_id": task["conversation_id"],
        "total_tokens": task["total_tokens"],
        "duration_ms": task["duration_ms"],
        "created_at": task["created_at"],
        "steps": [
            {
                "step_number": s["step_number"], "status": s["status"],
                "step_type": s["step_type"], "tool_name": s["tool_name"],
                "tool_args": s["tool_args"], "tool_result": s["tool_result"],
                "thought": s["thought"], "duration_ms": s["duration_ms"],
            }
            for s in steps
        ],
    }


@router.delete("/agent/tasks/{task_id}")
def delete_agent_task(task_id: int, user: dict = Depends(get_current_user)):
    conn = get_db("agent")
    task = conn.execute(
        "SELECT id FROM agent_tasks WHERE id = ? AND user_id = ?",
        (task_id, user["id"]),
    ).fetchone()
    if not task:
        conn.close()
        raise HTTPException(status_code=404, detail="任务不存在")
    conn.execute("DELETE FROM agent_tasks WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


# ── Tool confirmation ─────────────────────────────────────────────────

@router.post("/agent/confirm/{task_id}")
def confirm_tool(task_id: int, body: dict):
    step = body.get("step_number")
    approved = body.get("approved", False)
    key = f"{task_id}_{step}"
    ws_manager.confirmations[key] = {"approved": approved}
    return {"ok": True, "task_id": task_id, "step": step, "approved": approved}


# ── 用户干预 (HTTP POST) ──────────────────────────────────────────────

@router.post("/agent/tasks/{task_id}/intervene")
async def agent_intervene(task_id: int, body: dict, user: dict = Depends(get_current_user)):
    fb_id = body.get("id", "")
    content = body.get("content", "")
    priority = body.get("priority", "normal")

    if not fb_id or not content.strip():
        raise HTTPException(400, "id and content are required")

    fb, merged_ids = await intervention_handler.receive(str(task_id), {
        "id": fb_id, "content": content, "priority": priority,
    })

    if fb is None:
        return {"ok": False, "reason": "queue_full_or_empty"}

    return {"ok": True, "feedback_id": fb.id, "priority": fb.priority, "merged_ids": merged_ids}


@router.post("/agent/tasks/{task_id}/retract")
async def agent_retract_intervention(task_id: int, body: dict, user: dict = Depends(get_current_user)):
    fb_id = body.get("feedback_id", "")
    if not fb_id:
        raise HTTPException(400, "feedback_id is required")
    ok = await intervention_handler.retract(str(task_id), fb_id)
    return {"ok": ok}


# ── 前端兼容 SSE Agent 流式 ───────────────────────────────────────────

@router.post("/agent/stream")
async def compat_agent_stream(req: LegacyAgentRequest, authorization: str | None = Header(None)):
    current_user = verify_token_from_header(authorization) if authorization else None
    if deepseek is None:
        return StreamingResponse(
            iter([f"data: {json.dumps({'error': 'DeepSeek 未配置，Agent 不可用'})}\n\n"]),
            media_type="text/event-stream",
        )

    user_msg = ""
    history_context = ""
    system_msgs = [m for m in req.messages if m.get("role") == "system"]
    chat_msgs = [m for m in req.messages if m.get("role") in ("user", "assistant")]

    recent = chat_msgs[-12:]
    if len(recent) > 1:
        history_lines = []
        for m in recent[:-1]:
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
            role_label = "用户" if m.get("role") == "user" else "助手"
            if str(content).strip():
                history_lines.append(f"{role_label}: {str(content)[:300]}")
        if history_lines:
            history_context = "## 对话历史\n" + "\n".join(history_lines) + "\n\n"

    for m in reversed(req.messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                user_msg = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
            else:
                user_msg = str(content)
            if user_msg.strip():
                break

    if not user_msg or not user_msg.strip():
        user_msg = "(empty)"

    agent_system = ""
    if system_msgs:
        agent_system = "## 自定义角色\n" + system_msgs[-1].get("content", "")[:500] + "\n\n"

    full_task = f"{agent_system}{history_context}## 当前任务\n{user_msg}"
    agent_mode = req.mode or "react"
    engine = get_engine(agent_mode)

    async def generate():
        conn = get_db("agent")
        user_id = current_user["id"] if current_user else 1

        from ..agent.tools import set_current_user
        set_current_user(user_id)

        cur = conn.execute(
            "INSERT INTO agent_tasks (user_id, title, description, status, agent_mode) VALUES (?, ?, ?, 'executing', ?)",
            (user_id, user_msg[:50], full_task, agent_mode),
        )
        conn.commit()
        task_id = cur.lastrowid
        conn.close()

        yield f"data: {json.dumps({'type': 'start', 'mode': agent_mode, 'task_id': task_id})}\n\n"

        try:
            bg_task = asyncio.create_task(engine.run(full_task, user_id, task_id))

            last_step = 0
            last_event_idx = 0
            while not bg_task.done():
                await asyncio.sleep(0.5)

                int_events, last_event_idx = await intervention_handler.poll_events(
                    str(task_id), last_event_idx
                )
                for evt in int_events:
                    yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

                conn = get_db("agent")
                steps = conn.execute(
                    "SELECT * FROM agent_steps WHERE task_id = ? AND step_number > ? ORDER BY step_number",
                    (task_id, last_step),
                ).fetchall()
                conn.close()
                for s in steps:
                    last_step = s["step_number"]

                    # Thinking step → emit thinking_end for frontend
                    if s["step_type"] == "thinking" and s.get("thought"):
                        yield f"data: {json.dumps({'type': 'thinking_end', 'content': s['thought']})}\n\n"
                        continue

                    if s["status"] == "confirming":
                        yield f"data: {json.dumps({'type': 'confirmation_required', 'task_id': task_id, 'step_num': s['step_number'], 'tool_name': s['tool_name'], 'args': s['tool_args']})}\n\n"
                        key = f"{task_id}_{s['step_number']}"
                        waited = 0
                        while key not in ws_manager.confirmations and waited < 120:
                            await asyncio.sleep(0.5)
                            waited += 1
                            if bg_task.done():
                                break
                        confirm = ws_manager.confirmations.pop(key, None)
                        if confirm and confirm.get("approved"):
                            yield f"data: {json.dumps({'type': 'step', 'step': {'turn': s['step_number'], 'type': 'tool_call', 'tool_name': s['tool_name'], 'thought': None, 'observation': '✅ 已允许执行'}})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'step', 'step': {'turn': s['step_number'], 'type': 'tool_call', 'tool_name': s['tool_name'], 'thought': None, 'observation': '⛔ 已取消'}})}\n\n"
                        continue

                    obs_text = ""
                    raw = s["tool_result"]
                    if raw:
                        try:
                            parsed = json.loads(raw)
                            inner = parsed.get("observation", "")
                            if isinstance(inner, str) and inner.startswith("{"):
                                try:
                                    inner_parsed = json.loads(inner)
                                    if inner_parsed.get("type") == "directory":
                                        obs_text = f"📁 {inner_parsed.get('path','.')} ({inner_parsed.get('count',0)} 项): " + ", ".join(inner_parsed.get("items", [])[:10])
                                    elif "results" in inner_parsed:
                                        items = inner_parsed["results"][:3]
                                        obs_text = f"🔍 搜索 '{inner_parsed.get('query','')}' ({inner_parsed.get('count',0)} 条)\n" + "\n".join(f"  {r.get('title','')} - {r.get('url','')}" for r in items)
                                    elif "content" in inner_parsed:
                                        obs_text = inner_parsed["content"][:500]
                                    elif "result" in inner_parsed:
                                        obs_text = f"🧮 {inner_parsed.get('expression','')} = {inner_parsed.get('result','')}"
                                    elif "output" in inner_parsed:
                                        obs_text = inner_parsed["output"][:500]
                                    elif "error" in inner_parsed:
                                        obs_text = f"❌ {inner_parsed['error']}"
                                    elif "clickable_link" in inner_parsed or "download_url" in inner_parsed:
                                        # Document/excel created → show download link prominently
                                        dl_url = inner_parsed.get("download_url", "")
                                        fname = inner_parsed.get("filename", "")
                                        obs_text = inner_parsed.get("clickable_link", f"📥 下载 {fname}: {dl_url}")
                                        # Emit file_created event for prominent UI card
                                        yield f"data: {json.dumps({'type': 'file_created', 'filename': fname, 'download_url': dl_url, 'size_bytes': inner_parsed.get('size_bytes', 0)})}\n\n"
                                    else:
                                        obs_text = inner[:500]
                                except (json.JSONDecodeError, TypeError):
                                    obs_text = inner[:500]
                            elif isinstance(inner, str):
                                obs_text = inner[:500]
                            else:
                                obs_text = str(inner)[:500]
                        except (json.JSONDecodeError, TypeError):
                            obs_text = raw[:500]

                    step_data = {
                        "turn": s["step_number"], "type": s["step_type"],
                        "tool_name": s["tool_name"], "tool_args": s["tool_args"],
                        "thought": s["thought"], "observation": obs_text,
                        "duration_ms": s["duration_ms"], "status": s["status"],
                    }
                    yield f"data: {json.dumps({'type': 'step', 'step': step_data})}\n\n"

            conn = get_db("agent")
            task = conn.execute("SELECT * FROM agent_tasks WHERE id = ?", (task_id,)).fetchone()
            conn.close()

            if task and task["final_answer"]:
                final = task["final_answer"]
            else:
                final = "任务已完成。"

            dialogue_logger.log(
                user="anonymous", question=user_msg, answer=final,
                source="agent", model=req.model,
                tokens=task["total_tokens"] if task else 0,
            )
            yield f"data: {json.dumps({'type': 'answer', 'answer': final})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )

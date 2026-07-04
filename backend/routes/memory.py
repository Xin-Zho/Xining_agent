"""记忆 + Agent 模式路由"""
from fastapi import APIRouter, Depends, HTTPException

from ..database import get_db
from ..auth import get_current_user
from ..memory import MemoryManager

router = APIRouter(prefix="/api", tags=["memory"])


@router.get("/memory")
async def list_memory(user: dict = Depends(get_current_user)):
    mgr = MemoryManager(user["id"])
    items = await mgr.search(query="", memory_types=["episodic", "semantic"], limit=50)
    return {"memories": [i.to_dict() for i in items]}


@router.post("/memory")
async def save_memory(body: dict, user: dict = Depends(get_current_user)):
    content = body.get("content", body.get("value", "")).strip()
    if not content:
        raise HTTPException(400, "content 不能为空")
    memory_type = body.get("memory_type", "semantic")
    importance = body.get("importance", 0.7)
    mgr = MemoryManager(user["id"])
    mid = await mgr.add(content=content, memory_type=memory_type, importance=importance)
    return {"ok": True, "memory_id": mid}


@router.delete("/memory/{memory_id}")
async def delete_memory(memory_id: str, user: dict = Depends(get_current_user)):
    mgr = MemoryManager(user["id"])
    ok = await mgr._episodic.delete(memory_id) or await mgr._semantic.delete(memory_id)
    if not ok:
        raise HTTPException(404, "记忆不存在")
    return {"ok": True}


@router.get("/agent/modes")
def agent_modes():
    return {
        "default": "react",
        "available": [
            {"id": "react", "name": "ReAct Agent",
             "description": "思考→行动→观察循环，Token预算 + 并行执行 + 自动反思"},
            {"id": "plan_solve", "name": "Plan-Solve",
             "description": "先制定计划，再逐步执行，最后汇总"},
        ],
    }

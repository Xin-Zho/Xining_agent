"""对话管理路由"""
from fastapi import APIRouter, Depends, HTTPException

from ..models import CreateConversationRequest
from ..database import get_db
from ..auth import get_current_user

router = APIRouter(prefix="/api", tags=["conversations"])


@router.get("/conversations")
def list_conversations(user: dict = Depends(get_current_user)):
    conn = get_db("chat")
    rows = conn.execute(
        """SELECT id, title, created_at FROM conversations
           WHERE user_id = ? ORDER BY created_at DESC""",
        (user["id"],),
    ).fetchall()
    conn.close()
    return [
        {"id": r["id"], "title": r["title"], "created_at": r["created_at"]}
        for r in rows
    ]


@router.post("/conversations")
def create_conversation(
    body: CreateConversationRequest, user: dict = Depends(get_current_user)
):
    conn = get_db("chat")
    cur = conn.execute(
        "INSERT INTO conversations (user_id, title) VALUES (?, ?)",
        (user["id"], body.title),
    )
    conn.commit()
    conv_id = cur.lastrowid
    conn.close()
    return {"id": conv_id, "title": body.title}


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: int, user: dict = Depends(get_current_user)):
    conn = get_db("chat")
    conv = conn.execute(
        "SELECT id, title, created_at FROM conversations WHERE id = ? AND user_id = ?",
        (conversation_id, user["id"]),
    ).fetchone()
    if not conv:
        conn.close()
        raise HTTPException(status_code=404, detail="对话不存在")

    msgs = conn.execute(
        """SELECT role, content, created_at FROM messages
           WHERE conversation_id = ? ORDER BY created_at""",
        (conversation_id,),
    ).fetchall()
    conn.close()
    return {
        "id": conv["id"],
        "title": conv["title"],
        "created_at": conv["created_at"],
        "messages": [
            {"role": m["role"], "content": m["content"], "created_at": m["created_at"]}
            for m in msgs
        ],
    }

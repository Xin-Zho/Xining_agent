"""对话路由 — 非流式、流式、文件上传、legacy 流式"""
import io
import json

from fastapi import APIRouter, Depends, UploadFile, File, Header, HTTPException
from fastapi.responses import StreamingResponse

from ..models import ChatRequest
from ..database import get_db
from ..auth import get_current_user
from ..dependencies import (
    deepseek, SYSTEM_PROMPT, MAX_HISTORY_ROUNDS, MAX_FILE_SIZE,
    ALLOWED_IMAGE, ALLOWED_TEXT, dialogue_logger, llm_client, ctx_manager,
    verify_token_from_header,
)
from ..llm_client import LLMClient

router = APIRouter(prefix="/api", tags=["chat"])

LegacyChatRequest = __import__('pydantic').BaseModel  # 占位，实际在 dependencies 引用


# ── Non-streaming chat ─────────────────────────────────────────────────

@router.post("/chat")
def chat(body: ChatRequest, user: dict = Depends(get_current_user)):
    conn = get_db("chat")

    if deepseek is None:
        conn.close()
        raise HTTPException(status_code=503, detail="DeepSeek chat is not configured.")

    conv = conn.execute(
        "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
        (body.conversation_id, user["id"]),
    ).fetchone()
    if not conv:
        conn.close()
        raise HTTPException(status_code=404, detail="对话不存在")

    conn.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (?, 'user', ?)",
        (body.conversation_id, body.message),
    )

    current_title = conn.execute(
        "SELECT title FROM conversations WHERE id = ?", (body.conversation_id,)
    ).fetchone()["title"]
    if current_title == "新对话":
        title = body.message[:30] + ("..." if len(body.message) > 30 else "")
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (title, body.conversation_id),
        )

    rows = conn.execute(
        """SELECT role, content FROM messages
           WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?""",
        (body.conversation_id, MAX_HISTORY_ROUNDS * 2),
    ).fetchall()
    rows = list(reversed(rows))

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for r in rows:
        messages.append({"role": r["role"], "content": r["content"]})

    try:
        resp = deepseek.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
        )
        reply = resp.choices[0].message.content
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=502, detail=f"AI API 调用失败: {e}")

    conn.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (?, 'assistant', ?)",
        (body.conversation_id, reply),
    )
    conn.commit()
    conn.close()

    dialogue_logger.log(
        user=user["username"], question=body.message, answer=reply,
        source="chat", model="deepseek-chat",
        tokens=resp.usage.total_tokens if resp.usage else 0,
    )

    return {"reply": reply, "conversation_id": body.conversation_id}


# ── Streaming chat ─────────────────────────────────────────────────────

@router.post("/chat/stream")
async def chat_stream(req: dict | ChatRequest, authorization: str | None = Header(None)):
    """流式对话 — SSE 格式。兼容旧前端 {messages} 和移动端 {conversation_id, message}"""
    if deepseek is None:
        raise HTTPException(status_code=503, detail="DeepSeek chat is not configured.")

    if isinstance(req, dict) and "messages" in req:
        messages = list(req["messages"])
        model_id = "deepseek-reasoner" if req.get("model") == "reasoner" else "deepseek-chat"
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, list):
                    user_msg = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
                else:
                    user_msg = str(content)
                break
    elif hasattr(req, 'conversation_id'):
        current_user = verify_token_from_header(authorization)
        if not current_user:
            raise HTTPException(status_code=401, detail="未登录或 token 已过期")

        conn = get_db("chat")
        conv = conn.execute(
            "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
            (req.conversation_id, current_user["id"]),
        ).fetchone()
        conn.close()
        if not conv:
            raise HTTPException(status_code=404, detail="对话不存在")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        conn2 = get_db("chat")
        rows = conn2.execute(
            """SELECT role, content FROM messages
               WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?""",
            (req.conversation_id, MAX_HISTORY_ROUNDS * 2),
        ).fetchall()
        conn2.close()
        for r in reversed(rows):
            messages.append({"role": r["role"], "content": r["content"]})
        user_msg = req.message
        model_id = "deepseek-chat"
    else:
        raise HTTPException(status_code=422, detail="无效的请求格式")

    cl = deepseek

    def generate():
        full_reply = ""
        try:
            stream = cl.chat.completions.create(
                model=model_id, messages=messages, temperature=0.7,
                max_tokens=4096, stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_reply += delta.content
                    yield f"data: {json.dumps({'token': delta.content})}\n\n"

            dialogue_logger.log(
                user="anonymous", question=user_msg[:200], answer=full_reply[:2000],
                source="chat", model=model_id,
            )
            yield f"data: {json.dumps({'done': True, 'full_reply': full_reply})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ── File upload ────────────────────────────────────────────────────────

@router.post("/upload")
async def upload(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    content_bytes = await file.read()

    if len(content_bytes) > MAX_FILE_SIZE:
        raise HTTPException(400, f"文件过大，最大 {MAX_FILE_SIZE // 1024 // 1024}MB")

    content_type = file.content_type or "application/octet-stream"

    if content_type in ALLOWED_IMAGE:
        import base64
        b64 = base64.b64encode(content_bytes).decode("ascii")
        return {
            "ok": True, "filename": file.filename, "content_type": content_type,
            "is_image": True, "content": f"data:{content_type};base64,{b64}",
        }

    if content_type in ALLOWED_TEXT:
        try:
            text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content_bytes.decode("gbk")
            except Exception:
                raise HTTPException(400, "无法解码文件内容，请确认文件编码为 UTF-8")

        if content_type == "application/pdf":
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(io.BytesIO(content_bytes))
                text_parts = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                pdf_text = "\n--- 分页 ---\n".join(text_parts)
                if not pdf_text.strip():
                    raise HTTPException(400, "PDF 文件中未提取到文字（可能是扫描件或图片型 PDF）")
                return {
                    "ok": True, "filename": file.filename, "content_type": content_type,
                    "is_image": False, "content": f"[PDF: {file.filename}]\n{pdf_text[:80000]}",
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(400, f"PDF 解析失败：{str(e)[:100]}")

        return {
            "ok": True, "filename": file.filename, "content_type": content_type,
            "is_image": False, "content": text,
        }

    allowed = ALLOWED_IMAGE | ALLOWED_TEXT
    raise HTTPException(400, f"不支持的文件类型。允许: {', '.join(sorted(allowed))}")


# ── Legacy streaming ───────────────────────────────────────────────────

from pydantic import BaseModel

class LegacyChatRequest(BaseModel):
    messages: list[dict]
    model: str = "chat"


@router.post("/chat/stream-legacy")
async def chat_stream_legacy(req: LegacyChatRequest):
    if deepseek is None:
        return StreamingResponse(
            iter([f"data: {json.dumps({'error': 'DeepSeek 未配置'})}\n\n"]),
            media_type="text/event-stream",
        )

    cl = deepseek
    model_id = "deepseek-reasoner" if req.model == "reasoner" else "deepseek-chat"
    compressed, token_info = ctx_manager.maybe_compress(req.messages)

    def generate():
        yield f"data: {json.dumps({'token_info': token_info})}\n\n"
        full_reply = ""
        try:
            stream = cl.chat.completions.create(
                model=model_id, messages=compressed, temperature=0.7,
                max_tokens=4096, stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_reply += delta.content
                    yield f"data: {json.dumps({'token': delta.content})}\n\n"
            dialogue_logger.log(
                user="anonymous", question=str(req.messages[-1].get("content", ""))[:200],
                answer=full_reply[:2000], source="chat", model=model_id,
            )
            yield f"data: {json.dumps({'done': True, 'full_reply': full_reply, 'token_info': token_info})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )

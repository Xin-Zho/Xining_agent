"""下载 + 静态文件 + 根路由"""
import os
from urllib.parse import quote

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from ..database import get_db
from ..auth import get_current_user, verify_token
from ..dependencies import STATIC_DIR, verify_token_from_header

router = APIRouter(tags=["downloads"])


# ── 文件下载 ──────────────────────────────────────────────────────────

@router.get("/api/download/{filename:path}")
def download_file(filename: str, token: str = None, authorization: str | None = Header(None)):
    user = None
    if token:
        try:
            user = verify_token(token)
        except Exception:
            user = None
    if not user and authorization:
        user = verify_token_from_header(authorization)
    if not user:
        raise HTTPException(401, "请登录后下载文件")

    dl_dir = os.path.join(STATIC_DIR, "downloads")
    filepath = os.path.join(dl_dir, os.path.basename(filename))
    if not os.path.isfile(filepath):
        raise HTTPException(404, "文件不存在")

    if user.get("username") == "admin":
        return FileResponse(filepath, filename=os.path.basename(filename))

    conn = get_db("memory")
    row = conn.execute(
        "SELECT user_id FROM downloads WHERE filepath = ? ORDER BY id DESC LIMIT 1",
        (filepath,),
    ).fetchone()
    conn.close()

    if row and row["user_id"] != user["id"]:
        raise HTTPException(403, "无权访问此文件")

    return FileResponse(filepath, filename=os.path.basename(filename))


@router.get("/api/downloads")
def list_downloads(user: dict = Depends(get_current_user)):
    conn = get_db("memory")
    rows = conn.execute(
        "SELECT filename, filepath, size_bytes, created_at FROM downloads WHERE user_id = ? ORDER BY created_at DESC",
        (user["id"],),
    ).fetchall()
    conn.close()
    files = [
        {"name": r["filename"], "size": r["size_bytes"],
         "url": f"/api/download/{quote(r['filename'], safe='/')}",
         "created": r["created_at"]}
        for r in rows
    ]
    return {"files": files, "count": len(files)}


# ── 静态文件 & Web App ────────────────────────────────────────────────

if os.path.isdir(STATIC_DIR):

    @router.get("/app")
    async def web_app():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    @router.get("/downloads")
    async def downloads_page(token: str = None, authorization: str | None = Header(None)):
        user = None
        if token:
            try:
                user = verify_token(token)
            except Exception:
                pass
        if not user and authorization:
            user = verify_token_from_header(authorization)
        if not user:
            return HTMLResponse("<h2>请先登录 <a href='/app'>返回登录</a></h2>", status_code=401)

        conn = get_db("memory")
        rows = conn.execute(
            "SELECT filename, size_bytes, created_at FROM downloads WHERE user_id = ? ORDER BY created_at DESC",
            (user["id"],),
        ).fetchall()
        conn.close()
        files_html = ""
        for r in rows:
            sz = r["size_bytes"]
            sz_str = f"{sz/1024:.0f}KB" if sz > 1024 else f"{sz}B"
            encoded = quote(r["filename"], safe='/')
            files_html += f'<tr><td><a href="/api/download/{encoded}">📥 {r["filename"]}</a></td><td>{sz_str}</td><td>{r["created_at"]}</td></tr>'
        html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><title>下载文件</title>
<style>body{{font-family:system-ui,sans-serif;max-width:800px;margin:40px auto;padding:0 20px;background:#FAFAFB;color:#18181B}}
h1{{font-size:24px;margin-bottom:8px}}table{{width:100%;border-collapse:collapse;margin-top:20px}}
th{{text-align:left;padding:10px 12px;border-bottom:2px solid #E5E7EB;font-size:12px;color:#71717A;text-transform:uppercase;letter-spacing:0.5px}}td{{padding:10px 12px;border-bottom:1px solid #E5E7EB}}a{{color:#5B6AF0;text-decoration:none}}a:hover{{text-decoration:underline}}
.back{{display:inline-block;margin-top:24px;color:#71717A;font-size:14px}}</style></head><body>
<h1>📂 下载文件</h1><p>Agent 生成的所有可下载文件</p>
<table><tr><th>文件</th><th>大小</th><th>时间</th></tr>{files_html or '<tr><td colspan=3>暂无文件</td></tr>'}</table>
<a class="back" href="/app">← 返回聊天</a></body></html>"""
        return HTMLResponse(content=html)


# ── Root endpoint ─────────────────────────────────────────────────────

@router.get("/")
def root():
    return {
        "name": "Agent Learning — Unified",
        "version": "2.0.0",
        "agent_mode": "react",
        "endpoints": {
            "auth": ["/api/register", "/api/login"],
            "chat": ["/api/chat", "/api/chat/stream"],
            "upload": ["/api/upload"],
            "conversations": ["/api/conversations", "/api/conversations/{id}"],
            "agent_tasks": ["/api/agent/tasks", "/api/agent/tasks/{id}"],
            "agent_modes": ["/api/agent/modes"],
            "logs": ["/api/logs/stats", "api/logs/suggestions"],
            "memory": ["/api/memory"],
            "websocket": ["/ws/agent/{task_id}"],
        },
    }

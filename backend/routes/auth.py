"""认证路由 — 注册、登录、前端兼容"""
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse

from ..models import AuthRequest
from ..database import get_db
from ..auth import create_token, get_current_user, hash_password, verify_password
from ..dependencies import ALLOW_REGISTRATION, INVITE_CODE, verify_token_from_header

router = APIRouter(prefix="/api", tags=["auth"])


# ── 核心路由 ──────────────────────────────────────────────────────────

@router.post("/register")
def register(body: AuthRequest):
    if not ALLOW_REGISTRATION:
        raise HTTPException(status_code=403, detail="注册已关闭")
    if getattr(body, 'invite_code', '') != INVITE_CODE:
        raise HTTPException(status_code=403, detail="认证码错误")
    if not body.username.strip() or len(body.password) < 8:
        raise HTTPException(status_code=422, detail="用户名不能为空，密码至少4位")

    conn = get_db("chat")
    existing = conn.execute(
        "SELECT id FROM users WHERE username = ?", (body.username.strip(),)
    ).fetchone()
    if existing:
        conn.close()
        raise HTTPException(status_code=409, detail="用户名已被注册")

    hash_ = hash_password(body.password)
    cur = conn.execute(
        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
        (body.username.strip(), hash_),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return {"token": create_token(user_id), "username": body.username.strip()}


@router.post("/login")
def login(body: AuthRequest):
    conn = get_db("chat")
    user = conn.execute(
        "SELECT id, username, password_hash FROM users WHERE username = ?",
        (body.username.strip(),),
    ).fetchone()
    conn.close()

    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    return {"token": create_token(user["id"]), "username": user["username"]}


# ── 前端兼容路由 ──────────────────────────────────────────────────────

@router.post("/auth/register")
def compat_register(body: AuthRequest):
    if not ALLOW_REGISTRATION:
        return JSONResponse({"ok": False, "error": "注册已关闭"}, 403)
    if getattr(body, 'invite_code', '') != INVITE_CODE:
        return JSONResponse({"ok": False, "error": "认证码错误"}, 403)
    if not body.username.strip() or len(body.password) < 8:
        return JSONResponse({"ok": False, "error": "用户名不能为空，密码至少8位"}, 422)

    conn = get_db("chat")
    existing = conn.execute(
        "SELECT id FROM users WHERE username = ?", (body.username.strip(),)
    ).fetchone()
    if existing:
        conn.close()
        return JSONResponse({"ok": False, "error": "用户名已存在"}, 409)

    hash_ = hash_password(body.password)
    cur = conn.execute(
        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
        (body.username.strip(), hash_),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return {"ok": True, "token": create_token(user_id), "username": body.username.strip()}


@router.post("/auth/login")
def compat_login(body: AuthRequest):
    conn = get_db("chat")
    user = conn.execute(
        "SELECT id, username, password_hash FROM users WHERE username = ?",
        (body.username.strip(),),
    ).fetchone()
    conn.close()

    if not user or not verify_password(body.password, user["password_hash"]):
        return JSONResponse({"ok": False, "error": "用户名或密码错误"}, 401)

    return {"ok": True, "token": create_token(user["id"]), "username": user["username"]}


@router.get("/auth/me")
def compat_me(authorization: str | None = Header(None)):
    user = verify_token_from_header(authorization)
    if not user:
        return JSONResponse({"ok": False, "error": "未登录或 token 已过期"}, 401)
    return {"ok": True, "username": user["username"]}

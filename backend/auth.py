import os
from datetime import datetime, timezone

from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt
from jose import jwt, JWTError

from .database import get_db

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-secret-change-in-production")
security = HTTPBearer()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hash_: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hash_.encode("utf-8"))


def create_token(user_id: int) -> str:
    payload = {"sub": str(user_id), "iat": datetime.now(timezone.utc)}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="无效的认证令牌")

    conn = get_db()
    user = conn.execute("SELECT id, username FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    return {"id": user["id"], "username": user["username"]}


def verify_token(token: str) -> dict:
    """Verify JWT token string, return user dict. Used for WebSocket auth."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="无效的认证令牌")

    conn = get_db()
    user = conn.execute("SELECT id, username FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    return {"id": user["id"], "username": user["username"]}

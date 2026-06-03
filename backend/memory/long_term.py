"""
长期记忆 — SQLite 持久化，跨会话保留关键信息

用法：
    ltm = LongTermMemory(user_id=1)
    ltm.save("用户偏好", "喜欢用 Python")
    result = ltm.get("用户偏好")
    keys = ltm.list_keys()
    results = ltm.search("Python")
"""
from datetime import datetime, timezone

from ..database import get_db


class LongTermMemory:
    """
    长期记忆：基于 SQLite 的跨会话记忆系统。

    每条记忆 = user_id + key + value + 时间戳
    """

    def __init__(self, user_id: int):
        self.user_id = user_id

    def save(self, key: str, value: str):
        """保存或更新一条记忆"""
        conn = get_db("memory")
        conn.execute(
            """INSERT INTO agent_memory (user_id, key, value, updated_at)
               VALUES (?, ?, ?, datetime('now'))
               ON CONFLICT(user_id, key) DO UPDATE SET
               value = excluded.value, updated_at = datetime('now')""",
            (self.user_id, key, value),
        )
        conn.commit()
        conn.close()

    def get(self, key: str) -> str:
        """读取一条记忆，不存在返回空字符串"""
        conn = get_db("memory")
        row = conn.execute(
            "SELECT value FROM agent_memory WHERE user_id = ? AND key = ?",
            (self.user_id, key),
        ).fetchone()
        conn.close()
        return row["value"] if row else ""

    def list_keys(self) -> list[str]:
        """列出所有记忆的 key"""
        conn = get_db("memory")
        rows = conn.execute(
            "SELECT key FROM agent_memory WHERE user_id = ? ORDER BY updated_at DESC",
            (self.user_id,),
        ).fetchall()
        conn.close()
        return [r["key"] for r in rows]

    def list_all(self) -> list[dict]:
        """列出所有记忆（含内容）"""
        conn = get_db("memory")
        rows = conn.execute(
            "SELECT key, value, updated_at FROM agent_memory WHERE user_id = ? ORDER BY updated_at DESC",
            (self.user_id,),
        ).fetchall()
        conn.close()
        return [
            {"key": r["key"], "value": r["value"], "updated_at": r["updated_at"]}
            for r in rows
        ]

    def search(self, keyword: str) -> list[dict]:
        """简单关键词搜索（大小写不敏感）"""
        conn = get_db("memory")
        rows = conn.execute(
            "SELECT key, value FROM agent_memory WHERE user_id = ? "
            "AND (key LIKE ? OR value LIKE ?)",
            (self.user_id, f"%{keyword}%", f"%{keyword}%"),
        ).fetchall()
        conn.close()
        return [{"key": r["key"], "value": r["value"]} for r in rows]

    def delete(self, key: str):
        """删除一条记忆"""
        conn = get_db("memory")
        conn.execute(
            "DELETE FROM agent_memory WHERE user_id = ? AND key = ?",
            (self.user_id, key),
        )
        conn.commit()
        conn.close()

    def clear(self):
        """清空所有记忆"""
        conn = get_db("memory")
        conn.execute("DELETE FROM agent_memory WHERE user_id = ?", (self.user_id,))
        conn.commit()
        conn.close()

    def get_context_for_prompt(self, max_items: int = 5) -> str:
        """
        把记忆格式化为一段上下文文本，可注入到 System Prompt。
        返回最近更新的 max_items 条记忆。
        """
        conn = get_db("memory")
        rows = conn.execute(
            "SELECT key, value FROM agent_memory WHERE user_id = ? "
            "ORDER BY updated_at DESC LIMIT ?",
            (self.user_id, max_items),
        ).fetchall()
        conn.close()

        if not rows:
            return ""

        lines = ["[长期记忆]"]
        for r in rows:
            lines.append(f"- {r['key']}: {r['value']}")
        return "\n".join(lines)

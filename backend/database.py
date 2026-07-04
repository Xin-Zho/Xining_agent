import os
import sqlite3

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# 三 Bank 分区：chat.db(用户+对话) / agent.db(任务+步骤) / memory.db(记忆+下载)
def _db_path(name: str) -> str:
    return os.path.join(DATA_DIR, f"{name}.db")

def get_db(name: str = "chat") -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(name), check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    if name == "chat":
        conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    os.makedirs(DATA_DIR, exist_ok=True)

    # chat.db — 用户 + 对话 + 消息
    c = get_db("chat")
    c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            title TEXT NOT NULL DEFAULT '新对话',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
    """)
    c.commit(); c.close()

    # agent.db — Agent 任务 + 步骤
    a = get_db("agent")
    a.executescript("""
        CREATE TABLE IF NOT EXISTS agent_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending','planning','executing','observing','completed','failed','cancelled')),
            agent_mode TEXT NOT NULL DEFAULT 'react'
                CHECK(agent_mode IN ('react','plan_solve')),
            plan_json TEXT,
            final_answer TEXT,
            conversation_id INTEGER,
            total_tokens INTEGER DEFAULT 0,
            duration_ms INTEGER DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS agent_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL REFERENCES agent_tasks(id) ON DELETE CASCADE,
            step_number INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending','running','completed','failed','skipped','confirming','cancelled')),
            step_type TEXT NOT NULL
                CHECK(step_type IN ('plan','thought','tool_call','observation','response','thinking')),
            tool_name TEXT,
            tool_args TEXT,
            tool_result TEXT,
            thought TEXT,
            duration_ms INTEGER,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_agent_tasks_user ON agent_tasks(user_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_agent_steps_task ON agent_steps(task_id, step_number);
    """)
    a.commit()
    # Migration: fix existing agent_steps CHECK constraint to include 'thinking'
    try:
        a.execute("INSERT INTO agent_steps (task_id, step_number, status, step_type) VALUES (-1, -1, 'failed', 'thinking')")
        a.execute("DELETE FROM agent_steps WHERE task_id = -1")
    except sqlite3.IntegrityError:
        # Rebuild table with new constraint
        a.executescript("""
            CREATE TABLE IF NOT EXISTS agent_steps_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER NOT NULL REFERENCES agent_tasks(id) ON DELETE CASCADE,
                step_number INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending','running','completed','failed','skipped','confirming','cancelled')),
                step_type TEXT NOT NULL
                    CHECK(step_type IN ('plan','thought','tool_call','observation','response','thinking')),
                tool_name TEXT,
                tool_args TEXT,
                tool_result TEXT,
                thought TEXT,
                duration_ms INTEGER,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            INSERT INTO agent_steps_new SELECT * FROM agent_steps;
            DROP TABLE agent_steps;
            ALTER TABLE agent_steps_new RENAME TO agent_steps;
            CREATE INDEX IF NOT EXISTS idx_agent_steps_task ON agent_steps(task_id, step_number);
        """)
    a.commit(); a.close()

    # memory.db — 长期记忆 + 下载记录
    m = get_db("memory")
    m.executescript("""
        CREATE TABLE IF NOT EXISTS agent_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(user_id, key)
        );
        CREATE INDEX IF NOT EXISTS idx_agent_memory_user ON agent_memory(user_id, key);
        CREATE TABLE IF NOT EXISTS downloads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            size_bytes INTEGER DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_downloads_user ON downloads(user_id, created_at);
    """)
    m.commit(); m.close()

    # memory.db — episodic + semantic tables (NEW)
    m2 = get_db("memory")
    m2.executescript("""
        CREATE TABLE IF NOT EXISTS episodic_memory (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            importance REAL DEFAULT 0.5,
            created_at REAL NOT NULL DEFAULT (unixepoch()),
            last_accessed_at REAL NOT NULL DEFAULT (unixepoch()),
            access_count INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_episodic_user ON episodic_memory(user_id, created_at);

        CREATE TABLE IF NOT EXISTS semantic_memory (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            importance REAL DEFAULT 0.5,
            created_at REAL NOT NULL DEFAULT (unixepoch()),
            last_accessed_at REAL NOT NULL DEFAULT (unixepoch()),
            access_count INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}',
            source_episodic_id TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_semantic_user ON semantic_memory(user_id, created_at);
    """)
    m2.commit(); m2.close()

    # evaluation.db 表（task_evaluations + eval_aggregates + task_feedback）
    from .evaluation.db import init_eval_db
    init_eval_db()

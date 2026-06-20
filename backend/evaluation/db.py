"""
评估数据库 — 建表 + CRUD 辅助。

新增 3 张表（全部在 agent.db，与 agent_tasks/steps 同库以支持 FK + JOIN）：
  - task_evaluations: 单任务评估结果
  - eval_aggregates: 时间窗口汇总快照
  - task_feedback: 用户显式评分
"""
import sqlite3

from ..database import get_db


def init_eval_db():
    """初始化评估相关表（在 database.init_db() 中调用）"""
    conn = get_db("agent")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS task_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL REFERENCES agent_tasks(id) ON DELETE CASCADE,
            -- 定量指标
            success BOOLEAN NOT NULL DEFAULT 0,
            tool_total INTEGER DEFAULT 0,
            tool_success INTEGER DEFAULT 0,
            tool_error INTEGER DEFAULT 0,
            tool_error_rate REAL DEFAULT 0.0,
            step_count INTEGER DEFAULT 0,
            avg_step_duration_ms REAL DEFAULT 0.0,
            token_efficiency REAL DEFAULT 0.0,
            completion_ratio REAL DEFAULT 0.0,
            plan_coverage REAL,
            -- 定性评分 (LLM Judge)
            quality_score REAL,
            accuracy_score REAL,
            completeness_score REAL,
            relevance_score REAL,
            tool_usage_score REAL,
            judge_rationale TEXT,
            judge_model TEXT,
            judge_tokens INTEGER DEFAULT 0,
            -- 元数据
            evaluated_at TEXT NOT NULL DEFAULT (datetime('now')),
            evaluation_mode TEXT NOT NULL DEFAULT 'auto'
        );
        CREATE INDEX IF NOT EXISTS idx_eval_task ON task_evaluations(task_id);
        CREATE INDEX IF NOT EXISTS idx_eval_quality ON task_evaluations(quality_score);
        CREATE INDEX IF NOT EXISTS idx_eval_evaluated ON task_evaluations(evaluated_at);

        CREATE TABLE IF NOT EXISTS eval_aggregates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            period_start TEXT NOT NULL,
            period_type TEXT NOT NULL CHECK(period_type IN ('daily','weekly','monthly')),
            agent_mode TEXT,
            task_count INTEGER DEFAULT 0,
            success_count INTEGER DEFAULT 0,
            success_rate REAL DEFAULT 0.0,
            avg_quality_score REAL,
            avg_tool_error_rate REAL DEFAULT 0.0,
            avg_tokens_per_task REAL DEFAULT 0.0,
            avg_duration_ms REAL DEFAULT 0.0,
            avg_token_efficiency REAL DEFAULT 0.0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(period_start, period_type, agent_mode)
        );
        CREATE INDEX IF NOT EXISTS idx_agg_period ON eval_aggregates(period_start, period_type);

        CREATE TABLE IF NOT EXISTS task_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL REFERENCES agent_tasks(id) ON DELETE CASCADE,
            user_id INTEGER NOT NULL,
            rating INTEGER CHECK(rating BETWEEN 1 AND 5),
            comment TEXT,
            tags TEXT DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_fb_task ON task_feedback(task_id);

        CREATE TABLE IF NOT EXISTS optimization_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL CHECK(category IN ('tool_config','parameter','mode_routing','prompt')),
            severity TEXT NOT NULL DEFAULT 'medium' CHECK(severity IN ('high','medium','low')),
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            evidence TEXT DEFAULT '{}',
            suggested_change TEXT NOT NULL,
            auto_applicable BOOLEAN DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','applied','dismissed')),
            applied_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_rec_status ON optimization_recommendations(status, severity);
    """)
    conn.commit()
    conn.close()


# ── task_evaluations CRUD ──────────────────────────────────

def create_task_evaluation(task_id: int, metrics: dict, mode: str = "auto") -> int:
    """插入评估记录，返回 id"""
    conn = get_db("agent")
    cur = conn.execute(
        """INSERT INTO task_evaluations (
            task_id, success, tool_total, tool_success, tool_error,
            tool_error_rate, step_count, avg_step_duration_ms,
            token_efficiency, completion_ratio, plan_coverage,
            evaluation_mode
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            task_id,
            int(metrics.get("success", False)),
            metrics.get("tool_total", 0),
            metrics.get("tool_success", 0),
            metrics.get("tool_error", 0),
            metrics.get("tool_error_rate", 0.0),
            metrics.get("step_count", 0),
            metrics.get("avg_step_duration_ms", 0.0),
            metrics.get("token_efficiency", 0.0),
            metrics.get("completion_ratio", 0.0),
            metrics.get("plan_coverage"),
            mode,
        ),
    )
    conn.commit()
    eval_id = cur.lastrowid
    conn.close()
    return eval_id


def update_evaluation_quality(eval_id: int, quality: dict):
    """写入 LLM Judge 结果"""
    conn = get_db("agent")
    conn.execute(
        """UPDATE task_evaluations SET
            quality_score = ?, accuracy_score = ?, completeness_score = ?,
            relevance_score = ?, tool_usage_score = ?,
            judge_rationale = ?, judge_model = ?, judge_tokens = ?
        WHERE id = ?""",
        (
            quality.get("overall"),
            quality.get("accuracy"),
            quality.get("completeness"),
            quality.get("relevance"),
            quality.get("tool_usage"),
            quality.get("rationale", ""),
            quality.get("model", ""),
            quality.get("tokens", 0),
            eval_id,
        ),
    )
    conn.commit()
    conn.close()


def get_task_evaluation(task_id: int) -> dict | None:
    """按 task_id 查询最新评估"""
    conn = get_db("agent")
    row = conn.execute(
        "SELECT * FROM task_evaluations WHERE task_id = ? ORDER BY evaluated_at DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_evaluation_for_task(task_id: int) -> int | None:
    """检查任务是否已有评估，返回 evaluation id 或 None"""
    conn = get_db("agent")
    row = conn.execute(
        "SELECT id FROM task_evaluations WHERE task_id = ? LIMIT 1",
        (task_id,),
    ).fetchone()
    conn.close()
    return row["id"] if row else None


def list_task_evaluations(
    status: str = None,
    evaluation_mode: str = None,
    date_from: str = None,
    date_to: str = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """分页查询评估列表"""
    conn = get_db("agent")
    q = """SELECT e.* FROM task_evaluations e
           JOIN agent_tasks t ON e.task_id = t.id WHERE 1=1"""
    params = []

    if status:
        q += " AND t.status = ?"
        params.append(status)
    if evaluation_mode:
        q += " AND e.evaluation_mode = ?"
        params.append(evaluation_mode)
    if date_from:
        q += " AND e.evaluated_at >= ?"
        params.append(date_from)
    if date_to:
        q += " AND e.evaluated_at <= ?"
        params.append(date_to)

    q += " ORDER BY e.evaluated_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_task_evaluation(task_id: int) -> bool:
    """删除评估记录（允许重新评估）"""
    conn = get_db("agent")
    cur = conn.execute("DELETE FROM task_evaluations WHERE task_id = ?", (task_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


# ── task_feedback CRUD ─────────────────────────────────────

def save_feedback(user_id: int, task_id: int, rating: int,
                  comment: str = None, tags: list[str] = None) -> int:
    import json
    conn = get_db("agent")
    cur = conn.execute(
        """INSERT INTO task_feedback (task_id, user_id, rating, comment, tags)
           VALUES (?, ?, ?, ?, ?)""",
        (task_id, user_id, rating, comment or "", json.dumps(tags or [])),
    )
    conn.commit()
    fid = cur.lastrowid
    conn.close()
    return fid


def get_feedback(task_id: int) -> list[dict]:
    import json
    conn = get_db("agent")
    rows = conn.execute(
        "SELECT * FROM task_feedback WHERE task_id = ? ORDER BY created_at DESC",
        (task_id,),
    ).fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        try:
            d["tags"] = json.loads(d["tags"])
        except (json.JSONDecodeError, TypeError):
            d["tags"] = []
        results.append(d)
    return results


# ── eval_aggregates ────────────────────────────────────────

def upsert_aggregate(period_start: str, period_type: str, agent_mode: str | None,
                     metrics: dict):
    conn = get_db("agent")
    conn.execute(
        """INSERT INTO eval_aggregates (
            period_start, period_type, agent_mode,
            task_count, success_count, success_rate,
            avg_quality_score, avg_tool_error_rate,
            avg_tokens_per_task, avg_duration_ms, avg_token_efficiency
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(period_start, period_type, agent_mode) DO UPDATE SET
            task_count = excluded.task_count,
            success_count = excluded.success_count,
            success_rate = excluded.success_rate,
            avg_quality_score = excluded.avg_quality_score,
            avg_tool_error_rate = excluded.avg_tool_error_rate,
            avg_tokens_per_task = excluded.avg_tokens_per_task,
            avg_duration_ms = excluded.avg_duration_ms,
            avg_token_efficiency = excluded.avg_token_efficiency,
            created_at = datetime('now')""",
        (
            period_start, period_type, agent_mode,
            metrics.get("task_count", 0),
            metrics.get("success_count", 0),
            metrics.get("success_rate", 0.0),
            metrics.get("avg_quality_score"),
            metrics.get("avg_tool_error_rate", 0.0),
            metrics.get("avg_tokens_per_task", 0.0),
            metrics.get("avg_duration_ms", 0.0),
            metrics.get("avg_token_efficiency", 0.0),
        ),
    )
    conn.commit()
    conn.close()


# ── optimization_recommendations CRUD ─────────────────────

def create_recommendation(rec: dict) -> int:
    """创建优化建议"""
    import json
    conn = get_db("agent")
    cur = conn.execute(
        """INSERT INTO optimization_recommendations
           (category, severity, title, description, evidence, suggested_change, auto_applicable)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            rec["category"],
            rec["severity"],
            rec["title"],
            rec["description"],
            json.dumps(rec.get("evidence", {})),
            json.dumps(rec.get("suggested_change", {})),
            int(rec.get("auto_applicable", False)),
        ),
    )
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid


def list_recommendations(status: str = None, severity: str = None,
                         limit: int = 50) -> list[dict]:
    """查询建议列表"""
    import json
    conn = get_db("agent")
    q = "SELECT * FROM optimization_recommendations WHERE 1=1"
    params = []
    if status:
        q += " AND status = ?"
        params.append(status)
    if severity:
        q += " AND severity = ?"
        params.append(severity)
    q += " ORDER BY CASE severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, created_at DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(q, params).fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        for field in ("evidence", "suggested_change"):
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                d[field] = {}
        results.append(d)
    return results


def update_recommendation_status(rec_id: int, status: str):
    """更新建议状态（apply/dismiss）"""
    conn = get_db("agent")
    if status == "applied":
        conn.execute(
            "UPDATE optimization_recommendations SET status=?, applied_at=datetime('now') WHERE id=?",
            (status, rec_id),
        )
    else:
        conn.execute(
            "UPDATE optimization_recommendations SET status=? WHERE id=?",
            (status, rec_id),
        )
    conn.commit()
    conn.close()


def get_recommendation_stats() -> dict:
    """建议统计"""
    conn = get_db("agent")
    row = conn.execute(
        """SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) as pending,
            SUM(CASE WHEN status='applied' THEN 1 ELSE 0 END) as applied,
            SUM(CASE WHEN status='dismissed' THEN 1 ELSE 0 END) as dismissed
        FROM optimization_recommendations"""
    ).fetchone()
    conn.close()
    return {
        "total": row["total"] or 0,
        "pending": row["pending"] or 0,
        "applied": row["applied"] or 0,
        "dismissed": row["dismissed"] or 0,
    }


def clear_stale_recommendations():
    """清理超过 30 天的 pending 建议（避免堆积）"""
    conn = get_db("agent")
    conn.execute(
        "DELETE FROM optimization_recommendations WHERE status='pending' AND created_at < datetime('now', '-30 days')"
    )
    conn.commit()
    conn.close()

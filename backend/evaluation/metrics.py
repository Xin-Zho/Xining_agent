"""
定量指标计算 — 纯 SQL 查询 agent_tasks + agent_steps，零 LLM 成本。

每个指标都有独立的 SQL 实现，避免一次查询扫描全表。
"""
import math

from ..database import get_db


def compute_quantitative_metrics(task_id: int) -> dict:
    """
    计算单个任务的定量指标。

    返回字典包含：success, tool_total, tool_success, tool_error,
    tool_error_rate, step_count, avg_step_duration_ms,
    token_efficiency, completion_ratio, plan_coverage
    """
    conn = get_db("agent")

    # ── 任务基本信息 ──
    task = conn.execute(
        "SELECT status, agent_mode, final_answer, total_tokens, plan_json "
        "FROM agent_tasks WHERE id = ?",
        (task_id,),
    ).fetchone()

    if not task:
        conn.close()
        return _empty_metrics()

    success = task["status"] == "completed"
    final_answer = task["final_answer"] or ""
    total_tokens = task["total_tokens"] or 0
    plan_json = task["plan_json"]
    agent_mode = task["agent_mode"]

    # ── 步骤统计 ──
    # 所有步骤
    all_steps = conn.execute(
        "SELECT status, step_type, duration_ms FROM agent_steps WHERE task_id = ?",
        (task_id,),
    ).fetchall()

    step_count = len(all_steps)
    if step_count == 0:
        conn.close()
        return _empty_metrics(success=success, step_count=0)

    completed_steps = sum(1 for s in all_steps if s["status"] == "completed")
    completion_ratio = completed_steps / step_count

    # ── 工具调用统计 ──
    tool_steps = [s for s in all_steps if s["step_type"] == "tool_call"]
    tool_total = len(tool_steps)
    tool_success = sum(1 for s in tool_steps if s["status"] == "completed")
    tool_error = sum(1 for s in tool_steps if s["status"] == "failed")
    tool_error_rate = tool_error / tool_total if tool_total > 0 else 0.0

    # 工具步骤平均耗时
    tool_durations = [s["duration_ms"] for s in tool_steps if s["duration_ms"]]
    avg_step_duration_ms = (
        sum(tool_durations) / len(tool_durations) if tool_durations else 0.0
    )

    # ── Token 效率 ──
    answer_chars = len(final_answer)
    token_efficiency = answer_chars / total_tokens if total_tokens > 0 else 0.0

    # ── Plan 覆盖率 (plan_solve 专用) ──
    plan_coverage = None
    if agent_mode == "plan_solve" and plan_json:
        try:
            import json
            plan = json.loads(plan_json)
            if isinstance(plan, list):
                planned_count = len(plan)
                executed_count = tool_total  # 用工具调用数近似
                plan_coverage = (
                    min(executed_count / planned_count, 1.0)
                    if planned_count > 0 else None
                )
        except (json.JSONDecodeError, TypeError):
            pass

    conn.close()

    return {
        "success": success,
        "tool_total": tool_total,
        "tool_success": tool_success,
        "tool_error": tool_error,
        "tool_error_rate": round(tool_error_rate, 4),
        "step_count": step_count,
        "avg_step_duration_ms": round(avg_step_duration_ms, 1),
        "token_efficiency": round(token_efficiency, 4),
        "completion_ratio": round(completion_ratio, 4),
        "plan_coverage": round(plan_coverage, 4) if plan_coverage is not None else None,
    }


def _empty_metrics(success: bool = False, step_count: int = 0) -> dict:
    return {
        "success": success,
        "tool_total": 0,
        "tool_success": 0,
        "tool_error": 0,
        "tool_error_rate": 0.0,
        "step_count": step_count,
        "avg_step_duration_ms": 0.0,
        "token_efficiency": 0.0,
        "completion_ratio": 0.0,
        "plan_coverage": None,
    }


# ── 看板：总览 ────────────────────────────────────────────

def compute_dashboard_summary(
    date_from: str = None, date_to: str = None
) -> dict:
    """
    总览看板：成功率、质量分、模式对比、失败工具排行。

    如果未指定日期范围，默认最近 30 天。
    """
    conn = get_db("agent")

    date_filter = ""
    date_params = []
    if date_from:
        date_filter += " AND t.created_at >= ?"
        date_params.append(date_from)
    if date_to:
        date_filter += " AND t.created_at <= ?"
        date_params.append(date_to)

    # ── 总体统计 ──
    params = date_params.copy()
    row = conn.execute(
        f"""SELECT
            COUNT(*) as total,
            SUM(CASE WHEN t.status = 'completed' THEN 1 ELSE 0 END) as completed,
            AVG(CASE WHEN t.status IN ('completed','failed') THEN t.duration_ms END) as avg_duration,
            AVG(CASE WHEN t.status IN ('completed','failed') THEN t.total_tokens END) as avg_tokens
        FROM agent_tasks t
        WHERE t.status IN ('completed', 'failed') {date_filter}""",
        params,
    ).fetchone()

    total = row["total"] or 0
    completed = row["completed"] or 0
    success_rate = completed / total if total > 0 else 0.0

    # ── 平均质量分 ──
    params = date_params.copy()
    qual_row = conn.execute(
        f"""SELECT AVG(e.quality_score) as avg_q
        FROM task_evaluations e
        JOIN agent_tasks t ON e.task_id = t.id
        WHERE e.quality_score IS NOT NULL {date_filter}""",
        params,
    ).fetchone()

    # ── 平均工具错误率 ──
    params = date_params.copy()
    tool_row = conn.execute(
        f"""SELECT AVG(e.tool_error_rate) as avg_ter
        FROM task_evaluations e
        JOIN agent_tasks t ON e.task_id = t.id
        WHERE t.status IN ('completed','failed') {date_filter}""",
        params,
    ).fetchone()

    # ── 平均 Token 效率 ──
    params = date_params.copy()
    eff_row = conn.execute(
        f"""SELECT AVG(e.token_efficiency) as avg_eff
        FROM task_evaluations e
        JOIN agent_tasks t ON e.task_id = t.id
        WHERE t.status IN ('completed','failed') {date_filter}""",
        params,
    ).fetchone()

    # ── 按模式对比 ──
    params = date_params.copy()
    mode_rows = conn.execute(
        f"""SELECT
            t.agent_mode,
            COUNT(*) as count,
            CAST(SUM(CASE WHEN t.status = 'completed' THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as sr,
            AVG(e.quality_score) as avg_q,
            AVG(e.tool_error_rate) as avg_ter,
            AVG(t.duration_ms) as avg_dur,
            AVG(t.total_tokens) as avg_tok
        FROM agent_tasks t
        LEFT JOIN task_evaluations e ON e.task_id = t.id
        WHERE t.status IN ('completed','failed') {date_filter}
        GROUP BY t.agent_mode""",
        params,
    ).fetchall()

    by_mode = {}
    for r in mode_rows:
        by_mode[r["agent_mode"]] = {
            "count": r["count"],
            "success_rate": round(r["sr"], 3) if r["sr"] else 0.0,
            "avg_quality": round(r["avg_q"], 1) if r["avg_q"] else None,
            "avg_tool_error_rate": round(r["avg_ter"], 3) if r["avg_ter"] else 0.0,
            "avg_duration_ms": round(r["avg_dur"], 1) if r["avg_dur"] else 0.0,
            "avg_tokens": round(r["avg_tok"], 1) if r["avg_tok"] else 0.0,
        }

    # ── 失败工具排行 (Top 5) ──
    params = date_params.copy()
    tool_rows = conn.execute(
        f"""SELECT
            s.tool_name,
            COUNT(*) as total,
            SUM(CASE WHEN s.status = 'completed' THEN 1 ELSE 0 END) as success_count,
            SUM(CASE WHEN s.status = 'failed' THEN 1 ELSE 0 END) as error_count
        FROM agent_steps s
        JOIN agent_tasks t ON s.task_id = t.id
        WHERE s.step_type = 'tool_call' AND s.tool_name IS NOT NULL {date_filter}
        GROUP BY s.tool_name
        ORDER BY CAST(error_count AS REAL) / COUNT(*) DESC
        LIMIT 5""",
        params,
    ).fetchall()

    top_failing = []
    for r in tool_rows:
        total_calls = r["total"]
        error_count = r["error_count"]
        top_failing.append({
            "tool_name": r["tool_name"],
            "total_calls": total_calls,
            "success_count": r["success_count"],
            "error_count": error_count,
            "error_rate": round(error_count / total_calls, 3) if total_calls > 0 else 0.0,
        })

    conn.close()

    return {
        "period": {"from": date_from or "30d ago", "to": date_to or "now"},
        "total_tasks": total,
        "success_rate": round(success_rate, 3),
        "avg_quality_score": round(qual_row["avg_q"], 1) if qual_row and qual_row["avg_q"] else None,
        "avg_tool_error_rate": round(tool_row["avg_ter"], 3) if tool_row and tool_row["avg_ter"] else 0.0,
        "avg_token_efficiency": round(eff_row["avg_eff"], 4) if eff_row and eff_row["avg_eff"] else 0.0,
        "avg_duration_ms": round(row["avg_duration"], 1) if row["avg_duration"] else 0.0,
        "by_mode": by_mode,
        "top_failing_tools": top_failing,
    }


# ── 工具级错误统计 ────────────────────────────────────────

def compute_tool_error_stats(
    date_from: str = None, date_to: str = None
) -> list[dict]:
    """每个工具的错误率、调用次数、平均耗时"""
    conn = get_db("agent")

    date_join = ""
    date_where = ""
    params = []
    if date_from or date_to:
        date_join = "JOIN agent_tasks t ON s.task_id = t.id"
        conditions = []
        if date_from:
            conditions.append("t.created_at >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("t.created_at <= ?")
            params.append(date_to)
        date_where = " AND " + " AND ".join(conditions)

    rows = conn.execute(
        f"""SELECT
            s.tool_name,
            COUNT(*) as total_calls,
            SUM(CASE WHEN s.status = 'completed' THEN 1 ELSE 0 END) as success_count,
            SUM(CASE WHEN s.status = 'failed' THEN 1 ELSE 0 END) as error_count,
            AVG(s.duration_ms) as avg_duration
        FROM agent_steps s
        {date_join}
        WHERE s.step_type = 'tool_call' AND s.tool_name IS NOT NULL{date_where}
        GROUP BY s.tool_name
        ORDER BY CAST(SUM(CASE WHEN s.status = 'failed' THEN 1 ELSE 0 END) AS REAL) / COUNT(*) DESC""",
        params,
    ).fetchall()

    conn.close()

    return [
        {
            "tool_name": r["tool_name"],
            "total_calls": r["total_calls"],
            "success_count": r["success_count"],
            "error_count": r["error_count"],
            "error_rate": round(
                r["error_count"] / r["total_calls"], 3
            ) if r["total_calls"] > 0 else 0.0,
            "avg_duration_ms": round(r["avg_duration"], 1) if r["avg_duration"] else 0.0,
        }
        for r in rows
    ]


# ── 趋势 ──────────────────────────────────────────────────

def compute_trends(
    days: int = 30, period_type: str = "daily",
    agent_mode: str = None,
) -> list[dict]:
    """按时间窗口汇总趋势数据"""
    conn = get_db("agent")

    # 确定分组格式
    if period_type == "daily":
        group_fmt = "%Y-%m-%d"
    elif period_type == "weekly":
        group_fmt = "%Y-%W"
    else:  # monthly
        group_fmt = "%Y-%m"

    params = []
    mode_filter = ""
    if agent_mode:
        mode_filter = " AND t.agent_mode = ?"
        params.append(agent_mode)

    rows = conn.execute(
        f"""SELECT
            strftime('{group_fmt}', t.created_at) as period_start,
            COUNT(*) as task_count,
            CAST(SUM(CASE WHEN t.status = 'completed' THEN 1 ELSE 0 END) AS REAL)
                / COUNT(*) as success_rate,
            AVG(e.quality_score) as avg_quality,
            AVG(e.tool_error_rate) as avg_tool_error,
            AVG(t.total_tokens) as avg_tokens,
            AVG(t.duration_ms) as avg_duration
        FROM agent_tasks t
        LEFT JOIN task_evaluations e ON e.task_id = t.id
        WHERE t.status IN ('completed','failed')
          AND t.created_at >= datetime('now', ? || ' days')
          {mode_filter}
        GROUP BY period_start
        ORDER BY period_start ASC""",
        [f"-{days}"] + params,
    ).fetchall()

    conn.close()

    return [
        {
            "period_start": r["period_start"],
            "task_count": r["task_count"],
            "success_rate": round(r["success_rate"], 3) if r["success_rate"] else 0.0,
            "avg_quality_score": round(r["avg_quality"], 1) if r["avg_quality"] else None,
            "avg_tool_error_rate": round(r["avg_tool_error"], 3) if r["avg_tool_error"] else 0.0,
            "avg_tokens_per_task": round(r["avg_tokens"], 1) if r["avg_tokens"] else 0.0,
            "avg_duration_ms": round(r["avg_duration"], 1) if r["avg_duration"] else 0.0,
        }
        for r in rows
    ]


# ── 弱点检测 ──────────────────────────────────────────────

def compute_weaknesses(
    date_from: str = None, date_to: str = None
) -> dict:
    """基于启发式规则的弱点检测，不依赖 LLM"""
    conn = get_db("agent")

    date_join = ""
    date_where = ""
    params = []
    if date_from:
        date_join = "JOIN agent_tasks t ON s.task_id = t.id"
        date_where = " AND t.created_at >= ?"
        params.append(date_from)
    if date_to:
        if not date_join:
            date_join = "JOIN agent_tasks t ON s.task_id = t.id"
        date_where += " AND t.created_at <= ?"
        params.append(date_to)

    weaknesses = []

    # 1. 工具错误率 > 20%
    tool_rows = conn.execute(
        f"""SELECT
            s.tool_name,
            COUNT(*) as total,
            SUM(CASE WHEN s.status = 'failed' THEN 1 ELSE 0 END) as errors,
            CAST(SUM(CASE WHEN s.status = 'failed' THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as error_rate
        FROM agent_steps s
        {date_join}
        WHERE s.step_type = 'tool_call' AND s.tool_name IS NOT NULL{date_where}
        GROUP BY s.tool_name
        HAVING error_rate > 0.2
        ORDER BY error_rate DESC""",
        params,
    ).fetchall()

    for r in tool_rows:
        weaknesses.append({
            "severity": "high" if r["error_rate"] > 0.3 else "medium",
            "category": "tool_error",
            "description": f"工具 {r['tool_name']} 错误率 {r['error_rate']:.0%} ({r['errors']}/{r['total']} 次)",
            "evidence": f"共 {r['total']} 次调用，{r['errors']} 次失败",
        })

    # 2. 模式对比：如果 plan_solve 成功率比 react 低 10%+ 且 react 任务数 > 10
    mode_rows = conn.execute(
        f"""SELECT
            t.agent_mode,
            COUNT(*) as cnt,
            CAST(SUM(CASE WHEN t.status = 'completed' THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as sr
        FROM agent_tasks t
        WHERE t.status IN ('completed','failed')
        GROUP BY t.agent_mode""",
    ).fetchall()

    modes = {r["agent_mode"]: r for r in mode_rows}
    if "react" in modes and "plan_solve" in modes:
        react_sr = modes["react"]["sr"]
        ps_sr = modes["plan_solve"]["sr"]
        if react_sr - ps_sr > 0.1 and modes["react"]["cnt"] > 10:
            weaknesses.append({
                "severity": "medium",
                "category": "mode",
                "description": (
                    f"plan_solve 成功率 ({ps_sr:.0%}) 显著低于 react ({react_sr:.0%})，"
                    f"差异 {react_sr - ps_sr:.0%}"
                ),
                "evidence": f"react={modes['react']['cnt']}次, plan_solve={modes['plan_solve']['cnt']}次",
            })

    # 3. Token 效率低（< 0.01 意味着每字符消耗 > 100 tokens）
    eff_row = conn.execute(
        f"""SELECT AVG(e.token_efficiency) as avg_eff
        FROM task_evaluations e
        JOIN agent_tasks t ON e.task_id = t.id
        WHERE t.status = 'completed'""",
    ).fetchone()
    if eff_row and eff_row["avg_eff"] and eff_row["avg_eff"] < 0.01:
        weaknesses.append({
            "severity": "low",
            "category": "efficiency",
            "description": f"Token 效率偏低 ({eff_row['avg_eff']:.4f})，Agent 消耗大量 token 但输出较短",
            "evidence": "检查 System Prompt 长度和工具结果截断策略",
        })

    # 4. 步骤数过多（> 10 步的任务比例 > 30%）
    step_row = conn.execute(
        """SELECT
            CAST(SUM(CASE WHEN cnt > 10 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as over10_pct
        FROM (SELECT s.task_id, COUNT(*) as cnt FROM agent_steps s GROUP BY s.task_id)""",
    ).fetchone()
    if step_row and step_row["over10_pct"] and step_row["over10_pct"] > 0.3:
        weaknesses.append({
            "severity": "medium",
            "category": "efficiency",
            "description": f"{step_row['over10_pct']:.0%} 的任务超过 10 个步骤，Agent 可能效率偏低",
            "evidence": "考虑减少工具调用轮次上限或优化工具选择策略",
        })

    conn.close()

    # 排序：高严重度优先
    severity_order = {"high": 0, "medium": 1, "low": 2}
    weaknesses.sort(key=lambda w: severity_order.get(w["severity"], 99))

    return {
        "period": {"from": date_from or "all", "to": date_to or "now"},
        "weaknesses": weaknesses[:10],
        "summary": "",  # LLM judge 填充
    }

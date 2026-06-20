"""
时间窗口聚合器 — 从 task_evaluations 汇总到 eval_aggregates。

使用场景：
  - 手动触发: POST /api/evaluation/batch 后自动 rollup
  - 定时任务: 每日 rollup 前一天数据（可选 cron）
  - 趋势查询: GET /api/evaluation/trends 直接查 task_evaluations（不依赖 aggregate 表）
"""
import logging
from datetime import datetime, timedelta

from ..database import get_db

logger = logging.getLogger(__name__)


class Aggregator:
    """时间窗口汇总——可选缓存层，加速趋势查询。

    eval_aggregates 表是物化视图，trends 端点直接查询 task_evaluations
    也能正常工作，只在数据量大时才需要 aggregator。
    """

    @staticmethod
    def rollup_period(period_start: str, period_type: str = "daily",
                      agent_mode: str = None):
        """
        汇总指定时间窗口的数据到 eval_aggregates。

        period_start: ISO 日期字符串，如 '2026-06-17'
        period_type: 'daily' | 'weekly' | 'monthly'
        """
        from .db import upsert_aggregate

        conn = get_db("agent")

        # 确定时间段
        if period_type == "daily":
            next_start = (datetime.fromisoformat(period_start) + timedelta(days=1)).isoformat()
        elif period_type == "weekly":
            next_start = (datetime.fromisoformat(period_start) + timedelta(days=7)).isoformat()
        else:  # monthly
            d = datetime.fromisoformat(period_start)
            if d.month == 12:
                next_start = d.replace(year=d.year + 1, month=1).isoformat()
            else:
                next_start = d.replace(month=d.month + 1).isoformat()

        mode_filter = ""
        mode_params = []
        if agent_mode:
            mode_filter = " AND t.agent_mode = ?"
            mode_params = [agent_mode]

        params = [period_start, next_start] + mode_params

        row = conn.execute(
            f"""SELECT
                COUNT(*) as task_count,
                SUM(CASE WHEN t.status = 'completed' THEN 1 ELSE 0 END) as success_count,
                CAST(SUM(CASE WHEN t.status = 'completed' THEN 1 ELSE 0 END) AS REAL)
                    / NULLIF(COUNT(*), 0) as success_rate,
                AVG(e.quality_score) as avg_quality_score,
                AVG(e.tool_error_rate) as avg_tool_error_rate,
                AVG(t.total_tokens) as avg_tokens_per_task,
                AVG(t.duration_ms) as avg_duration_ms,
                AVG(e.token_efficiency) as avg_token_efficiency
            FROM agent_tasks t
            LEFT JOIN task_evaluations e ON e.task_id = t.id
            WHERE t.status IN ('completed','failed')
              AND t.created_at >= ? AND t.created_at < ?{mode_filter}""",
            params,
        ).fetchone()

        conn.close()

        if not row or row["task_count"] == 0:
            logger.debug(f"No tasks in {period_type} {period_start}, skipping rollup")
            return

        metrics = {
            "task_count": row["task_count"],
            "success_count": row["success_count"] or 0,
            "success_rate": round(row["success_rate"], 4) if row["success_rate"] else 0.0,
            "avg_quality_score": round(row["avg_quality_score"], 1) if row["avg_quality_score"] else None,
            "avg_tool_error_rate": round(row["avg_tool_error_rate"], 4) if row["avg_tool_error_rate"] else 0.0,
            "avg_tokens_per_task": round(row["avg_tokens_per_task"], 1) if row["avg_tokens_per_task"] else 0.0,
            "avg_duration_ms": round(row["avg_duration_ms"], 1) if row["avg_duration_ms"] else 0.0,
            "avg_token_efficiency": round(row["avg_token_efficiency"], 4) if row["avg_token_efficiency"] else 0.0,
        }

        upsert_aggregate(period_start, period_type, agent_mode, metrics)
        logger.info(
            f"Rollup {period_type} {period_start} mode={agent_mode or 'all'}: "
            f"{metrics['task_count']} tasks, sr={metrics['success_rate']:.1%}"
        )

    @staticmethod
    def rollup_recent(days: int = 7):
        """
        汇总最近 N 天的数据（按天），用于补全 eval_aggregates。

        每个 agent_mode 单独汇总 + 全量汇总。
        """
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        for i in range(days):
            day = today - timedelta(days=i)
            period_start = day.isoformat()

            # All modes
            Aggregator.rollup_period(period_start, "daily")

            # Per mode
            for mode in ["react", "plan_solve"]:
                Aggregator.rollup_period(period_start, "daily", agent_mode=mode)

    @staticmethod
    def get_aggregates(period_type: str = "daily", days: int = 30,
                       agent_mode: str = None) -> list[dict]:
        """从 eval_aggregates 表查询已汇总的数据"""
        conn = get_db("agent")

        params = [period_type]
        mode_filter = ""
        if agent_mode:
            mode_filter = " AND agent_mode = ?"
            params.append(agent_mode)

        rows = conn.execute(
            f"""SELECT * FROM eval_aggregates
            WHERE period_type = ?{mode_filter}
            AND period_start >= date('now', ? || ' days')
            ORDER BY period_start ASC""",
            params + [f"-{days}"],
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

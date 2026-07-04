"""
反馈闭环控制器 — 评估数据 → 优化建议 → 人工确认 → 迭代。

从 weaknesses / tools / dashboard / trends / feedback 五个维度提取
可执行建议，写入 optimization_recommendations 表。

触发方式：
  - 自动：hooks.py 每 20 个任务触发一次
  - 手动：POST /api/evaluation/recommendations/generate
"""
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class Optimizer:
    """评估反馈闭环——将弱点转化为结构化建议。"""

    @staticmethod
    async def generate(date_from: str = None, date_to: str = None) -> dict:
        """
        主入口：从评估数据中生成优化建议。

        返回: {generated_at, summary, recommendations_count, stats}
        """
        from .db import create_recommendation, get_recommendation_stats, clear_stale_recommendations

        # 清理超过 30 天的 stale 建议
        clear_stale_recommendations()

        generated = []

        # ── 1. 工具错误率 → tool_config 建议 ────────────
        try:
            generated += Optimizer._check_tool_errors(date_from, date_to)
        except Exception as e:
            logger.error(f"Tool error check failed: {e}")

        # ── 2. 模式对比 → mode_routing 建议 ────────────
        try:
            generated += Optimizer._check_mode_gap(date_from, date_to)
        except Exception as e:
            logger.error(f"Mode gap check failed: {e}")

        # ── 3. Token 效率 → parameter 建议 ─────────────
        try:
            generated += Optimizer._check_token_efficiency(date_from, date_to)
        except Exception as e:
            logger.error(f"Token efficiency check failed: {e}")

        # ── 4. 步骤数过多 → parameter 建议 ───────────
        try:
            generated += Optimizer._check_excessive_steps(date_from, date_to)
        except Exception as e:
            logger.error(f"Step count check failed: {e}")

        # ── 5. 用户低分反馈 → prompt 建议 ────────────
        try:
            generated += Optimizer._check_user_feedback(date_from, date_to)
        except Exception as e:
            logger.error(f"Feedback check failed: {e}")

        # ── 写入数据库 ─────────────────────────────────
        for rec in generated:
            try:
                create_recommendation(rec)
            except Exception as e:
                logger.error(f"Failed to save recommendation: {e}")

        stats = get_recommendation_stats()

        # 生成自然语言摘要
        summary = Optimizer._build_summary(generated)

        logger.info(
            f"Optimizer generated {len(generated)} recommendations: "
            f"{stats['pending']} pending, {stats['applied']} applied"
        )

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "recommendations_count": len(generated),
            "stats": stats,
        }

    # ── 各类检测逻辑 ──────────────────────────────────

    @staticmethod
    def _check_tool_errors(date_from, date_to) -> list[dict]:
        """工具错误率 > 20% → 建议加确认步骤"""
        from .metrics import compute_tool_error_stats
        tools = compute_tool_error_stats(date_from, date_to)
        results = []

        for t in tools:
            if t["error_rate"] <= 0.2:
                continue
            if t["total_calls"] < 5:  # 样本太小不触发
                continue

            severity = "high" if t["error_rate"] > 0.3 else "medium"
            results.append({
                "category": "tool_config",
                "severity": severity,
                "title": f"工具 {t['tool_name']} 错误率 {t['error_rate']:.0%}，建议加确认步骤",
                "description": (
                    f"{t['tool_name']} 在 {t['total_calls']} 次调用中失败 {t['error_count']} 次"
                    f"（错误率 {t['error_rate']:.1%}），"
                    f"平均耗时 {t['avg_duration_ms']:.0f}ms。"
                ),
                "evidence": t,
                "suggested_change": {
                    "target": "tool_confirmation",
                    "tool_name": t["tool_name"],
                    "current": "false",
                    "proposed": "true",
                    "rationale": f"错误率 {t['error_rate']:.0%} 超过 20% 阈值，建议每次调用前让用户确认",
                },
                "auto_applicable": True,  # require_confirmation 是安全无副作用参数
            })

        return results

    @staticmethod
    def _check_mode_gap(date_from, date_to) -> list[dict]:
        """plan_solve 成功率比 react 低 10%+ → 建议默认用 react"""
        from .metrics import compute_dashboard_summary
        dash = compute_dashboard_summary(date_from, date_to)
        by_mode = dash.get("by_mode", {})

        react = by_mode.get("react", {})
        ps = by_mode.get("plan_solve", {})

        if not react or not ps:
            return []

        react_sr = react.get("success_rate", 0)
        ps_sr = ps.get("success_rate", 0)
        react_count = react.get("count", 0)
        ps_count = ps.get("count", 0)

        gap = react_sr - ps_sr
        if gap < 0.1 or react_count < 10 or ps_count < 5:
            return []

        severity = "high" if gap > 0.2 else "medium"
        return [{
            "category": "mode_routing",
            "severity": severity,
            "title": f"plan_solve 成功率 ({ps_sr:.0%}) 比 react ({react_sr:.0%}) 低 {gap:.0%}",
            "description": (
                f"react 模式 {react_count} 次任务成功率 {react_sr:.0%}，"
                f"plan_solve 模式 {ps_count} 次任务成功率 {ps_sr:.0%}，"
                f"差距 {gap:.0%}。建议对简单查询默认使用 react 模式。"
            ),
            "evidence": {"react": react, "plan_solve": ps, "gap": round(gap, 3)},
            "suggested_change": {
                "target": "mode_default",
                "current": "用户选择",
                "proposed": "react（简单任务自动路由）",
                "rationale": f"react 成功率显著更高（+{gap:.0%}），应作为默认引擎",
            },
            "auto_applicable": False,  # 模式切换需人工确认
        }]

    @staticmethod
    def _check_token_efficiency(date_from, date_to) -> list[dict]:
        """token_efficiency < 0.01 → 建议降低 token 预算"""
        from ..database import get_db
        conn = get_db("agent")

        params = []
        date_filter = ""
        if date_from:
            date_filter += " AND t.created_at >= ?"
            params.append(date_from)
        if date_to:
            date_filter += " AND t.created_at <= ?"
            params.append(date_to)

        row = conn.execute(
            f"""SELECT AVG(e.token_efficiency) as avg_eff, COUNT(*) as cnt
            FROM task_evaluations e
            JOIN agent_tasks t ON e.task_id = t.id
            WHERE t.status = 'completed' AND e.token_efficiency > 0{date_filter}""",
            params,
        ).fetchone()
        conn.close()

        if not row or not row["avg_eff"] or row["avg_eff"] >= 0.01 or row["cnt"] < 10:
            return []

        avg_eff = row["avg_eff"]
        return [{
            "category": "parameter",
            "severity": "low",
            "title": f"Token 效率偏低 ({avg_eff:.4f})，Agent 消耗大量 token 但输出较短",
            "description": (
                f"平均每字符消耗 {1/avg_eff:.0f} tokens（{row['cnt']} 个完成任务的均值）。"
                f"建议降低 TOKEN_BUDGET 或 MAX_OBS_TOKENS。"
            ),
            "evidence": {"avg_token_efficiency": round(avg_eff, 5), "task_count": row["cnt"]},
            "suggested_change": {
                "target": "parameter_value",
                "current": "TOKEN_BUDGET=90000, MAX_OBS_TOKENS=2000",
                "proposed": "TOKEN_BUDGET=70000, MAX_OBS_TOKENS=1500",
                "rationale": f"当前效率仅 {avg_eff:.4f}，降低预算可迫使 Agent 更精简",
            },
            "auto_applicable": False,
        }]

    @staticmethod
    def _check_excessive_steps(date_from, date_to) -> list[dict]:
        """>30% 任务超过 10 步 → 建议降低 MAX_ITERATIONS"""
        from ..database import get_db
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

        row = conn.execute(
            f"""SELECT
                CAST(SUM(CASE WHEN cnt > 10 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as over10_pct,
                COUNT(*) as total
            FROM (
                SELECT s.task_id, COUNT(*) as cnt
                FROM agent_steps s
                {date_join}
                WHERE 1=1{date_where}
                GROUP BY s.task_id
            )""",
            params,
        ).fetchone()
        conn.close()

        if not row or not row["over10_pct"] or row["over10_pct"] < 0.3 or row["total"] < 10:
            return []

        pct = row["over10_pct"]
        return [{
            "category": "parameter",
            "severity": "medium",
            "title": f"{pct:.0%} 的任务超过 10 个步骤，Agent 效率偏低",
            "description": (
                f"在 {row['total']} 个任务中，{pct:.0%} 超过 10 个步骤。"
                f"建议降低 MAX_ITERATIONS 从 5 到 3 或增强早停规则。"
            ),
            "evidence": {"over10_pct": round(pct, 3), "total_tasks": row["total"]},
            "suggested_change": {
                "target": "parameter_value",
                "current": "MAX_ITERATIONS=5",
                "proposed": "MAX_ITERATIONS=3",
                "rationale": f"{pct:.0%} 任务步数过多，减少最大轮次以提升效率",
            },
            "auto_applicable": False,
        }]

    @staticmethod
    def _check_user_feedback(date_from, date_to) -> list[dict]:
        """用户评分 ≤ 2 或有负面标签 → prompt 建议"""
        from ..database import get_db
        conn = get_db("agent")

        params = []
        date_filter = ""
        if date_from:
            date_filter += " AND created_at >= ?"
            params.append(date_from)
        if date_to:
            date_filter += " AND created_at <= ?"
            params.append(date_to)

        # 低分反馈
        low_ratings = conn.execute(
            f"SELECT COUNT(*) as cnt, GROUP_CONCAT(comment, ' | ') as comments FROM task_feedback WHERE rating <= 2{date_filter}",
            params,
        ).fetchone()

        # 负面标签统计
        import json
        tag_rows = conn.execute(
            f"SELECT tags FROM task_feedback WHERE tags != '[]'{date_filter}",
            params,
        ).fetchall()
        conn.close()

        results = []

        if low_ratings and low_ratings["cnt"] >= 3:
            results.append({
                "category": "prompt",
                "severity": "high",
                "title": f"收到 {low_ratings['cnt']} 条低分反馈（评分 ≤ 2），用户满意度下降",
                "description": (
                    f"用户反馈: {low_ratings['comments'][:200]}。"
                    f"建议检查最近低分任务的 judge_rationale，针对性优化 System Prompt。"
                ),
                "evidence": {"low_rating_count": low_ratings["cnt"]},
                "suggested_change": {
                    "target": "prompt_text",
                    "current": "当前 AGENT_SYSTEM_PROMPT",
                    "proposed": "注入「优先使用 react 模式」「答案需包含来源引用」「工具失败时主动降级」",
                    "rationale": "低分反馈表明 agent 在准确性或完整性上有问题",
                },
                "auto_applicable": False,
            })

        # 负面标签
        tag_counts = {}
        for tr in tag_rows:
            try:
                tags = json.loads(tr["tags"])
            except (json.JSONDecodeError, TypeError):
                continue
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        frequent_tags = {k: v for k, v in tag_counts.items() if v >= 3}
        if frequent_tags:
            tags_str = ", ".join(f"{k}({v}次)" for k, v in frequent_tags.items())
            results.append({
                "category": "prompt",
                "severity": "medium",
                "title": f"用户高频负面标签: {tags_str}",
                "description": (
                    f"用户提交的反馈标签中，以下标签出现 3 次以上: {tags_str}。"
                    f"建议针对性地调整 System Prompt 以减少这些问题。"
                ),
                "evidence": {"frequent_tags": frequent_tags},
                "suggested_change": {
                    "target": "prompt_text",
                    "current": "当前 AGENT_SYSTEM_PROMPT",
                    "proposed": f"针对 {list(frequent_tags.keys())} 增加约束规则",
                    "rationale": "用户反馈标签揭示高频问题模式",
                },
                "auto_applicable": False,
            })

        return results

    # ── 摘要生成 ──────────────────────────────────────

    @staticmethod
    def _build_summary(recommendations: list[dict]) -> str:
        if not recommendations:
            return "当前未检测到需要优化的建议。系统运行良好。"

        high = sum(1 for r in recommendations if r["severity"] == "high")
        med = sum(1 for r in recommendations if r["severity"] == "medium")
        low = sum(1 for r in recommendations if r["severity"] == "low")

        parts = [f"基于评估数据分析，生成 {len(recommendations)} 条优化建议"]
        if high:
            parts.append(f"{high} 条高优")
        if med:
            parts.append(f"{med} 条中优")
        if low:
            parts.append(f"{low} 条低优")

        return "（" + "、".join(parts[1:]) + "）。请审核后 apply 或 dismiss。"

    @staticmethod
    def get_summary_from_db() -> str:
        """从 DB 读取 pending 建议生成摘要（用于推荐列表展示）"""
        from .db import list_recommendations, get_recommendation_stats
        stats = get_recommendation_stats()
        pending = list_recommendations(status="pending", limit=20)

        if not pending:
            return "无待处理建议。系统运行良好，或尝试 POST /generate 重新生成。"

        high = sum(1 for r in pending if r["severity"] == "high")
        return (
            f"共 {stats['total']} 条建议（{stats['applied']} 已应用，"
            f"{stats['dismissed']} 已忽略），当前 {stats['pending']} 条待处理"
            f"{'，其中 ' + str(high) + ' 条高优' if high else ''}。"
        )

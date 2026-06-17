"""
评估 API 路由 — /api/evaluation/...
"""
import asyncio
import json

from fastapi import APIRouter, HTTPException, Query, Depends

from ..dependencies import verify_token_from_header
from ..auth import verify_token as _verify_token_direct
from .db import (
    get_task_evaluation, get_evaluation_for_task,
    list_task_evaluations, delete_task_evaluation,
    create_task_evaluation, save_feedback, get_feedback,
)
from .models import (
    TaskEvaluationOut, DashboardSummary, ToolErrorStats,
    TrendPoint, WeaknessesOut, WeaknessItem,
    BatchEvalRequest, BatchEvalResponse,
    TaskFeedbackIn, TaskFeedbackOut,
)
from .metrics import (
    compute_quantitative_metrics,
    compute_dashboard_summary,
    compute_tool_error_stats,
    compute_trends,
    compute_weaknesses,
)

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


# ── 辅助：从 token 提取 user_id ────────────────────────────

def _get_user_id(authorization: str | None = None) -> int:
    """从 Authorization header 提取 user_id（宽松模式，无 token 返回 0）"""
    if not authorization:
        return 0
    user = verify_token_from_header(authorization)
    return user["user_id"] if user else 0


# ── 单任务评估 ─────────────────────────────────────────────

@router.post("/tasks/{task_id}")
async def evaluate_task(task_id: int, include_judge: bool = False):
    """评估单个任务。重新评估会先删除旧记录。"""
    delete_task_evaluation(task_id)

    metrics = compute_quantitative_metrics(task_id)
    eval_id = create_task_evaluation(task_id, metrics, mode="manual")

    result = get_task_evaluation(task_id)

    if include_judge:
        try:
            from .judge import LLMJudge
            judge = LLMJudge()
            quality = await judge.evaluate_task(task_id)
            from .db import update_evaluation_quality
            update_evaluation_quality(eval_id, quality)
            result = get_task_evaluation(task_id)
        except ImportError:
            raise HTTPException(501, "LLM Judge not available yet")

    return {"evaluation": result}


@router.get("/tasks/{task_id}")
async def get_evaluation(task_id: int):
    """查询任务评估"""
    ev = get_task_evaluation(task_id)
    if not ev:
        raise HTTPException(404, f"No evaluation for task {task_id}")
    return {"evaluation": ev}


@router.delete("/tasks/{task_id}")
async def delete_evaluation(task_id: int):
    """删除评估（允许重新评估）"""
    deleted = delete_task_evaluation(task_id)
    return {"task_id": task_id, "deleted": deleted}


# ── 评估列表 ───────────────────────────────────────────────

@router.get("/tasks")
async def list_evaluations(
    status: str = Query(None, description="任务状态: completed|failed"),
    evaluation_mode: str = Query(None, description="评估模式: auto|manual|batch"),
    date_from: str = Query(None),
    date_to: str = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
):
    """分页查询评估记录"""
    rows = list_task_evaluations(
        status=status,
        evaluation_mode=evaluation_mode,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        offset=offset,
    )
    return {"evaluations": rows, "count": len(rows)}


# ── 批量评估 ───────────────────────────────────────────────

@router.post("/batch")
async def batch_evaluate(req: BatchEvalRequest):
    """批量评估任务——按 ID 列表或日期范围"""
    from ..database import get_db

    tasks_to_eval = []

    if req.task_ids:
        # 按 ID 列表
        conn = get_db("agent")
        placeholders = ",".join("?" for _ in req.task_ids)
        rows = conn.execute(
            f"SELECT id FROM agent_tasks WHERE id IN ({placeholders}) AND status IN ('completed','failed')",
            req.task_ids,
        ).fetchall()
        conn.close()
        tasks_to_eval = [r["id"] for r in rows]
    else:
        # 按日期范围
        conn = get_db("agent")
        q = "SELECT id FROM agent_tasks WHERE status IN ('completed','failed')"
        params = []
        if req.date_from:
            q += " AND created_at >= ?"
            params.append(req.date_from)
        if req.date_to:
            q += " AND created_at <= ?"
            params.append(req.date_to)
        if req.status:
            q += " AND status = ?"
            params.append(req.status)
        q += f" ORDER BY created_at DESC LIMIT {req.limit}"
        rows = conn.execute(q, params).fetchall()
        conn.close()
        tasks_to_eval = [r["id"] for r in rows]

    result = BatchEvalResponse(total=len(tasks_to_eval), evaluated=0, skipped=0)

    for tid in tasks_to_eval:
        existing = get_evaluation_for_task(tid)
        if existing:
            result.skipped += 1
            continue

        try:
            metrics = compute_quantitative_metrics(tid)
            create_task_evaluation(tid, metrics, mode="batch")
            result.evaluated += 1

            if req.include_judge:
                try:
                    from .judge import LLMJudge
                    from .db import update_evaluation_quality, get_evaluation_for_task
                    judge = LLMJudge()
                    quality = await judge.evaluate_task(tid)
                    eid = get_evaluation_for_task(tid)
                    if eid:
                        update_evaluation_quality(eid, quality)
                except ImportError:
                    pass
        except Exception as e:
            result.errors.append(f"task {tid}: {e}")

    return result.model_dump()


# ── 看板 ──────────────────────────────────────────────────

@router.get("/dashboard")
async def dashboard(
    date_from: str = Query(None),
    date_to: str = Query(None),
):
    """总览看板——成功率/质量分/模式对比/失败工具"""
    summary = compute_dashboard_summary(date_from=date_from, date_to=date_to)
    return summary


# ── 趋势 ──────────────────────────────────────────────────

@router.get("/trends")
async def trends(
    days: int = Query(30, le=365),
    period: str = Query("daily", description="daily|weekly|monthly"),
    agent_mode: str = Query(None, description="react|plan_solve"),
):
    """时间趋势——按天/周/月汇总"""
    data = compute_trends(days=days, period_type=period, agent_mode=agent_mode)
    return {"period": period, "days": days, "agent_mode": agent_mode, "trends": data}


# ── 工具错误统计 ──────────────────────────────────────────

@router.get("/tools")
async def tool_stats(
    date_from: str = Query(None),
    date_to: str = Query(None),
):
    """工具级错误率统计"""
    stats = compute_tool_error_stats(date_from=date_from, date_to=date_to)
    return {"tools": stats}


# ── 弱点分析 ──────────────────────────────────────────────

@router.get("/weaknesses")
async def weaknesses(
    date_from: str = Query(None),
    date_to: str = Query(None),
):
    """弱点分析——启发式 + LLM 增强（如果可用）"""
    result = compute_weaknesses(date_from=date_from, date_to=date_to)

    # 尝试用 LLM 生成自然语言总结
    try:
        from .judge import LLMJudge
        judge = LLMJudge()
        summary = await judge.analyze_weaknesses(result["weaknesses"])
        result["summary"] = summary
    except ImportError:
        if result["weaknesses"]:
            items = "; ".join(w["description"] for w in result["weaknesses"][:3])
            result["summary"] = f"检测到 {len(result['weaknesses'])} 个潜在问题: {items}"
        else:
            result["summary"] = "当前时间范围内未检测到明显弱项。"

    return result


# ── 优化建议（反馈闭环）──────────────────────────────────

@router.post("/recommendations/generate")
async def generate_recommendations(
    date_from: str = Query(None),
    date_to: str = Query(None),
):
    """手动触发建议生成——从评估数据中提取可执行优化建议"""
    from .optimizer import Optimizer
    result = await Optimizer.generate(date_from=date_from, date_to=date_to)
    return result


@router.get("/recommendations")
async def list_recommendations(
    status: str = Query(None, description="pending|applied|dismissed"),
    severity: str = Query(None, description="high|medium|low"),
    limit: int = Query(50, le=200),
):
    """获取优化建议列表"""
    from .db import list_recommendations, get_recommendation_stats
    from .optimizer import Optimizer

    recs = list_recommendations(status=status, severity=severity, limit=limit)
    stats = get_recommendation_stats()
    summary = Optimizer.get_summary_from_db()

    return {
        "generated_at": recs[0]["created_at"] if recs else "",
        "summary": summary,
        "recommendations": recs,
        "stats": stats,
    }


@router.post("/recommendations/{rec_id}/apply")
async def apply_recommendation(rec_id: int):
    """确认应用建议（标记 applied）"""
    from .db import update_recommendation_status
    update_recommendation_status(rec_id, "applied")
    return {"id": rec_id, "status": "applied"}


@router.post("/recommendations/{rec_id}/dismiss")
async def dismiss_recommendation(rec_id: int):
    """忽略建议（标记 dismissed）"""
    from .db import update_recommendation_status
    update_recommendation_status(rec_id, "dismissed")
    return {"id": rec_id, "status": "dismissed"}


# ── 用户反馈 ──────────────────────────────────────────────

@router.post("/feedback")
async def submit_feedback(
    req: TaskFeedbackIn,
    authorization: str | None = None,
):
    """提交任务反馈评分"""
    user_id = _get_user_id(authorization)
    fid = save_feedback(
        user_id=user_id,
        task_id=req.task_id,
        rating=req.rating,
        comment=req.comment,
        tags=req.tags,
    )
    return {"id": fid, "task_id": req.task_id, "rating": req.rating}


@router.get("/feedback/{task_id}")
async def get_task_feedback(task_id: int):
    """查询任务反馈"""
    fb = get_feedback(task_id)
    return {"task_id": task_id, "feedback": fb}

"""
引擎完成钩子 — 从 AgentEngine/PlanSolveEngine 的任务结束路径调用。

职责：
  1. 计算定量指标写入 task_evaluations（每次任务结束，零 LLM 成本）
  2. 按采样率触发 LLM Judge 定性评估（异步，不阻塞引擎）

用法：
  在 engine.py / plan_solve_engine.py 的 completed/failed 路径：
      from ..evaluation.hooks import on_task_completed
      asyncio.create_task(on_task_completed(task_id))
"""
import asyncio
import logging

logger = logging.getLogger(__name__)

# LLM Judge 采样率（20%）
JUDGE_SAMPLING_RATE = 0.2
# Optimizer 触发频率（每 N 个任务）
OPTIMIZER_TRIGGER_EVERY = 20
# 计数器（不依赖 random 模块）
_sampling_counter = 0
_task_counter = 0


def _should_judge(task_id: int) -> bool:
    """基于 task_id 的确定性采样，保证可复现"""
    global _sampling_counter
    _sampling_counter += 1
    return (task_id % 5 == 0)  # 每 5 个任务评估 1 个 = 20%


async def on_task_completed(task_id: int):
    """
    任务完成后触发。异步 fire-and-forget，不阻塞引擎。

    步骤：
      1. 计算定量指标 → 写入 task_evaluations
      2. 按采样率 → 触发 LLM Judge
    """
    try:
        from .metrics import compute_quantitative_metrics
        from .db import create_task_evaluation, get_evaluation_for_task

        # 避免重复评估
        existing = get_evaluation_for_task(task_id)
        if existing:
            logger.debug(f"Task {task_id} already evaluated (id={existing}), skipping")
            return

        # 1. 定量指标（始终执行）
        metrics = compute_quantitative_metrics(task_id)
        eval_id = create_task_evaluation(task_id, metrics, mode="auto")

        mode_str = "✅" if metrics["success"] else "❌"
        logger.info(
            f"Evaluation {eval_id}: task={task_id} {mode_str} "
            f"success={metrics['success']} tool_err={metrics['tool_error_rate']:.2%} "
            f"steps={metrics['step_count']}"
        )

        # 2. 定性评分（采样触发）
        if _should_judge(task_id):
            logger.info(f"Task {task_id} selected for LLM judge (sampling rate={JUDGE_SAMPLING_RATE})")
            asyncio.create_task(_run_judge(task_id, eval_id))

        # 3. 优化建议（每 N 个任务触发一次）
        global _task_counter
        _task_counter += 1
        if _task_counter % OPTIMIZER_TRIGGER_EVERY == 0:
            logger.info(f"Optimizer triggered after {_task_counter} tasks")
            asyncio.create_task(_run_optimizer())

    except Exception as e:
        logger.error(f"on_task_completed failed for task {task_id}: {e}", exc_info=True)


async def _run_judge(task_id: int, eval_id: int):
    """运行 LLM Judge（异步，不阻塞主流程）"""
    try:
        from .judge import LLMJudge
        judge = LLMJudge()
        quality = await judge.evaluate_task(task_id)

        from .db import update_evaluation_quality
        update_evaluation_quality(eval_id, quality)

        logger.info(
            f"LLM Judge completed for task {task_id}: "
            f"overall={quality.get('overall')}/10 "
            f"(tokens={quality.get('tokens', 0)})"
        )
    except Exception as e:
        logger.error(f"LLM Judge failed for task {task_id}: {e}")


async def _run_optimizer():
    """运行优化器（异步，不阻塞主流程）"""
    try:
        from .optimizer import Optimizer
        result = await Optimizer.generate()
        logger.info(
            f"Optimizer completed: {result['recommendations_count']} recommendations, "
            f"{result['stats']['pending']} pending"
        )
    except Exception as e:
        logger.error(f"Optimizer failed: {e}")

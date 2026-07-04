"""
Agent 性能评估模块。

两层架构：
  - 定量指标 (metrics.py): 纯 SQL 查询 agent_tasks/agent_steps，零 LLM 成本
  - 定性评分 (judge.py): LLM-as-Judge，结构化 rubric，采样触发

入口：
  - hooks.on_task_completed(task_id) → 引擎钩子
  - router → REST API (/api/evaluation/...)
"""
from .db import (
    init_eval_db, create_task_evaluation, get_task_evaluation,
    save_feedback, get_feedback, get_evaluation_for_task,
)
from .models import (
    TaskEvaluationOut, DashboardSummary, ToolErrorStats,
    TaskFeedbackIn, TaskFeedbackOut,
    WeaknessItem, WeaknessesOut, TrendPoint,
)
from .metrics import (
    compute_quantitative_metrics,
    compute_dashboard_summary,
    compute_tool_error_stats,
    compute_trends,
    compute_weaknesses,
)
from .judge import LLMJudge
from .aggregator import Aggregator
from .hooks import on_task_completed

"""评估模块 — Pydantic 数据模型"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── 任务评估 ──────────────────────────────────────────────

class TaskEvaluationOut(BaseModel):
    """单任务评估结果（API 响应）"""
    id: int
    task_id: int
    # 定量
    success: bool
    tool_total: int = 0
    tool_success: int = 0
    tool_error: int = 0
    tool_error_rate: float = 0.0
    step_count: int = 0
    avg_step_duration_ms: float = 0.0
    token_efficiency: float = 0.0
    completion_ratio: float = 0.0
    plan_coverage: Optional[float] = None
    # 定性（LLM Judge，可能为空）
    quality_score: Optional[float] = None
    accuracy_score: Optional[float] = None
    completeness_score: Optional[float] = None
    relevance_score: Optional[float] = None
    tool_usage_score: Optional[float] = None
    judge_rationale: Optional[str] = None
    judge_model: Optional[str] = None
    judge_tokens: int = 0
    # 元数据
    evaluated_at: str
    evaluation_mode: str = "auto"


class BatchEvalRequest(BaseModel):
    """批量评估请求"""
    task_ids: Optional[list[int]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    status: Optional[str] = "completed"
    limit: int = Field(default=100, le=500)
    include_judge: bool = False


class BatchEvalResponse(BaseModel):
    """批量评估结果"""
    total: int
    evaluated: int
    skipped: int  # 已有评估跳过的
    errors: list[str] = []


# ── 看板 ──────────────────────────────────────────────────

class ModeComparison(BaseModel):
    """按 Agent 模式的对比"""
    count: int = 0
    success_rate: float = 0.0
    avg_quality: Optional[float] = None
    avg_tool_error_rate: float = 0.0
    avg_duration_ms: float = 0.0
    avg_tokens: float = 0.0


class DashboardSummary(BaseModel):
    """总览看板"""
    period: dict  # {from, to}
    total_tasks: int = 0
    success_rate: float = 0.0
    avg_quality_score: Optional[float] = None
    avg_tool_error_rate: float = 0.0
    avg_token_efficiency: float = 0.0
    avg_duration_ms: float = 0.0
    by_mode: dict[str, ModeComparison] = {}
    top_failing_tools: list[dict] = []


class ToolErrorStats(BaseModel):
    """工具级错误统计"""
    tool_name: str
    total_calls: int
    success_count: int
    error_count: int
    error_rate: float
    avg_duration_ms: float = 0.0


class TrendPoint(BaseModel):
    """单个时间点数据"""
    period_start: str
    task_count: int = 0
    success_rate: float = 0.0
    avg_quality_score: Optional[float] = None
    avg_tool_error_rate: float = 0.0
    avg_tokens_per_task: float = 0.0
    avg_duration_ms: float = 0.0


# ── 弱点 ──────────────────────────────────────────────────

class WeaknessItem(BaseModel):
    severity: str  # high | medium | low
    category: str  # tool_error | quality | efficiency | mode
    description: str
    evidence: Optional[str] = None  # 支撑数据


class WeaknessesOut(BaseModel):
    period: dict  # {from, to}
    weaknesses: list[WeaknessItem]
    summary: str  # LLM 生成的自然语言分析（如果可用）


# ── 反馈 ──────────────────────────────────────────────────

class TaskFeedbackIn(BaseModel):
    task_id: int
    rating: int = Field(ge=1, le=5)
    comment: Optional[str] = None
    tags: list[str] = []


class TaskFeedbackOut(BaseModel):
    id: int
    task_id: int
    user_id: int
    rating: int
    comment: Optional[str] = None
    tags: list[str] = []
    created_at: str


# ── 优化建议 ──────────────────────────────────────────────

class SuggestedChange(BaseModel):
    target: str  # tool_confirmation | parameter_value | mode_default | prompt_text
    current: str = ""
    proposed: str = ""
    rationale: str = ""


class OptimizationRecommendation(BaseModel):
    id: int
    category: str  # tool_config | parameter | mode_routing | prompt
    severity: str  # high | medium | low
    title: str
    description: str
    evidence: dict = {}
    suggested_change: SuggestedChange
    auto_applicable: bool = False
    status: str = "pending"
    applied_at: Optional[str] = None
    created_at: str = ""


class RecommendationsResponse(BaseModel):
    generated_at: str
    summary: str
    recommendations: list[OptimizationRecommendation]
    stats: dict  # {total, pending, applied, dismissed}

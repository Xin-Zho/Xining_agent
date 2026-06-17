"""日志统计 + Prompt 优化建议"""
from fastapi import APIRouter, Depends

from ..auth import get_current_user
from ..dependencies import dialogue_logger
from ..training import RuleExtractor

router = APIRouter(prefix="/api", tags=["logs"])


@router.get("/logs/stats")
def logs_stats(user: dict = Depends(get_current_user)):
    return dialogue_logger.stats()


@router.get("/logs/suggestions")
def logs_suggestions(user: dict = Depends(get_current_user)):
    logs = dialogue_logger.get_agent_logs(days=30)
    extractor = RuleExtractor(logs)
    suggestions = extractor.generate_prompt_suggestions()
    return {"ok": True, "total_logs": len(logs), "suggestions": suggestions}

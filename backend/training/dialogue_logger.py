"""
对话监听器 — 记录所有 Claude/Agent 交互到结构化 JSONL 日志

用法：
    logger = DialogueLogger()
    logger.log(user="alice", question="...", answer="...", source="claude", tokens=500)
    logger.log(user="alice", question="...", answer="...", source="agent", steps=[...])
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path


class DialogueLogger:
    """
    每条交互一行 JSON，增量追加到日志文件。
    文件名按天分：YYYY-MM-DD.jsonl
    """

    def __init__(self, log_dir: str = None):
        if log_dir is None:
            log_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "dialogue_logs"
            )
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def log(self, user: str, question: str, answer: str,
            source: str = "claude", model: str = "",
            tokens: int = 0, steps: list = None,
            reflection: str = "", metadata: dict = None):
        """
        source: "claude" | "agent" | "chat"
        """
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "user": user,
            "model": model,
            "question": question[:5000],
            "answer": answer[:10000],
            "answer_length": len(answer),
            "tokens": tokens,
            "steps": steps or [],
            "reflection": reflection,
            "metadata": metadata or {},
        }

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filepath = os.path.join(self.log_dir, f"{today}.jsonl")
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_recent_logs(self, days: int = 7) -> list[dict]:
        """读取最近 N 天的日志"""
        import glob as glob_module
        logs = []
        pattern = os.path.join(self.log_dir, "*.jsonl")
        files = sorted(glob_module.glob(pattern), reverse=True)[:days]
        for fpath in files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            logs.append(json.loads(line))
            except Exception:
                continue
        return logs

    def get_claude_logs(self, days: int = 7) -> list[dict]:
        """只取 Claude 来源的日志"""
        all_logs = self.get_recent_logs(days)
        return [l for l in all_logs if l.get("source") == "claude"]

    def stats(self) -> dict:
        """日志统计概览"""
        logs = self.get_recent_logs(7)
        claude_logs = [l for l in logs if l.get("source") == "claude"]
        agent_logs = [l for l in logs if l.get("source") == "agent"]
        return {
            "total": len(logs),
            "claude": len(claude_logs),
            "agent": len(agent_logs),
            "avg_claude_tokens": sum(l.get("tokens", 0) for l in claude_logs) // max(len(claude_logs), 1),
            "avg_agent_tokens": sum(l.get("tokens", 0) for l in agent_logs) // max(len(agent_logs), 1),
        }

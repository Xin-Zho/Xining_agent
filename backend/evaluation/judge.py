"""
LLM-as-Judge — 结构化 rubric 定性评估。

两套 prompt：
  - evaluate_task: 单任务 4 维度评分（Accuracy/Completeness/Relevance/Tool Usage）
  - analyze_weaknesses: 弱点列表 → 自然语言总结

使用 DeepSeek API（与系统共用同一个 LLMClient）。
"""
import json
import logging

from ..llm_client import LLMClient, estimate_tokens

logger = logging.getLogger(__name__)

# ── Judge System Prompt ──────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are an agent quality evaluator. Score the following agent task execution on four dimensions (0-10 each):

1. ACCURACY (0-10): Are the facts correct? Did the agent fabricate information or hallucinate?
   0 = completely fabricated, 10 = verifiably correct
2. COMPLETENESS (0-10): Did the agent fully answer the user's question without omission?
   0 = missed the point entirely, 10 = comprehensive
3. RELEVANCE (0-10): Did the agent stay on topic? Was tool use appropriate to the question?
   0 = wasted steps on irrelevant tools, 10 = laser-focused
4. TOOL USAGE (0-10): Were the right tools selected? Unnecessary or redundant calls?
   0 = wrong tools, excessive calls, 10 = optimal selection

CRITICAL RULES:
- Score HONESTLY. Don't inflate scores just because the answer looks polished.
- If the final answer is empty or an error message, score all dimensions 0.
- If the agent used NO tools on a question that doesn't need tools, that's correct.
- If the agent called the same tool with the same args multiple times, deduct tool_usage.

Respond in JSON ONLY (no markdown, no extra text):
{"accuracy": <0-10>, "completeness": <0-10>, "relevance": <0-10>, "tool_usage": <0-10>, "overall": <0-10>, "rationale": "<2-3 sentence explanation in Chinese>"}"""


WEAKNESS_ANALYSIS_PROMPT = """You are an agent performance analyst. Given a list of detected weaknesses, write a concise 3-5 sentence summary in Chinese that:

1. Highlights the most critical issues (severity=high first)
2. Explains the practical impact on users
3. Suggests the most impactful fix (pick ONE)

Weaknesses detected:
{weaknesses_text}

Respond in Chinese, plain text only (no JSON, no markdown). Keep under 200 characters."""


class LLMJudge:
    """LLM 评估器——使用 DeepSeek 作为评判模型"""

    def __init__(self, model_id: str = None):
        self.model_id = model_id or os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
        self._client = None  # lazy init

    def _get_client(self):
        if self._client is None:
            self._client = LLMClient(model_id=self._model)
        return self._client

    # ── 单任务评分 ─────────────────────────────────────

    async def evaluate_task(self, task_id: int) -> dict:
        """
        对单个任务进行结构化评分。

        返回: {
            accuracy, completeness, relevance, tool_usage,
            overall, rationale, model, tokens
        }
        """
        context = self._build_task_context(task_id)
        if not context:
            return self._empty_score("Task not found")

        prompt = self._build_judge_prompt(context)

        try:
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            response = self._get_client().chat(messages)
            result = self._parse_judge_response(response)
            result["model"] = self._model
            result["tokens"] = estimate_tokens(
                JUDGE_SYSTEM_PROMPT + prompt + response
            )
            return result
        except Exception as e:
            logger.error(f"LLM Judge failed for task {task_id}: {e}")
            return self._empty_score(f"Judge error: {e}")

    def _build_task_context(self, task_id: int) -> dict | None:
        """从 DB 构建评估上下文"""
        from ..database import get_db
        conn = get_db("agent")

        task = conn.execute(
            "SELECT * FROM agent_tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if not task:
            conn.close()
            return None

        steps = conn.execute(
            """SELECT step_number, status, step_type, tool_name, tool_args,
                      tool_result, thought, duration_ms
               FROM agent_steps WHERE task_id = ?
               ORDER BY step_number""",
            (task_id,),
        ).fetchall()
        conn.close()

        return {
            "task_id": task_id,
            "description": task["description"],
            "agent_mode": task["agent_mode"],
            "status": task["status"],
            "final_answer": task["final_answer"] or "",
            "total_tokens": task["total_tokens"] or 0,
            "duration_ms": task["duration_ms"] or 0,
            "steps": [dict(s) for s in steps],
        }

    def _build_judge_prompt(self, ctx: dict) -> str:
        """构建 judge prompt（含任务描述、步骤摘要、最终答案）"""
        lines = [
            f"Task: {ctx['description'][:300]}",
            f"Agent Mode: {ctx['agent_mode']}",
            f"Status: {ctx['status']} | Tokens: {ctx['total_tokens']} | Duration: {ctx['duration_ms']}ms",
            "",
            "Steps:",
        ]

        for step in ctx["steps"]:
            sn = step["step_number"]
            stype = step["step_type"]
            status = step["status"]
            tool = step.get("tool_name") or "-"

            if stype == "tool_call":
                args_preview = (step.get("tool_args") or "")[:120]
                result_preview = (step.get("tool_result") or "")[:200]
                lines.append(
                    f"  [{sn}] tool_call: {tool}({args_preview}) → {status} "
                    f"result={result_preview}"
                )
            elif stype == "thought":
                thought = (step.get("thought") or "")[:150]
                lines.append(f"  [{sn}] thought: {thought}")
            elif stype == "response":
                lines.append(f"  [{sn}] response (see final answer)")
            elif stype == "plan":
                lines.append(f"  [{sn}] plan: created")
            elif stype == "observation":
                obs = (step.get("tool_result") or "")[:150]
                lines.append(f"  [{sn}] observation: {obs}")

        lines.append("")
        lines.append(f"Final Answer:\n{ctx['final_answer'][:2000]}")

        return "\n".join(lines)

    def _parse_judge_response(self, response: str) -> dict:
        """解析 LLM JSON 响应，带容错"""
        try:
            # 尝试直接解析
            data = json.loads(response.strip())
            return {
                "accuracy": float(data.get("accuracy", 0)),
                "completeness": float(data.get("completeness", 0)),
                "relevance": float(data.get("relevance", 0)),
                "tool_usage": float(data.get("tool_usage", 0)),
                "overall": float(data.get("overall", 0)),
                "rationale": data.get("rationale", ""),
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # 回退：用正则提取
        import re
        def _extract(key):
            m = re.search(rf'"{key}"\s*:\s*([\d.]+)', response)
            return float(m.group(1)) if m else 0.0

        rationale = ""
        m = re.search(r'"rationale"\s*:\s*"([^"]+)"', response)
        if m:
            rationale = m.group(1)

        return {
            "accuracy": _extract("accuracy"),
            "completeness": _extract("completeness"),
            "relevance": _extract("relevance"),
            "tool_usage": _extract("tool_usage"),
            "overall": _extract("overall"),
            "rationale": rationale,
        }

    def _empty_score(self, rationale: str = "") -> dict:
        return {
            "accuracy": 0.0, "completeness": 0.0,
            "relevance": 0.0, "tool_usage": 0.0,
            "overall": 0.0, "rationale": rationale,
            "model": "", "tokens": 0,
        }

    # ── 弱点分析 ───────────────────────────────────────

    async def analyze_weaknesses(self, weaknesses: list[dict]) -> str:
        """将弱点列表转为自然语言总结"""
        if not weaknesses:
            return "当前时间范围内未检测到明显弱项。"

        text_items = []
        for i, w in enumerate(weaknesses[:5]):
            text_items.append(
                f"{i+1}. [{w['severity'].upper()}] {w['category']}: {w['description']}"
            )

        prompt = WEAKNESS_ANALYSIS_PROMPT.format(
            weaknesses_text="\n".join(text_items)
        )

        try:
            messages = [
                {"role": "user", "content": prompt},
            ]
            response = self._get_client().chat(messages)
            return response.strip()[:300]
        except Exception as e:
            logger.error(f"Weakness analysis failed: {e}")
            items = "; ".join(w["description"] for w in weaknesses[:3])
            return f"检测到 {len(weaknesses)} 个潜在问题: {items}"

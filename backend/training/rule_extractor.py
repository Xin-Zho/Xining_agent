"""
规则提取器 — 从 Agent 对话日志中提取 Prompt 优化建议

分析维度：
  1. 工具选择模式 — Agent 面对不同类型问题选什么工具
  2. 回答结构 — Agent 如何组织回答
  3. 失败恢复 — Agent 如何从错误中恢复
  4. Token 效率 — Agent 如何保持简洁

输出：可直接用于 Prompt 优化的规则列表
"""
from collections import Counter


class RuleExtractor:
    """从对话日志中提取可复用的规则"""

    def __init__(self, logs: list[dict]):
        self.logs = [l for l in logs if l.get("source") in ("agent", "chat")]

    def extract_tool_usage_patterns(self) -> dict:
        """分析 Agent 的工具使用模式"""
        if not self.logs:
            return {"patterns": [], "recommendations": ["暂无日志，多通过 Agent 对话后会自动分析。"]}

        tool_counter = Counter()
        question_words = Counter()

        for entry in self.logs:
            steps = entry.get("steps", [])
            question = entry.get("question", "")

            for step in steps:
                action = step.get("action", "")
                tool_name = action.split("(")[0] if "(" in action else ""
                if tool_name:
                    tool_counter[tool_name] += 1

            if "帮我看" in question or "看看" in question or "有什么" in question:
                question_words["浏览/查看"] += 1
            elif "搜索" in question or "查" in question or "找" in question:
                question_words["搜索/查找"] += 1
            elif "改" in question or "修改" in question or "重构" in question:
                question_words["编辑/修改"] += 1
            elif "创建" in question or "新建" in question or "写" in question:
                question_words["创建/编写"] += 1
            else:
                question_words["问答/咨询"] += 1

        patterns = {
            "tool_distribution": dict(tool_counter.most_common(10)),
            "question_types": dict(question_words.most_common(5)),
        }

        recommendations = [
            f"最常用工具是 '{tool_counter.most_common(1)[0][0]}'（{tool_counter.most_common(1)[0][1]} 次），确保该工具的描述精准。",
            f"最常见问题类型是 '{question_words.most_common(1)[0][0]}'，优化此类场景的 Prompt 示例。",
        ]

        return {"patterns": patterns, "recommendations": recommendations}

    def extract_answer_patterns(self) -> dict:
        """分析回答结构特征"""
        if not self.logs:
            return {"avg_length": 0, "has_code_blocks": 0, "recommendations": []}

        total_len = sum(l.get("answer_length", 0) for l in self.logs)
        avg_len = total_len // max(len(self.logs), 1)
        code_count = sum(1 for l in self.logs if "```" in str(l.get("answer", "")))
        code_pct = code_count / max(len(self.logs), 1) * 100

        recommendations = []
        if avg_len > 2000:
            recommendations.append(f"Agent 平均回答 {avg_len} 字符，较长。检查你的 Agent 回答是否过于冗长。")
        if code_pct > 50:
            recommendations.append(f"{code_pct:.0f}% 的回答包含代码块，在 Agent Prompt 中强调代码示例格式。")

        return {
            "avg_length": avg_len,
            "code_block_pct": round(code_pct, 1),
            "recommendations": recommendations,
        }

    def extract_failure_recovery(self) -> dict:
        """分析错误恢复模式"""
        recovery_patterns = []
        for entry in self.logs:
            answer = entry.get("answer", "")
            steps = entry.get("steps", [])
            if any("失败" in s.get("observation", "") or "错误" in s.get("observation", "")
                   for s in steps):
                if len(steps) > 1:
                    recovery_patterns.append({
                        "question": entry.get("question", "")[:200],
                        "recovery_strategy": "多步尝试",
                        "steps": len(steps)
                    })

        recommendations = []
        if recovery_patterns:
            recommendations.append(
                f"在 {len(recovery_patterns)} 个案例中，Agent 遇到错误后通过多步尝试恢复。"
                f"在 Agent Prompt 中强调'工具失败后换策略，不要重复失败调用'。"
            )
        else:
            recommendations.append("未发现错误恢复案例。")

        return {"recovery_count": len(recovery_patterns), "recommendations": recommendations}

    def generate_prompt_suggestions(self, eval_summary: dict = None) -> dict:
        """
        综合所有分析，生成 Prompt 优化建议。

        Args:
            eval_summary: 可选，来自 compute_weaknesses() + compute_dashboard_summary() 的评估数据。
                         传入后输出会增加 structured 字段代替纯 markdown。

        Returns:
            dict: {"prompt": "markdown 字符串", "structured": [...]} 如果 eval_summary 传入
                  str: 纯 markdown（兼容旧行为）如果 eval_summary 为空
        """
        tool = self.extract_tool_usage_patterns()
        answer = self.extract_answer_patterns()
        recovery = self.extract_failure_recovery()

        lines = [
            "# Prompt 优化建议",
            f"分析时间：{len(self.logs)} 条 Agent 对话记录",
            "",
            "## 工具使用",
            *[f"- {r}" for r in tool["recommendations"]],
            "",
            "## 回答风格",
            *[f"- {r}" for r in answer["recommendations"]],
            "",
            "## 错误恢复",
            *[f"- {r}" for r in recovery["recommendations"]],
            "",
            "## 下一步",
            "1. 根据以上建议修改 backend/agent/engine.py 中的 AGENT_SYSTEM_PROMPT",
            "2. 用相同问题测试 Agent，对比回答质量",
            "3. 继续通过 Agent 管道对话积累更多日志",
        ]

        # 如果有评估数据，附加结构化建议
        if eval_summary:
            structured = []
            weaknesses = eval_summary.get("weaknesses", [])
            dashboard = eval_summary.get("dashboard", {})

            for w in weaknesses[:5]:
                structured.append({
                    "source": "evaluation",
                    "category": w.get("category", ""),
                    "severity": w.get("severity", ""),
                    "finding": w.get("description", ""),
                    "evidence": w.get("evidence", ""),
                })

            by_mode = dashboard.get("by_mode", {})
            if by_mode:
                lines.append("")
                lines.append("## 评估数据增强")
                for mode, stats in by_mode.items():
                    lines.append(f"- {mode}: 成功率 {stats.get('success_rate', 0):.0%}, 任务数 {stats.get('count', 0)}")

            return {"prompt": "\n".join(lines), "structured": structured}

        return "\n".join(lines)

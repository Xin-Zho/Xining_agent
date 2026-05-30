"""
Plan-and-Solve Agent — 先规划再执行

流程：
  用户提问 → 规划阶段（列出步骤）→ 逐步执行 → 汇总回答
"""
from agent_framework.core.prompt import PLAN_SOLVE_PROMPT
from agent_framework.tools.registry import ToolRegistry


class PlanAndSolveAgent:
    """
    Plan-and-Solve：对复杂任务先列计划，再逐步执行。

    用法：
        agent = PlanAndSolveAgent(llm_client, tool_registry)
        result = agent.run("帮我搭建一个 Flask 项目并写一个 API")
    """

    def __init__(self, llm_client, tool_registry: ToolRegistry = None, max_steps: int = 5):
        self.client = llm_client
        self.registry = tool_registry or ToolRegistry()
        self.max_steps = max_steps

    def run(self, user_message: str) -> dict:
        """
        执行 Plan-and-Solve。

        返回：{"answer": str, "plan": list, "steps": list}
        """
        tools_desc = self._format_tools()
        sys_prompt = PLAN_SOLVE_PROMPT.format(tools_description=tools_desc)

        # ========== 阶段 1：规划 ==========
        plan_messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"请为以下任务制定计划（只列步骤，不要执行）：\n\n{user_message}"}
        ]

        tools_def = self.registry.get_definitions() if self.registry.list_tools() else None

        plan_response = self.client.client.chat.completions.create(
            model=self.client.model_id,
            messages=plan_messages
        )
        plan_text = plan_response.choices[0].message.content or ""

        # 简单解析计划步骤
        import re
        steps_match = re.findall(r'步骤\s*[一二三四五1-5]\s*[：:]\s*(.+)', plan_text)
        if not steps_match:
            steps_match = re.findall(r'(\d+)[\.、]\s*(.+)', plan_text)
            steps_match = [s[1] for s in steps_match]
        if not steps_match:
            # 没有明确步骤格式，整段作计划
            steps_match = [plan_text]

        plan = steps_match[:self.max_steps]

        # ========== 阶段 2：逐步执行 ==========
        results = []
        context = f"任务：{user_message}\n\n计划：\n" + "\n".join(
            f"{i}. {s}" for i, s in enumerate(plan, 1)
        )

        for i, step in enumerate(plan, 1):
            step_messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"{context}\n\n请执行第 {i} 步：{step}"}
            ]

            # 每步用一次 LLM 调用（可能带工具调用）
            step_response = self.client.client.chat.completions.create(
                model=self.client.model_id,
                messages=step_messages,
                tools=tools_def
            )

            msg = step_response.choices[0].message
            step_result = msg.content or ""

            # 如果这步调了工具，执行并获取最终回复
            if msg.tool_calls:
                import json
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    obs = self.registry.execute(tc.function.name, args)
                    step_messages.append({"role": "tool", "tool_call_id": tc.id, "content": obs})

                final = self.client.chat(step_messages)
                step_result = final

            results.append({"step": i, "description": step, "result": step_result[:2000]})

        # ========== 阶段 3：汇总 ==========
        summary_messages = [
            {"role": "system", "content": "你是一个善于总结的助手。"},
            {"role": "user", "content": f"任务：{user_message}\n\n各步骤结果：\n" + "\n".join(
                f"步骤 {r['step']}: {r['result'][:500]}" for r in results
            ) + "\n\n请汇总成最终回答。"}
        ]
        answer = self.client.chat(summary_messages)

        return {
            "answer": answer,
            "plan": plan,
            "steps": results
        }

    def _format_tools(self) -> str:
        if not self.registry.list_tools():
            return "（暂无可用工具）"
        lines = []
        for name in self.registry.list_tools():
            tool = self.registry._tools[name]["definition"]["function"]
            params = tool["parameters"]["properties"]
            param_str = ", ".join(f"{k}: {v.get('type','?')}" for k, v in params.items())
            lines.append(f"- **{name}**({param_str}): {tool['description']}")
        return "\n".join(lines)

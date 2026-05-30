"""
ReAct Agent — 思考(Thought) → 行动(Action) → 观察(Observation) 循环

核心概念：
  Agent 不是"一次 LLM 调用"，而是一个 while 循环。
  每次循环：模型决定用哪个工具 → 执行 → 看到结果 → 再决定 → 直到能回答。
"""
import json
from agent_framework.core.prompt import REACT_SYSTEM_PROMPT
from agent_framework.tools.registry import ToolRegistry


class ReactAgent:
    """
    ReAct Agent：自主决定何时使用工具。

    用法：
        agent = ReactAgent(llm_client, tool_registry)
        result = agent.run("帮我看看当前目录有什么文件")
        # result = {"answer": "...", "steps": [...], "turns": 3}
    """

    def __init__(self, llm_client, tool_registry: ToolRegistry, max_turns: int = 10):
        self.client = llm_client
        self.registry = tool_registry
        self.max_turns = max_turns

    def run(self, user_message: str, system_prompt: str = None) -> dict:
        """
        运行 ReAct 循环。

        user_message: 用户问题
        system_prompt: 自定义 System Prompt（可选，不传用默认）
        返回：{"answer": str, "steps": list, "turns": int}
        """
        tools_desc = self._format_tools()
        sys_prompt = (system_prompt or REACT_SYSTEM_PROMPT).format(
            tools_description=tools_desc
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_message}
        ]

        tools_def = self.registry.get_definitions() if self.registry.list_tools() else None
        steps = []

        for turn in range(1, self.max_turns + 1):
            # 调 LLM
            response = self.client.client.chat.completions.create(
                model=self.client.model_id,
                messages=messages,
                tools=tools_def
            )

            choice = response.choices[0]
            msg = choice.message

            # 如果模型要调工具
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                    # 执行工具
                    observation = self.registry.execute(tool_name, arguments)

                    step = {
                        "turn": turn,
                        "thought": f"调用工具 {tool_name}",
                        "action": f"{tool_name}({json.dumps(arguments, ensure_ascii=False)})",
                        "observation": observation[:1000]  # 展示时截断
                    }
                    steps.append(step)

                    # 把工具调用和结果加入消息
                    messages.append({
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [{
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": observation
                    })
            else:
                # 模型给出最终回答
                answer = msg.content or ""
                return {
                    "answer": answer,
                    "steps": steps,
                    "turns": turn,
                }

        # 达到最大轮次，强制总结
        messages.append({
            "role": "user",
            "content": "你已经用完了所有轮次。请基于已有信息给出最终回答。"
        })
        final = self.client.chat(messages)
        return {
            "answer": final,
            "steps": steps,
            "turns": self.max_turns,
            "truncated": True
        }

    def _format_tools(self) -> str:
        """格式化工具列表为可读文本（给 System Prompt 用）"""
        if not self.registry.list_tools():
            return "（暂无可用工具）"
        lines = []
        for name in self.registry.list_tools():
            tool = self.registry._tools[name]["definition"]["function"]
            params = tool["parameters"]["properties"]
            param_str = ", ".join(f"{k}: {v.get('type','?')}" for k, v in params.items())
            lines.append(f"- **{name}**({param_str}): {tool['description']}")
        return "\n".join(lines)

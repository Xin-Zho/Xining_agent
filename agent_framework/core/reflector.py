"""
Reflection Agent — 执行 → 自我评审 → 改进

流程：
  执行任务 → LLM 评审回答质量 → 不够好就带着反馈重新执行 → 最多 N 次
"""
from agent_framework.core.prompt import REFLECTION_PROMPT
from agent_framework.tools.registry import ToolRegistry


class ReflectionAgent:
    """
    Reflection：对回答自我评审、迭代改进。

    用法：
        agent = ReflectionAgent(llm_client, tool_registry)
        result = agent.run("帮我写一个排序算法")
        # result = {"answer": "...", "reflections": [...], "iterations": 2}
    """

    def __init__(self, llm_client, tool_registry: ToolRegistry = None, max_iterations: int = 3):
        self.client = llm_client
        self.registry = tool_registry or ToolRegistry()
        self.max_iterations = max_iterations

    def run(self, user_message: str, system_prompt: str = None) -> dict:
        """
        执行-反思循环。

        返回：{"answer": str, "reflections": list, "iterations": int}
        """
        reflections = []
        current_answer = ""
        feedback = ""

        for i in range(self.max_iterations + 1):
            # 构建消息
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            if i == 0:
                # 首次执行
                messages.append({"role": "user", "content": user_message})
            else:
                # 带着反思重新执行
                messages.append({
                    "role": "user",
                    "content": f"原始问题：{user_message}\n\n"
                              f"你之前的回答：{current_answer}\n\n"
                              f"评审反馈：{feedback}\n\n"
                              f"请根据反馈重新回答，确保修正所有问题。"
                })

            # 调 LLM
            tools_def = self.registry.get_definitions() if self.registry.list_tools() else None
            response = self.client.client.chat.completions.create(
                model=self.client.model_id,
                messages=messages,
                tools=tools_def
            )
            current_answer = response.choices[0].message.content or ""

            if i >= self.max_iterations:
                break

            # 反思阶段
            reflect_prompt = REFLECTION_PROMPT.format(
                user_question=user_message,
                answer=current_answer
            )
            reflect_resp = self.client.chat([{"role": "user", "content": reflect_prompt}])

            if "通过" in reflect_resp and len(reflect_resp) < 10:
                # 评审通过
                reflections.append(f"第 {i+1} 次：通过")
                break

            feedback = reflect_resp
            reflections.append(f"第 {i+1} 次：\n{feedback[:500]}")

        return {
            "answer": current_answer,
            "reflections": reflections,
            "iterations": len(reflections)
        }

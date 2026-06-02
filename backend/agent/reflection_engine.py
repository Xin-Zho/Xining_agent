"""
Reflection Agent — 执行 → 自我评审 → 迭代改进

流程：
  执行任务 → LLM 评审回答质量 → 不够好就带着反馈重新执行 → 最多 N 次
"""
import asyncio
import time

from .tools import Tool
from .websocket_manager import WebSocketManager, _save_step, _update_step, _update_task
from ..llm_client import estimate_tokens

REFLECTION_SYSTEM_PROMPT = """你是一个善于自我改进的AI助手。你会：
1. 认真回答用户的问题
2. 对自己的回答进行反思
3. 如果发现不足，立即改进

请用中文回复。"""

REFLECTION_CHECK_PROMPT = """请用一句话评估以下回答：
- 如果回答准确完整，回复"通过"
- 如果有问题，用一句中文指出最关键的缺失/错误

用户问题：{user_question}
待评估回答：{answer}

评估："""


class ReflectionEngine:
    """Reflection：对回答自我评审、迭代改进"""

    def __init__(self, deepseek, tools: list[Tool], ws_manager: WebSocketManager,
                 max_iterations: int = 3):
        self.deepseek = deepseek
        self.tools = tools
        self.tool_map = {t.name: t for t in tools}
        self.ws = ws_manager
        self.max_iterations = max_iterations
        self._cancellations: set[int] = set()

    async def run(self, task_description: str, user_id: int, task_id: int):
        start_time = time.time()
        _update_task(task_id, status="executing")

        await self.ws.broadcast(task_id, "task_started", {
            "task_id": task_id,
            "title": task_description[:50],
            "mode": "reflection",
        })

        tool_schemas = [t.to_openai_schema() for t in self.tools]
        step_number = 0
        total_tokens = 0
        reflections = []
        current_answer = ""
        feedback = ""

        try:
            for i in range(self.max_iterations + 1):
                if task_id in self._cancellations:
                    _update_task(task_id, status="cancelled")
                    await self.ws.broadcast(task_id, "task_cancelled", {"task_id": task_id})
                    self._cancellations.discard(task_id)
                    return

                step_number += 1
                iteration_label = "首次执行" if i == 0 else f"第{i}次改进"
                _save_step(task_id, step_number, "thought", status="running",
                           thought=f"{iteration_label}...")

                await self.ws.broadcast(task_id, "step_start", {
                    "step_num": step_number,
                    "type": "thought",
                    "message": f"{iteration_label}...",
                })

                # 构建消息
                messages = [{"role": "system", "content": REFLECTION_SYSTEM_PROMPT}]

                if i == 0:
                    messages.append({"role": "user", "content": task_description})
                else:
                    messages.append({
                        "role": "user",
                        "content": f"原始问题：{task_description}\n\n"
                                  f"你之前的回答：{current_answer}\n\n"
                                  f"评审反馈：{feedback}\n\n"
                                  f"请根据反馈重新回答，确保修正所有问题。",
                    })

                # 调 LLM
                resp = await self._call_llm(messages, tool_schemas)
                total_tokens += resp.usage.total_tokens if resp.usage else 0
                current_answer = resp.choices[0].message.content or ""

                _update_step(task_id, step_number, "completed",
                             tool_result={"iteration": i + 1, "answer": current_answer[:1000]},
                             duration_ms=0)

                await self.ws.broadcast(task_id, "step_complete", {
                    "step_num": step_number,
                    "type": "thought",
                    "content": current_answer[:1000],
                })

                if i >= self.max_iterations:
                    break

                # 反思阶段
                step_number += 1
                await self.ws.broadcast(task_id, "step_start", {
                    "step_num": step_number,
                    "type": "thought",
                    "message": "自我评审中...",
                })

                reflect_prompt = REFLECTION_CHECK_PROMPT.format(
                    user_question=task_description,
                    answer=current_answer,
                )
                reflect_resp = await self._call_llm(
                    [{"role": "user", "content": reflect_prompt}]
                )
                total_tokens += reflect_resp.usage.total_tokens if reflect_resp.usage else 0
                eval_text = reflect_resp.choices[0].message.content.strip()

                if "通过" in eval_text and len(eval_text) < 10:
                    reflections.append(f"第{i + 1}次：通过 ✓")
                    _update_step(task_id, step_number, "completed",
                                 tool_result={"reflection": "通过"}, duration_ms=0)
                    await self.ws.broadcast(task_id, "step_complete", {
                        "step_num": step_number,
                        "type": "thought",
                        "content": "✓ 评审通过",
                    })
                    break

                feedback = eval_text
                reflections.append(f"第{i + 1}次：{feedback[:300]}")
                _update_step(task_id, step_number, "completed",
                             tool_result={"reflection": feedback[:500]}, duration_ms=0)

                await self.ws.broadcast(task_id, "step_complete", {
                    "step_num": step_number,
                    "type": "thought",
                    "content": f"发现改进点: {feedback[:300]}",
                })

            # 完成
            duration_ms = int((time.time() - start_time) * 1000)
            _update_task(task_id, status="completed", final_answer=current_answer,
                         total_tokens=total_tokens, duration_ms=duration_ms)

            await self.ws.broadcast(task_id, "task_complete", {
                "final_answer": current_answer,
                "total_steps": step_number,
                "total_tokens": total_tokens,
                "duration_ms": duration_ms,
                "reflections": reflections,
            })

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            error_msg = f"任务执行失败: {str(e)}"
            _update_task(task_id, status="failed", final_answer=error_msg,
                         total_tokens=total_tokens, duration_ms=duration_ms)
            await self.ws.broadcast(task_id, "task_error", {
                "error": str(e),
                "last_step": step_number,
            })

    async def _call_llm(self, messages: list[dict], tools: list[dict] = None):
        kwargs = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
        }
        if tools:
            kwargs["tools"] = tools
        return await asyncio.to_thread(
            self.deepseek.chat.completions.create,
            **kwargs,
        )

    async def cancel(self, task_id: int):
        self._cancellations.add(task_id)

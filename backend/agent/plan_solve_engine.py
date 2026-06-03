"""
Plan-and-Solve Agent — 先规划再逐步执行

流程：
  用户提问 → 规划阶段（列出步骤）→ 逐步执行 → 汇总回答
"""
import asyncio
import json
import re
import time

from .tools import Tool
from .websocket_manager import WebSocketManager, _save_step, _update_step, _update_task
from ..llm_client import estimate_tokens

PLAN_SOLVE_SYSTEM_PROMPT = """You are a planning+execution agent. Plan→Execute→Synthesize in ≤5 steps. Think English, answer Chinese. Never guess—use tools. Batch calls, fail fast, cite sources. Use create_document for reports/tables. Output with Markdown tables and source URLs."""


class PlanSolveEngine:
    """Plan-and-Solve：对复杂任务先列计划，再逐步执行"""

    def __init__(self, deepseek, tools: list[Tool], ws_manager: WebSocketManager):
        self.deepseek = deepseek
        self.tools = tools
        self.tool_map = {t.name: t for t in tools}
        self.ws = ws_manager
        self._cancellations: set[int] = set()

    async def run(self, task_description: str, user_id: int, task_id: int, max_steps: int = 5):
        start_time = time.time()
        _update_task(task_id, status="planning")

        await self.ws.broadcast(task_id, "task_started", {
            "task_id": task_id,
            "title": task_description[:50],
            "mode": "plan_solve",
        })

        tool_schemas = [t.to_openai_schema() for t in self.tools]
        step_number = 0
        total_tokens = 0
        messages = []  # 消息连续性

        try:
            # ── 阶段 1：制定计划 ──────────────────────────
            step_number += 1
            _save_step(task_id, step_number, "plan", status="running",
                       thought="Analyzing task, creating execution plan...")

            await self.ws.broadcast(task_id, "step_start", {
                "step_num": step_number,
                "type": "plan",
                "message": "Analyzing task & creating plan...",
            })

            plan_messages = [
                {"role": "system", "content": PLAN_SOLVE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Create a plan (list steps only, do NOT execute):\n\n{task_description}"},
            ]

            plan_resp = await self._call_llm(plan_messages)
            total_tokens += plan_resp.usage.total_tokens if plan_resp.usage else 0
            plan_text = plan_resp.choices[0].message.content or ""

            steps_match = re.findall(r'(\d+)[\.、)]\s*(.+)', plan_text)
            steps_match = [s[1] for s in steps_match if len(s[1]) > 5]
            if not steps_match:
                steps_match = re.findall(r'步骤\s*[一二三四五1-5]\s*[：:]\s*(.+)', plan_text)
            if not steps_match:
                steps_match = [plan_text[:200]]

            plan = steps_match[:max_steps]

            _update_step(task_id, step_number, "completed",
                         tool_result={"plan": plan, "raw": plan_text[:1000]}, duration_ms=0)

            await self.ws.broadcast(task_id, "step_complete", {
                "step_num": step_number,
                "type": "plan",
                "content": plan_text[:1000],
                "plan": plan,
            })

            # ── 阶段 2：逐步执行（带消息连续性）───────
            results = []
            context = f"Task: {task_description}\n\nPlan:\n" + "\n".join(
                f"{i}. {s}" for i, s in enumerate(plan, 1)
            )

            for i, step_desc in enumerate(plan, 1):
                if task_id in self._cancellations:
                    _update_task(task_id, status="cancelled")
                    await self.ws.broadcast(task_id, "task_cancelled", {"task_id": task_id})
                    self._cancellations.discard(task_id)
                    return

                step_number += 1
                _save_step(task_id, step_number, "thought", status="running",
                           thought=f"Executing step {i}: {step_desc}")

                await self.ws.broadcast(task_id, "step_start", {
                    "step_num": step_number,
                    "type": "thought",
                    "message": f"Step {i}/{len(plan)}: {step_desc[:100]}",
                })

                # 构建带历史的步骤消息
                step_messages = [
                    {"role": "system", "content": PLAN_SOLVE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"{context}\n\nExecute step {i}: {step_desc}\nBATCH all tool calls for this step in ONE response."},
                ]

                step_resp = await self._call_llm(step_messages, tool_schemas)
                total_tokens += step_resp.usage.total_tokens if step_resp.usage else 0
                msg = step_resp.choices[0].message
                step_result = msg.content or ""

                # 并行执行所有工具调用
                if msg.tool_calls:
                    call_tasks = []
                    for tc in msg.tool_calls:
                        tool = self.tool_map.get(tc.function.name)
                        if tool:
                            try:
                                args = json.loads(tc.function.arguments)
                            except json.JSONDecodeError:
                                args = {}
                            call_tasks.append((tc, tool, args))

                    # 记录工具调用
                    step_messages.append({
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                            for tc, _, _ in call_tasks
                        ]
                    })

                    # 并行执行
                    async def exec_one(tc, tool, args):
                        try:
                            result = await tool.handler(**args)
                            return tc.id, result
                        except Exception as e:
                            return tc.id, f"Error: {e}"

                    parallel_tasks = [exec_one(tc, tool, args) for tc, tool, args in call_tasks]
                    completed = await asyncio.gather(*parallel_tasks)

                    for tc_id, obs in completed:
                        obs_str = json.dumps(obs, ensure_ascii=False) if not isinstance(obs, str) else obs
                        step_messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": obs_str[:4000],
                        })

                    final = await self._call_llm(step_messages)
                    total_tokens += final.usage.total_tokens if final.usage else 0
                    step_result = final.choices[0].message.content or step_result

                results.append({"step": i, "description": step_desc, "result": step_result[:2000]})

                _update_step(task_id, step_number, "completed",
                             tool_result={"step_result": step_result[:2000]}, duration_ms=0)

                await self.ws.broadcast(task_id, "step_complete", {
                    "step_num": step_number,
                    "type": "thought",
                    "content": step_result[:1000],
                })

            # ── 阶段 3：汇总 ──────────────────────────────
            step_number += 1
            await self.ws.broadcast(task_id, "step_start", {
                "step_num": step_number,
                "type": "thought",
                "message": "Synthesizing final answer...",
            })

            summary_messages = [
                {"role": "system", "content": "You are a synthesis expert. Combine all step results into a polished final answer. Use tables for data. Include sources. Speak Chinese."},
                {"role": "user", "content": f"Task: {task_description}\n\nStep Results:\n" +
                    "\n".join(f"Step {r['step']}: {r['result'][:500]}" for r in results) +
                    "\n\nSynthesize a comprehensive final answer with tables, analysis, and sources."},
            ]
            summary_resp = await self._call_llm(summary_messages)
            total_tokens += summary_resp.usage.total_tokens if summary_resp.usage else 0
            final_answer = summary_resp.choices[0].message.content or "Task completed."

            duration_ms = int((time.time() - start_time) * 1000)
            _update_task(task_id, status="completed", final_answer=final_answer,
                         total_tokens=total_tokens, duration_ms=duration_ms,
                         plan_json=json.dumps({"plan": plan, "results": results}, ensure_ascii=False))

            await self.ws.broadcast(task_id, "task_complete", {
                "final_answer": final_answer,
                "total_steps": step_number,
                "total_tokens": total_tokens,
                "duration_ms": duration_ms,
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
        cached_messages = list(messages)
        if tools:
            tool_desc = "Tools: " + ", ".join(
                t["function"]["name"] + "(" + t["function"]["description"][:50] + ")"
                for t in tools
            )
            cached_messages.insert(1, {"role": "system", "content": tool_desc})
        kwargs = {
            "model": "deepseek-chat",
            "messages": cached_messages,
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

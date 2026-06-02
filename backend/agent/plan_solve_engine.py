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

PLAN_SOLVE_SYSTEM_PROMPT = """你是一个善于规划的执行助手。

## 工作流程

**第 1 步：输出计划**
分析任务，列出 2~5 个步骤。格式：
【计划】
1. xxx
2. xxx

**第 2 步：逐步执行**
每步执行后汇报结果。某步失败则调整计划。

**第 3 步：汇总**
整合结果给出最终回答。

## 规则
- 步骤 ≤5 个
- 某步失败不要全放弃
- 使用可用工具完成任务
- 请用中文回复"""


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

        try:
            # ── 阶段 1：制定计划 ──────────────────────────
            step_number += 1
            _save_step(task_id, step_number, "plan", status="running",
                       thought="制定执行计划...")

            await self.ws.broadcast(task_id, "step_start", {
                "step_num": step_number,
                "type": "plan",
                "message": "制定执行计划...",
            })

            plan_messages = [
                {"role": "system", "content": PLAN_SOLVE_SYSTEM_PROMPT},
                {"role": "user", "content": f"请为以下任务制定计划（只列步骤，不要执行）：\n\n{task_description}"},
            ]

            plan_resp = await self._call_llm(plan_messages)
            total_tokens += plan_resp.usage.total_tokens if plan_resp.usage else 0
            plan_text = plan_resp.choices[0].message.content or ""

            # 解析计划步骤
            steps_match = re.findall(r'步骤\s*[一二三四五1-5]\s*[：:]\s*(.+)', plan_text)
            if not steps_match:
                steps_match = re.findall(r'(\d+)[\.、]\s*(.+)', plan_text)
                steps_match = [s[1] for s in steps_match]
            if not steps_match:
                steps_match = [plan_text]

            plan = steps_match[:max_steps]

            _update_step(task_id, step_number, "completed",
                         tool_result={"plan": plan, "raw": plan_text[:1000]}, duration_ms=0)

            await self.ws.broadcast(task_id, "step_complete", {
                "step_num": step_number,
                "type": "plan",
                "content": plan_text[:1000],
                "plan": plan,
            })

            # ── 阶段 2：逐步执行 ──────────────────────────
            results = []
            context = f"任务：{task_description}\n\n计划：\n" + "\n".join(
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
                           thought=f"执行第{i}步: {step_desc}")

                await self.ws.broadcast(task_id, "step_start", {
                    "step_num": step_number,
                    "type": "thought",
                    "message": f"执行第{i}步: {step_desc}",
                })

                step_messages = [
                    {"role": "system", "content": PLAN_SOLVE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"{context}\n\n请执行第 {i} 步：{step_desc}"},
                ]

                step_resp = await self._call_llm(step_messages, tool_schemas)
                total_tokens += step_resp.usage.total_tokens if step_resp.usage else 0
                msg = step_resp.choices[0].message
                step_result = msg.content or ""

                # 如果有工具调用，执行它
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool = self.tool_map.get(tc.function.name)
                        if tool:
                            try:
                                args = json.loads(tc.function.arguments)
                            except json.JSONDecodeError:
                                args = {}
                            obs = await tool.handler(**args)
                            step_messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps(obs, ensure_ascii=False),
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
                "message": "汇总最终结果...",
            })

            summary_messages = [
                {"role": "system", "content": "你是一个善于总结的助手。"},
                {"role": "user", "content": f"任务：{task_description}\n\n各步骤结果：\n" +
                    "\n".join(f"步骤 {r['step']}: {r['result'][:500]}" for r in results) +
                    "\n\n请汇总成最终回答。"},
            ]
            summary_resp = await self._call_llm(summary_messages)
            total_tokens += summary_resp.usage.total_tokens if summary_resp.usage else 0
            final_answer = summary_resp.choices[0].message.content or "任务已完成。"

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

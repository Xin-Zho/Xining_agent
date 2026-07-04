"""
Plan-and-Solve Agent — 先规划再逐步执行

流程：
  用户提问 → 规划阶段（列出步骤）→ 逐步执行 → 汇总回答
"""
import asyncio
import json
import re
import time

from ..protocols.mcp.tool_adapter import ToolProtocol
from .websocket_manager import WebSocketManager, _save_step, _update_step, _update_task
from .review_prompt import REVIEW_SYSTEM_PROMPT
from ..llm_client import estimate_tokens
from ..evaluation.hooks import on_task_completed

PLAN_SOLVE_SYSTEM_PROMPT = """You are a planning+execution agent for scientific computing tasks.

## Clarification Rule (HIGHEST PRIORITY)

If the user's task is AMBIGUOUS or under-specified, do NOT guess. Instead, ask 2-3 specific clarifying questions BEFORE any planning or execution.

Examples of when to ask:
- "帮我计算反应速率" → Which reaction? What temperature/pressure? Is it elementary or complex?
- "分析这个分子" → Which molecule? What properties (HOMO/LUMO, dipole, vibrational modes)?
- "优化这个计算" → Which method/basis set? What property? Accuracy or speed priority?

Format your clarifying questions as a numbered list. After the user answers, proceed with the clarified task.

## Workflow

1. Is the task ambiguous? → Ask clarifying questions, STOP
2. Is the task simple? → Answer directly
3. Is the task complex? → Plan(≤5 steps) → Execute(batched tools) → Synthesize

## Plan quality rules
- Each step must name the specific tool(s) it will use
- Each step must state its measurable output
- Dependencies between steps must be explicit
- Max 5 steps; if more needed, the task should be split

Think English, answer Chinese. Use tools, cite sources."""


class PlanSolveEngine:
    """Plan-and-Solve：对复杂任务先列计划，再逐步执行"""

    def __init__(self, deepseek, tools: list[ToolProtocol], ws_manager: WebSocketManager):
        self.deepseek = deepseek
        self.tools = tools
        self.tool_map = {t.name: t for t in tools}
        self.ws = ws_manager
        self._cancellations: set[int] = set()
        self._tool_desc = "Tools: " + ", ".join(
            t.to_openai_schema()["function"]["name"] + "("
            + t.to_openai_schema()["function"]["description"][:50] + ")"
            for t in tools
        ) if tools else ""

    async def run(self, task_description: str, user_id: int, task_id: int, max_steps: int = 5):
        start_time = time.time()
        _update_task(task_id, status="planning")

        # 注入记忆上下文
        from ..memory.manager import MemoryManager
        mem_mgr = MemoryManager(user_id)
        self._mem_mgr = mem_mgr

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
            # ── 阶段 0：复杂度判断 ──────────────────────────
            # 简单问题直接回答，不需要规划
            check_messages = [
                {"role": "system", "content": "Is this task complex (requires multiple steps/tools)? Answer ONLY 'simple' or 'complex'."},
                {"role": "user", "content": task_description},
            ]
            check_resp = await self._call_llm(check_messages, None)
            complexity = (check_resp.choices[0].message.content or "").strip().lower()

            if "simple" in complexity:
                # 简单问题，直接调用 Agent 逻辑
                step_number += 1
                _save_step(task_id, step_number, "thought", status="running",
                           thought="Simple question, answering directly...")
                direct_messages = [
                    {"role": "system", "content": "Answer concisely in Chinese. Think English but respond in Chinese. Use tools if needed. Never guess."},
                    {"role": "user", "content": task_description},
                ]
                direct_resp = await self._call_llm(direct_messages, tool_schemas)
                total_tokens += direct_resp.usage.total_tokens if direct_resp.usage else 0
                msg = direct_resp.choices[0].message
                final_answer = msg.content or ""

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
                            obs_str = json.dumps(obs, ensure_ascii=False)[:4000]
                            direct_messages.append({"role": "assistant", "content": msg.content or "",
                                "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}]})
                            direct_messages.append({"role": "tool", "tool_call_id": tc.id, "content": obs_str})
                    final_resp = await self._call_llm(direct_messages, None)
                    total_tokens += final_resp.usage.total_tokens if final_resp.usage else 0
                    final_answer = final_resp.choices[0].message.content or final_answer

                _update_step(task_id, step_number, "completed", duration_ms=0)

                duration_ms = int((time.time() - start_time) * 1000)
                _update_task(task_id, status="completed", final_answer=final_answer,
                             total_tokens=total_tokens, duration_ms=duration_ms)
                await self.ws.broadcast(task_id, "task_complete", {
                    "final_answer": final_answer, "total_steps": step_number,
                    "total_tokens": total_tokens, "duration_ms": duration_ms,
                })
                return

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

            # ── 阶段 1.5：副 Agent 审校方案 ──────────────
            step_number += 1
            _save_step(task_id, step_number, "review", status="running",
                       thought="副 Agent 审校方案中...")

            await self.ws.broadcast(task_id, "step_start", {
                "step_num": step_number,
                "type": "review",
                "message": "Reviewing plan for gaps...",
            })

            plan_for_review = json.dumps({
                "plan": plan,
            }, ensure_ascii=False)

            review_messages = [
                {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": f"Review this plan:\n\n{plan_for_review}"},
            ]

            review_resp = await self._call_llm(review_messages)
            total_tokens += review_resp.usage.total_tokens if review_resp.usage else 0
            review_text = review_resp.choices[0].message.content or "{}"

            try:
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', review_text)
                if json_match:
                    review_text = json_match.group(1).strip()
                review_data = json.loads(review_text)
            except json.JSONDecodeError:
                review_data = {
                    "missing_steps": [],
                    "flawed_logic": [],
                    "boundary_gaps": [],
                    "suggestions": [],
                    "_parse_error": review_text[:200]
                }

            has_findings = any(
                review_data.get(k)
                for k in ["missing_steps", "flawed_logic", "boundary_gaps", "suggestions"]
            )

            _update_step(task_id, step_number, "completed",
                         tool_result={"review": review_data}, duration_ms=0)

            finding_count = sum(len(review_data.get(k, [])) for k in ['missing_steps', 'flawed_logic', 'boundary_gaps', 'suggestions'])

            await self.ws.broadcast(task_id, "step_complete", {
                "step_num": step_number,
                "type": "review",
                "content": "方案审校完成" if not has_findings else f"发现 {finding_count} 条改进建议",
                "review": review_data,
            })

            # ── 阶段 2：逐步执行（带消息连续性）───────
            results = []
            context = f"Task: {task_description}\n\nPlan:\n" + "\n".join(
                f"{i}. {s}" for i, s in enumerate(plan, 1)
            )

            # 将审校结果注入执行上下文
            if has_findings:
                review_summary = f"""\n\n[方案审校结果]
副 Agent 对方案进行了审校，发现以下改进点：

遗漏步骤：{json.dumps(review_data.get('missing_steps', []), ensure_ascii=False, indent=2)}
逻辑缺陷：{json.dumps(review_data.get('flawed_logic', []), ensure_ascii=False, indent=2)}
边界缺口：{json.dumps(review_data.get('boundary_gaps', []), ensure_ascii=False, indent=2)}
优化建议：{json.dumps(review_data.get('suggestions', []), ensure_ascii=False, indent=2)}

请逐条判断是否采纳，修正方案后继续执行。"""
                context += review_summary

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

            # 保存任务总结到情景记忆
            try:
                await self._mem_mgr.add(
                    content=f"Plan-Solve任务: {task_description[:80]}\n结论: {final_answer[:300]}",
                    memory_type="episodic",
                    importance=0.5,
                    metadata={"task_id": task_id, "mode": "plan_solve"},
                )
            except Exception:
                pass

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
            asyncio.create_task(on_task_completed(task_id))

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            error_msg = f"任务执行失败: {str(e)}"
            _update_task(task_id, status="failed", final_answer=error_msg,
                         total_tokens=total_tokens, duration_ms=duration_ms)
            await self.ws.broadcast(task_id, "task_error", {
                "error": str(e),
                "last_step": step_number,
            })
            asyncio.create_task(on_task_completed(task_id))

    async def _call_llm(self, messages: list[dict], tools: list[dict] = None):
        cached_messages = list(messages)
        if tools and self._tool_desc:
            if len(cached_messages) < 2 or cached_messages[1].get("content", "")[:6] != "Tools:":
                cached_messages.insert(1, {"role": "system", "content": self._tool_desc})
        kwargs = {
            "model": os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"),
            "messages": cached_messages,
            "temperature": 0.7,
            "max_tokens": 4096,
        }
        if tools:
            kwargs["tools"] = tools
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self.deepseek.chat.completions.create,
                    **kwargs,
                ),
                timeout=300,
            )
        except asyncio.TimeoutError:
            raise Exception("LLM 调用超时（300秒），请稍后重试或简化问题。")

    async def cancel(self, task_id: int):
        self._cancellations.add(task_id)

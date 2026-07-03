"""
增强版 ReAct Agent — 融合两个项目的精华

从 agent_learning 融入:
  - Token 预算管理 (90K 上限 + 紧急压缩)
  - 工具结果智能截断 (保留头 60% + 尾 20%)
  - 并行工具执行 (asyncio.gather)
  - 重复调用检测 (同一参数 ≥2 次阻止)
  - 自动反思 (最终回答前自检)
  - 及早退出 (够用了就答)

从 codex_test 保留:
  - 异步架构 (asyncio)
  - WebSocket 实时推送
  - 任务取消机制
  - SQLite 持久化
"""
import asyncio
import json
import logging
import os
import time

logger = logging.getLogger(__name__)

from ..protocols.mcp.tool_adapter import ToolProtocol
from .websocket_manager import WebSocketManager, _save_step, _update_step, _update_task
from .intervention import InterventionHandler
from ..llm_client import estimate_tokens
from ..evaluation.hooks import on_task_completed

# Token 预算
TOKEN_BUDGET = 90_000
MAX_OBS_TOKENS = 2000
LLM_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b")

AGENT_SYSTEM_PROMPT = """You are a scientific computing agent. You answer chemistry and physics questions using computational tools — NOT web search first. Your answers MUST be accurate, sourced, and formatted for students.

## Tool selection priority
1. COMPUTE first:
   - calculator: symbolic math — diff, integrate, solve, limit, series, unit conversion
   - balance_equation: balance chemical equations
   - element_lookup: element properties, molar mass, electronegativity
   - solution_chem: pH, buffers, titrations
   - kinetics: reaction rates, half-life, Arrhenius
   - electrochem: Nernst equation, cell potentials
   - mechanics: kinematics, forces, energy
   - electromagnetism: Coulomb, Biot-Savart
   - quantum: infinite well, harmonic oscillator, H-atom (analytically solvable ONLY)
   - optics: lens equation, interference
   - thermodynamics: Carnot cycle, ideal gas
   - error_propagation: uncertainty synthesis

2. VERIFY second:
   - verify_claim: check dimensional correctness, order-of-magnitude, back-substitution
   - back_substitute: plug solutions back into original equations

3. SEARCH third (only when computation tools cannot answer):
   - web_search: current events, definitions, real-world data not in science_kb
   - rag_search: search the scientific knowledge base for constants, formulas, theories

4. OUTPUT last:
   - create_document: generate .md/.csv reports
   - create_excel: generate .xlsx data tables

## Accuracy rules (CRITICAL)
- Every calculation result MUST be verified when possible. Use verify_claim or back_substitute.
- Every factual claim (constant value, formula, theory) MUST cite its source. Use rag_search to find the citation.
- If a quantum system is NOT analytically solvable (e.g., helium atom), say "not analytically solvable — suggest numerical methods (HF/DFT)" INSTEAD of guessing.
- When two methods disagree, report both results and flag the discrepancy. Do NOT silently pick one.
- Use LaTeX for all math: $inline$ for short expressions, $$block$$ for equations.

## When to stop
Stop computing when:
- You have a verified numeric answer with correct units
- One credible source (peer-reviewed, NIST, textbook) directly answers the question
- The same computation with the same inputs was already done this task

Do NOT keep computing because:
- "Let me double-check with another method" — only if the first result is suspicious (wrong units, wrong order-of-magnitude)
- "Let me search for context" — the knowledge base has all standard constants

## Output format
- Math: ALWAYS use LaTeX ($...$ inline, $$...$$ block)
- Tables: Markdown with aligned columns
- Citations: mark each factual claim with its source like [NIST WebBook] or [IUPAC Gold Book]
- Language: answer in Chinese (中文)

## Anti-patterns — NEVER
- ✗ web_search("hydrogen ground state energy") — use quantum tool instead
- ✗ web_search("pH of 0.1M HCl") — use solution_chem instead
- ✗ Guessing a number without computation — always compute
- ✗ "According to Wikipedia..." without a specific URL or revision date
- ✗ Silently returning a wrong number — flag uncertainty explicitly"""

REFLECTION_PROMPT = """Assess this scientific answer for accuracy. Check:
1. Are all calculations verified (back-substitution or dimensional analysis)?
2. Are all factual claims cited with a source?
3. Is the answer free of silent errors (wrong numbers, wrong units, wrong order-of-magnitude)?
4. Does the answer use LaTeX for all math expressions?

If the answer passes all checks, reply ONLY 'pass'.
If there are issues, state the most critical one in one sentence, then reply 'pass' so the task completes.
Do NOT block the answer for minor formatting issues.

用户问题：{user_question}
回答：{answer}

评估："""

MAX_ITERATIONS = 3
STEP_TIMEOUT = 120
LLM_CALL_TIMEOUT = 300


class AgentEngine:
    """增强版 ReAct Agent — Token 预算 + 并行执行 + 自动反思"""

    def __init__(self, deepseek, tools: list[ToolProtocol], ws_manager: WebSocketManager,
                 intervention: InterventionHandler = None):
        self.deepseek = deepseek
        self.tools = tools
        self.tool_map = {t.name: t for t in tools}
        self.ws = ws_manager
        self.intervention = intervention
        self._cancellations: set[int] = set()
        self._called_history: list[str] = []  # 防重复调用
        self._consecutive_failures: dict[str, int] = {}  # 连续失败计数（per tool）
        # 预计算工具参考（嵌入 system prompt 尾部，利用 prompt cache）
        _tool_lines = []
        for t in tools:
            schema = t.to_openai_schema()["function"]
            params = list(schema.get("parameters", {}).get("properties", {}).keys())
            _tool_lines.append(f"- **{schema['name']}**({', '.join(params[:3])}): {schema['description'][:80]}")
        self._tool_list = "## Tool Reference\n" + "\n".join(_tool_lines) if _tool_lines else ""

    async def run(self, task_description: str, user_id: int, task_id: int,
                  max_iterations: int = MAX_ITERATIONS):
        start_time = time.time()
        self._called_history = []
        self._consecutive_failures = {}  # 每任务重置连续失败计数
        self._current_task_id = task_id  # 供 _execute_tool 使用
        self._current_step_num = 0       # 供 _execute_tool 使用
        task_id_str = str(task_id)

        # 生命周期：注册任务
        if self.intervention:
            await self.intervention.register_task(task_id_str)

        _update_task(task_id, status="executing")

        await self.ws.broadcast(task_id, "task_started", {
            "task_id": task_id,
            "title": task_description[:50],
        })

        # 注入记忆上下文（三层记忆系统）
        from ..memory.manager import MemoryManager
        mem_mgr = MemoryManager(user_id)
        self._mem_mgr = mem_mgr
        memory_context = await mem_mgr.get_context_for_prompt(task_description[:100])
        # 拼接 system prompt：基础指令 + 工具参考 + 记忆上下文（单条消息，利于 prompt cache）
        system_prompt = AGENT_SYSTEM_PROMPT + "\n\n" + self._tool_list
        if memory_context:
            system_prompt = system_prompt + "\n\n" + memory_context

        tool_schemas = [t.to_openai_schema() for t in self.tools]

        # ── 科学计算 Agent：所有问题都走工具路径 ──
        # 科学问题明确性高，跳过复杂度判断，直接给 LLM 全套工具
        raw_question = task_description.split("## 当前任务\n")[-1] if "## 当前任务" in task_description else task_description
        is_simple = False

        print(f"[ENGINE] SCI-COMPUTE path, raw_question='{raw_question[:80]}' task_id={task_id}", flush=True)

        print(f"[ENGINE] Taking SCI-COMPUTE path for task {task_id}, system_prompt_len={len(system_prompt)}", flush=True)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请完成以下任务：\n\n{task_description}\n\n先分析任务，然后逐步执行。每个步骤都要记录。最后给出完整的总结。"},
        ]

        step_number = 0
        total_tokens = estimate_tokens(AGENT_SYSTEM_PROMPT) + estimate_tokens(task_description)

        try:
            for iteration in range(max_iterations):
                # ── 取消检查 ──────────────────────────────
                if task_id in self._cancellations:
                    _update_task(task_id, status="cancelled")
                    await self.ws.broadcast(task_id, "task_cancelled", {"task_id": task_id})
                    self._cancellations.discard(task_id)
                    return

                # ── 取消快车道（WebSocket 干预取消）─────
                if self.intervention and await self.intervention.has_cancellation(task_id_str):
                    _update_task(task_id, status="cancelled")
                    await self.ws.broadcast(task_id, "task_cancelled", {"task_id": task_id})
                    self._cancellations.discard(task_id)
                    return

                # ── Token 预算检查 ─────────────────────────
                if total_tokens > TOKEN_BUDGET:
                    messages = await self._compress_context(messages)
                    total_tokens = estimate_tokens(
                        " ".join(m.get("content", "") or "" for m in messages)
                    )

                # ── 思考步骤 ───────────────────────────────
                step_number += 1
                self._current_step_num = step_number
                _save_step(task_id, step_number, "thought", status="running",
                           thought=f"第{iteration + 1}轮思考...")

                await self.ws.broadcast(task_id, "step_start", {
                    "step_num": step_number,
                    "type": "thought",
                    "message": f"第{iteration + 1}轮推理中...",
                })

                response = await self._call_llm_streaming(messages, tool_schemas)
                total_tokens += response.usage.total_tokens if response.usage else 0

                msg = response.choices[0].message
                print(f"[ENGINE] LLM response: content_len={len(msg.content or '')} tool_calls={len(msg.tool_calls) if msg.tool_calls else 0}", flush=True)

                # 记录 LLM 思考
                if msg.content:
                    _update_step(task_id, step_number, "completed",
                                 tool_result={"thought": msg.content}, duration_ms=0)
                    await self.ws.broadcast(task_id, "step_complete", {
                        "step_num": step_number,
                        "type": "thought",
                        "content": msg.content,
                    })

                # ── 无工具调用 → 任务完成 ─────────────────
                if not msg.tool_calls:
                    final_answer = msg.content or ""
                    thought_content = msg.content or ""

                    # content 为空 → 让 LLM 基于历史总结回答
                    if not final_answer.strip():
                        messages.append({"role": "user", "content": "请基于以上所有搜索结果和对话历史，给出完整的最终回答。用中文。"})
                        summary_resp = await self._call_llm(messages, None)
                        total_tokens += summary_resp.usage.total_tokens if summary_resp.usage else 0
                        final_answer = summary_resp.choices[0].message.content or "抱歉，未能完成此任务。请重新描述您的问题。"

                    # 消化追踪
                    if self.intervention:
                        digest_events = await self.intervention.mark_digested(
                            task_id_str, thought_content
                        )
                        for event in digest_events:
                            await self.ws.broadcast(task_id, event["type"], {
                                "id": event["id"],
                                "decision": event["decision"],
                                "summary": event["summary"],
                            })

                    # 自动保存任务总结到情景记忆
                    try:
                        await self._mem_mgr.add(
                            content=f"任务: {task_description[:80]}\n结论: {final_answer[:300]}",
                            memory_type="episodic",
                            importance=0.5,
                            metadata={"task_id": task_id},
                        )
                    except Exception:
                        pass  # 记忆保存失败不阻塞

                    # 自动反思：只润色文字，不搜新数据，结果投评估
                    reflection = await self._reflect(task_description, final_answer)
                    if reflection and "pass" not in reflection.lower() and len(reflection) > 5:
                        # 反思发现问题 → 仅基于已有信息修正文字，不给工具
                        polish_messages = [
                            {"role": "system", "content": "你是一个文字润色专家。根据评审意见修正以下回答，只修正文字表达，不补充新信息。用中文。"},
                            {"role": "user", "content": f"评审意见：{reflection}\n\n原始回答：{final_answer}\n\n修正后的回答："},
                        ]
                        try:
                            polish_resp = await self._call_llm(polish_messages, None)
                            total_tokens += polish_resp.usage.total_tokens if polish_resp.usage else 0
                            final_answer = polish_resp.choices[0].message.content or final_answer
                        except Exception:
                            pass  # 润色失败不阻塞
                    # 反思结果存入 DB（供评估系统聚合分析）
                    try:
                        from ..evaluation.db import create_task_evaluation
                        reflection_quality = {"reflection": reflection, "final_answer_len": len(final_answer)}
                        create_task_evaluation(task_id, reflection_quality, mode="auto")
                    except Exception:
                        pass

                    duration_ms = int((time.time() - start_time) * 1000)
                    _update_task(task_id, status="completed", final_answer=final_answer,
                                 total_tokens=total_tokens, duration_ms=duration_ms)
                    await self.ws.broadcast(task_id, "task_complete", {
                        "final_answer": final_answer,
                        "total_steps": step_number,
                        "total_tokens": total_tokens,
                        "duration_ms": duration_ms,
                    })
                    asyncio.create_task(on_task_completed(task_id))
                    return

                # ── 解析工具调用 (重复检测) ─────────────
                call_tasks = []
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                    call_key = f"{tool_name}:{json.dumps(arguments, ensure_ascii=False)}"
                    if self._called_history.count(call_key) >= 2:
                        arguments = {"__blocked__": True, "__reason__": "重复调用已超过 2 次"}
                    else:
                        self._called_history.append(call_key)

                    call_tasks.append((tc, tool_name, arguments))

                # ── 并行调用上限：每轮最多 3 个 ──────────
                MAX_PARALLEL = 3
                overflow = call_tasks[MAX_PARALLEL:]
                call_tasks = call_tasks[:MAX_PARALLEL]
                for tc, tool_name, arguments in overflow:
                    call_tasks.append((tc, tool_name, {"__blocked__": True, "__reason__": f"本轮并行已达上限（{MAX_PARALLEL}），排队到下一轮"}))

                # ── 构建执行任务（含正确的 step_num）─────
                exec_tasks = []
                for i, (tc, tool_name, arguments) in enumerate(call_tasks):
                    sn = step_number + 1 + i
                    if not arguments.get("__blocked__"):
                        self._current_step_num = sn
                        _save_step(task_id, sn, "tool_call", status="running",
                                   tool_name=tool_name, tool_args=arguments)
                        await self.ws.broadcast(task_id, "step_start", {
                            "step_num": sn,
                            "type": "tool_call",
                            "tool_name": tool_name,
                            "args": arguments,
                        })
                    exec_tasks.append((tc, tool_name, arguments, sn))

                # ── 并行执行工具 ─────────────────────────
                async def exec_one(tc, tool_name, arguments, step_num):
                    if arguments.get("__blocked__"):
                        return tc.id, {"error": f"工具调用已被阻止：{arguments.get('__reason__', '重复调用')}"}

                    if task_id in self._cancellations:
                        return tc.id, {"error": "任务已取消"}

                    tool = self.tool_map.get(tool_name)
                    if not tool:
                        return tc.id, {"error": f"未知工具: {tool_name}"}

                    # 连续失败检查：同一工具连续失败 2 次则本轮跳过
                    if self._consecutive_failures.get(tool_name, 0) >= 2:
                        return tc.id, {"error": f"工具 '{tool_name}' 连续失败 2 次，本轮跳过，请换用其他工具"}

                    tool_start = time.time()
                    try:
                        result = await self._execute_tool_with_retry(
                            tool, arguments, tool_name, max_retries=3,
                            step_num=step_num,
                        )
                        duration = int((time.time() - tool_start) * 1000)
                        return tc.id, {"result": result, "duration_ms": duration, "tool_name": tool_name}
                    except Exception as e:
                        return tc.id, {"error": str(e), "tool_name": tool_name}

                # 并行等待所有工具完成
                results = {}
                tasks = [exec_one(tc, tn, args, sn) for tc, tn, args, sn in exec_tasks]
                completed = await asyncio.gather(*tasks)

                for tc_id, result_data in completed:
                    results[tc_id] = result_data

                # ── 收集结果 ─────────────────────────────
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                })

                for i, (tc, tool_name, arguments) in enumerate(call_tasks):
                    sn = step_number + 1 + i
                    result_data = results.get(tc.id, {"error": "工具执行无响应"})

                    if "result" in result_data:
                        raw_obs = result_data["result"]
                        # 智能截断
                        observation = self._smart_truncate(json.dumps(raw_obs, ensure_ascii=False))
                        duration = result_data.get("duration_ms", 0)
                        status = "completed"
                        # 成功：重置连续失败计数
                        self._consecutive_failures[tool_name] = 0
                    else:
                        observation = result_data.get("error", "未知错误")
                        duration = 0
                        status = "failed"
                        # 非阻塞类失败：递增连续失败计数
                        if not arguments.get("__blocked__"):
                            self._consecutive_failures[tool_name] = self._consecutive_failures.get(tool_name, 0) + 1

                    step_data = {
                        "turn": iteration + 1,
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "observation": observation[:800],
                    }

                    _update_step(task_id, sn, status,
                                 tool_result={"observation": observation[:2000],
                                              "tool_name": tool_name},
                                 duration_ms=duration,
                                 tool_args=arguments)

                    total_tokens += estimate_tokens(observation)

                    await self.ws.broadcast(task_id, "step_complete", {
                        "step_num": sn,
                        "type": "tool_call",
                        "tool_name": tool_name,
                        "result": observation[:1000],
                        "duration_ms": duration,
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": observation
                    })

                step_number += len(call_tasks)

                # ── 反馈注入点 ─────────────────────────
                if self.intervention:
                    fb_msgs, injected_ids = await self.intervention.drain(task_id_str)
                    if fb_msgs:
                        messages.extend(fb_msgs)
                        for fb_id in injected_ids:
                            await self.ws.broadcast(task_id, "intervention_applied", {
                                "id": fb_id,
                                "step_number": step_number,
                            })

            # ── 达到最大轮次 ────────────────────────────
            messages.append({
                "role": "user",
                "content": "工具调用轮次已用完。请基于以上所有搜索结果，立即给出完整的最终回答。用 Markdown 表格整理数据，标注来源。不要再搜了。"
            })
            final_resp = await self._call_llm(messages, None)  # 不给工具，强制总结
            final_answer = final_resp.choices[0].message.content or "抱歉，未能完成此任务。请重新描述您的问题。"
            total_tokens += final_resp.usage.total_tokens if final_resp.usage else 0

            # 消化追踪（最大轮次路径也需要）
            if self.intervention:
                digest_events = await self.intervention.mark_digested(
                    task_id_str, final_answer
                )
                for event in digest_events:
                    await self.ws.broadcast(task_id, event["type"], {
                        "id": event["id"],
                        "decision": event["decision"],
                        "summary": event["summary"],
                    })

            if len(final_answer) > 30:
                try:
                    await self._mem_mgr.add(
                        content=f"任务(达最大轮次): {task_description[:80]}\n结论: {final_answer[:300]}",
                        memory_type="episodic",
                        importance=0.5,
                        metadata={"task_id": task_id},
                    )
                    await self._mem_mgr.consolidate()
                    await self._mem_mgr.forget()
                except Exception:
                    pass

            duration_ms = int((time.time() - start_time) * 1000)
            _update_task(task_id, status="completed", final_answer=final_answer,
                         total_tokens=total_tokens, duration_ms=duration_ms)
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

        finally:
            # 生命周期：清理干预资源
            if self.intervention:
                pending_rejections = await self.intervention.complete_task(task_id_str)
                for fb_id, reason in pending_rejections:
                    await self.ws.broadcast(task_id, "intervention_rejected", {
                        "id": fb_id, "reason": reason,
                    })

    async def _reflect(self, user_question: str, answer: str) -> str:
        """自动反思：检查回答是否足够好。返回 'pass' 或改进建议。"""
        try:
            prompt = REFLECTION_PROMPT.format(
                user_question=user_question,
                answer=answer[:2000]
            )
            resp = await self._call_llm(
                [{"role": "user", "content": prompt}],
                None,  # 反思不需要工具
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return "pass"

    async def _call_llm(self, messages: list[dict], tools: list[dict] = None):
        """统一流式调用：stream=True 收集全部 chunk 后拼成完整 response 返回。
        与 _call_llm_streaming 共享 DeepSeek prompt cache。
        """
        kwargs = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
        try:
            stream = await asyncio.wait_for(
                asyncio.to_thread(
                    self.deepseek.chat.completions.create,
                    **kwargs,
                ),
                timeout=LLM_CALL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error(f"LLM streaming timed out after {LLM_CALL_TIMEOUT}s")
            raise Exception(f"LLM 调用超时（{LLM_CALL_TIMEOUT}秒），请稍后重试或简化问题。")

        # 收集流式 chunk，拼成非流式 response 对象
        collected_content = ""
        collected_tool_calls = []
        usage = type('Usage', (), {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})()

        for chunk in stream:
            if chunk.usage:
                for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                    setattr(usage, k, getattr(usage, k) + getattr(chunk.usage, k, 0))
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                collected_content += delta.content
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    while len(collected_tool_calls) <= tc.index:
                        collected_tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    entry = collected_tool_calls[tc.index]
                    if tc.id:
                        entry["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            entry["function"]["name"] = tc.function.name
                        if tc.function.arguments:
                            entry["function"]["arguments"] += tc.function.arguments

        choice = type('Choice', (), {
            'message': type('Message', (), {
                'content': collected_content or None,
                'tool_calls': collected_tool_calls or None,
            })(),
        })()
        return type('Response', (), {
            'choices': [choice],
            'usage': usage,
        })()

    async def _call_llm_streaming(self, messages: list[dict], tools: list[dict] = None):
        """
        LLM 流式调用 → WebSocket 推送 thinking_delta。

        与 _call_llm 相同的 messages 预处理，但使用 stream=True。
        返回完整响应对象（兼容现有 tool_calls 解析逻辑）。
        """
        from openai import OpenAI

        task_id = getattr(self, '_current_task_id', 0)
        should_stream = task_id > 0

        if not should_stream:
            # 无 task_id（简单路径）→ 直接用非流式
            return await self._call_llm(messages, tools)

        # ── 流式调用 ──────────────────────────
        await self.ws.broadcast(task_id, "thinking_start", {
            "message": "Agent 正在思考...",
        })

        kwargs = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools

        # 在 executor 中运行整个流式调用（避免阻塞 event loop）
        async def _stream_and_collect():
            loop = asyncio.get_event_loop()
            content_parts = []
            tool_call_chunks: dict[int, dict] = {}
            total_tokens = 0

            def _sync_stream():
                nonlocal total_tokens
                stream = self.deepseek.chat.completions.create(**kwargs)
                for chunk in stream:
                    if hasattr(chunk, 'usage') and chunk.usage:
                        total_tokens = getattr(chunk.usage, 'total_tokens', 0)
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue
                    if delta.content:
                        content_parts.append(delta.content)
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_call_chunks:
                                tool_call_chunks[idx] = {"id": "", "function_name": "", "function_args": ""}
                            if tc_delta.id:
                                tool_call_chunks[idx]["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    tool_call_chunks[idx]["function_name"] += tc_delta.function.name
                                if tc_delta.function.arguments:
                                    tool_call_chunks[idx]["function_args"] += tc_delta.function.arguments
                return content_parts, tool_call_chunks, total_tokens

            result = await loop.run_in_executor(None, _sync_stream)
            return result

        try:
            content_parts, tool_call_chunks, total_tokens = await asyncio.wait_for(
                _stream_and_collect(), timeout=LLM_CALL_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"LLM streaming timed out after {LLM_CALL_TIMEOUT}s for task {task_id}")
            # 降级：尝试非流式调用
            try:
                tools_arg = kwargs.get("tools")
                fallback = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.deepseek.chat.completions.create,
                        model=kwargs["model"],
                        messages=kwargs["messages"],
                        temperature=kwargs["temperature"],
                        max_tokens=kwargs["max_tokens"],
                        tools=tools_arg,
                    ),
                    timeout=LLM_CALL_TIMEOUT,
                )
                msg = fallback.choices[0].message
                thinking_text = msg.content or ""
                await self.ws.broadcast(task_id, "thinking_end", {
                    "content": thinking_text[:500],
                })
                if thinking_text.strip():
                    _save_step(task_id, self._current_step_num, "thinking",
                                   status="completed", thought=thinking_text[:2000])
                # 重建 tool_calls
                tool_calls = []
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append(type('ToolCall', (), {
                            'id': tc.id,
                            'type': 'function',
                            'function': type('Function', (), {
                                'name': tc.function.name,
                                'arguments': tc.function.arguments,
                            }),
                        }))
                message = type('Message', (), {
                    'content': thinking_text,
                    'tool_calls': tool_calls if tool_calls else None,
                })()
                choice = type('Choice', (), {'message': message})()
                usage = type('Usage', (), {
                    'total_tokens': getattr(fallback.usage, 'total_tokens', 0) if fallback.usage else 0,
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                })()
                return type('Response', (), {'choices': [choice], 'usage': usage})()
            except Exception as e2:
                logger.error(f"Fallback LLM call also failed: {e2}")
                raise Exception(f"LLM 调用超时（{LLM_CALL_TIMEOUT}秒）且降级失败，请稍后重试。")

        thinking_text = "".join(content_parts)
        await self.ws.broadcast(task_id, "thinking_end", {
            "content": thinking_text[:500],
        })

        # Save thinking as a DB step so SSE poller picks it up
        if thinking_text.strip():
            _save_step(task_id, self._current_step_num, "thinking",
                       status="completed", thought=thinking_text[:2000])

        # ── 构造兼容的响应对象 ─────────────────
        content = thinking_text

        # 重建 tool_calls
        tool_calls = []
        for idx in sorted(tool_call_chunks.keys()):
            tc = tool_call_chunks[idx]
            tool_calls.append(type('ToolCall', (), {
                'id': tc["id"],
                'type': 'function',
                'function': type('Function', (), {
                    'name': tc["function_name"],
                    'arguments': tc["function_args"],
                }),
            }))

        # 构造 response-like 对象
        message = type('Message', (), {
            'content': content,
            'tool_calls': tool_calls if tool_calls else None,
        })()
        choice = type('Choice', (), {
            'message': message,
        })()
        usage = type('Usage', (), {
            'total_tokens': total_tokens,
            'prompt_tokens': 0,
            'completion_tokens': 0,
        })()
        response = type('Response', (), {
            'choices': [choice],
            'usage': usage,
        })()

        return response

    def _smart_truncate(self, text: str, max_tokens: int = MAX_OBS_TOKENS) -> str:
        """
        智能截断工具结果：
        - 短结果：原样返回
        - 长结果：保留前 60% + 后 20%（头尾关键，中间省略）
        """
        tokens = estimate_tokens(text)
        if tokens <= max_tokens:
            return text

        head_size = int(len(text) * 0.6)
        tail_size = int(len(text) * 0.2)

        head = text[:head_size]
        tail = text[-tail_size:] if tail_size > 0 else ""

        omitted = tokens - estimate_tokens(head) - estimate_tokens(tail)
        return (
            f"{head}\n\n"
            f"...（省略 {omitted} tokens 的中间内容）...\n\n"
            f"{tail}"
        )

    async def _compress_context(self, messages: list[dict]) -> list[dict]:
        """智能上下文压缩：用 LLM 将中间轮次总结为摘要，保留 system + 最近消息"""
        if len(messages) <= 4:
            return messages

        system = [m for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]

        # 保留最近 4 条（保证 tool_calls/tool 配对完整）
        keep = []
        seen_tool = False
        for m in reversed(rest):
            keep.insert(0, m)
            if m.get("role") == "tool":
                seen_tool = True
            elif m.get("role") == "assistant" and m.get("tool_calls"):
                seen_tool = False
            if len(keep) >= 4 and not seen_tool:
                break

        # 中间部分是需要压缩的
        to_compress = rest[:-len(keep)] if len(keep) < len(rest) else []
        if not to_compress:
            return messages

        # 用 LLM 生成摘要
        try:
            compress_prompt = "Summarize the key findings and actions from this conversation history in 2-3 sentences. Focus on what data was found, what tools were used, and what conclusions were reached. Reply in English, be concise:\n\n"
            for m in to_compress:
                content = str(m.get("content", ""))[:500]
                role = m.get("role", "unknown")
                compress_prompt += f"[{role}] {content}\n"

            summary_resp = await self._call_llm(
                [{"role": "user", "content": compress_prompt}],
                None  # 不需要工具
            )
            summary = summary_resp.choices[0].message.content.strip()
        except Exception:
            summary = "Previous conversation context compressed due to length."

        # 构建新消息列表：system + 摘要 + 最近消息
        compressed = system + [
            {"role": "system", "content": f"[Previous context summary]\n{summary}"}
        ] + keep

        return compressed

    async def _execute_tool(self, tool: ToolProtocol, args: dict, step_num: int = 0) -> dict:
        """执行工具调用，确认检查在主进程拦截"""
        if tool.require_confirmation:
            # 使用传入的 step_num，不用 self._current_step_num（并行调用时会错乱）
            sn = step_num or self._current_step_num
            # 更新已有步骤状态（主循环已通过 _save_step 创建），不重复插入
            _update_step(self._current_task_id, sn, "confirming",
                         tool_name=tool.name, tool_args=args)
            await self.ws.broadcast(self._current_task_id, "confirmation_required", {
                "step_num": sn,
                "tool_name": tool.name,
                "args": args,
            })
            confirmed = await self._wait_for_confirmation(
                self._current_task_id, sn,
            )
            if not confirmed:
                _update_step(self._current_task_id, sn, "skipped",
                             tool_result={"msg": "用户取消执行"}, duration_ms=0)
                await self.ws.broadcast(self._current_task_id, "step_complete", {
                    "step_num": sn, "type": "tool_call",
                    "tool_name": tool.name, "result": "⛔ 已取消",
                })
                return {"status": "rejected", "message": "User denied confirmation"}
            else:
                _update_step(self._current_task_id, sn, "running",
                             tool_args=args)

        return await tool.handler(**args)

    async def _execute_tool_with_retry(self, tool: ToolProtocol, args: dict,
                                       tool_name: str, max_retries: int = 3,
                                       step_num: int = 0) -> dict:
        """
        带指数退避的工具执行重试。

        仅对超时/网络类错误重试，不对工具逻辑错误（如文件不存在）重试。
        Backoff: 1s → 2s → 4s
        """
        task_id = getattr(self, '_current_task_id', 0)
        _sn = step_num or getattr(self, '_current_step_num', 0)

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    self._execute_tool(tool, dict(args), step_num=_sn),
                    timeout=STEP_TIMEOUT,
                )
                if attempt > 0 and task_id:
                    await self.ws.broadcast(task_id, "retry_success", {
                        "step_num": step_num,
                        "tool_name": tool_name,
                        "attempt": attempt + 1,
                        "message": f"{tool_name} 重试成功（第{attempt}次重试）",
                    })
                return result
            except asyncio.TimeoutError:
                last_error = f"工具执行超时（{STEP_TIMEOUT}秒）"
            except Exception as e:
                err_str = str(e)
                # 不重试的错误类型：工具不存在、用户拒绝、参数错误
                if any(kw in err_str.lower() for kw in
                       ["unknown tool", "用户取消", "rejected", "invalid argument",
                        "not found", "no such file", "permission denied"]):
                    raise
                last_error = f"工具执行失败：{e}"

            if attempt < max_retries:
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} for {tool_name}: {last_error}, "
                    f"waiting {wait}s"
                )
                if task_id:
                    await self.ws.broadcast(task_id, "retry_attempt", {
                        "step_num": step_num,
                        "tool_name": tool_name,
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "wait_seconds": wait,
                        "error": last_error,
                    })
                await asyncio.sleep(wait)

        raise Exception(f"{tool_name}: {last_error}（重试{max_retries}次后仍失败）")

    async def _wait_for_confirmation(self, task_id: int, step_num: int) -> bool:
        """等待用户确认工具执行，最长 60 秒"""
        key = f"{task_id}_{step_num}"
        for _ in range(120):  # 60 秒，每 0.5 秒检查
            confirm = self.ws.confirmations.get(key)
            if confirm is not None:
                # 标记已处理，避免 SSE 流竞争
                self.ws.confirmations[key] = {"approved": confirm.get("approved", False), "handled": True}
                return confirm.get("approved", False)
            await asyncio.sleep(0.5)
        return False  # 超时默认拒绝

    async def cancel(self, task_id: int):
        self._cancellations.add(task_id)

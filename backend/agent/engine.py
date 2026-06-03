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
import time

from .tools import Tool
from .websocket_manager import WebSocketManager, _save_step, _update_step, _update_task
from ..llm_client import estimate_tokens

# Token 预算
TOKEN_BUDGET = 90_000
MAX_OBS_TOKENS = 2000

AGENT_SYSTEM_PROMPT = """You are an autonomous agent with tools. NEVER fabricate data — always use tools for real-time info. Think in English, answer in Chinese.

Rules: 1) Batch date+queries together 2) Never guess 3) Parallel calls only 4) Synthesize into tables 5) Fail→retry different approach same round 6) Anticipate next question 7) Code over manual 8) English think/Chinese answer 9) Cite sources 10) Use create_document for reports/tables. For PDF files, use read_pdf to extract text.

Output: Lead with conclusion, use Markdown tables, cite sources with URLs."""

REFLECTION_PROMPT = """请用一句话评估以下回答是否准确完整。
如果回答没问题，只回复'pass'。如果有问题，指出最关键的缺失。

用户问题：{user_question}
回答：{answer}

评估："""

MAX_ITERATIONS = 10
STEP_TIMEOUT = 60


class AgentEngine:
    """增强版 ReAct Agent — Token 预算 + 并行执行 + 自动反思"""

    def __init__(self, deepseek, tools: list[Tool], ws_manager: WebSocketManager):
        self.deepseek = deepseek
        self.tools = tools
        self.tool_map = {t.name: t for t in tools}
        self.ws = ws_manager
        self._cancellations: set[int] = set()
        self._called_history: list[str] = []  # 防重复调用

    async def run(self, task_description: str, user_id: int, task_id: int,
                  max_iterations: int = MAX_ITERATIONS):
        start_time = time.time()
        self._called_history = []

        _update_task(task_id, status="executing")

        await self.ws.broadcast(task_id, "task_started", {
            "task_id": task_id,
            "title": task_description[:50],
        })

        # 注入跨会话记忆上下文
        from ..memory.long_term import LongTermMemory
        ltm = LongTermMemory(user_id)
        memory_context = ltm.get_context_for_prompt(max_items=5)
        system_prompt = AGENT_SYSTEM_PROMPT
        if memory_context:
            system_prompt = AGENT_SYSTEM_PROMPT + "\n\n" + memory_context

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请完成以下任务：\n\n{task_description}\n\n先分析任务，然后逐步执行。每个步骤都要记录。最后给出完整的总结。"},
        ]

        tool_schemas = [t.to_openai_schema() for t in self.tools]
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

                # ── Token 预算检查 ─────────────────────────
                if total_tokens > TOKEN_BUDGET:
                    messages = await self._compress_context(messages)
                    total_tokens = estimate_tokens(
                        " ".join(m.get("content", "") or "" for m in messages)
                    )

                # ── 思考步骤 ───────────────────────────────
                step_number += 1
                _save_step(task_id, step_number, "thought", status="running",
                           thought=f"第{iteration + 1}轮思考...")

                await self.ws.broadcast(task_id, "step_start", {
                    "step_num": step_number,
                    "type": "thought",
                    "message": f"第{iteration + 1}轮推理中...",
                })

                response = await self._call_llm(messages, tool_schemas)
                total_tokens += response.usage.total_tokens if response.usage else 0

                msg = response.choices[0].message

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
                    final_answer = msg.content or "任务已完成。"

                    # 自动保存关键发现到长期记忆
                    try:
                        from ..memory.long_term import LongTermMemory
                        ltm = LongTermMemory(user_id)
                        # 提取任务关键词作为记忆标题
                        mem_key = task_description[:50].replace("\n", " ").strip()
                        mem_value = final_answer[:800]
                        ltm.save(mem_key, mem_value)
                    except Exception:
                        pass  # 记忆保存失败不阻塞

                    # 自动反思
                    reflection = await self._reflect(task_description, final_answer)
                    if reflection and "pass" not in reflection.lower() and len(reflection) > 5:
                        # 反思发现问题，再试一次
                        messages.append({"role": "assistant", "content": final_answer})
                        messages.append({
                            "role": "user",
                            "content": f"评审反馈：{reflection}\n\n请根据反馈修正你的回答。"
                        })
                        step_number += 1
                        await self.ws.broadcast(task_id, "step_start", {
                            "step_num": step_number,
                            "type": "thought",
                            "message": "自我反思中，改进回答...",
                        })
                        response2 = await self._call_llm(messages, tool_schemas)
                        total_tokens += response2.usage.total_tokens if response2.usage else 0
                        final_answer = response2.choices[0].message.content or final_answer

                    duration_ms = int((time.time() - start_time) * 1000)
                    _update_task(task_id, status="completed", final_answer=final_answer,
                                 total_tokens=total_tokens, duration_ms=duration_ms)
                    await self.ws.broadcast(task_id, "task_complete", {
                        "final_answer": final_answer,
                        "total_steps": step_number,
                        "total_tokens": total_tokens,
                        "duration_ms": duration_ms,
                    })
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

                # ── 并行执行工具 ─────────────────────────
                async def exec_one(tc, tool_name, arguments):
                    if arguments.get("__blocked__"):
                        return tc.id, {"error": f"工具调用已被阻止：{arguments.get('__reason__', '重复调用')}"}

                    if task_id in self._cancellations:
                        return tc.id, {"error": "任务已取消"}

                    tool = self.tool_map.get(tool_name)
                    if not tool:
                        return tc.id, {"error": f"未知工具: {tool_name}"}

                    # WebSocket 推送工具调用开始
                    step_start_num = step_number + call_tasks.index((tc, tool_name, arguments)) + 1
                    if step_start_num <= step_number + len(call_tasks):
                        pass  # step number tracking handled below

                    tool_start = time.time()
                    try:
                        result = await asyncio.wait_for(
                            tool.handler(**arguments),
                            timeout=STEP_TIMEOUT,
                        )
                        duration = int((time.time() - tool_start) * 1000)
                        return tc.id, {"result": result, "duration_ms": duration, "tool_name": tool_name}
                    except asyncio.TimeoutError:
                        return tc.id, {"error": f"工具执行超时（{STEP_TIMEOUT}秒）", "tool_name": tool_name}
                    except Exception as e:
                        return tc.id, {"error": f"工具执行失败：{e}", "tool_name": tool_name}

                # 记录工具调用步骤（需要确认的工具先请求用户确认）
                for i, (tc, tool_name, arguments) in enumerate(call_tasks):
                    if arguments.get("__blocked__"):
                        continue
                    sn = step_number + 1 + i
                    tool = self.tool_map.get(tool_name)

                    # 检查是否需要用户确认
                    if tool and getattr(tool, 'require_confirmation', False):
                        _save_step(task_id, sn, "tool_call", status="confirming",
                                   tool_name=tool_name, tool_args=arguments)
                        await self.ws.broadcast(task_id, "confirmation_required", {
                            "step_num": sn,
                            "tool_name": tool_name,
                            "args": arguments,
                        })
                        # 等用户确认（最多 60 秒）
                        confirmed = await self._wait_for_confirmation(task_id, sn)
                        if not confirmed:
                            arguments = {"__blocked__": True, "__reason__": "用户取消了此操作"}
                            _update_step(task_id, sn, "skipped",
                                         tool_result={"msg": "用户取消执行"}, duration_ms=0)
                            await self.ws.broadcast(task_id, "step_complete", {
                                "step_num": sn, "type": "tool_call",
                                "tool_name": tool_name, "result": "⛔ 已取消",
                            })
                            continue
                        else:
                            _update_step(task_id, sn, "running",
                                         tool_args=arguments)

                    _save_step(task_id, sn, "tool_call", status="running",
                               tool_name=tool_name, tool_args=arguments)
                    await self.ws.broadcast(task_id, "step_start", {
                        "step_num": sn,
                        "type": "tool_call",
                        "tool_name": tool_name,
                        "args": arguments,
                    })

                # 并行等待所有工具完成
                results = {}
                tasks = [exec_one(tc, tn, args) for tc, tn, args in call_tasks]
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
                    else:
                        observation = result_data.get("error", "未知错误")
                        duration = 0
                        status = "failed"

                    # 失败提示
                    if any(w in observation for w in ["失败", "错误", "超时", "不支持", "安全限制", "异常", "阻止"]):
                        observation += "\n（此工具未成功，请换一个方法）"

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

            # ── 达到最大轮次 ────────────────────────────
            messages.append({
                "role": "user",
                "content": "轮次已用完。请基于以上信息，用一句话给出最终回答。"
            })
            final_resp = await self._call_llm(messages, tool_schemas)
            final_answer = final_resp.choices[0].message.content or "任务达到最大执行轮数。"
            total_tokens += final_resp.usage.total_tokens if final_resp.usage else 0

            duration_ms = int((time.time() - start_time) * 1000)
            _update_task(task_id, status="completed", final_answer=final_answer,
                         total_tokens=total_tokens, duration_ms=duration_ms)
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
        # 构建带缓存优化的消息列表：system prompt + inline tools → 稳定前缀可被 DeepSeek 缓存
        cached_messages = list(messages)
        if tools:
            tool_desc = "Available tools: " + ", ".join(
                t["function"]["name"] + "(" + t["function"]["description"][:60] + ")"
                for t in tools
            )
            # 在 system 之后插入稳定的 tools 消息（作为第二个 system 消息，缓存友好）
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
        if len(messages) <= 8:
            return messages

        system = [m for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]

        # 保留最近 6 条（保证 tool_calls/tool 配对完整）
        keep = []
        seen_tool = False
        for m in reversed(rest):
            keep.insert(0, m)
            if m.get("role") == "tool":
                seen_tool = True
            elif m.get("role") == "assistant" and m.get("tool_calls"):
                seen_tool = False
            if len(keep) >= 8 and not seen_tool:
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

    async def _wait_for_confirmation(self, task_id: int, step_num: int) -> bool:
        """等待用户确认工具执行，最长 60 秒"""
        key = f"{task_id}_{step_num}"
        for _ in range(120):  # 60 秒，每 0.5 秒检查
            confirm = self.ws.confirmations.pop(key, None)
            if confirm is not None:
                return confirm.get("approved", False)
            await asyncio.sleep(0.5)
        return False  # 超时默认拒绝

    async def cancel(self, task_id: int):
        self._cancellations.add(task_id)

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
from .intervention import InterventionHandler
from ..llm_client import estimate_tokens

# Token 预算
TOKEN_BUDGET = 90_000
MAX_OBS_TOKENS = 2000

AGENT_SYSTEM_PROMPT = """You are an autonomous agent with tools: list_downloads, execute_command, memory_search, web_search, web_fetch, stock_query, read_file, read_pdf, create_excel, create_docx, create_document, calculator, grep_files, glob_files, edit_file.

## Examples (IMITATE THIS PATTERN)

Q: "show me my generated files"
→ call list_downloads() → return table of files with download links

Q: "search my files for 'stock'"
→ call list_downloads(user_search='stock') → return matching files

Q: "yesterday's top 5 A-share gainers"
→ call execute_command('date') + stock_query(action='top', count=5) in ONE response → table with rankings, codes, names, %, prices + source

Q: "make an Excel report of top stocks"
→ call stock_query(action='top', count=20) → call create_excel(filename='stocks.xlsx', data_json='...') → return download link

Q: "remember I like Python"
→ call memory_search(action='save', key='user preference', value='likes Python') → confirm saved

Q: "what did I tell you before about my preferences"
→ call memory_search(action='list') → return list of saved memories

Q: "what earthquake happened recently"
→ call execute_command('date') + web_search(query='latest earthquakes', fresh='d') in ONE response → synthesize findings with table + source URLs

Q: "read this PDF report"
→ call read_pdf(path='docs/report.pdf') → summarize content

Q: "Apple stock price today" or "Tesla market cap"
→ stock_query ONLY covers China A-shares. For US/HK/foreign stocks, use web_search(query='Apple stock AAPL today') + web_fetch(url='...') to get real-time data from finance websites. NEVER use stock_query for foreign stocks.

Q: "who will win World Cup match X vs Y" or "NBA game prediction"
→ Use web_search(query='X vs Y match preview odds') + web_fetch to gather: (1) recent form and head-to-head results, (2) betting odds / implied probability, (3) expert analysis. Then synthesize: estimate win probability for each side, likely score range, key factors. State that predictions are based on odds and public data, not guarantees.

## Rules

0. **EXAMPLES FIRST.** Scan Examples. If task matches, follow that pattern EXACTLY.
1. **TOOL SCOPE.** stock_query = China A-shares only (沪深/科创板/创业板). For foreign stocks, crypto, forex: use web_search + web_fetch. For sports/events: web_search + web_fetch → analyze odds/form → give probability and score estimate with data sources.
2. **ASK WITH OPTIONS, DON'T GUESS.** If the user request is ambiguous, ask 2-4 specific questions with concrete OPTIONS for each — like A/B/C choices. NEVER ask open-ended "请描述..." questions. Examples:
   - User uploads a contract with no instructions → "请选择需要我做什么：A. 审阅法律条款 B. 提取关键日期金额 C. 总结内容概要 D. 修改特定条款"
   - User says "帮我看看这个" → "你想了解哪个方面？A. 内容总结 B. 数据提取 C. 问题检查 D. 格式优化"
   - User says "做个分析" → "分析哪个维度？A. 趋势对比 B. 数据统计 C. 风险评估 D. 竞品对标"
   Always add a final option "E. 补充描述（以上都不对，我来说明）". Format as numbered list with lettered options.
3. Batch date+queries together — ONE response
4. NEVER answer without tools if question needs data
5. Synthesize into Markdown tables with sources
6. **FAIL → FALLBACK.** If tool returns empty, error, or irrelevant results: IMMEDIATELY try web_search. stock_query got nothing? → web_search. web_fetch blocked? → different keywords. NEVER stop after one failed attempt.
7. **STOP AND SYNTHESIZE.** Max 5 tool-call rounds total. After gathering enough data (usually 2-3 rounds), you MUST stop calling tools and write a comprehensive answer with Markdown tables and sources. DO NOT keep searching — the user needs the answer, not more research.
8. Think English, answer Chinese
9. Use create_document for reports/tables — include download link. NEVER use file:// protocol — use /api/download/filename only."""

REFLECTION_PROMPT = """请用一句话评估以下回答是否准确完整。
如果回答没问题，只回复'pass'。如果有问题，指出最关键的缺失。

用户问题：{user_question}
回答：{answer}

评估："""

MAX_ITERATIONS = 5
STEP_TIMEOUT = 60


class AgentEngine:
    """增强版 ReAct Agent — Token 预算 + 并行执行 + 自动反思"""

    def __init__(self, deepseek, tools: list[Tool], ws_manager: WebSocketManager,
                 intervention: InterventionHandler = None):
        self.deepseek = deepseek
        self.tools = tools
        self.tool_map = {t.name: t for t in tools}
        self.ws = ws_manager
        self.intervention = intervention
        self._cancellations: set[int] = set()
        self._called_history: list[str] = []  # 防重复调用
        # 预计算工具描述（缓存优化：每次 call_llm 复用同一段文本，不重算）
        self._tool_desc = "Tools: " + ", ".join(
            t.to_openai_schema()["function"]["name"] + "("
            + t.to_openai_schema()["function"]["description"][:50] + ")"
            for t in tools
        ) if tools else ""

    async def run(self, task_description: str, user_id: int, task_id: int,
                  max_iterations: int = MAX_ITERATIONS):
        start_time = time.time()
        self._called_history = []
        task_id_str = str(task_id)

        # 生命周期：注册任务
        if self.intervention:
            await self.intervention.register_task(task_id_str)

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

        tool_schemas = [t.to_openai_schema() for t in self.tools]

        # ── 简单问题直接回答（无工具、无规划）──
        # 剥掉 SSE 包装前缀，用原始用户问题做判断
        raw_question = task_description.split("## 当前任务\n")[-1] if "## 当前任务" in task_description else task_description
        # 明确需要搜索/工具的关键词 → 不走简单路径
        needs_tools = any(w in raw_question for w in [
            "搜", "查", "找", "分析", "生成", "创建", "下载", "股票",
            "天气", "新闻", "最新", "实时", "今天", "现在", "当前",
            "帮我写", "帮我做", "帮我查", "计算", "预测", "比较",
            "世界杯", "球赛", "比分", "谁会赢", "比赛", "走势",
            "推荐", "评测", "攻略", "教程", "价格", "多少钱",
        ])
        if needs_tools:
            # 本地关键词命中 → 直接走复杂路径，不让 LLM 推翻
            is_simple = False
        else:
            # 本地判为"可能简单"，LLM 二次确认
            is_simple = (
                len(raw_question) <= 15 or
                any(w in raw_question for w in ["你好", "谢谢", "再见", "哈哈", "嗯", "哦", "好", "OK", "Hi", "hi"])
            )
            if not is_simple:
                # 边界情况 → LLM 判断
                raw_question2 = task_description.split("## 当前任务\n")[-1] if "## 当前任务" in task_description else task_description
                try:
                    check_resp = await self._call_llm([
                        {"role": "system", "content": "Does this task require web_search, file operations, or external data? If yes → 'complex'. If pure conversation/knowledge question → 'simple'. Answer ONLY one word."},
                        {"role": "user", "content": raw_question2},
                    ], None)
                    is_simple = "simple" in (check_resp.choices[0].message.content or "").strip().lower()
                except Exception:
                    is_simple = False

        if is_simple:
            direct_messages = [
                {"role": "system", "content": "你是一个智能聊天助手。直接回答用户的问题，不要提'任务'或'完成'。用自然的口语。可以用工具查事实但要快。用中文回答。"},
                {"role": "user", "content": raw_question},
            ]
            direct_resp = await self._call_llm(direct_messages, tool_schemas if len(raw_question) > 20 else None)
            total_tokens = (direct_resp.usage.total_tokens if direct_resp.usage else 0)
            final_answer = direct_resp.choices[0].message.content or ""

            # 如果 LLM 调了工具，执行并追答
            msg = direct_resp.choices[0].message
            if msg.tool_calls:
                for tc in msg.tool_calls[:2]:  # 最多2个工具
                    tool = self.tool_map.get(tc.function.name)
                    if tool:
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            args = {}
                        try:
                            obs = await asyncio.wait_for(tool.handler(**args), timeout=STEP_TIMEOUT)
                            obs_str = json.dumps(obs, ensure_ascii=False)[:2000]
                            direct_messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}]})
                            direct_messages.append({"role": "tool", "tool_call_id": tc.id, "content": obs_str})
                        except Exception as e:
                            direct_messages.append({"role": "tool", "tool_call_id": tc.id, "content": f"Error: {e}"})
                final_resp = await self._call_llm(direct_messages, None)
                total_tokens += (final_resp.usage.total_tokens if final_resp.usage else 0)
                final_answer = final_resp.choices[0].message.content or final_answer

            if not final_answer.strip():
                final_answer = "你好！有什么可以帮你的？"

            duration_ms = int((time.time() - start_time) * 1000)
            _update_task(task_id, status="completed", final_answer=final_answer,
                         total_tokens=total_tokens, duration_ms=duration_ms)
            await self.ws.broadcast(task_id, "task_complete", {
                "final_answer": final_answer,
                "total_steps": 1,
                "total_tokens": total_tokens,
                "duration_ms": duration_ms,
            })
            if self.intervention:
                await self.intervention.complete_task(task_id_str)
            return

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
        # 稳定前缀 = system_prompt + tool_desc（固定不变）→ DeepSeek 自动缓存命中
        cached_messages = list(messages)
        if tools and self._tool_desc:
            # 只在原消息还没有 tool_desc 时插入（避免 _compress_context 导致的重复）
            if len(cached_messages) < 2 or cached_messages[1].get("content", "")[:6] != "Tools:":
                cached_messages.insert(1, {"role": "system", "content": self._tool_desc})

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

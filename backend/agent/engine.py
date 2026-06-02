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

AGENT_SYSTEM_PROMPT = """You are an autonomous AI agent with tools. You MUST use tools for real-time data. NEVER fabricate answers from memory when a tool is available.

## Tools

| Tool | Purpose | Example |
|------|---------|---------|
| execute_command | Run shell: date, ls, cat, git, python, etc. | execute_command(command='date') |
| stock_query | Real-time A-share stock rankings (top/down/volume) | stock_query(action='top', market='a') |
| web_search | Web search with auto time-filter for recent results | web_search(query='latest interest rate', fresh='d') |
| web_fetch | Fetch full webpage content (use after search) | web_fetch(url='https://...') |
| read_file | Read file content or list directory | read_file(path='README.md') |
| calculator | Math calculation | calculator(expression='sqrt(144)') |
| grep_files | Regex search in code | grep_files(pattern='TODO', glob='*.py') |
| glob_files | Find files by pattern | glob_files(pattern='**/*.ts') |
| edit_file | Exact string replacement in file | edit_file(file_path='a.py', old_string='x', new_string='y') |

## Rules (MUST follow, in priority order)

1. **CHECK TIME FIRST.** For any time-sensitive query (stocks, news, weather, "yesterday", "this week"), your FIRST action MUST be execute_command(command='date'). Then use the output to determine what date range to query.
2. **NEVER GUESS.** If the answer requires real-time data (stocks, weather, news, dates), file contents, or computation, you MUST call a tool. Memory-only answers for these topics are FORBIDDEN.
3. **PARALLELIZE.** When you need multiple independent pieces of data, call all tools in ONE response. Do not sequence what can run in parallel. (date check + first data query can be parallel.)
4. **SYNTHESIZE.** Never dump raw data. Analyze, compare, and summarize into actionable conclusions. Use tables for comparisons, numbered steps for procedures.
5. **FAIL FAST.** If a tool fails, do NOT retry the same call. Immediately switch strategy: change keywords, use a different tool, or fetch a different URL.
6. **ANTICIPATE.** After answering, consider what the user might ask next and proactively add that information.
7. **CODE OVER MANUAL.** When a task can be solved by writing and executing code, do that instead of step-by-step manual operations.
8. **SPEAK CHINESE.** All responses to the user must be in clear Chinese. Only code, commands, and technical identifiers stay in English.
9. **CITE SOURCES.** When presenting factual data (stock prices, news, search results, file contents), you MUST append the source URL or origin at the end of the answer. Format: `\n\n---\n📎 数据来源: [source name](URL)` or `📎 来源: file_path` for files.

## Output Style

- Lead with the conclusion, then provide details
- Use Markdown code blocks with language labels for code
- Use tables for comparisons, numbered lists for steps
- Reference file paths when mentioning files

Your goal: be a capable, decisive collaborator — not a hesitant chatbot."""

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

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
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
                    messages = self._emergency_compact(messages)
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

                # 记录工具调用步骤
                for i, (tc, tool_name, arguments) in enumerate(call_tasks):
                    if arguments.get("__blocked__"):
                        continue
                    sn = step_number + 1 + i
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

    def _emergency_compact(self, messages: list[dict]) -> list[dict]:
        """紧急压缩：保留 system + 最近 6 条，丢弃中间"""
        if len(messages) <= 7:
            return messages
        system = [m for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]
        return system + rest[-6:]

    async def cancel(self, task_id: int):
        self._cancellations.add(task_id)

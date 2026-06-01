"""
ReAct Agent — 思考(Thought) → 行动(Action) → 观察(Observation) 循环

优化要点：
  - Token 预算管理（防止超限）
  - 工具结果智能截断（保留关键部分，丢弃噪音）
  - 重复调用检测（防止死循环）
  - 及早退出（够用了就答）
"""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent_framework.core.prompt import REACT_SYSTEM_PROMPT
from agent_framework.tools.registry import ToolRegistry
from chat.context_manager import estimate_tokens

# 约 100K token 预算（留余量给模型回复）
TOKEN_BUDGET = 90_000
# 工具结果最大 token（保留头尾，中间截掉）
MAX_OBS_TOKENS = 2000


class ReactAgent:
    """
    ReAct Agent：自主决定何时使用工具 + token 预算管理。

    用法：
        agent = ReactAgent(llm_client, tool_registry)
        result = agent.run("帮我看看当前目录有什么文件")
    """

    def __init__(self, llm_client, tool_registry: ToolRegistry, max_turns: int = 8):
        self.client = llm_client
        self.registry = tool_registry
        self.max_turns = max_turns
        self._called_history = []  # 防重复调用

    def run(self, user_message: str, system_prompt: str = None) -> dict:
        """
        运行 ReAct 循环。返回：
        {
            "answer": str,
            "steps": list,
            "turns": int,
            "total_tokens": int,      # 估算总 token
            "truncated": bool          # 是否因 token 超限被截
        }
        """
        self._called_history = []

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
        total_tokens = estimate_tokens(sys_prompt) + estimate_tokens(user_message)

        for turn in range(1, self.max_turns + 1):
            # Token 预算检查
            if total_tokens > TOKEN_BUDGET:
                # 超限：注入摘要并继续
                messages = self._emergency_compact(messages)
                total_tokens = estimate_tokens(
                    " ".join(m.get("content", "") or "" for m in messages)
                )

            # 调 LLM
            response = self.client.client.chat.completions.create(
                model=self.client.model_id,
                messages=messages,
                tools=tools_def,
                timeout=60
            )

            choice = response.choices[0]
            msg = choice.message

            # 统计 token
            if hasattr(response, 'usage') and response.usage:
                total_tokens += getattr(response.usage, 'total_tokens', 0)

            # 模型要调工具
            if msg.tool_calls:
                # 解析所有工具调用
                call_tasks = []  # [(tc, tool_name, arguments)]
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

                # ========== 并行执行所有工具 ==========
                from concurrent.futures import ThreadPoolExecutor, as_completed

                results = {}  # tc.id -> observation
                with ThreadPoolExecutor(max_workers=min(8, len(call_tasks))) as pool:
                    futures = {}
                    for tc, tool_name, arguments in call_tasks:
                        if arguments.get("__blocked__"):
                            # 被阻止，不执行
                            results[tc.id] = f"工具调用已被阻止：{arguments.get('__reason__', '重复调用')}"
                        else:
                            futures[pool.submit(
                                self.registry.execute, tool_name, arguments
                            )] = tc.id

                    for future in as_completed(futures):
                        tc_id = futures[future]
                        try:
                            results[tc_id] = future.result(timeout=30)
                        except Exception as e:
                            results[tc_id] = f"工具执行异常：{type(e).__name__}: {e}"

                # ========== 收集结果 ==========
                # 一条 assistant 消息带所有 tool_calls
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

                # 每个 tool call 一条 tool 消息
                for tc, tool_name, arguments in call_tasks:
                    raw_obs = results.get(tc.id, "工具执行无响应")
                    observation = self._smart_truncate(raw_obs)

                    if any(w in observation for w in ["失败", "错误", "超时", "不支持", "安全限制", "异常", "阻止"]):
                        observation += "\n（此工具未成功，请换一个方法）"

                    step = {
                        "turn": turn,
                        "thought": f"调用 {tool_name}",
                        "action": f"{tool_name}({json.dumps(arguments, ensure_ascii=False)})",
                        "observation": observation[:800]
                    }
                    steps.append(step)
                    total_tokens += estimate_tokens(observation)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": observation
                    })
            else:
                # 最终回答
                answer = msg.content or ""
                return {
                    "answer": answer,
                    "steps": steps,
                    "turns": turn,
                    "total_tokens": total_tokens,
                    "truncated": total_tokens > TOKEN_BUDGET
                }

        # 达到最大轮次
        messages.append({
            "role": "user",
            "content": "轮次已用完。请基于以上信息，用一句话给出最终回答。"
        })
        final = self.client.chat(messages)
        return {
            "answer": final,
            "steps": steps,
            "turns": self.max_turns,
            "total_tokens": total_tokens,
            "truncated": True
        }

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
            f"...（省略 {omitted} tokens 的中间内容，完整结果请用 read_file 分段读取）...\n\n"
            f"{tail}"
        )

    def _emergency_compact(self, messages: list[dict]) -> list[dict]:
        """紧急压缩：保留 system + 最近 6 条，丢弃中间"""
        if len(messages) <= 7:
            return messages
        system = [m for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]
        return system + rest[-6:]

    def _format_tools(self) -> str:
        if not self.registry.list_tools():
            return "（无可用工具）"
        lines = []
        for name in self.registry.list_tools():
            tool = self.registry._tools[name]["definition"]["function"]
            desc = tool["description"]
            params = tool["parameters"]["properties"]
            param_str = ", ".join(
                f"{k}({v.get('type','str')})" for k, v in params.items()
            )
            lines.append(f"- {name}({param_str}): {desc}")
        return "\n".join(lines)

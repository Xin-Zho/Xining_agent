import asyncio
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .websocket_manager import WebSocketManager, _save_step, _update_step, _update_task
from ..claude_config import load_config

DEFAULT_APPEND_SYSTEM_PROMPT = (
    "Reply in Chinese. Use tools when they help. Stop as soon as you have enough "
    "evidence to answer well. Avoid repeating the same tool call."
)


@dataclass
class _MessageUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )


@dataclass
class _ThoughtState:
    step_number: int
    started_at: float
    text_parts: list[str] = field(default_factory=list)


@dataclass
class _ToolState:
    step_number: int
    started_at: float
    name: str
    parsed_args: Any = None
    input_parts: list[str] = field(default_factory=list)


@dataclass
class _RunState:
    next_step_number: int = 0
    total_tokens: int = 0
    current_usage: _MessageUsage | None = None
    active_thought: _ThoughtState | None = None
    active_tool: _ToolState | None = None
    last_assistant_text: str = ""
    final_answer: str | None = None
    error_message: str | None = None
    stderr_lines: list[str] = field(default_factory=list)


class ClaudeCodeEngine:
    def __init__(self, ws_manager: WebSocketManager, repo_root: str | Path | None = None):
        self.ws = ws_manager
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[2])
        self.claude_bin = self._resolve_claude_bin()

        cfg = load_config()
        self.model = cfg.model or None
        self.permission_mode = cfg.permission_mode or "bypassPermissions"
        self.append_system_prompt = cfg.append_system_prompt or DEFAULT_APPEND_SYSTEM_PROMPT
        self.allowed_tools = cfg.allowed_tools or ""
        self.extra_dirs = [
            item.strip()
            for item in (cfg.extra_dirs or [])
            if item.strip()
        ]

        self._processes: dict[int, asyncio.subprocess.Process] = {}
        self._cancellations: set[int] = set()

    def _resolve_claude_bin(self) -> str:
        candidates = [
            os.environ.get("CLAUDE_CODE_BIN", "").strip(),
            shutil.which("claude.cmd"),
            shutil.which("claude.exe"),
            shutil.which("claude"),
            r"C:\Users\Administrator\AppData\Roaming\npm\claude.cmd",
            r"C:\node_global\claude.cmd",
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return str(candidate)
        raise RuntimeError("Claude Code CLI not found. Set CLAUDE_CODE_BIN or install `claude`.")

    def _build_command(self, task_description: str) -> list[str]:
        command = [
            self.claude_bin,
            "-p",
            "--verbose",
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--permission-mode",
            self.permission_mode,
            "--add-dir",
            str(self.repo_root),
        ]
        for directory in self.extra_dirs:
            command.extend(["--add-dir", directory])
        if self.model:
            command.extend(["--model", self.model])
        if self.allowed_tools:
            command.extend(["--allowedTools", self.allowed_tools])
        if self.append_system_prompt:
            command.extend(["--append-system-prompt", self.append_system_prompt])
        command.append(task_description)
        return command

    async def run(self, task_description: str, user_id: int, task_id: int):
        start_time = time.time()
        state = _RunState()
        _update_task(task_id, status="executing")

        await self.ws.broadcast(
            task_id,
            "task_started",
            {
                "task_id": task_id,
                "title": task_description[:50],
            },
        )

        env = os.environ.copy()
        process = await asyncio.create_subprocess_exec(
            *self._build_command(task_description),
            cwd=str(self.repo_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        self._processes[task_id] = process

        stdout_task = asyncio.create_task(self._consume_stdout(task_id, state, process))
        stderr_task = asyncio.create_task(self._consume_stderr(state, process))

        try:
            return_code = await process.wait()
            await stdout_task
            await stderr_task

            await self._finalize_open_steps(task_id, state)

            if task_id in self._cancellations:
                _update_task(task_id, status="cancelled")
                await self.ws.broadcast(task_id, "task_cancelled", {"task_id": task_id})
                return

            duration_ms = int((time.time() - start_time) * 1000)
            final_answer = (
                state.final_answer
                or state.last_assistant_text.strip()
                or "Claude Code 已完成任务。"
            )

            if return_code == 0 and not state.error_message:
                _update_task(
                    task_id,
                    status="completed",
                    final_answer=final_answer,
                    total_tokens=state.total_tokens,
                    duration_ms=duration_ms,
                )
                await self.ws.broadcast(
                    task_id,
                    "task_complete",
                    {
                        "final_answer": final_answer,
                        "total_steps": state.next_step_number,
                        "total_tokens": state.total_tokens,
                        "duration_ms": duration_ms,
                    },
                )
                return

            error_message = state.error_message or self._build_process_error(return_code, state)
            _update_task(
                task_id,
                status="failed",
                final_answer=error_message,
                total_tokens=state.total_tokens,
                duration_ms=duration_ms,
            )
            await self.ws.broadcast(
                task_id,
                "task_error",
                {
                    "error": error_message,
                    "last_step": state.next_step_number,
                },
            )
        except Exception as exc:
            duration_ms = int((time.time() - start_time) * 1000)
            error_message = f"Claude Code 执行失败: {exc}"
            _update_task(
                task_id,
                status="failed",
                final_answer=error_message,
                total_tokens=state.total_tokens,
                duration_ms=duration_ms,
            )
            await self.ws.broadcast(
                task_id,
                "task_error",
                {
                    "error": error_message,
                    "last_step": state.next_step_number,
                },
            )
        finally:
            self._processes.pop(task_id, None)
            self._cancellations.discard(task_id)

    async def _consume_stdout(
        self,
        task_id: int,
        state: _RunState,
        process: asyncio.subprocess.Process,
    ):
        if process.stdout is None:
            return

        async for raw_line in process.stdout:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            await self._handle_stdout_line(task_id, state, line)

    async def _consume_stderr(self, state: _RunState, process: asyncio.subprocess.Process):
        if process.stderr is None:
            return

        async for raw_line in process.stderr:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line:
                state.stderr_lines.append(line)

    async def _handle_stdout_line(self, task_id: int, state: _RunState, line: str):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            if state.active_thought is None:
                await self._start_thought(task_id, state)
            await self._append_thought(task_id, state, line)
            return

        payload_type = payload.get("type")
        if payload_type == "stream_event":
            await self._handle_stream_event(task_id, state, payload.get("event", {}))
            return
        if payload_type == "assistant":
            text = self._extract_text(payload.get("message") or payload)
            if text:
                state.last_assistant_text = text
                state.final_answer = text
            total_tokens = self._extract_total_tokens(payload)
            if total_tokens:
                state.total_tokens = max(state.total_tokens, total_tokens)
            return
        if payload_type == "result":
            result_text = payload.get("result")
            if isinstance(result_text, str) and result_text.strip():
                state.final_answer = result_text.strip()
            total_tokens = self._extract_total_tokens(payload)
            if total_tokens:
                state.total_tokens = max(state.total_tokens, total_tokens)
            if payload.get("subtype") == "error":
                state.error_message = payload.get("message") or payload.get("result")
            return
        if payload_type == "error":
            state.error_message = payload.get("message") or line
            return
        if payload_type == "system":
            subtype = payload.get("subtype")
            if subtype == "api_retry":
                retry_line = f"Claude Code API retry {payload.get('attempt')}/{payload.get('max_retries')}"
                if state.active_thought is None:
                    await self._start_thought(task_id, state)
                await self._append_thought(task_id, state, retry_line)
            elif subtype == "status" and payload.get("status") == "error":
                state.error_message = line
            return

        total_tokens = self._extract_total_tokens(payload)
        if total_tokens:
            state.total_tokens = max(state.total_tokens, total_tokens)

    async def _handle_stream_event(self, task_id: int, state: _RunState, event: dict):
        event_type = event.get("type")

        if event_type == "message_start":
            state.current_usage = self._usage_from_event(event.get("message", {}).get("usage"))
            return

        if event_type == "content_block_start":
            block = event.get("content_block", {})
            block_type = block.get("type")
            if block_type == "text":
                await self._finalize_tool(task_id, state)
                await self._start_thought(task_id, state)
                if block.get("text"):
                    await self._append_thought(task_id, state, block["text"])
            elif block_type == "tool_use":
                await self._finalize_thought(task_id, state)
                await self._start_tool(task_id, state, block.get("name") or "tool", block.get("input"))
            return

        if event_type == "content_block_delta":
            delta = event.get("delta", {})
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                if state.active_thought is None:
                    await self._start_thought(task_id, state)
                await self._append_thought(task_id, state, delta.get("text", ""))
            elif delta_type == "input_json_delta" and state.active_tool is not None:
                state.active_tool.input_parts.append(delta.get("partial_json", ""))
            return

        if event_type == "content_block_stop":
            if state.active_tool is not None:
                await self._finalize_tool(task_id, state)
            elif state.active_thought is not None:
                await self._finalize_thought(task_id, state)
            return

        if event_type == "message_delta":
            usage = event.get("usage")
            if state.current_usage is not None and isinstance(usage, dict):
                state.current_usage.output_tokens = max(
                    state.current_usage.output_tokens,
                    int(usage.get("output_tokens") or 0),
                )
            return

        if event_type == "message_stop":
            if state.current_usage is not None:
                state.total_tokens += state.current_usage.total()
                state.current_usage = None
            return

    async def _start_thought(self, task_id: int, state: _RunState):
        if state.active_thought is not None:
            return
        state.next_step_number += 1
        step_number = state.next_step_number
        state.active_thought = _ThoughtState(step_number=step_number, started_at=time.time())
        _save_step(
            task_id,
            step_number,
            "thought",
            status="running",
            thought="",
        )
        await self.ws.broadcast(
            task_id,
            "step_start",
            {
                "step_num": step_number,
                "type": "thought",
                "message": "",
            },
        )

    async def _append_thought(self, task_id: int, state: _RunState, text: str):
        if not text or state.active_thought is None:
            return
        state.active_thought.text_parts.append(text)
        content = "".join(state.active_thought.text_parts).strip()
        if not content:
            return
        state.last_assistant_text = content
        await self.ws.broadcast(
            task_id,
            "step_complete",
            {
                "step_num": state.active_thought.step_number,
                "type": "thought",
                "content": content,
            },
        )

    async def _finalize_thought(self, task_id: int, state: _RunState):
        if state.active_thought is None:
            return
        thought = state.active_thought
        content = "".join(thought.text_parts).strip()
        duration_ms = int((time.time() - thought.started_at) * 1000)
        _update_step(
            task_id,
            thought.step_number,
            "completed",
            tool_result={"thought": content},
            duration_ms=duration_ms,
        )
        await self.ws.broadcast(
            task_id,
            "step_complete",
            {
                "step_num": thought.step_number,
                "type": "thought",
                "content": content,
                "duration_ms": duration_ms,
            },
        )
        state.last_assistant_text = content
        state.active_thought = None

    async def _start_tool(self, task_id: int, state: _RunState, name: str, args: Any):
        if state.active_tool is not None:
            await self._finalize_tool(task_id, state)
        state.next_step_number += 1
        step_number = state.next_step_number
        state.active_tool = _ToolState(
            step_number=step_number,
            started_at=time.time(),
            name=name,
            parsed_args=args,
        )
        tool_args = args if isinstance(args, dict) else {}
        _save_step(
            task_id,
            step_number,
            "tool_call",
            status="running",
            tool_name=name,
            tool_args=tool_args,
        )
        await self.ws.broadcast(
            task_id,
            "step_start",
            {
                "step_num": step_number,
                "type": "tool_call",
                "tool_name": name,
                "args": tool_args,
            },
        )

    async def _finalize_tool(self, task_id: int, state: _RunState):
        if state.active_tool is None:
            return
        tool = state.active_tool
        parsed_args = self._parse_tool_args(tool)
        duration_ms = int((time.time() - tool.started_at) * 1000)
        result = {
            "status": "completed",
            "provider": "claude-code",
            "summary": f"{tool.name} 已由 Claude Code 调用",
        }
        _update_step(
            task_id,
            tool.step_number,
            "completed",
            tool_result=result,
            duration_ms=duration_ms,
            tool_args=parsed_args,
        )
        await self.ws.broadcast(
            task_id,
            "step_complete",
            {
                "step_num": tool.step_number,
                "type": "tool_call",
                "tool_name": tool.name,
                "result": result,
                "duration_ms": duration_ms,
            },
        )
        state.active_tool = None

    async def _finalize_open_steps(self, task_id: int, state: _RunState):
        await self._finalize_tool(task_id, state)
        await self._finalize_thought(task_id, state)

    def _parse_tool_args(self, tool: _ToolState) -> dict:
        if isinstance(tool.parsed_args, dict):
            return tool.parsed_args
        raw = "".join(tool.input_parts).strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"raw": raw}

    def _usage_from_event(self, usage: Any) -> _MessageUsage:
        if not isinstance(usage, dict):
            return _MessageUsage()
        return _MessageUsage(
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens") or 0),
            cache_read_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
        )

    def _extract_text(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, list):
            parts = [self._extract_text(item) for item in payload]
            return "".join(part for part in parts if part)
        if isinstance(payload, dict):
            payload_type = payload.get("type")
            if payload_type in {"text", "thinking"} and isinstance(payload.get("text"), str):
                return payload["text"]
            if isinstance(payload.get("content"), list):
                return self._extract_text(payload["content"])
            if isinstance(payload.get("message"), dict):
                return self._extract_text(payload["message"])
            if isinstance(payload.get("text"), str):
                return payload["text"]
        return ""

    def _extract_total_tokens(self, payload: Any) -> int:
        if isinstance(payload, dict):
            if isinstance(payload.get("totalTokens"), int):
                return payload["totalTokens"]
            usage = payload.get("usage")
            if isinstance(usage, dict):
                cache_creation = usage.get("cache_creation")
                cache_creation_total = 0
                if isinstance(cache_creation, dict):
                    cache_creation_total = int(
                        cache_creation.get("ephemeral_1h_input_tokens") or 0
                    ) + int(cache_creation.get("ephemeral_5m_input_tokens") or 0)
                return (
                    int(usage.get("input_tokens") or 0)
                    + int(usage.get("output_tokens") or 0)
                    + int(usage.get("cache_creation_input_tokens") or 0)
                    + int(usage.get("cache_read_input_tokens") or 0)
                    + cache_creation_total
                )
            for value in payload.values():
                total = self._extract_total_tokens(value)
                if total:
                    return total
        elif isinstance(payload, list):
            for item in payload:
                total = self._extract_total_tokens(item)
                if total:
                    return total
        return 0

    def _build_process_error(self, return_code: int, state: _RunState) -> str:
        stderr_text = "\n".join(state.stderr_lines).strip()
        if stderr_text:
            return stderr_text
        return f"Claude Code exited with code {return_code}."

    async def cancel(self, task_id: int):
        self._cancellations.add(task_id)
        process = self._processes.get(task_id)
        if process is None:
            return
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "claude_config.json"

DEFAULT_CONFIG = {
    "model": "",
    "allowed_tools": "",
    "permission_mode": "bypassPermissions",
    "append_system_prompt": "",
    "extra_dirs": [],
}


@dataclass
class ClaudeCodeConfig:
    model: str = ""
    allowed_tools: str = ""
    permission_mode: str = "bypassPermissions"
    append_system_prompt: str = ""
    extra_dirs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _env_or(config_value: str, env_key: str, default: str = "") -> str:
    env_val = os.environ.get(env_key, "").strip()
    if env_val:
        return env_val
    if config_value:
        return config_value
    return default


def _resolve_extra_dirs(file_value: list) -> list[str]:
    env_val = os.environ.get("CLAUDE_CODE_ADD_DIRS", "").strip()
    if env_val:
        return [item.strip() for item in env_val.split(os.pathsep) if item.strip()]
    return [item.strip() for item in (file_value or []) if item.strip()]


def load_config() -> ClaudeCodeConfig:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    else:
        data = {}

    return ClaudeCodeConfig(
        model=_env_or(
            data.get("model", ""), "CLAUDE_CODE_MODEL"
        ),
        allowed_tools=_env_or(
            data.get("allowed_tools", ""), "CLAUDE_CODE_ALLOWED_TOOLS"
        ),
        permission_mode=_env_or(
            data.get("permission_mode", ""), "CLAUDE_CODE_PERMISSION_MODE",
            "bypassPermissions",
        ),
        append_system_prompt=_env_or(
            data.get("append_system_prompt", ""), "CLAUDE_CODE_APPEND_SYSTEM_PROMPT"
        ),
        extra_dirs=_resolve_extra_dirs(data.get("extra_dirs", [])),
    )


def save_config(config: ClaudeCodeConfig) -> None:
    data = config.to_dict()
    CONFIG_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_effective_config() -> dict:
    """Return the effective config with source annotations (env vs file)."""
    file_data = {}
    if CONFIG_PATH.exists():
        try:
            file_data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    result = {}
    for key, env_var, default in [
        ("model", "CLAUDE_CODE_MODEL", ""),
        ("allowed_tools", "CLAUDE_CODE_ALLOWED_TOOLS", ""),
        ("permission_mode", "CLAUDE_CODE_PERMISSION_MODE", "bypassPermissions"),
        ("append_system_prompt", "CLAUDE_CODE_APPEND_SYSTEM_PROMPT", ""),
    ]:
        env_val = os.environ.get(env_var, "").strip()
        file_val = file_data.get(key, "")
        if env_val:
            result[key] = {"value": env_val, "source": "env"}
        elif file_val:
            result[key] = {"value": file_val, "source": "file"}
        else:
            result[key] = {"value": default, "source": "default"}

    if os.environ.get("CLAUDE_CODE_ADD_DIRS", "").strip():
        result["extra_dirs"] = {"value": _resolve_extra_dirs([]), "source": "env"}
    else:
        result["extra_dirs"] = {"value": file_data.get("extra_dirs", []) or [], "source": "file"}
    return result

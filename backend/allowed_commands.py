"""命令白名单 — 独立模块，避免 import 链污染"""
ALLOWED_COMMANDS = frozenset({
    "ls", "dir", "cat", "type", "echo", "head", "tail", "wc",
    "python", "python3", "pip", "curl", "wget", "git", "find",
    "grep", "sort", "uniq", "date", "whoami", "pwd", "which",
    "mkdir", "touch", "cp", "mv", "tree", "nano", "vim", "code",
})

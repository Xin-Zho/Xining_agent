"""
长期记忆 — JSON 文件持久化，跨会话保留关键信息

用法：
    ltm = LongTermMemory("data/memory.json")
    ltm.save("用户偏好", "喜欢用 Python")
    result = ltm.get("用户偏好")
    keys = ltm.list_keys()
    results = ltm.search("Python")
"""
import json
import os
from datetime import datetime, timezone


class LongTermMemory:
    """
    长期记忆：存到 JSON 文件，跨会话保留。

    每条记忆 = key + value + 时间戳
    """

    def __init__(self, filepath: str = None):
        """
        filepath: JSON 文件路径（默认 agent_framework/data/long_term_memory.json）
        """
        if filepath is None:
            filepath = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "long_term_memory.json"
            )
        self.filepath = filepath
        self._data: dict = {}
        self._load()

    def _load(self):
        """从文件读取"""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._data = {}

    def _save(self):
        """写入文件"""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def save(self, key: str, value: str):
        """保存一条记忆"""
        self._data[key] = {
            "value": value,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        self._save()

    def get(self, key: str) -> str:
        """读取一条记忆，不存在返回空字符串"""
        item = self._data.get(key)
        return item["value"] if item else ""

    def list_keys(self) -> list[str]:
        """列出所有记忆的 key"""
        return list(self._data.keys())

    def list_all(self) -> list[dict]:
        """列出所有记忆（含内容）"""
        return [{"key": k, "value": v["value"], "updated_at": v.get("updated_at", "")}
                for k, v in self._data.items()]

    def search(self, keyword: str) -> list[dict]:
        """简单关键词搜索（大小写不敏感）"""
        keyword_lower = keyword.lower()
        results = []
        for k, v in self._data.items():
            if keyword_lower in k.lower() or keyword_lower in v["value"].lower():
                results.append({"key": k, "value": v["value"]})
        return results

    def delete(self, key: str):
        """删除一条记忆"""
        if key in self._data:
            del self._data[key]
            self._save()

    def clear(self):
        """清空所有记忆"""
        self._data = {}
        self._save()

    def get_context_for_prompt(self, max_items: int = 5) -> str:
        """
        把记忆格式化为一段上下文文本，可注入到 System Prompt。
        返回最近更新的 max_items 条记忆。
        """
        if not self._data:
            return ""

        # 按更新时间排序，取最近 N 条
        sorted_items = sorted(
            self._data.items(),
            key=lambda x: x[1].get("updated_at", ""),
            reverse=True
        )[:max_items]

        lines = ["[长期记忆]"]
        for k, v in sorted_items:
            lines.append(f"- {k}: {v['value']}")
        return "\n".join(lines)

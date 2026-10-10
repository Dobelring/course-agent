"""评测结果持久化：按 suite 合并写入 results.json。

独立成模块（无环境变量副作用）以便单测覆盖：
单独跑任一 suite 只更新自己的 key，不覆盖其他 suite 的历史结果。
"""
from __future__ import annotations

import json
from pathlib import Path


def merge_results(path: Path, results: dict) -> dict:
    """把 results 合并进 path 指向的 JSON 文件，返回合并后的完整内容。"""
    merged: dict = {}
    if path.exists():
        try:
            merged = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            merged = {}
    merged.update(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged

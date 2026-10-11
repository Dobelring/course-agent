"""向量存储格式：float32 BLOB（主）+ JSON 文本（旧数据只读兼容）。

BLOB 优势（实测见 scripts/bench_retrieval.py）：
- 512 维 float32 ≈ 2KB，JSON 文本约 5-8KB——体积约 1/3
- np.frombuffer 零解析开销，JSON 每候选一次 json.loads
读取顺序：embedding_blob 优先，为空回退解析 embedding JSON（迁移期双读，
旧数据无需重刷即可继续检索）。
"""
from __future__ import annotations

import json

import numpy as np


def encode_vec(vec: list[float]) -> bytes:
    """向量 → float32 BLOB。"""
    return np.asarray(vec, dtype=np.float32).tobytes()


def decode_vec(raw: bytes | None) -> list[float] | None:
    """BLOB → floats；空/脏数据返回 None（调用方跳过）。"""
    if not raw:
        return None
    try:
        arr = np.frombuffer(raw, dtype=np.float32)
        return arr.tolist() if arr.size else None
    except Exception:  # noqa: BLE001 — 长度不是 4 的倍数等脏数据
        return None


def decode_json_vec(text: str | None) -> list[float] | None:
    """旧 JSON 格式向量 → floats；空/脏数据返回 None。"""
    if not text:
        return None
    try:
        vec = json.loads(text)
        return [float(x) for x in vec] if vec else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None

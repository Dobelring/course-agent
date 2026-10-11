"""检索管线基准：JSON vs BLOB 解析、纯 Python 余弦 vs numpy 批量点积。

合成 2000 候选 × 512 维（与 bge-small 同维度），
结果打印到 stdout——数字用于 README/设计文档（写明环境与规模）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

N = 2000
DIM = 512

rng = np.random.default_rng(42)
vectors = rng.standard_normal((N, DIM)).astype(np.float32)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # 模拟 BGE 已归一化
query = vectors[0]  # 第一条为查询
json_blobs = [json.dumps([round(float(x), 6) for x in v]) for v in vectors]
byte_blobs = [v.astype(np.float32).tobytes() for v in vectors]


def bench(label: str, fn, repeat: int = 5) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    print(f"{label:<44} {best * 1000:8.2f} ms")
    return best


def decode_all_json() -> None:
    for b in json_blobs:
        json.loads(b)


def decode_all_blob() -> None:
    for b in byte_blobs:
        np.frombuffer(b, dtype=np.float32)


def cosine_loop() -> None:
    from app.rag.retriever import cosine

    for v in vectors:
        cosine(query.tolist(), v.tolist())


def numpy_dot() -> None:
    mat = np.stack(vectors)
    _ = (mat @ query).tolist()


def main() -> None:
    print(f"\n== 检索基准：{N} 候选 × {DIM} 维（float32，已归一化；取 5 次最优）==\n")
    t_json = bench(f"JSON 文本解码 × {N}", decode_all_json)
    t_blob = bench(f"BLOB frombuffer × {N}", decode_all_blob)
    t_loop = bench("纯 Python 余弦循环（旧实现）", cosine_loop)
    t_np = bench("numpy 批量点积（新实现，含归一化）", numpy_dot)
    print("-" * 56)
    print(f"解码加速 {t_json / t_blob:.0f}x；余弦加速 {t_loop / t_np:.0f}x")


if __name__ == "__main__":
    main()

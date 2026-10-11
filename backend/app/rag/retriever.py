"""检索管线：混合召回（BM25 + 向量 RRF 融合）→ 交叉编码器重排。

三阶段（均按 user_id 过滤，避免越权读取他人资料）：
1. 召回：向量余弦一路 + BM25 关键词一路，各取 rag_recall_k 候选
2. 融合：RRF（Reciprocal Rank Fusion）把两路排名合成统一得分——
   不依赖分数绝对值可比（余弦是 0~1、BM25 无上界），只看名次，鲁棒
3. 重排：融合后的前 rag_rerank_top 名送交叉编码器逐对打分，取 top_k

为什么混合：纯向量对「查询与文档无词汇重合」的语义改写敏感（评测 r26
bad case），BM25 恰好补上精确词汇匹配；两路互补，RRF 免调权重。
中文分词不引 jieba，用「中文二元组 + 英文单词」零依赖切分。
"""
import json
import logging
import math
import re

import numpy as np
from sqlalchemy.orm import Session

from app.config import settings
from app.models.document import CourseDocument, DocumentChunk
from app.rag.embedder import embed_query
from app.rag.vecfmt import decode_json_vec, decode_vec

logger = logging.getLogger(__name__)

RRF_K = 60  # RRF 常用平滑常数（Cormack et al.）

# 中文二元组 + 英文/数字单词；单个中文字也保留（常见虚词如「课」「的」）
_TOKEN_RE = re.compile(r"[一-鿿]+|[a-zA-Z0-9_]+")
_ASCII_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    """零依赖分词：CJK 连续串 → 二元组 + 单字；ASCII → 小写单词。"""
    tokens: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        seg = m.group()
        if _ASCII_RE.fullmatch(seg):
            tokens.append(seg.lower())
        else:
            tokens.extend(seg)  # 单字
            tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))  # 二元组
    return tokens


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def bm25_scores(
    query_tokens: list[str],
    docs_tokens: list[list[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """标准 BM25 打分（Okapi）。docs_tokens[i] 是第 i 篇文档的词序列。"""
    n = len(docs_tokens)
    if n == 0:
        return []
    avgdl = sum(len(d) for d in docs_tokens) / n
    df: dict[str, int] = {}
    for d in docs_tokens:
        for t in set(d):
            df[t] = df.get(t, 0) + 1

    scores = []
    for d in docs_tokens:
        tf: dict[str, int] = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t in query_tokens:
            if t not in tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * (tf[t] * (k1 + 1)) / (
                tf[t] + k1 * (1 - b + b * len(d) / avgdl)
            )
        scores.append(s)
    return scores


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """RRF 融合多路排名：每路中位置越靠前得分越高，求和。"""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


def _load_candidates(db: Session, user_id: int) -> list[dict]:
    """加载当前用户 ready 文档的切片候选：BLOB 优先、旧 JSON 只读兼容。"""
    rows = (
        db.query(DocumentChunk, CourseDocument.filename)
        .join(CourseDocument, DocumentChunk.document_id == CourseDocument.id)
        .filter(CourseDocument.user_id == user_id, CourseDocument.status == "ready")
        .all()
    )
    candidates: list[dict] = []
    for chunk, filename in rows:
        vec = decode_vec(chunk.embedding_blob) or decode_json_vec(chunk.embedding)
        if vec is None:
            # 脏数据（空/坏 BLOB 与坏 JSON）跳过，不让一条坏切片拖垮整个检索
            if chunk.embedding_blob or chunk.embedding:
                logger.warning("切片向量解析失败，已跳过 chunk_id=%s", chunk.id)
            continue
        candidates.append({
            "chunk_id": chunk.id,
            "chunk_index": chunk.chunk_index,
            "filename": filename,
            "content": chunk.content,
            "vec": vec,
        })
    return candidates


def _vector_scores(qv: list[float], candidates: list[dict]) -> list[float]:
    """批量余弦：numpy 矩阵点积替代 Python 逐元素循环。

    BGE 向量本身已 L2 归一化，余弦退化为点积；为防御未归一化数据，
    显式按模长归一化后再点积（纯零向量按 1 处理，得分 0）。
    """
    mat = np.asarray([c["vec"] for c in candidates], dtype=np.float32)
    q = np.asarray(qv, dtype=np.float32)
    mat_norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat_norms[mat_norms == 0.0] = 1.0
    q_norm = float(np.linalg.norm(q)) or 1.0
    sims = (mat / mat_norms) @ (q / q_norm)
    return [float(s) for s in sims]


def retrieve(db: Session, user_id: int, query: str, top_k: int = 3) -> list[dict]:
    """检索当前用户已 ready 的文档切片：混合召回 → RRF → 重排 → top-k。"""
    candidates = _load_candidates(db, user_id)
    if not candidates:
        return []

    # ---- 第一路：向量召回 ----
    qv = embed_query(query)
    # 维度不齐（历史模型更换残留）的候选不进矩阵，避免形状错误
    candidates = [c for c in candidates if len(c["vec"]) == len(qv)]
    if not candidates:
        return []
    sims = _vector_scores(qv, candidates)
    for c, s in zip(candidates, sims):
        c["cos"] = s
    vec_ranked = sorted(candidates, key=lambda c: c["cos"], reverse=True)

    if not settings.rag_hybrid:
        return _finalize(vec_ranked[:top_k], [c["cos"] for c in vec_ranked[:top_k]])

    # ---- 第二路：BM25 关键词召回 ----
    q_tokens = tokenize(query)
    docs_tokens = [tokenize(c["content"]) for c in candidates]
    bm25 = bm25_scores(q_tokens, docs_tokens)
    for c, s in zip(candidates, bm25):
        c["bm25"] = s
    bm25_ranked = sorted(candidates, key=lambda c: c["bm25"], reverse=True)

    # ---- RRF 融合两路排名 ----
    fused = rrf_fuse([
        [c["chunk_id"] for c in vec_ranked],
        [c["chunk_id"] for c in bm25_ranked],
    ])
    by_id = {c["chunk_id"]: c for c in candidates}
    fused_ranked = sorted(
        (by_id[cid] for cid in fused),
        key=lambda c: fused[c["chunk_id"]],
        reverse=True,
    )

    # ---- 多样性：同一文档最多贡献 rag_max_per_doc 个候选进重排池 ----
    # 避免 top-k 被同一文档的相邻切片占满，把位置留给其他相关文档
    cap = max(1, settings.rag_max_per_doc)
    per_doc: dict[str, int] = {}
    diverse: list[dict] = []
    for c in fused_ranked:
        n = per_doc.get(c["filename"], 0)
        if n >= cap:
            continue
        per_doc[c["filename"]] = n + 1
        diverse.append(c)

    # ---- 重排：融合 top 候选送交叉编码器 ----
    pool = diverse[: max(settings.rag_rerank_top, top_k)]
    if settings.rag_rerank and len(pool) > top_k:
        try:
            from app.rag.reranker import rerank
            r_scores = rerank(query, [c["content"] for c in pool])
            for c, s in zip(pool, r_scores):
                c["rerank"] = s
            pool.sort(key=lambda c: c["rerank"], reverse=True)
        except Exception:  # noqa: BLE001 — 重排失败回退融合排名，不阻断检索
            # 必须留痕：静默降级会掩盖模型/依赖故障，排障时无从下手
            logger.warning("重排失败，回退融合排名", exc_info=True)

    final = pool[:top_k]
    logger.info("检索完成 user=%s query=%r 候选=%d 返回=%d 混合=%s 重排=%s",
                user_id, query[:40], len(candidates), len(final),
                settings.rag_hybrid, settings.rag_rerank)
    return _finalize(
        final,
        [c.get("rerank", fused[c["chunk_id"]]) for c in final],
    )


# 相邻合并后单条结果的 content 上限（合并块允许比单块更长，仍是提示词经济性约束）
_MERGED_CONTENT_CAP = 800


def _finalize(order: list[dict], scores: list[float]) -> list[dict]:
    """统一输出格式：score 为当前排名依据（余弦 / 融合分 / 重排分）。

    多样性收尾：同一文档的相邻切片（chunk_index 连续）合并为一条——
    它们本就是一个连续段落，分开占 top-k 名额只会重复上下文；
    合并块 content 允许更长（_MERGED_CONTENT_CAP），score 取相邻块中较高者。
    """
    merged: list[dict] = []
    for c, s in zip(order, scores):
        score = round(s, 4)
        if (merged and merged[-1]["filename"] == c["filename"]
                and c["chunk_index"] == merged[-1]["chunk_index"] + 1):
            prev = merged[-1]
            prev["content"] = (prev["content"] + "\n" + c["content"])[:_MERGED_CONTENT_CAP]
            prev["score"] = max(prev["score"], score)
            continue
        merged.append({
            "score": score,
            "filename": c["filename"],
            "chunk_index": c["chunk_index"],
            "content": c["content"][:500],
        })
    return merged

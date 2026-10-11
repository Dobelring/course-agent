"""交叉编码器重排（fastembed CrossEncoder，本地 ONNX 推理）。

与向量召回互补：向量召回看「语义相近」，交叉编码器看「查询-文档逐对相关性」，
对查询词与文档无词汇重合但语义相关的 bad case（如 r26）有直接提升。

模型默认 BAAI/bge-reranker-v2-m3（多语言，中文效果好），
首次调用会从 HuggingFace 下载（国内默认走 hf-mirror）。
"""
import logging
import os
import threading

# 必须在导入 fastembed 之前设置（与 embedder 同策略）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: E402

from app.config import settings  # noqa: E402

logger = logging.getLogger(__name__)

# 逐对交叉编码的显存/内存峰值与批大小线性相关，分批控制
_RERANK_BATCH = settings.rag_rerank_batch

_model: TextCrossEncoder | None = None
_model_name: str | None = None
_model_lock = threading.Lock()


def get_reranker() -> TextCrossEncoder:
    global _model, _model_name
    if _model is None or _model_name != settings.rag_rerank_model:
        with _model_lock:
            if _model is None or _model_name != settings.rag_rerank_model:
                _model = TextCrossEncoder(model_name=settings.rag_rerank_model)
                _model_name = settings.rag_rerank_model
    return _model


def warmup() -> None:
    """启动预热：加载重排模型（best-effort，失败不阻断启动）。"""
    try:
        get_reranker().rerank("预热", ["预热文本"])
    except Exception:  # noqa: BLE001 — 预热失败不阻断启动
        logger.warning("重排模型预热失败，将在首次检索时重试", exc_info=True)


def rerank(query: str, documents: list[str]) -> list[float]:
    """返回每个文档与查询的相关性得分（越高越相关）；分批推理控制内存峰值。"""
    if not documents:
        return []
    scores: list[float] = []
    model = get_reranker()
    for i in range(0, len(documents), _RERANK_BATCH):
        batch = documents[i:i + _RERANK_BATCH]
        scores.extend(float(s) for s in model.rerank(query, batch))
    return scores

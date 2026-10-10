"""交叉编码器重排（fastembed CrossEncoder，本地 ONNX 推理）。

与向量召回互补：向量召回看「语义相近」，交叉编码器看「查询-文档逐对相关性」，
对查询词与文档无词汇重合但语义相关的 bad case（如 r26）有直接提升。

模型默认 BAAI/bge-reranker-v2-m3（多语言，中文效果好），
首次调用会从 HuggingFace 下载（国内默认走 hf-mirror）。
"""
import os

# 必须在导入 fastembed 之前设置（与 embedder 同策略）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: E402

from app.config import settings  # noqa: E402

_model: TextCrossEncoder | None = None
_model_name: str | None = None


def get_reranker() -> TextCrossEncoder:
    global _model, _model_name
    if _model is None or _model_name != settings.rag_rerank_model:
        _model = TextCrossEncoder(model_name=settings.rag_rerank_model)
        _model_name = settings.rag_rerank_model
    return _model


def rerank(query: str, documents: list[str]) -> list[float]:
    """返回每个文档与查询的相关性得分（越高越相关）。"""
    if not documents:
        return []
    scores = get_reranker().rerank(query, documents)
    return [float(s) for s in scores]

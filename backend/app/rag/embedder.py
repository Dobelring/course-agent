"""本地向量化：fastembed + BGE 中文小模型，无需额外 API Key。

首次使用会自动从 HuggingFace 下载模型（约 100MB）；
国内网络默认走 hf-mirror 镜像，如已配置 HF_ENDPOINT 则尊重用户设置。
"""
import logging
import os
import threading

# 必须在导入 huggingface_hub（fastembed 依赖）之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from fastembed import TextEmbedding  # noqa: E402

logger = logging.getLogger(__name__)

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

_model: TextEmbedding | None = None
# 双重检查锁：首次并发调用只加载一次模型，防止双份内存
_model_lock = threading.Lock()


def get_model() -> TextEmbedding:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5")
    return _model


def warmup() -> None:
    """启动预热：加载模型并跑一次空编码，把首次开销移出请求路径。

    best-effort：预热失败（如离线环境）只记日志，首次检索时再加载。
    """
    try:
        get_model().embed(["预热"])
    except Exception:  # noqa: BLE001 — 预热失败不阻断启动
        logger.warning("embedding 模型预热失败，将在首次检索时重试", exc_info=True)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量向量化（用于文档切片），统一转成 Python float 便于 JSON 存储。"""
    return [[float(x) for x in v] for v in get_model().embed(texts)]


def embed_query(query: str) -> list[float]:
    """向量化查询（BGE 中文模型要求查询语句带指令前缀）。"""
    vec = next(iter(get_model().embed([QUERY_PREFIX + query])))
    return [float(x) for x in vec]

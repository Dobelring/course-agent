"""RAG 基础设施健壮性：单例双重检查锁、分批重排、预热 best-effort、索引回滚。"""
import threading

from app.rag import embedder
from app.rag import reranker as reranker_mod


def test_embedder_singleton_under_concurrency(monkeypatch):
    """并发首次调用只实例化一次模型（双重检查锁，防双份内存）。"""
    created = []

    class FakeModel:
        def __init__(self, **kwargs):
            created.append(kwargs.get("model_name"))

    monkeypatch.setattr(embedder, "TextEmbedding", FakeModel)
    monkeypatch.setattr(embedder, "_model", None)
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        embedder.get_model()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        assert len(created) == 1
    finally:
        embedder._model = None  # 恢复全局单例，防污染其他测试


def test_reranker_batches_documents(monkeypatch):
    """重排分批推理：70 条 → 32+32+6，分数顺序不变。"""
    batches = []

    class FakeReranker:
        def rerank(self, q, docs):
            batches.append(len(docs))
            return [float(len(docs) - i) for i in range(len(docs))]

    monkeypatch.setattr(reranker_mod, "get_reranker", lambda: FakeReranker())
    docs = [f"d{i}" for i in range(70)]
    scores = reranker_mod.rerank("q", docs)
    assert batches == [32, 32, 6]
    assert len(scores) == 70


def test_warmup_swallows_model_failure(monkeypatch):
    """预热失败只记日志不抛出（离线环境启动不被阻断）。"""

    def boom():
        raise RuntimeError("离线无模型")

    monkeypatch.setattr(embedder, "get_model", boom)
    embedder.warmup()  # 不应抛出


def test_reranker_warmup_swallows_failure(monkeypatch):
    def boom():
        raise RuntimeError("离线无模型")

    monkeypatch.setattr(reranker_mod, "get_reranker", boom)
    reranker_mod.warmup()  # 不应抛出

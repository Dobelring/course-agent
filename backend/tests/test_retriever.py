"""RAG 检索：余弦相似度、按 user_id 隔离、只检索 ready 文档、top-k 截断。"""
import json

from app.config import settings
from app.models.document import CourseDocument, DocumentChunk
from app.rag.retriever import _load_candidates, _vector_scores, cosine, retrieve


def test_cosine_identical_vectors():
    assert abs(cosine([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) - 1.0) < 1e-9


def test_cosine_orthogonal_vectors():
    assert abs(cosine([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])) < 1e-9


def test_cosine_zero_vector_is_zero():
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_retrieve_returns_own_docs_ranked_by_score(db, seeded, monkeypatch):
    monkeypatch.setattr("app.rag.retriever.embed_query", lambda q: [1.0, 0.0, 0.0])
    results = retrieve(db, seeded.users.student1.id, "什么是装饰器", top_k=3)
    # [1,0,0] 的切片应排在 [0,1,0] 之前
    assert results[0]["filename"] == "u1_notes.md"
    assert results[0]["chunk_index"] == 0
    # 相邻切片合并后 score 可能并列（RRF 融合分），按非升序断言
    assert all(results[i]["score"] >= results[i + 1]["score"]
               for i in range(len(results) - 1))


def test_retrieve_isolates_users(db, seeded, monkeypatch):
    """student2 的查询绝不能命中 student1 的切片（水平越权防护）。"""
    monkeypatch.setattr("app.rag.retriever.embed_query", lambda q: [1.0, 0.0, 0.0])
    mine = retrieve(db, seeded.users.student2.id, "装饰器", top_k=5)
    assert all(r["filename"] == "u2_notes.md" for r in mine)
    assert not any("u1_" in r["filename"] for r in mine)

    other = retrieve(db, seeded.users.student1.id, "私有", top_k=5)
    assert all(r["filename"] != "u2_notes.md" for r in other)


def test_retrieve_skips_pending_docs(db, seeded, monkeypatch):
    monkeypatch.setattr("app.rag.retriever.embed_query", lambda q: [1.0, 0.0, 0.0])
    results = retrieve(db, seeded.users.student1.id, "pending", top_k=5)
    assert all(r["filename"] != "u1_draft.md" for r in results)


def test_retrieve_respects_top_k(db, seeded, monkeypatch):
    monkeypatch.setattr("app.rag.retriever.embed_query", lambda q: [1.0, 0.0, 0.0])
    assert len(retrieve(db, seeded.users.student1.id, "任意查询", top_k=1)) == 1


def test_retrieve_skips_empty_embedding(db, seeded, monkeypatch):
    doc = CourseDocument(user_id=seeded.users.student1.id, filename="u1_noembed.md",
                         file_path="/tmp/x.md", file_type="md", status="ready")
    db.add(doc)
    db.flush()
    db.add(DocumentChunk(document_id=doc.id, chunk_index=0, content="无向量", embedding="[]"))
    db.commit()

    monkeypatch.setattr("app.rag.retriever.embed_query", lambda q: [1.0, 0.0, 0.0])
    results = retrieve(db, seeded.users.student1.id, "无向量", top_k=10)
    assert all(r["filename"] != "u1_noembed.md" for r in results)
    # sanity：embedding 存为 JSON 字符串
    chunk = db.query(DocumentChunk).filter(DocumentChunk.content == "无向量").one()
    assert json.loads(chunk.embedding) == []


def test_retrieve_skips_corrupt_embedding_json(db, seeded, monkeypatch):
    """脏 JSON 向量不拖垮检索：坏切片被剔除，其余照常返回。"""
    doc = CourseDocument(user_id=seeded.users.student1.id, filename="u1_bad.md",
                         file_path="/tmp/bad.md", file_type="md", status="ready")
    db.add(doc)
    db.flush()
    db.add(DocumentChunk(document_id=doc.id, chunk_index=0,
                         content="坏向量切片", embedding="{not-json"))
    db.commit()

    monkeypatch.setattr("app.rag.retriever.embed_query", lambda q: [1.0, 0.0, 0.0])
    results = retrieve(db, seeded.users.student1.id, "坏向量", top_k=10)
    assert all(r["filename"] != "u1_bad.md" for r in results)


def test_vector_scores_matches_reference_cosine():
    """numpy 批量点积与纯 Python 参考实现给出一致排名与数值。"""
    import random

    from app.rag.retriever import _vector_scores

    random.seed(7)
    qv = [1.0, 0.0, 0.0]
    candidates = [
        {"vec": [1.0, 0.0, 0.0]},
        {"vec": [0.6, 0.8, 0.0]},   # 未归一化
        {"vec": [0.0, 1.0, 0.0]},
        {"vec": [0.0, 0.0, 0.0]},   # 零向量 → 0
    ]
    scores = _vector_scores(qv, candidates)
    ref = [cosine(qv, c["vec"]) for c in candidates]
    for got, want in zip(scores, ref):
        assert abs(got - want) < 1e-6
    assert scores[0] == 1.0 and scores[2] == 0.0 and scores[3] == 0.0


def test_retrieve_reads_legacy_json_embedding(db, seeded, monkeypatch):
    """旧格式（JSON 文本向量）在 BLOB 为空时仍可检索（迁移期双读兼容）。"""
    doc = CourseDocument(user_id=seeded.users.student1.id, filename="u1_legacy.md",
                         file_path="/tmp/legacy.md", file_type="md", status="ready")
    db.add(doc)
    db.flush()
    db.add(DocumentChunk(document_id=doc.id, chunk_index=0, content="旧格式切片",
                         embedding="[0.9, 0.1]", embedding_blob=None))
    db.commit()

    monkeypatch.setattr("app.rag.retriever.embed_query", lambda q: [1.0, 0.0])
    results = retrieve(db, seeded.users.student1.id, "旧格式", top_k=5)
    assert any(r["filename"] == "u1_legacy.md" for r in results)


def test_retrieve_prefers_blob_over_json(db, seeded, monkeypatch):
    """两列同时存在时以 BLOB 为准（新格式权威）。"""
    doc = CourseDocument(user_id=seeded.users.student1.id, filename="u1_both.md",
                         file_path="/tmp/both.md", file_type="md", status="ready")
    db.add(doc)
    db.flush()
    from app.rag.vecfmt import encode_vec
    db.add(DocumentChunk(document_id=doc.id, chunk_index=0, content="双格式切片",
                         embedding="[0.0, 1.0]",
                         embedding_blob=encode_vec([1.0, 0.0])))
    db.commit()

    # 直接验证向量来源：BLOB 优先于 JSON（decode 后是 [1,0] 而非 [0,1]）
    cands = _load_candidates(db, seeded.users.student1.id)
    c = next(c for c in cands if c["filename"] == "u1_both.md")
    assert c["vec"] == [1.0, 0.0]


def test_finalize_merges_adjacent_chunks(db, seeded, monkeypatch):
    """同一文档相邻切片合并为一条，content 拼接、score 取较高者。"""
    doc = CourseDocument(user_id=seeded.users.student1.id, filename="u1_series.md",
                         file_path="/tmp/series.md", file_type="md", status="ready")
    db.add(doc)
    db.flush()
    from app.models.document import DocumentChunk
    db.add(DocumentChunk(document_id=doc.id, chunk_index=0, content="上半段内容",
                         embedding="[1.0, 0.0]"))
    db.add(DocumentChunk(document_id=doc.id, chunk_index=1, content="下半段内容",
                         embedding="[0.9, 0.1]"))
    db.commit()

    monkeypatch.setattr("app.rag.retriever.embed_query", lambda q: [1.0, 0.0])
    monkeypatch.setattr(settings, "rag_max_per_doc", 5)
    results = retrieve(db, seeded.users.student1.id, "连续内容", top_k=5)
    series = [r for r in results if r["filename"] == "u1_series.md"]
    assert len(series) == 1                    # 相邻两条合并为一条
    assert "上半段内容" in series[0]["content"]
    assert "下半段内容" in series[0]["content"]


def test_per_doc_cap_keeps_results_diverse(db, seeded, monkeypatch):
    """每文档限流：同一文档的大量切片不会占满 top-k，位置留给其他文档。"""
    doc = CourseDocument(user_id=seeded.users.student1.id, filename="u1_flood.md",
                         file_path="/tmp/flood.md", file_type="md", status="ready")
    db.add(doc)
    db.flush()
    from app.models.document import DocumentChunk
    # 同一文档塞 5 个与查询完全一致的切片
    for i in range(5):
        db.add(DocumentChunk(document_id=doc.id, chunk_index=i,
                             content=f"重复命中{i}", embedding="[1.0, 0.0]"))
    # 另一文档 1 个次优切片
    other = CourseDocument(user_id=seeded.users.student1.id, filename="u2_other.md",
                           file_path="/tmp/o.md", file_type="md", status="ready")
    db.add(other)
    db.flush()
    db.add(DocumentChunk(document_id=other.id, chunk_index=0, content="其他文档",
                         embedding="[0.8, 0.6]"))
    db.commit()

    monkeypatch.setattr("app.rag.retriever.embed_query", lambda q: [1.0, 0.0])
    monkeypatch.setattr(settings, "rag_max_per_doc", 2)
    results = retrieve(db, seeded.users.student1.id, "多样性", top_k=3)
    flood = sum(1 for r in results if r["filename"] == "u1_flood.md")
    assert flood <= 2                          # 限流生效
    assert any(r["filename"] == "u2_other.md" for r in results)  # 其他文档有位置

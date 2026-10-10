"""RAG 检索：余弦相似度、按 user_id 隔离、只检索 ready 文档、top-k 截断。"""
import json

from app.models.document import CourseDocument, DocumentChunk
from app.rag.retriever import cosine, retrieve


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
    assert results[0]["score"] > results[-1]["score"]


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

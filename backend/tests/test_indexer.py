"""文档切片与向量化入库（indexer）。"""
from app.models.document import CourseDocument, DocumentChunk
from app.rag.indexer import CHUNK_SIZE, index_document, split_text


def test_split_empty_text_returns_no_chunks():
    assert split_text("") == []
    assert split_text("\n\n  \n") == []


def test_split_short_text_single_chunk():
    chunks = split_text("第一行\n第二行")
    assert len(chunks) == 1
    assert "第一行" in chunks[0] and "第二行" in chunks[0]


def test_split_long_text_respects_chunk_size():
    text = "\n".join(f"这是第{i}行内容，用于验证分块长度约束。" for i in range(200))
    chunks = split_text(text)
    assert len(chunks) > 1
    # 每块接近 CHUNK_SIZE（单行追加可能略微超出一行的余量）
    for c in chunks:
        assert len(c) <= CHUNK_SIZE + 100
    # 内容不丢
    joined = "".join(chunks)
    assert "这是第0行内容" in joined and "这是第199行内容" in joined


def test_index_document_writes_chunks_and_marks_ready(db, make_users, monkeypatch):
    monkeypatch.setattr("app.rag.indexer.embed_texts",
                        lambda texts: [[0.5, 0.5] for _ in texts])
    user = make_users.student1
    doc = CourseDocument(user_id=user.id, filename="a.md", file_path="/tmp/a.md",
                         file_type="md", status="pending")
    db.add(doc)
    db.commit()

    index_document(db, doc, "第一段\n第二段")
    db.refresh(doc)

    assert doc.status == "ready"
    rows = db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).all()
    assert len(rows) == 1
    assert json_list(rows[0].embedding) == [0.5, 0.5]


def test_index_document_empty_text_marks_failed(db, make_users, monkeypatch):
    monkeypatch.setattr("app.rag.indexer.embed_texts", lambda texts: [])
    user = make_users.student1
    doc = CourseDocument(user_id=user.id, filename="b.md", file_path="/tmp/b.md",
                         file_type="md", status="pending")
    db.add(doc)
    db.commit()

    index_document(db, doc, "")
    db.refresh(doc)
    assert doc.status == "failed"
    assert db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).count() == 0


def json_list(raw: str) -> list:
    import json

    return json.loads(raw)


def test_index_document_rolls_back_on_failure(db, make_users, monkeypatch):
    """向量化中途失败 → 回滚半成品 chunk，文档状态 failed，异常上抛。"""
    from app.models.document import CourseDocument, DocumentChunk

    def boom(texts):
        raise RuntimeError("模型加载失败")

    monkeypatch.setattr("app.rag.indexer.embed_texts", boom)
    user = make_users.student1
    doc = CourseDocument(user_id=user.id, filename="x.md", file_path="/tmp/x.md",
                         file_type="md", status="pending")
    db.add(doc)
    db.commit()

    import pytest as _pytest
    with _pytest.raises(RuntimeError):
        index_document(db, doc, "第一段内容")

    assert doc.status == "failed"
    # 半成品切片没有残留
    assert db.query(DocumentChunk).filter(
        DocumentChunk.document_id == doc.id).count() == 0

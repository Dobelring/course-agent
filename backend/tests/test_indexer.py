"""文档切片与向量化入库（indexer）：递归切分 + overlap + BLOB 写入 + 回滚。"""
import pytest

from app.config import settings
from app.models.document import CourseDocument, DocumentChunk
from app.rag.indexer import index_document, split_text
from app.rag.vecfmt import decode_vec


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
    # 每块不超过 目标长度 + overlap 余量（overlap 前缀可能临时超出）
    for c in chunks:
        assert len(c) <= settings.rag_chunk_size + settings.rag_chunk_overlap + 10
    # 内容不丢
    joined = "".join(chunks)
    assert "这是第0行内容" in joined and "这是第199行内容" in joined


def test_split_long_single_line_is_force_split():
    """无分隔符的超长单行按长度强制切分（保证终止）。"""
    text = "字" * 1200
    chunks = split_text(text, size=500, overlap=75)
    assert len(chunks) >= 3
    assert all(len(c) <= 500 for c in chunks)
    # 内容不丢（overlap 会让拼接总量略大于原文）
    assert "".join(chunks).count("字") >= 1200


def test_split_prefers_sentence_boundary():
    """句号优先于逗号：块在句子边界断开，不切断句子内部。"""
    sentences = [f"这是第{i}句话。" for i in range(30)]
    text = "，".join(sentences)  # 全篇只有逗号和句号
    chunks = split_text(text, size=120, overlap=0)
    assert len(chunks) > 1
    for c in chunks:
        # 每块以句号或逗号结尾（句子内部不被切断）
        assert c.endswith("。") or c.endswith("，"), c[-20:]


def test_split_overlap_shares_context_between_adjacent_chunks():
    """相邻块共享 overlap 尾部：边界处的上下文在两块中都可见。"""
    text = "\n".join(f"段落{i}：" + "内容填充" * 30 for i in range(6))
    chunks = split_text(text, size=300, overlap=60)
    assert len(chunks) > 1
    # 第二块的开头来自第一块的尾部（overlap 生效）
    assert chunks[1][:60] in chunks[0]


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
    # 新格式写入 float32 BLOB；旧 JSON 列不再写入
    assert rows[0].embedding is None
    assert decode_vec(rows[0].embedding_blob) == [0.5, 0.5]


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

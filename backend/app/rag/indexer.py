"""文档切片 + 向量化 + 写入 document_chunks。"""
import json
import logging

from sqlalchemy.orm import Session

from app.models.document import CourseDocument, DocumentChunk
from app.rag.embedder import embed_texts

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500  # 每块约 500 字


def split_text(text: str) -> list[str]:
    """按换行段落切分，接近 CHUNK_SIZE 时收束为一块。"""
    chunks: list[str] = []
    current = ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(current) + len(line) > CHUNK_SIZE and current:
            chunks.append(current)
            current = ""
        current += line + "\n"
    if current.strip():
        chunks.append(current)
    return chunks


def index_document(db: Session, doc: CourseDocument, text: str) -> None:
    """切片、向量化并落库；成功后把文档状态置为 ready。

    失败回滚：向量化/写入中途异常时先 rollback 清掉会话里的半成品 chunk，
    再把文档状态置为 failed 落库。调用方的异常处理若直接 commit，
    会把半成品一起提交——回滚必须在本函数内完成。
    """
    try:
        chunks = split_text(text)
        if not chunks:
            doc.status = "failed"
            db.commit()
            return

        vectors = embed_texts(chunks)
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            db.add(
                DocumentChunk(
                    document_id=doc.id,
                    chunk_index=i,
                    content=chunk,
                    embedding=json.dumps(vec),
                )
            )
        doc.status = "ready"
        db.commit()
    except Exception:
        logger.warning("文档索引失败，已回滚半成品切片 doc_id=%s", doc.id, exc_info=True)
        db.rollback()
        doc.status = "failed"
        db.commit()
        raise

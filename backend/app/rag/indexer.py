"""文档切片 + 向量化 + 写入 document_chunks。

切片策略（递归字符切分 + overlap）：
- 分隔符按「段落 → 换行 → 句号 → 逗号 → 空格」逐级降级，尽量在自然边界断开
- 相邻块保留 rag_chunk_overlap 字尾部内容——关键信息落在边界时不再被割裂
- 超长单行（无任何分隔符）按长度强制切分，保证终止
"""
import logging

from sqlalchemy.orm import Session

from app.config import settings
from app.models.document import CourseDocument, DocumentChunk
from app.rag.embedder import embed_texts
from app.rag.vecfmt import encode_vec

logger = logging.getLogger(__name__)

# 递归切分的分隔符优先级：越靠前越保留语义完整性
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " "]


def _split_long_run(run: str, size: int, overlap: int) -> list[str]:
    """无分隔符的超长文本按长度强制切分（带 overlap）。"""
    step = max(1, size - overlap)
    return [run[i:i + size] for i in range(0, len(run), step)]


def _split_keep_sep(text: str, sep: str) -> list[str]:
    """按分隔符切分并把分隔符附着在**前段尾部**——重组无损（句子带着句号走）。"""
    parts = text.split(sep)
    if len(parts) == 1:
        return parts
    pieces = [p + sep for p in parts[:-1]]
    if parts[-1]:
        pieces.append(parts[-1])
    return pieces


def _recursive_split(text: str, size: int, overlap: int, seps: list[str]) -> list[str]:
    """递归字符切分：当前分隔符切出的段仍超长时，降级到下一个分隔符。

    分块上限为 size + overlap（overlap 前缀临时超出目标长度，由下一次
    收束消化）；块内文本由原始片段无损拼接，不在句子内部额外断开。
    """
    if len(text) <= size:
        return [text] if text.strip() else []
    if not seps:
        return _split_long_run(text, size, overlap)

    sep, rest = seps[0], seps[1:]
    pieces = _split_keep_sep(text, sep)
    if len(pieces) == 1:  # 该分隔符切不动 → 降级
        return _recursive_split(text, size, overlap, rest)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not piece.strip():
            continue
        # 单段本身就超长 → 先递归细分再并入
        if len(piece) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_recursive_split(piece, size, overlap, rest))
            continue
        if current and len(current) + len(piece) > size + overlap:
            chunks.append(current)
            # overlap：上一块尾部进入下一块开头，保留跨块上下文
            current = current[-overlap:] if overlap > 0 else ""
        current += piece
    if current.strip():
        chunks.append(current)
    return chunks


def split_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """递归字符切分（段落→句→子句逐级降级）+ 相邻块 overlap。"""
    size = settings.rag_chunk_size if size is None else size
    overlap = settings.rag_chunk_overlap if overlap is None else overlap
    return _recursive_split(text or "", size, max(0, overlap), _SEPARATORS)


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
                    embedding_blob=encode_vec(vec),
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

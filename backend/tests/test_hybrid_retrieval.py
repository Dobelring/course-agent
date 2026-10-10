"""混合检索：分词、BM25、RRF 融合，以及 r26 bad case 的修复验证。"""
import json

from app.config import settings
from app.models.document import CourseDocument, DocumentChunk
from app.rag.retriever import bm25_scores, retrieve, rrf_fuse, tokenize


# ---------- 分词 ----------

def test_tokenize_chinese_bigrams_and_unigrams():
    toks = tokenize("性能计时")
    assert "性能" in toks and "计时" in toks   # 二元组
    assert "性" in toks and "能" in toks        # 单字


def test_tokenize_ascii_lowercased():
    toks = tokenize("BM25 algorithm 与 Transformer")
    assert "bm25" in toks
    assert "algorithm" in toks
    assert "transformer" in toks


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize("。，！") == []  # 纯标点不产出


# ---------- BM25 ----------

def test_bm25_ranks_term_match_higher():
    docs = [tokenize("装饰器是语法糖"), tokenize("二叉树遍历"), tokenize("装饰器日志")]
    scores = bm25_scores(tokenize("装饰器"), docs)
    assert scores[0] > scores[1]
    assert scores[2] > scores[1]  # 「装饰器」出现两次的文档得分更高


def test_bm25_no_match_scores_zero():
    docs = [tokenize("装饰器是语法糖")]
    scores = bm25_scores(tokenize("量子力学"), docs)
    assert scores == [0.0]


def test_bm25_empty_docs():
    assert bm25_scores(["x"], []) == []


# ---------- RRF ----------

def test_rrf_favors_items_ranked_high_in_both_lists():
    fused = rrf_fuse([[1, 2, 3], [2, 1, 3]])
    assert fused[2] > fused[3]      # 两路都靠前的 2 胜出
    assert fused[1] > fused[3]


def test_rrf_single_list_equals_rank_order():
    fused = rrf_fuse([[10, 20, 30]])
    assert fused[10] > fused[20] > fused[30]


# ---------- 集成：r26 bad case 场景 ----------

def _add_chunk(db, doc, index, content, embedding):
    db.add(DocumentChunk(document_id=doc.id, chunk_index=index,
                         content=content, embedding=json.dumps(embedding)))


def test_hybrid_recovers_keyword_hit_missed_by_vector(db, make_users, monkeypatch):
    """向量排错但关键词命中的切片（r26 场景），BM25 一路应把它捞到前列。

    注：纯 RRF 对「完全互补」的排名（如 (1,4) vs (4,1)）必然平票——两路排名
    的多重集相同则 RRF 和相等，这是 RRF 的固有性质。因此场景设计为真实非
    对称情形：目标切片向量第 3（语义未完全跑丢），干扰项 BM25 第 4（关键词
    零命中排末位），目标以 {1,3} 对 {4,1} 的排名组合胜出。
    """
    user = make_users.student1
    docs = {}
    for fname in ["f1.md", "f2.md", "干扰文档.md", "装饰器讲义.md"]:
        doc = CourseDocument(user_id=user.id, filename=fname,
                             file_path=f"/tmp/{fname}", file_type="md", status="ready")
        db.add(doc)
        docs[fname] = doc
    db.flush()

    # 插入顺序 = BM25 零分时的稳定次序：f1/f2 先于干扰项
    _add_chunk(db, docs["f1.md"], 0, "机器学习模型的训练流程", [0.9, 0.43588989, 0.0])
    _add_chunk(db, docs["f2.md"], 0, "数据库连接池的配置方法", [0.5, 0.8660254, 0.0])
    _add_chunk(db, docs["干扰文档.md"], 0, "模型训练与推理部署流程", [1.0, 0.0, 0.0])
    _add_chunk(db, docs["装饰器讲义.md"], 0,
               "装饰器常用于性能计时与缓存的实现方式，不修改源码增强函数。",
               [0.7, 0.7141428, 0.0])
    db.commit()

    # 查询向量与干扰文档完全一致 → 纯向量把干扰项排第一（复现 bad case）
    monkeypatch.setattr("app.rag.retriever.embed_query", lambda q: [1.0, 0.0, 0.0])
    query = "性能计时和缓存怎么实现"

    monkeypatch.setattr(settings, "rag_hybrid", False)
    vector_only = retrieve(db, user.id, query, top_k=1)
    assert vector_only[0]["filename"] == "干扰文档.md"

    # 混合路径：查询词精确命中讲义（BM25 第 1），RRF 应把它捞到第一
    monkeypatch.setattr(settings, "rag_hybrid", True)
    hybrid = retrieve(db, user.id, query, top_k=1)
    assert hybrid[0]["filename"] == "装饰器讲义.md"


def test_hybrid_keeps_permission_isolation(db, seeded, monkeypatch):
    """混合管线不破坏 user_id 隔离。"""
    monkeypatch.setattr("app.rag.retriever.embed_query", lambda q: [1.0, 0.0, 0.0])
    monkeypatch.setattr(settings, "rag_hybrid", True)
    results = retrieve(db, seeded.users.student2.id, "装饰器 语法糖 计时", top_k=5)
    assert results and all(r["filename"] == "u2_notes.md" for r in results)


def test_hybrid_disables_bm25_when_flag_off(db, seeded, monkeypatch):
    monkeypatch.setattr("app.rag.retriever.embed_query", lambda q: [1.0, 0.0, 0.0])
    monkeypatch.setattr(settings, "rag_hybrid", False)
    results = retrieve(db, seeded.users.student1.id, "任意查询", top_k=2)
    assert results  # 纯向量路径仍工作
    assert results[0]["score"] <= 1.0  # score 是余弦值

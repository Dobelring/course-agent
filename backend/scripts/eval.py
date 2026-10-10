r"""评测脚本：检索质量（离线）+ 工具选择/规划质量（需 LLM_API_KEY）。

用法（backend 目录）：
    python scripts/eval.py retrieval      # 检索评测，无需 API Key
    python scripts/eval.py tools          # 工具选择评测，需要 LLM_API_KEY
    python scripts/eval.py plan           # 规划质量评测（分类+拆解），需要 Key
    python scripts/eval.py all            # 全部

检索指标：Hit@1、Hit@3、MRR。工具指标：工具命中率。
规划指标：复杂度分类准确率、计划工具覆盖率、完全覆盖率。
语料：data/eval/corpus/ 下 5 份文档，独立评测用户 eval_user，不污染业务数据。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

# 评测使用独立的 SQLite 库，与开发库（course_agent.db）完全隔离；
# 必须在导入任何 app.* 模块（进而实例化 settings）之前设置。
_EVAL_DB = BACKEND_DIR / "data" / "eval" / "eval.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_EVAL_DB.as_posix()}"

EVAL_SET = BACKEND_DIR / "data" / "eval" / "eval_set.jsonl"
CORPUS_DIR = BACKEND_DIR / "data" / "eval" / "corpus"
EVAL_USER_USERNAME = "eval_user"

# 评测检索固定 top_k=3（与业务默认一致），MRR 取第 1 个命中位置的倒数
TOP_K = 3


def load_cases(etype: str) -> list[dict]:
    cases = [
        json.loads(line)
        for line in EVAL_SET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [c for c in cases if c["type"] == etype]


# ---------------------------------------------------------------------------
# 评测库准备：建表 + 评测用户 + 语料入库
# ---------------------------------------------------------------------------
def prepare_eval_user(db):
    """返回 (eval_user, 是否新建)。语料索引幂等：已 ready 则跳过。"""
    from app.core.security import hash_password
    from app.models.document import CourseDocument
    from app.models.user import User

    user = db.query(User).filter(User.username == EVAL_USER_USERNAME).first()
    if user is None:
        user = User(
            username=EVAL_USER_USERNAME,
            password_hash=hash_password("eval-only"),
            name="评测用户",
            role="student",
            student_no="EVAL001",
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    existing = {
        d.filename
        for d in db.query(CourseDocument)
        .filter(CourseDocument.user_id == user.id, CourseDocument.status == "ready")
        .all()
    }
    return user, existing


def index_corpus(db, user, existing: set[str]) -> int:
    """把语料目录下的文档索引进库，返回本次新索引数量。"""
    from app.models.document import CourseDocument
    from app.rag.indexer import index_document

    indexed = 0
    for path in sorted(CORPUS_DIR.glob("*.md")):
        if path.name in existing:
            continue
        doc = CourseDocument(
            user_id=user.id,
            filename=path.name,
            file_path=str(path),
            file_type="md",
            status="pending",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        index_document(db, doc, path.read_text(encoding="utf-8"))
        indexed += 1
    return indexed


# ---------------------------------------------------------------------------
# 检索评测（离线）
# ---------------------------------------------------------------------------
def run_retrieval_eval(db, user) -> dict:
    from app.rag.retriever import retrieve

    cases = load_cases("retrieval")
    tiers: dict[str, list] = {}
    for c in cases:
        tiers.setdefault(c.get("difficulty", "normal"), []).append(c)

    all_metrics: dict[str, dict] = {}
    print(f"\n== 检索评测（共 {len(cases)} 条查询, top_k={TOP_K}）==")

    grand = {"hits_at_1": 0, "hits_at_3": 0, "rr_sum": 0.0, "n": 0, "misses": []}

    for tier_name, tier_cases in tiers.items():
        label = "常规查询" if tier_name == "normal" else "困难查询（换措辞/无关键词重合）"
        print(f"\n-- {label}（{len(tier_cases)} 条）--")
        print(f"{'ID':<5} {'Hit@1':<6} {'Hit@3':<6} {'RR':<6} 查询")
        print("-" * 78)

        hits_at_1 = hits_at_3 = 0
        rr_sum = 0.0
        misses: list[dict] = []

        for case in tier_cases:
            results = retrieve(db, user.id, case["query"], top_k=TOP_K)
            filenames = [r["filename"] for r in results]
            expected = case["expected_doc"]

            rank = None
            if expected in filenames:
                rank = filenames.index(expected) + 1

            hit1 = int(rank == 1)
            hit3 = int(rank is not None)
            rr = 1.0 / rank if rank else 0.0

            hits_at_1 += hit1
            hits_at_3 += hit3
            rr_sum += rr

            if not hit3:
                misses.append({"id": case["id"], "query": case["query"],
                               "expected": expected, "got": filenames})

            mark = "✓" if hit3 else "✗"
            print(f"{case['id']:<5} {hit1:<6} {hit3:<6} {rr:<6.2f} {mark} {case['query'][:30]}")

        n = len(tier_cases)
        all_metrics[tier_name] = {
            "queries": n,
            "hit_at_1": round(hits_at_1 / n, 4),
            "hit_at_3": round(hits_at_3 / n, 4),
            "mrr": round(rr_sum / n, 4),
            "misses": misses,
        }
        print("-" * 78)
        print(f"Hit@1 = {all_metrics[tier_name]['hit_at_1']:.2%}   "
              f"Hit@3 = {all_metrics[tier_name]['hit_at_3']:.2%}   "
              f"MRR = {all_metrics[tier_name]['mrr']:.4f}")
        if misses:
            print(f"未命中 {len(misses)} 条：")
            for m in misses:
                print(f"  {m['id']} 期望 {m['expected']}  实得 {m['got']}")

        grand["hits_at_1"] += hits_at_1
        grand["hits_at_3"] += hits_at_3
        grand["rr_sum"] += rr_sum
        grand["n"] += n
        grand["misses"].extend(misses)

    n = grand["n"]
    all_metrics["overall"] = {
        "queries": n,
        "hit_at_1": round(grand["hits_at_1"] / n, 4),
        "hit_at_3": round(grand["hits_at_3"] / n, 4),
        "mrr": round(grand["rr_sum"] / n, 4),
        "misses": grand["misses"],
    }
    print("\n== 总体 ==")
    print(f"Hit@1 = {all_metrics['overall']['hit_at_1']:.2%}   "
          f"Hit@3 = {all_metrics['overall']['hit_at_3']:.2%}   "
          f"MRR = {all_metrics['overall']['mrr']:.4f}")
    return all_metrics


# ---------------------------------------------------------------------------
# 工具选择评测（需 LLM）
# ---------------------------------------------------------------------------
async def run_tool_eval() -> dict:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    from app.agent.engine import bind_user, build_agent, build_llm, mcp_client_config
    from app.config import settings
    from app.database import SessionLocal
    from app.models.user import User

    if not settings.llm_api_key:
        print("\n[跳过] 未配置 LLM_API_KEY，工具选择评测需要真实模型。")
        print("       配置后执行：python scripts/eval.py tools")
        return {"skipped": True}

    try:
        build_llm()
    except ValueError as e:
        print(f"\n[跳过] {e}")
        return {"skipped": True}

    cases = load_cases("tool")
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == EVAL_USER_USERNAME).first()
        if user is None:
            print("[错误] 请先运行 retrieval 评测或 `python scripts/eval.py all` 创建评测用户。")
            return {"skipped": True}

        client = MultiServerMCPClient(mcp_client_config())
        raw_tools = await client.get_tools()
        tool_names = {t.name for t in raw_tools}
        tools = bind_user(raw_tools, user.id)
        agent = build_agent(user, tools)

        correct = 0
        details: list[dict] = []
        print(f"\n== 工具选择评测（{len(cases)} 条查询）==\n")

        for case in cases:
            called: list[str] = []
            started: set[str] = set()

            async for event in agent.astream_events(
                {"messages": [("user", case["query"])]}, version="v2"
            ):
                if event.get("event") == "on_tool_start":
                    name = event.get("name") or ""
                    if name in tool_names and name not in started:
                        started.add(name)
                        called.append(name.split("__")[-1])

            expected = case["expected_tools"]
        # 判分：期望工具出现在实际调用序列中即算正确
            hit = any(t in called for t in expected)
            correct += int(hit)
            details.append({"id": case["id"], "query": case["query"],
                            "expected": expected, "called": called, "hit": hit})
            mark = "✓" if hit else "✗"
            print(f"{case['id']:<5} {mark} 期望={','.join(expected):<28} 实调={','.join(called) or '（无）'}")

        n = len(cases)
        metrics = {
            "queries": n,
            "tool_accuracy": round(correct / n, 4),
            "skipped": False,
            "details": details,
        }
        print("-" * 78)
        print(f"工具命中率 = {metrics['tool_accuracy']:.2%}  ({correct}/{n})")
        return metrics
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 规划质量评测（需 LLM）：分类准确率 + 计划工具覆盖率
# ---------------------------------------------------------------------------
async def run_plan_eval() -> dict:
    """对 plan 类用例：① 复杂度分类是否正确 ② 拆解计划是否覆盖期望工具。"""
    from app.agent.engine import build_llm
    from app.agent.planner import classify_complexity, create_plan
    from app.config import settings

    if not settings.llm_api_key:
        print("\n[跳过] 未配置 LLM_API_KEY，规划评测需要真实模型。")
        return {"skipped": True}

    llm = build_llm()
    cases = load_cases("plan")
    tools = ["get_my_schedule", "search_course_documents", "list_my_documents",
             "save_study_plan", "get_upcoming_assignments", "get_my_scores"]

    cls_correct = 0
    coverage_sum = 0.0
    plan_ok = 0
    details: list[dict] = []

    print(f"\n== 规划评测（{len(cases)} 条查询）==\n")
    print(f"{'ID':<5} {'分类':<4} {'工具覆盖':<8} 查询")
    print("-" * 78)

    for case in cases:
        is_complex, reason = await classify_complexity(llm, case["query"])
        cls_hit = int(is_complex == case["expect_complex"])
        cls_correct += cls_hit

        steps = await create_plan(llm, case["query"], tools,
                                  context="", max_steps=settings.agent_max_steps)
        planned_tools = {s.tool_hint for s in steps if s.tool_hint}
        expected = set(case["expected_tools"])
        coverage = len(expected & planned_tools) / len(expected) if expected else 1.0
        coverage_sum += coverage
        if coverage == 1.0:
            plan_ok += 1

        details.append({
            "id": case["id"], "query": case["query"],
            "expect_complex": case["expect_complex"], "got_complex": is_complex,
            "classify_hit": bool(cls_hit),
            "expected_tools": sorted(expected), "planned_tools": sorted(planned_tools),
            "coverage": round(coverage, 4),
            "steps": [s.title for s in steps],
        })
        mark_c = "✓" if cls_hit else "✗"
        mark_t = "✓" if coverage == 1.0 else ("~" if coverage > 0 else "✗")
        print(f"{case['id']:<5} {mark_c:<4} {mark_t} {coverage:.0%}      {case['query'][:32]}")

    n = len(cases)
    metrics = {
        "queries": n,
        "classify_accuracy": round(cls_correct / n, 4),
        "tool_coverage": round(coverage_sum / n, 4),
        "full_coverage_rate": round(plan_ok / n, 4),
        "skipped": False,
        "details": details,
    }
    print("-" * 78)
    print(f"分类准确率 = {metrics['classify_accuracy']:.2%}   "
          f"平均工具覆盖率 = {metrics['tool_coverage']:.2%}   "
          f"完全覆盖 = {metrics['full_coverage_rate']:.2%}")
    return metrics


# ---------------------------------------------------------------------------
def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which not in ("retrieval", "tools", "plan", "all"):
        print(__doc__)
        return 2

    from app.database import Base, SessionLocal, engine
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user, existing = prepare_eval_user(db)
        indexed = index_corpus(db, user, existing)
        if indexed:
            print(f"语料索引：新入库 {indexed} 篇（共 {len(list(CORPUS_DIR.glob('*.md')))} 篇）")
        else:
            print(f"语料索引：{len(list(CORPUS_DIR.glob('*.md')))} 篇已就绪，跳过重复索引")

        results: dict = {}
        if which in ("retrieval", "all"):
            results["retrieval"] = run_retrieval_eval(db, user)
        if which in ("tools", "all"):
            results["tool"] = asyncio.run(run_tool_eval())
        if which in ("plan", "all"):
            results["plan"] = asyncio.run(run_plan_eval())

        out = BACKEND_DIR / "data" / "eval" / "results.json"
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入 {out.relative_to(BACKEND_DIR)}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

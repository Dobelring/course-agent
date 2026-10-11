r"""评测脚本：检索 / 工具选择 / 规划 / 端到端执行 四套件。

用法（backend 目录）：
    python scripts/eval.py retrieval      # 检索评测，无需 API Key
    python scripts/eval.py tools          # 工具选择评测，需要 LLM_API_KEY
    python scripts/eval.py plan           # 规划质量评测（分类+拆解），需要 Key
    python scripts/eval.py execution      # 端到端完成率（真实执行+断言），需要 Key
    python scripts/eval.py all            # 全部

检索指标：Hit@1、Hit@3、MRR。工具指标：工具命中率。
规划指标：复杂度分类准确率、计划工具覆盖率、完全覆盖率。
执行指标：任务完成率（答案内容断言 + DB 状态断言；写操作由自动审批器批准）。
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
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ 自身（eval_report）

from eval_report import merge_results  # noqa: E402

# 评测使用独立的 SQLite 库，与开发库（course_agent.db）完全隔离；
# 必须在导入任何 app.* 模块（进而实例化 settings）之前设置。
_EVAL_DB = BACKEND_DIR / "data" / "eval" / "eval.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_EVAL_DB.as_posix()}"
# 评测连续压测，关闭限流与审批超时等待（审批由脚本自动批准）
os.environ["RATE_LIMIT_CHAT_PER_MIN"] = "0"
os.environ["RATE_LIMIT_LOGIN_PER_MIN"] = "0"

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


def _layer_metrics(details: list[dict], ok_fn) -> dict:
    """按 tag 分层聚合命中率：{layer: {n, hit, rate}}（分层抽样报告）。"""
    layers: dict[str, dict] = {}
    for d in details:
        k = d.get("tag") or "未分类"
        layer = layers.setdefault(k, {"n": 0, "hit": 0})
        layer["n"] += 1
        layer["hit"] += int(bool(ok_fn(d)))
    return {k: {"n": v["n"], "hit": v["hit"], "rate": round(v["hit"] / v["n"], 4)}
            for k, v in sorted(layers.items())}


def _repeat_count() -> int:
    """解析 --repeat N（默认 1）：plan/execution 等 LLM 套件的多轮方差采样。"""
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--repeat" and i + 1 < len(argv):
            return max(1, int(argv[i + 1]))
    return 1


def _aggregate_rounds(rounds: list[dict], keys: list[str]) -> dict:
    """多轮评测聚合：关键指标 mean±std（LLM 非确定性的方差报告）。"""
    import statistics

    agg: dict = {}
    for k in keys:
        vals = [r[k] for r in rounds if not r.get("skipped") and k in r]
        if vals:
            agg[k] = {"mean": round(statistics.fmean(vals), 4),
                      "std": round(statistics.pstdev(vals) if len(vals) > 1 else 0.0, 4)}
    return agg


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
        _ranks: list = []
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

            _ranks.append(rank)
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
        grand.setdefault("case_rows", []).extend(
            {"expected_doc": c["expected_doc"], "hit1": h1, "hit3": h3, "rr": rr}
            for c, h1, h3, rr in zip(
                tier_cases,
                [int(x == 1) for x in _ranks],
                [int(x is not None) for x in _ranks],
                [1.0 / x if x else 0.0 for x in _ranks],
            )
        )

    n = grand["n"]
    by_doc: dict[str, dict] = {}
    for r in grand.get("case_rows", []):
        d = by_doc.setdefault(r["expected_doc"], {"n": 0, "hit1": 0, "hit3": 0, "rr": 0.0})
        d["n"] += 1
        d["hit1"] += r["hit1"]
        d["hit3"] += r["hit3"]
        d["rr"] += r["rr"]
    all_metrics["by_doc"] = {
        k: {"n": v["n"], "hit_at_1": round(v["hit1"] / v["n"], 4),
            "hit_at_3": round(v["hit3"] / v["n"], 4), "mrr": round(v["rr"] / v["n"], 4)}
        for k, v in sorted(by_doc.items())
    }
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
    for k, v in all_metrics["by_doc"].items():
        print(f"  [{k}] n={v['n']} Hit@1 {v['hit_at_1']:.0%} Hit@3 {v['hit_at_3']:.0%} MRR {v['mrr']:.2f}")
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
            error: str | None = None

            # 单例容错：模型循环调用触发 GraphRecursionError 等异常
            # 只判该例失败，不拖垮整个套件
            try:
                async for event in agent.astream_events(
                    {"messages": [("user", case["query"])]}, version="v2"
                ):
                    if event.get("event") == "on_tool_start":
                        name = event.get("name") or ""
                        if name in tool_names and name not in started:
                            started.add(name)
                            called.append(name.split("__")[-1])
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"[:120]

            expected = case["expected_tools"]
        # 判分：期望工具出现在实际调用序列中即算正确（异常=失败）
            hit = error is None and any(t in called for t in expected)
            correct += int(hit)
            details.append({"id": case["id"], "query": case["query"], "tag": case.get("tag"),
                            "expected": expected, "called": called, "hit": hit,
                            "error": error})
            mark = "✗!" if error else ("✓" if hit else "✗")
            note = f" [{error}]" if error else ""
            print(f"{case['id']:<5} {mark} 期望={','.join(expected):<28} "
                  f"实调={','.join(called) or '（无）'}{note}")

        n = len(cases)
        metrics = {
            "queries": n,
            "tool_accuracy": round(correct / n, 4),
            "by_layer": _layer_metrics(details, lambda d: d["hit"]),
            "skipped": False,
            "details": details,
        }
        print("-" * 78)
        print(f"工具命中率 = {metrics['tool_accuracy']:.2%}  ({correct}/{n})")
        for k, v in metrics["by_layer"].items():
            print(f"  [{k}] {v['hit']}/{v['n']} = {v['rate']:.0%}")
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
            "id": case["id"], "query": case["query"], "tag": case.get("tag"),
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
    by_layer = {}
    for d in details:
        layer = by_layer.setdefault(d.get("tag") or "未分类",
                                    {"n": 0, "cls_hit": 0, "cov": 0.0})
        layer["n"] += 1
        layer["cls_hit"] += int(d["classify_hit"])
        layer["cov"] += d["coverage"]
    metrics = {
        "queries": n,
        "classify_accuracy": round(cls_correct / n, 4),
        "tool_coverage": round(coverage_sum / n, 4),
        "full_coverage_rate": round(plan_ok / n, 4),
        "by_layer": {k: {"n": v["n"],
                         "classify_accuracy": round(v["cls_hit"] / v["n"], 4),
                         "avg_coverage": round(v["cov"] / v["n"], 4)}
                     for k, v in sorted(by_layer.items())},
        "skipped": False,
        "details": details,
    }
    print("-" * 78)
    print(f"分类准确率 = {metrics['classify_accuracy']:.2%}   "
          f"平均工具覆盖率 = {metrics['tool_coverage']:.2%}   "
          f"完全覆盖 = {metrics['full_coverage_rate']:.2%}")
    for k, v in metrics["by_layer"].items():
        print(f"  [{k}] n={v['n']} 分类 {v['classify_accuracy']:.0%} 覆盖 {v['avg_coverage']:.0%}")
    return metrics


# ---------------------------------------------------------------------------
# 端到端执行评测（真实 LLM + MCP + DB 断言）
# ---------------------------------------------------------------------------
def seed_execution_data(db, user) -> None:
    """为执行评测播种确定性业务数据（幂等）。

    断言依赖（与 eval_set.jsonl 的 e1-e10 对应）：
    - 成绩：数据结构 92、高数 61（保持最弱）、英语 78、机器学习 85
      → e1「最弱=高数」、e5「92 分」
    - 课表：周四数据结构 → e2；周一高等数学 → e6；周三大学英语 → e7
    - 作业：红黑树实验(+2d) → e5「实验」；极限习题集(+3d)；
      神经网络编程(-1d 已逾期) → e8「逾期」
    """
    from datetime import datetime, timedelta

    from app.models.assignment import Assignment
    from app.models.course import Course, Schedule
    from app.models.score import Score

    if db.query(Score).filter(Score.user_id == user.id).count() >= 4:
        return  # 已播种（幂等；e1-e5 断言依赖的 92/61/周四/实验 均不变）

    cs = Course(code="EVAL-CS201", name="数据结构", teacher="王老师", credit=3)
    ma = Course(code="EVAL-MA101", name="高等数学", teacher="赵老师", credit=5)
    en = Course(code="EVAL-EN101", name="大学英语", teacher="孙老师", credit=2)
    ml = Course(code="EVAL-ML101", name="机器学习", teacher="李老师", credit=3)
    db.add_all([cs, ma, en, ml])
    db.flush()
    db.add(Score(user_id=user.id, course_id=cs.id, score=92))
    db.add(Score(user_id=user.id, course_id=ma.id, score=61))
    db.add(Score(user_id=user.id, course_id=en.id, score=78))
    db.add(Score(user_id=user.id, course_id=ml.id, score=85))
    db.add(Schedule(user_id=user.id, course_id=cs.id, weekday=4,
                    start_period=5, end_period=6, week_start=1, week_end=16,
                    location="教三201"))
    db.add(Schedule(user_id=user.id, course_id=ma.id, weekday=1,
                    start_period=1, end_period=2, week_start=1, week_end=16,
                    location="教一105"))
    db.add(Schedule(user_id=user.id, course_id=en.id, weekday=3,
                    start_period=3, end_period=4, week_start=1, week_end=16,
                    location="文二楼301"))
    db.add(Assignment(user_id=user.id, course_id=cs.id,
                      title="数据结构实验：红黑树实现",
                      due_at=datetime.now() + timedelta(days=2), status="pending"))
    db.add(Assignment(user_id=user.id, course_id=ma.id,
                      title="高等数学作业：极限习题集",
                      due_at=datetime.now() + timedelta(days=3), status="pending"))
    db.add(Assignment(user_id=user.id, course_id=ml.id,
                      title="机器学习作业：神经网络编程",
                      due_at=datetime.now() - timedelta(days=1), status="pending"))
    db.commit()
    print("执行评测数据已播种：成绩/课表/作业（幂等）")


async def _auto_approver(stop: asyncio.Event, user_id: int) -> None:
    """评测自动审批器：批准 eval_user 的写操作审批（模拟用户点击「允许」）。

    与被测请求同处一个事件循环——请求阻塞在审批 Future 上时，本任务仍被调度，
    这正是 ASGI 单循环下人工审批的等价物。
    """
    from app.agent.approval import PENDING, decide

    while not stop.is_set():
        for aid in list(PENDING.keys()):
            try:
                decide(aid, True, user_id)
            except Exception:  # noqa: BLE001 — 评测容错
                pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.05)
        except asyncio.TimeoutError:
            continue


def _parse_chat_sse(body: str) -> dict:
    """解析 /chat 的 SSE 响应体 → {answer, done, errors, approvals}。"""
    answer, done, errors, approvals = [], None, [], []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            ev = json.loads(line[6:])
        except ValueError:
            continue
        t = ev.get("type")
        if t == "text":
            answer.append(ev.get("delta") or "")
        elif t == "done":
            done = ev
        elif t == "error":
            errors.append(ev.get("message"))
        elif t == "approval_required":
            approvals.append(ev)
    return {"answer": "".join(answer), "done": done,
            "errors": errors, "approvals": approvals}


async def run_execution_eval(db, user) -> dict:
    """端到端任务完成率：真实执行 chat → 断言答案内容与 DB 状态。"""
    from app.config import settings

    if not settings.llm_api_key:
        print("\n[跳过] 未配置 LLM_API_KEY，执行评测需要真实模型。")
        return {"skipped": True}

    import time as _time

    import httpx
    from httpx import ASGITransport, AsyncClient

    from app.agent.approval import PENDING
    from app.core.security import create_access_token
    from app.main import app
    from app.models.plan import StudyPlan, StudyPlanItem

    seed_execution_data(db, user)
    # 评测内关闭画像刷新（避免额外 LLM 调用引入不确定性与延迟）
    settings.memory_refresh_every = 10**9

    def _plan_count() -> int:
        return db.query(StudyPlan).filter(StudyPlan.user_id == user.id).count()

    def _latest_plan_items() -> int:
        latest = (
            db.query(StudyPlan)
            .filter(StudyPlan.user_id == user.id)
            .order_by(StudyPlan.id.desc())
            .first()
        )
        if latest is None:
            return 0
        return (
            db.query(StudyPlanItem)
            .filter(StudyPlanItem.plan_id == latest.id)
            .count()
        )

    token = create_access_token(user.id, user.username, user.role)
    headers = {"Authorization": f"Bearer {token}"}
    cases = load_cases("execution")

    stop = asyncio.Event()
    approver_task = asyncio.create_task(_auto_approver(stop, user.id))
    details: list[dict] = []
    passed_total = 0
    total_usage = {"input_tokens": 0, "output_tokens": 0, "cost_cny": 0.0}

    print(f"\n== 端到端执行评测（{len(cases)} 条，真实 LLM + MCP；写操作自动批准）==\n")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://eval.local",
                           timeout=httpx.Timeout(600.0)) as client:
        for case in cases:
            mode = case.get("mode", "direct")
            plans_before = _plan_count()
            started = _time.perf_counter()
            checks: list[dict] = []
            try:
                resp = await asyncio.wait_for(
                    client.post(
                        "/api/chat",
                        headers=headers,
                        json={"message": case["query"], "mode": mode},
                    ),
                    timeout=600,
                )
            except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
                checks.append({"name": "请求成功", "ok": False,
                               "note": f"{type(exc).__name__}: {exc}"})
                details.append({"id": case["id"], "query": case["query"],
                                "mode": mode, "tag": case.get("tag"),
                                "passed": False, "checks": checks})
                print(f"{case['id']:<4} ✗ 请求失败 {checks[0]['note'][:80]}")
                continue

            parsed = _parse_chat_sse(resp.text)
            answer, done = parsed["answer"], parsed["done"]
            latency_ms = int((_time.perf_counter() - started) * 1000)

            # ---- 通用断言 ----
            checks.append({"name": "无错误事件", "ok": not parsed["errors"],
                           "note": "; ".join(parsed["errors"])[:120]})
            checks.append({"name": "收到 done 事件", "ok": done is not None, "note": ""})
            if done and done.get("usage"):
                for k in total_usage:
                    total_usage[k] += (done["usage"].get(k) or 0)

            # ---- 内容断言 ----
            for kw in case.get("must_contain", []):
                checks.append({"name": f"答案含「{kw}」", "ok": kw in answer,
                               "note": (answer[:100] if not kw in answer else "")})

            # ---- DB 状态断言 ----
            if case.get("check") == "plan_saved":
                db.expire_all()
                plans_after = _plan_count()
                created = plans_after > plans_before
                checks.append({"name": "新增学习计划", "ok": created,
                               "note": f"{plans_before} → {plans_after}"})
                min_items = case.get("min_items", 1)
                items = _latest_plan_items()
                checks.append({"name": f"计划条目 ≥ {min_items}", "ok": items >= min_items,
                               "note": f"items={items}"})

            passed = all(c["ok"] for c in checks)
            passed_total += int(passed)
            details.append({
                "id": case["id"], "query": case["query"], "mode": mode,
                "tag": case.get("tag"),
                "passed": passed, "checks": checks,
                "approvals": len(parsed["approvals"]),
                "latency_ms": latency_ms,
                "answer_head": answer[:120],
                "usage": done.get("usage") if done else None,
            })
            mark = "✓" if passed else "✗"
            failed = [c["name"] for c in checks if not c["ok"]]
            print(f"{case['id']:<4} {mark} {case['query'][:34]}"
                  + (f"  [{', '.join(failed)}]" if failed else "")
                  + f"  ({latency_ms / 1000:.1f}s)")

    stop.set()
    await approver_task
    still_pending = len(PENDING)
    if still_pending:
        print(f"警告：退出时仍有 {still_pending} 个未决审批")

    n = len(cases)
    # 人工干预率（来自 approval_logs 落库）：触发审批的轮次占比 + 决策分布
    from sqlalchemy import func as _func

    from app.models.approval import ApprovalLog

    intervened = sum(1 for d in details if d.get("approvals"))
    decision_rows = (
        db.query(ApprovalLog.decision, _func.count(ApprovalLog.id))
        .group_by(ApprovalLog.decision)
        .all()
    )
    decisions = {(k or "pending"): v for k, v in decision_rows}
    metrics = {
        "queries": n,
        "passed": passed_total,
        "completion_rate": round(passed_total / n, 4) if n else 0.0,
        "by_layer": _layer_metrics(details, lambda d: d["passed"]),
        "intervention_rate": round(intervened / n, 4) if n else 0.0,
        "intervened_queries": intervened,
        "approval_decisions": decisions,
        "skipped": False,
        "eval_usage": total_usage,
        "details": details,
    }
    print("-" * 78)
    print(f"任务完成率 = {metrics['completion_rate']:.2%} ({passed_total}/{n})"
          f"  评测自身消耗 {total_usage['input_tokens']}+{total_usage['output_tokens']} tokens"
          f" / ¥{total_usage['cost_cny']:.4f}")
    print(f"人工干预率 = {metrics['intervention_rate']:.2%}（{intervened}/{n} 轮触发审批），"
          f"决策分布 {decisions}")
    for k, v in metrics["by_layer"].items():
        print(f"  [{k}] {v['hit']}/{v['n']} = {v['rate']:.0%}")
    return metrics


# ---------------------------------------------------------------------------
def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which not in ("retrieval", "tools", "plan", "execution", "all"):
        print(__doc__)
        return 2

    from app.database import Base, SessionLocal, engine
    from app import models  # noqa: F401
    from app.main import _ensure_columns

    Base.metadata.create_all(bind=engine)
    _ensure_columns()  # eval.db 可能是旧库：补 plan/token 等后加的列（create_all 不改已有表）
    db = SessionLocal()
    try:
        user, existing = prepare_eval_user(db)
        indexed = index_corpus(db, user, existing)
        if indexed:
            print(f"语料索引：新入库 {indexed} 篇（共 {len(list(CORPUS_DIR.glob('*.md')))} 篇）")
        else:
            print(f"语料索引：{len(list(CORPUS_DIR.glob('*.md')))} 篇已就绪，跳过重复索引")

        results: dict = {}
        repeat = _repeat_count()
        if which in ("retrieval", "all"):
            results["retrieval"] = run_retrieval_eval(db, user)
        if which in ("tools", "all"):
            rounds = [asyncio.run(run_tool_eval()) for _ in range(repeat)]
            tool_m = dict(rounds[-1])
            tool_m.update({"repeat": repeat,
                           "mean_std": _aggregate_rounds(rounds, ["tool_accuracy"])})
            results["tool"] = tool_m
            if repeat > 1:
                print(f"\n[tool 方差] {repeat} 轮：{tool_m['mean_std']}")
        if which in ("plan", "all"):
            rounds = [asyncio.run(run_plan_eval()) for _ in range(repeat)]
            plan_m = dict(rounds[-1])
            keys = ["classify_accuracy", "tool_coverage", "full_coverage_rate"]
            plan_m.update({"repeat": repeat,
                           "mean_std": _aggregate_rounds(rounds, keys)})
            results["plan"] = plan_m
            if repeat > 1:
                print(f"\n[plan 方差] {repeat} 轮：{plan_m['mean_std']}")
        if which in ("execution", "all"):
            rounds = [asyncio.run(run_execution_eval(db, user)) for _ in range(repeat)]
            exec_m = dict(rounds[-1])
            keys = ["completion_rate", "intervention_rate"]
            exec_m.update({"repeat": repeat,
                           "mean_std": _aggregate_rounds(rounds, keys)})
            results["execution"] = exec_m
            if repeat > 1:
                print(f"\n[execution 方差] {repeat} 轮：{exec_m['mean_std']}")

        # 按 suite 合并写入：单跑一个 suite 不会覆盖其他 suite 的历史结果
        out = BACKEND_DIR / "data" / "eval" / "results.json"
        merge_results(out, results)
        print(f"\n结果已合并写入 {out.relative_to(BACKEND_DIR)}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

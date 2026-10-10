"""并行子任务：依赖解析、就绪批次并发、阻塞跳过、重规划重编号。"""
import asyncio
import json
from types import SimpleNamespace

from app.agent.plan_executor import execute_plan
from app.agent.planner import Plan, PlanStep, parse_plan, renumber_steps


class FakeLLM:
    def __init__(self, *outputs):
        self.outputs = list(outputs)

    async def ainvoke(self, messages, **kw):
        out = self.outputs.pop(0) if self.outputs else json.dumps(
            {"steps": [{"title": "替代", "goal": "g", "depends_on": []}]})
        return SimpleNamespace(content=out)


def _collector():
    events: list[dict] = []

    async def emit(evt: dict):
        events.append(evt)

    return events, emit


def run(coro):
    return asyncio.run(coro)


# ---------- parse_plan 依赖解析 ----------

def test_parse_plan_defaults_to_sequential():
    """未提供 depends_on → 默认串行（依赖上一步，保持旧行为）。"""
    steps = parse_plan({"steps": [
        {"title": "a", "goal": "ga"},
        {"title": "b", "goal": "gb"},
        {"title": "c", "goal": "gc"},
    ]})
    assert steps[0].depends_on == []
    assert steps[1].depends_on == [1]
    assert steps[2].depends_on == [2]


def test_parse_plan_explicit_empty_means_parallel():
    steps = parse_plan({"steps": [
        {"title": "a", "goal": "ga", "depends_on": []},
        {"title": "b", "goal": "gb", "depends_on": []},
        {"title": "c", "goal": "gc", "depends_on": [1, 2]},
    ]})
    assert steps[0].depends_on == []
    assert steps[1].depends_on == []
    assert steps[2].depends_on == [1, 2]


def test_parse_plan_drops_forward_and_invalid_refs():
    """前向引用/自身引用/非法值全部剔除 → 结构上保证无环（DAG）。"""
    steps = parse_plan({"steps": [
        {"title": "a", "goal": "ga", "depends_on": [1, 2]},   # 自身+前向，全无效
        {"title": "b", "goal": "gb", "depends_on": ["x", 9]},  # 非法字符串+超界，全无效
        {"title": "c", "goal": "gc", "depends_on": [2, "2", 1]},  # 合法+去重
    ]})
    assert steps[0].depends_on == []
    assert steps[1].depends_on == []
    assert steps[2].depends_on == [2, 1]   # "2"→2 去重后保留 [2,1]


def test_renumber_shifts_ids_and_deps():
    steps = parse_plan({"steps": [
        {"title": "a", "goal": "ga", "depends_on": []},
        {"title": "b", "goal": "gb", "depends_on": [1]},
    ]})
    renumber_steps(steps, max_existing_id=5)
    assert [s.id for s in steps] == [6, 7]
    assert steps[1].depends_on == [6]      # 依赖同步平移


# ---------- 并发执行 ----------

def test_independent_steps_run_concurrently():
    """两个无依赖步骤必须并发：用事件门保证——若串行则死锁。"""
    events, emit = _collector()
    both_started = asyncio.Event()
    started_count = 0

    async def run_step(step: PlanStep) -> str:
        nonlocal started_count
        started_count += 1
        if started_count == 2:
            both_started.set()
        # 只有并发才会两个都到这；串行则第二个永远等不到 → 超时失败
        await asyncio.wait_for(both_started.wait(), timeout=1.0)
        return f"{step.title}完成"

    plan = Plan(steps=[
        PlanStep(id=1, title="查课表", goal="g", depends_on=[]),
        PlanStep(id=2, title="查成绩", goal="g", depends_on=[]),
    ])
    result = run(execute_plan(FakeLLM(), "目标", plan, run_step, emit=emit))
    assert result.success is True
    assert result.steps_done == 2
    # 两个 step_start 都先于 step_end（事件交织）
    types = [(e["type"], e.get("id")) for e in events if e["type"].startswith("step")]
    start_ids = [i for t, i in types if t == "step_start"]
    end_ids = [i for t, i in types if t == "step_end"]
    assert len(start_ids) == 2 and len(end_ids) == 2


def test_dependent_steps_run_in_order():
    """有依赖的步骤必须等前置完成。"""
    order = []

    async def run_step(step: PlanStep) -> str:
        order.append(f"start:{step.title}")
        await asyncio.sleep(0.02)
        order.append(f"end:{step.title}")
        return "ok"

    plan = Plan(steps=[
        PlanStep(id=1, title="查数据", goal="g", depends_on=[]),
        PlanStep(id=2, title="分析", goal="g", depends_on=[1]),
        PlanStep(id=3, title="写报告", goal="g", depends_on=[2]),
    ])
    result = run(execute_plan(FakeLLM(), "目标", plan, run_step,
                              emit=lambda e: asyncio.sleep(0)))
    assert result.success is True
    assert order == ["start:查数据", "end:查数据",
                     "start:分析", "end:分析",
                     "start:写报告", "end:写报告"]


def test_fan_out_fan_in_dag():
    """扇出（1→2,3）后扇入（2,3→4）：2/3 并发，4 等两者。"""
    events, emit = _collector()
    concurrent_at_23 = asyncio.Event()
    count = {"n": 0}

    async def run_step(step: PlanStep) -> str:
        if step.title in ("取成绩", "取作业"):
            count["n"] += 1
            if count["n"] == 2:
                concurrent_at_23.set()
            await asyncio.wait_for(concurrent_at_23.wait(), timeout=1.0)
        elif step.title == "汇总":
            # 汇总必须在 2、3 都完成后才被调度（依赖保证，无需等待）
            pass
        return f"{step.title}完成"

    plan = Plan(steps=[
        PlanStep(id=1, title="取课表", goal="g", depends_on=[]),
        PlanStep(id=2, title="取成绩", goal="g", depends_on=[1]),
        PlanStep(id=3, title="取作业", goal="g", depends_on=[1]),
        PlanStep(id=4, title="汇总", goal="g", depends_on=[2, 3]),
    ])
    result = run(execute_plan(FakeLLM(), "目标", plan, run_step, emit=emit))
    assert result.success is True
    assert result.steps_done == 4
    # 汇总是最后完成的
    end_order = [e["id"] for e in events if e["type"] == "step_end"]
    assert end_order[-1] == 4


def test_dependency_blocked_when_prerequisite_denied():
    """前置被拒 → 依赖它的步骤标记 skipped（前置步骤未完成），不挂死。"""
    from app.agent.approval import ApprovalDenied

    ran = []

    async def run_step(step: PlanStep) -> str:
        ran.append(step.title)
        if step.title == "保存计划":
            raise ApprovalDenied("用户拒绝执行该写操作")
        return "ok"

    plan = Plan(steps=[
        PlanStep(id=1, title="查课表", goal="g", depends_on=[]),
        PlanStep(id=2, title="保存计划", goal="g", depends_on=[1]),
        PlanStep(id=3, title="基于保存结果发通知", goal="g", depends_on=[2]),
    ])
    result = run(execute_plan(FakeLLM(), "目标", plan, run_step,
                              step_retries=0, emit=lambda e: asyncio.sleep(0)))
    statuses = {s.title: s.status for s in plan.steps}
    assert statuses["保存计划"] == "skipped"
    assert statuses["基于保存结果发通知"] == "skipped"
    assert "基于保存结果发通知" not in ran   # 绝不执行
    assert result.success is True           # 查课表完成（done_count>0）


def test_failure_blocks_dependents_but_not_independent():
    """步骤失败 → 依赖者阻塞跳过；独立步骤继续执行（重规划接管失败链）。"""
    from app.agent.planner import replan as _  # noqa: F401 确保模块加载

    events, emit = _collector()
    ran = []

    async def run_step(step: PlanStep) -> str:
        ran.append(step.title)
        if step.title == "查成绩":
            raise RuntimeError("工具故障")
        return f"{step.title}完成"

    llm = FakeLLM(json.dumps({"steps": [
        {"title": "改用查课表推断", "goal": "g2", "depends_on": []}]}))
    plan = Plan(steps=[
        PlanStep(id=1, title="查成绩", goal="g", depends_on=[]),
        PlanStep(id=2, title="基于成绩分析", goal="g", depends_on=[1]),
        PlanStep(id=3, title="查课表", goal="g", depends_on=[]),
    ])
    result = run(execute_plan(llm, "目标", plan, run_step,
                              step_retries=0, max_replans=2, emit=emit))

    assert "查课表" in ran                       # 独立分支不受影响
    assert "基于成绩分析" not in ran             # 依赖失败分支不执行
    types = [e["type"] for e in events]
    assert "replan" in types                    # 失败触发重规划
    # 重规划后新步骤 id 不与历史冲突
    ids = [s.id for s in result.plan.steps]
    assert len(ids) == len(set(ids))
    assert result.success is True


def test_batch_failure_triggers_single_replan():
    """同批次两个失败只触发一次重规划（批次级而非逐步）。"""
    events, emit = _collector()
    failed_titles = {"a", "b"}

    async def run_step(step: PlanStep) -> str:
        if step.title in failed_titles:
            raise RuntimeError("全部失败")
        return f"{step.title}完成"

    llm = FakeLLM(json.dumps({"steps": [
        {"title": "s1", "goal": "g", "depends_on": []},
        {"title": "s2", "goal": "g", "depends_on": []}]}))
    plan = Plan(steps=[
        PlanStep(id=1, title="a", goal="g", depends_on=[]),
        PlanStep(id=2, title="b", goal="g", depends_on=[]),
    ])
    result = run(execute_plan(llm, "目标", plan, run_step,
                              step_retries=0, max_replans=5, emit=emit))
    replans = [e for e in events if e["type"] == "replan"]
    assert len(replans) == 1                    # 两失败合并为一次批次重规划
    assert plan.revised == 1
    assert result.success is True               # 重规划后的步骤全部成功
    assert result.steps_done == 2

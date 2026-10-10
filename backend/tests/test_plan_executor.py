"""执行器：成功路径、单步重试、失败重规划、降级与全败回退。"""
import asyncio
import json
from types import SimpleNamespace

from app.agent.plan_executor import ExecutionResult, execute_plan
from app.agent.planner import Plan, PlanStep


class FakeLLM:
    """重规划用假 LLM：返回预设的修订步骤 JSON。"""

    def __init__(self, *outputs: str):
        self.outputs = list(outputs)

    async def ainvoke(self, messages, **kw):
        out = self.outputs.pop(0) if self.outputs else (
            json.dumps({"steps": [{"title": "替代步骤", "goal": "改用其他方式", "tool_hint": None}]})
        )
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(content=out)


def _plan(*titles: str) -> Plan:
    return Plan(steps=[PlanStep(id=i, title=t, goal=f"goal-{t}") for i, t in enumerate(titles, 1)])


def run(coro, **kw):
    return asyncio.run(coro)


def _collector():
    events: list[dict] = []

    async def emit(e: dict):
        events.append(e)

    return events, emit


# ---------- 成功路径 ----------

def test_happy_path_all_steps_done_and_synthesized():
    events, emit = _collector()
    llm = FakeLLM("综合后的最终答案")

    async def run_step(step):
        return f"{step.title} 的结果"

    result = run(execute_plan(llm, "目标", _plan("查课表", "查成绩"), run_step,
                              emit=emit))
    assert isinstance(result, ExecutionResult)
    assert result.success is True
    assert result.answer == "综合后的最终答案"
    assert result.steps_done == 2
    assert result.steps_failed == 0
    # 事件序列：plan → step_start → step_end → step_start → step_end
    types = [e["type"] for e in events]
    assert types[0] == "plan"
    assert types.count("step_start") == 2
    assert types.count("step_end") == 2
    assert all(e["status"] == "done" for e in events if e["type"] == "step_end")


# ---------- 单步重试 ----------

def test_step_retries_transient_failure():
    events, emit = _collector()
    attempts: dict[int, int] = {}

    async def flaky_step(step):
        attempts[step.id] = attempts.get(step.id, 0) + 1
        if attempts[step.id] < 2:
            raise ConnectionError("网络抖动")
        return "重试后成功"

    result = run(execute_plan(FakeLLM("答案"), "目标", _plan("步骤一"), flaky_step,
                              step_retries=2, emit=emit))
    assert result.success is True
    assert attempts[1] == 2
    # 至少发出过一次 step_retry（第一次失败触发）
    assert any(e["type"] == "step_retry" for e in events)


# ---------- 失败重规划 ----------

def test_failed_step_triggers_replan():
    events, emit = _collector()
    # 重规划输出需保留未受影响的后续步骤（符合 REPLAN_PROMPT 要求）
    llm = FakeLLM(json.dumps({"steps": [
        {"title": "替代查成绩", "goal": "换 get_my_scores", "tool_hint": "get_my_scores"},
        {"title": "写总结", "goal": "综合产出", "tool_hint": None},
    ]}))
    calls = []

    async def run_step(step):
        calls.append(step.title)
        if step.title == "查成绩":
            raise RuntimeError("工具不存在")
        return f"{step.title} 完成"

    result = run(execute_plan(llm, "分析成绩", _plan("查成绩", "写总结"), run_step,
                              step_retries=0, max_replans=2, emit=emit,
                              tools=["get_my_scores"]))
    types = [e["type"] for e in events]
    assert "replan" in types
    # 第一步失败 → 重规划；替代步骤「替代查成绩」+ 保留的「写总结」继续执行
    assert "替代查成绩" in calls
    assert "写总结" in calls
    assert result.success is True
    assert result.plan.revised == 1
    # 失败步骤保留作记录
    failed = [s for s in result.plan.steps if s.status == "failed"]
    assert failed and failed[0].error


def test_replan_context_feeds_failure_error():
    """重规划提示必须带上失败原因（在 replan 单测已验，这里验执行器传入的 plan 状态正确）。"""
    llm = FakeLLM()  # 输出默认替代步骤
    failed_seen = {}

    async def run_step(step):
        raise RuntimeError("boom")

    result = run(execute_plan(llm, "目标", _plan("必败步骤"), run_step,
                              step_retries=0, max_replans=0))
    assert result.success is False
    assert result.error is not None and "boom" in result.error


# ---------- 降级：重规划耗尽 ----------

def test_degrades_when_replans_exhausted():
    events, emit = _collector()
    llm = FakeLLM(
        json.dumps({"steps": [{"title": "替代1", "goal": "g", "tool_hint": None}]}),
        json.dumps({"steps": [{"title": "替代2", "goal": "g", "tool_hint": None}]}),
    )

    async def always_fail(step):
        raise RuntimeError("持续失败")

    result = run(execute_plan(llm, "目标", _plan("s1", "s2"), always_fail,
                              step_retries=0, max_replans=2, emit=emit))
    assert result.success is False
    assert result.degraded is True
    # 重规划耗尽事件 + 无步骤成功
    assert any(e["type"] == "degraded" for e in events)
    assert result.steps_done == 0


def test_partial_success_still_synthesizes():
    """第一步成功、第二步反复失败 → 部分成功也要产出综合答案。"""
    llm = FakeLLM(
        json.dumps({"steps": [{"title": "简化分析", "goal": "g", "tool_hint": None}]}),
        "基于已完成部分的答案",
    )

    async def run_step(step):
        if step.title == "深度分析":
            raise RuntimeError("坏步骤")
        return "查到了课表"

    result = run(execute_plan(llm, "查课表并分析", _plan("查课表", "深度分析"), run_step,
                              step_retries=0, max_replans=1))
    assert result.success is True
    assert result.answer == "基于已完成部分的答案"
    assert result.degraded is True          # 重规划次数耗尽
    assert result.steps_done == 2           # 查课表 + 简化分析
    assert result.steps_failed == 1         # 深度分析失败留痕


# ---------- 综合回退 ----------

def test_synthesis_failure_falls_back_to_concatenated_results():
    """综合 LLM 失败 → 回退为拼接步骤结果（保证有产出）。"""
    llm = FakeLLM(ValueError("综合超时"), ValueError("再超时"))

    async def run_step(step):
        return f"结果-{step.title}"

    result = run(execute_plan(llm, "目标", _plan("A"), run_step, step_retries=0))
    assert result.success is True
    assert "结果-A" in result.answer


def test_empty_step_output_counts_as_failure():
    async def empty_step(step):
        return "   "

    result = run(execute_plan(FakeLLM("x"), "目标", _plan("s"), empty_step, step_retries=0))
    assert result.success is False

"""规划器：复杂度分类、JSON 容错解析、计划构建、失败重规划。"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from app.agent.planner import (
    Plan,
    PlanStep,
    classify_complexity,
    create_plan,
    parse_plan,
    replan,
)


class FakeLLM:
    """按调用次序返回预设输出的假 LLM。"""

    def __init__(self, *outputs: str):
        self.outputs = list(outputs)
        self.calls: list[str] = []

    async def ainvoke(self, messages, **kw):
        self.calls.append(messages[-1]["content"])
        out = self.outputs.pop(0) if self.outputs else "{}"
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(content=out)


def run(coro):
    return asyncio.run(coro)


# ---------- 分类 ----------

def test_classify_complex():
    llm = FakeLLM('{"complexity": "complex", "reason": "需要多源汇总"}')
    is_complex, reason = run(classify_complexity(llm, "对比我所有科目成绩并制定计划"))
    assert is_complex is True
    assert reason == "需要多源汇总"


def test_classify_simple():
    llm = FakeLLM('{"complexity": "simple", "reason": "单步查询"}')
    is_complex, _ = run(classify_complexity(llm, "我今天有什么课"))
    assert is_complex is False


def test_classify_failure_defaults_to_simple():
    """分类失败不能阻断对话——保守走单循环（两次尝试均失败）。"""
    llm = FakeLLM(ValueError("网络错误"), ValueError("网络错误"))
    is_complex, reason = run(classify_complexity(llm, "任意输入"))
    assert is_complex is False
    assert reason == "classify_failed"


# ---------- JSON 容错 ----------

def test_parse_plan_from_fenced_json():
    data = {"steps": [{"title": "查课表", "goal": "查询用户课表", "tool_hint": "get_my_schedule"}]}
    steps = parse_plan(json.loads(json.dumps(data)))
    assert len(steps) == 1
    assert steps[0].id == 1
    assert steps[0].tool_hint == "get_my_schedule"


def test_parse_plan_rejects_empty_or_malformed():
    with pytest.raises(ValueError):
        parse_plan({"steps": []})
    with pytest.raises(ValueError):
        parse_plan({"steps": [{"title": "缺goal"}]})
    with pytest.raises(ValueError):
        parse_plan({"no_steps": True})


def test_parse_plan_null_tool_hint():
    steps = parse_plan({"steps": [{"title": "t", "goal": "g", "tool_hint": None}]})
    assert steps[0].tool_hint is None


# ---------- create_plan：含一次修复 ----------

def test_create_plan_repairs_non_json_output():
    """模型第一次输出带废话导致解析失败 → 自动追加修复提示重试。"""
    good = json.dumps({"steps": [{"title": "查成绩", "goal": "查询我的成绩", "tool_hint": "get_my_scores"}]})
    llm = FakeLLM("好的，我来规划：这不是JSON", good)
    steps = run(create_plan(llm, "帮我分析成绩", ["get_my_scores"], "", max_steps=5))
    assert len(steps) == 1
    assert len(llm.calls) == 2                     # 确实触发了修复调用
    assert "只输出合法 JSON" in llm.calls[1]


def test_create_plan_raises_after_repair_fails():
    llm = FakeLLM("还是不合法", "依旧不合法")
    with pytest.raises(ValueError):
        run(create_plan(llm, "目标", [], "", max_steps=5))


def test_create_plan_caps_at_max_steps():
    many = {"steps": [
        {"title": f"s{i}", "goal": f"g{i}", "tool_hint": None} for i in range(8)
    ]}
    llm = FakeLLM(json.dumps(many))
    steps = run(create_plan(llm, "目标", [], "", max_steps=5))
    assert len(steps) == 5


# ---------- replan：失败上下文喂回去 ----------

def _seed_plan() -> Plan:
    return Plan(steps=[
        PlanStep(id=1, title="查课表", goal="g1", status="done", result="周一有高数"),
        PlanStep(id=2, title="查成绩", goal="g2", status="failed", error="工具不存在"),
        PlanStep(id=3, title="写总结", goal="g3", status="pending"),
    ])


def test_replan_includes_failure_context():
    new_steps = {"steps": [{"title": "改用列表成绩", "goal": "通过其他接口取成绩", "tool_hint": "get_my_scores"}]}
    llm = FakeLLM(json.dumps(new_steps))
    plan = _seed_plan()
    steps = run(replan(llm, "分析成绩", plan, ["get_my_scores"], max_steps=5))

    prompt = llm.calls[0]
    assert "工具不存在" in prompt          # 失败原因回传
    assert "周一有高数" in prompt           # 已完成结果也带上
    assert "[3] 写总结" in prompt           # 未完成步骤保留供参考
    assert len(steps) == 1
    assert steps[0].id == 1                # 新计划重新编号


def test_replan_failed_step_status_flow():
    plan = _seed_plan()
    assert [s.title for s in plan.pending] == ["写总结"]
    assert [s.error for s in plan.failed] == ["工具不存在"]
    assert plan.steps[0].status == "done"

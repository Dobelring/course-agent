"""写操作审批：注册表、超时拒绝、归属校验、工具门控、执行器拒绝语义。"""
import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from app.agent.approval import (
    PENDING,
    ApprovalContext,
    ApprovalDenied,
    ApprovalGateTool,
    decide,
    gate_write_tools,
)
from app.agent.plan_executor import execute_plan
from app.agent.planner import Plan, PlanStep


def _collector():
    events: list[dict] = []

    async def emit(evt: dict):
        events.append(evt)

    return events, emit


# ---------- 审批上下文 ----------

def test_approval_granted_when_decided():
    events, emit = _collector()

    async def flow():
        ctx = ApprovalContext(emit, timeout=5, user_id=1)

        async def request():
            return await ctx.request("保存学习计划", {"title": "期末计划"})

        task = asyncio.create_task(request())
        # 等待审批注册
        for _ in range(100):
            if PENDING:
                break
            await asyncio.sleep(0.01)
        approval_id = next(iter(PENDING))
        assert decide(approval_id, True, user_id=1) is None
        return await task

    approved = asyncio.run(flow())
    assert approved is True
    types = [e["type"] for e in events]
    assert types == ["approval_required", "approval_resolved"]
    assert events[0]["tool"] == "保存学习计划"
    assert events[0]["approval_id"] not in PENDING  # 用完即清
    assert events[1]["approved"] is True


def test_approval_timeout_counts_as_denial():
    events, emit = _collector()

    async def flow():
        ctx = ApprovalContext(emit, timeout=0.1, user_id=1)
        result = await ctx.request("save", {})
        return ctx, result

    ctx, approved = asyncio.run(flow())
    assert approved is False
    assert events[1]["reason"] == "超时未决，按拒绝处理"
    assert ctx.denials_for(None) == 1        # 超时也计入拒绝归属


def test_denial_attributed_to_step():
    async def flow():
        ctx = ApprovalContext(lambda e: asyncio.sleep(0), timeout=0.05, user_id=1)
        from app.agent.approval import current_step_id
        token = current_step_id.set(7)
        try:
            await ctx.request("save", {})
        finally:
            current_step_id.reset(token)
        return ctx

    ctx = asyncio.run(flow())
    assert ctx.denials_for(7) == 1
    assert ctx.denials_for(None) == 0


def test_decide_rejects_foreign_owner():
    async def flow():
        ctx = ApprovalContext(lambda e: asyncio.sleep(0), timeout=5, user_id=1)
        task = asyncio.create_task(ctx.request("save", {}))
        for _ in range(100):
            if PENDING:
                break
            await asyncio.sleep(0.01)
        approval_id = next(iter(PENDING))
        err = decide(approval_id, True, user_id=999)  # 别人
        assert err == "无权处理他人的审批"
        # 真正的 owner 拒绝，解锁等待
        assert decide(approval_id, False, user_id=1) is None
        await task

    asyncio.run(flow())


def test_decide_unknown_id():
    assert decide("no-such-id", True, user_id=1) == "审批不存在或已过期"


# ---------- 工具门控 ----------

class _SaveArgs(BaseModel):
    user_id: int
    title: str = "计划"


class SaveTool(BaseTool):
    name: str = "save_study_plan"
    description: str = "保存计划"
    args_schema: type[BaseModel] = _SaveArgs
    executed: bool = False

    def _run(self, user_id: int, title: str = "") -> str:
        type(self).executed = True
        return f"saved:{user_id}:{title}"


def test_gate_wraps_only_configured_tools():
    events, emit = _collector()
    ctx = ApprovalContext(emit, timeout=5, user_id=1)
    tools = gate_write_tools([SaveTool()], ctx, {"save_study_plan"})
    assert isinstance(tools[0], ApprovalGateTool)


def test_gate_skips_unlisted_tools():
    class QueryTool(BaseTool):
        name: str = "get_my_schedule"
        description: str = "查课表"
        def _run(self) -> str:
            return "[]"
    events, emit = _collector()
    ctx = ApprovalContext(emit, timeout=5, user_id=1)
    tools = gate_write_tools([QueryTool()], ctx, {"save_study_plan"})
    assert not isinstance(tools[0], ApprovalGateTool)


def test_gated_tool_blocked_on_denial():
    """拒绝时返回拒绝文案给模型，底层工具绝不执行。"""
    SaveTool.executed = False
    events, emit = _collector()

    async def flow():
        ctx = ApprovalContext(emit, timeout=0.1, user_id=1)  # 快速超时=拒绝
        gated = gate_write_tools([SaveTool()], ctx, {"save_study_plan"})[0]
        return await gated.ainvoke({"user_id": 1, "title": "偷跑"})

    result = asyncio.run(flow())
    assert "拒绝" in result
    assert SaveTool.executed is False


def test_gated_tool_executes_on_approval():
    SaveTool.executed = False
    events, emit = _collector()

    async def flow():
        ctx = ApprovalContext(emit, timeout=5, user_id=1)
        gated = gate_write_tools([SaveTool()], ctx, {"save_study_plan"})[0]

        async def request():
            return await gated.ainvoke({"user_id": 1, "title": "正式计划"})

        task = asyncio.create_task(request())
        for _ in range(100):
            if PENDING:
                break
            await asyncio.sleep(0.01)
        decide(next(iter(PENDING)), True, user_id=1)
        return await task

    result = asyncio.run(flow())
    assert result == "saved:1:正式计划"
    assert SaveTool.executed is True


# ---------- 执行器拒绝语义 ----------

class FakeLLM:
    async def ainvoke(self, messages, **kw):
        return SimpleNamespace(content=json.dumps({
            "steps": [{"title": "替代步骤", "goal": "g", "tool_hint": None}]}))


def test_executor_marks_step_skipped_on_denial():
    """ApprovalDenied → skipped（不重试不重规划），且不计为降级。"""
    events, emit = _collector()

    async def run_step(step: PlanStep) -> str:
        raise ApprovalDenied("用户拒绝执行该写操作")

    plan = Plan(steps=[PlanStep(id=1, title="保存计划", goal="保存")])
    result = asyncio.run(execute_plan(
        FakeLLM(), "目标", plan, run_step,
        step_retries=2, max_replans=2, emit=emit,
    ))

    assert result.success is False
    assert result.user_denied is True
    assert result.degraded is False          # 用户拒绝不是系统降级
    end = [e for e in events if e["type"] == "step_end"][0]
    assert end["status"] == "skipped"
    # 没有 step_retry / replan / degraded 事件
    types = [e["type"] for e in events]
    assert "step_retry" not in types
    assert "replan" not in types
    assert "degraded" not in types


def test_denial_does_not_block_other_steps():
    """第一步被拒，后续步骤照常执行（只有依赖它的才被阻塞跳过）。"""
    events, emit = _collector()
    calls = []

    async def run_step(step: PlanStep) -> str:
        calls.append(step.title)
        if step.title == "保存计划":
            raise ApprovalDenied("用户拒绝执行该写操作")
        return f"{step.title}完成"

    plan = Plan(steps=[
        PlanStep(id=1, title="查课表", goal="查", depends_on=[]),
        PlanStep(id=2, title="保存计划", goal="存", depends_on=[1]),
        PlanStep(id=3, title="汇报", goal="说", depends_on=[]),  # 不依赖被拒步骤
    ])
    result = asyncio.run(execute_plan(
        FakeLLM(), "目标", plan, run_step, step_retries=0, emit=emit,
    ))

    statuses = {s.title: s.status for s in plan.steps}
    assert statuses["查课表"] == "done"
    assert statuses["保存计划"] == "skipped"
    assert statuses["汇报"] == "done"        # 独立步骤不被牵连
    assert result.success is True            # 有完成的步骤
    assert result.steps_done == 2


# ---------- 三态审批：修改后执行 ----------

def test_revise_returns_feedback_and_not_denial():
    """revise 决策返回用户意见（非拒绝），denials 不计数，日志记录 revise。"""
    events, emit = _collector()
    logs: list[dict] = []

    async def flow():
        ctx = ApprovalContext(emit, timeout=5, user_id=1,
                              max_revises=2, record=logs.append)
        task = asyncio.create_task(ctx.request("保存学习计划", {"title": "计划"}))
        for _ in range(100):
            if PENDING:
                break
            await asyncio.sleep(0.01)
        approval_id = next(iter(PENDING))
        assert decide(approval_id, "把第2条改具体些", user_id=1) is None
        return await task, ctx

    outcome, ctx = asyncio.run(flow())
    assert outcome == "把第2条改具体些"           # 意见回传给审批门 → 模型调整参数
    assert ctx.denials_for(None) == 0             # revise 不算拒绝
    kinds = [(l["kind"], l.get("decision")) for l in logs]
    assert ("request", None) in kinds and ("decision", "revise") in kinds
    assert logs[-1]["feedback"] == "把第2条改具体些"


def test_revise_exceeds_limit_converts_to_reject():
    """修改轮数超上限 → 转为拒绝（denials 计数，防无限拉扯）。"""
    events, emit = _collector()

    async def flow():
        ctx = ApprovalContext(emit, timeout=5, user_id=1, max_revises=1)

        async def request():
            return await ctx.request("save", {})

        async def decide_when_pending(value: bool | str):
            for _ in range(100):
                if PENDING:
                    break
                await asyncio.sleep(0.01)
            decide(next(iter(PENDING)), value, user_id=1)

        task = asyncio.create_task(request())
        await decide_when_pending("改一次")       # 第 1 次 revise：允许
        first = await task
        task2 = asyncio.create_task(request())
        await decide_when_pending("再改一次")     # 第 2 次 revise：超限 → 拒绝
        second = await task2
        return first, second, ctx

    first, second, ctx = asyncio.run(flow())
    assert first == "改一次"
    assert second is False
    assert ctx.denials_for(None) == 1
    reasons = [e["reason"] for e in events if e["type"] == "approval_resolved"]
    assert "修改次数已用尽，按拒绝处理" in reasons


def test_gate_tool_returns_revise_text_for_model():
    """审批门收到 revise：不执行底层工具，把修改意见作为工具结果回给模型。"""
    events, emit = _collector()

    class EchoTool(BaseTool):
        name: str = "echo"
        description: str = "echo"

        class Args(BaseModel):
            text: str

        args_schema: type = Args

        def _run(self, text: str) -> str:
            return f"executed:{text}"

    gate = ApprovalGateTool(underlying=EchoTool(), approval=None, pretty_name="echo",
                            name="echo", description="echo", args_schema=EchoTool.Args)

    async def flow():
        # 用一个假的 approval 对象：request 直接返回 revise 意见
        class FakeApproval:
            async def request(self, tool, args):
                return "把文本改成大写"

        gate.approval = FakeApproval()
        return await gate._arun(text="abc")

    out = asyncio.run(flow())
    assert "用户意见：把文本改成大写" in out
    assert "executed" not in out  # 底层工具未执行


def test_decide_accepts_bool_and_str_compat():
    """decide 同时接受旧 bool 与新三态；未知 id 返回错误。"""
    events, emit = _collector()

    async def flow():
        ctx = ApprovalContext(emit, timeout=5, user_id=1)
        task = asyncio.create_task(ctx.request("save", {}))
        for _ in range(100):
            if PENDING:
                break
            await asyncio.sleep(0.01)
        assert decide("no-such", True, user_id=1) == "审批不存在或已过期"
        decide(next(iter(PENDING)), False, user_id=1)  # 旧式 bool 拒绝
        return await task

    assert asyncio.run(flow()) is False

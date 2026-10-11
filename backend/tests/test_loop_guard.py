"""模型循环调用护栏：同参拦截、总数上限、正常调用不受影响、build_agent 接线。"""
import asyncio
from types import SimpleNamespace

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from app.agent import engine
from app.agent.engine import LoopGuardTool, build_agent
from app.config import settings


class CountingTool(BaseTool):
    """记录执行次数与参数的底层工具。"""

    name: str = "get_thing"
    description: str = "查询"
    calls: list = []

    class Args(BaseModel):
        q: str = ""

    args_schema: type = Args

    def _run(self, q: str = "") -> str:
        raise RuntimeError("同步路径不应被调用")

    async def _arun(self, q: str = "") -> str:
        type(self).calls.append(q)
        return f"结果-{q}"


def _guarded(ledger: dict) -> LoopGuardTool:
    return LoopGuardTool(underlying=CountingTool(), ledger=ledger,
                         name="get_thing", description="查询",
                         args_schema=CountingTool.Args)


def _run(coro):
    return asyncio.run(coro)


def test_identical_call_intercepted_on_third():
    """同参第 3 次调用被拦截：底层不再执行，返回提示文本。"""
    CountingTool.calls = []
    ledger: dict = {}
    tool = _guarded(ledger)

    async def flow():
        outs = []
        for _ in range(3):
            outs.append(await tool.ainvoke({"q": "同一问题"}))
        return outs

    outs = _run(flow())
    assert outs[0] == "结果-同一问题" and outs[1] == "结果-同一问题"
    assert "重复调用拦截" in outs[2]
    assert len(CountingTool.calls) == 2       # 底层只执行了 2 次


def test_different_args_not_intercepted():
    """参数不同（如 revise 后调整）不算重复：各自正常执行。"""
    CountingTool.calls = []
    tool = _guarded({})

    async def flow():
        return await asyncio.gather(
            tool.ainvoke({"q": "a"}),
            tool.ainvoke({"q": "b"}),
            tool.ainvoke({"q": "c"}),
        )

    outs = _run(flow())
    assert all(o.startswith("结果-") for o in outs)
    assert len(CountingTool.calls) == 3


def test_total_call_cap_intercepts():
    """总数超限即拦截：防止换着参数的循环调用。"""
    CountingTool.calls = []
    ledger: dict = {}
    tool = _guarded(ledger)
    tool.max_total = 5

    async def flow():
        outs = []
        for i in range(7):
            outs.append(await tool.ainvoke({"q": f"问题{i}"}))  # 参数全不同
        return outs

    outs = _run(flow())
    assert all("调用次数超限" not in o for o in outs[:5])   # 前 5 次正常
    assert all("调用次数超限" in o for o in outs[5:])       # 第 6 次起拦截
    assert len(CountingTool.calls) == 5


def test_interception_message_guides_model():
    """拦截文本明确指示停止调用、基于已有结果回答（打破循环的关键）。"""
    CountingTool.calls = []
    ledger: dict = {}
    tool = _guarded(ledger)

    async def flow():
        for _ in range(3):
            await tool.ainvoke({"q": "x"})
        return await tool.ainvoke({"q": "x"})

    out = _run(flow())
    assert "不要再调用任何工具" in out and "基于已有结果" in out


def test_build_agent_wraps_tools_with_shared_ledger():
    """build_agent 接线：工具被 LoopGuard 包裹且共享一本台账。"""
    from app.models.user import User

    user = User(username="u", password_hash="x", name="u", role="student")
    t1, t2 = CountingTool(name="a"), CountingTool(name="b")
    guarded = engine._guard_tools([t1, t2])
    assert all(isinstance(t, LoopGuardTool) for t in guarded)
    assert guarded[0].ledger is guarded[1].ledger  # 同一请求共享台账
    # 台账随调用累积：跨两个工具的计数互通
    async def flow():
        return await asyncio.gather(guarded[0].ainvoke({"q": "1"}),
                                    guarded[1].ainvoke({"q": "2"}))
    _run(flow())
    assert guarded[0].ledger["__total__"] == 2

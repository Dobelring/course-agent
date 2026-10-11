"""断点续跑：plan_from_dicts 还原、执行状态持久化、取消→暂停→继续执行。"""
import asyncio
import json
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from app.agent.plan_executor import execute_plan
from app.agent.planner import plan_from_dicts
from fakes import FakeLLM, _sse_data


# ---------------------------------------------------------------------------
# 1. plan_from_dicts：还原状态 + 续跑语义
# ---------------------------------------------------------------------------

def test_plan_from_dicts_preserves_step_state():
    plan = plan_from_dicts([
        {"id": 1, "title": "查课表", "goal": "查", "status": "done", "result": "课表数据"},
        {"id": 2, "title": "总结", "goal": "写", "status": "pending", "depends_on": [1]},
    ], revised=1)
    assert plan.revised == 1
    assert plan.steps[0].status == "done" and plan.steps[0].result == "课表数据"
    assert plan.steps[1].status == "pending"
    assert plan.pending == [plan.steps[1]]  # 只有 pending 会被续跑


def test_execute_plan_on_resumed_plan_only_runs_pending():
    """续跑语义：done 步骤不重跑，其产出进入综合；只执行 pending 部分。"""
    events = []

    async def emit(e):
        events.append(e)

    calls = []

    async def run_step(step):
        calls.append(step.id)
        return f"新产出-{step.id}"

    plan = plan_from_dicts([
        {"id": 1, "title": "查课表", "goal": "查", "status": "done", "result": "课表数据"},
        {"id": 2, "title": "总结", "goal": "写", "depends_on": [1]},
    ])

    async def flow():
        return await execute_plan(FakeLLM(), "目标", plan, run_step, emit=emit)

    result = asyncio.run(flow())
    assert result.success is True
    assert calls == [2]                       # 只有 pending 步骤真正执行
    assert plan.steps[0].result == "课表数据"  # done 产出原样保留
    types = [e["type"] for e in events]
    assert types.count("step_start") == 1
    assert types[0] == "plan"                 # 续跑也发 plan 事件（前端恢复卡片）


# ---------------------------------------------------------------------------
# 2. 聊天集成夹具
# ---------------------------------------------------------------------------

class RecordingAgent:
    """记录指令 + 正常完成。"""

    def __init__(self):
        self.instructions = []

    async def astream_events(self, inputs, version=None):
        self.instructions.append(inputs["messages"][-1].content)
        for ch in "步骤完成":
            yield {"event": "on_chat_model_stream",
                   "data": {"chunk": SimpleNamespace(content=ch)}}


class HangingAgent:
    """第一步挂起直到被取消（CancelledError 从 wait 中抛出）。

    started 事件在同进程内可观测——ASGITransport 会缓冲 SSE 响应体，
    测试无法从客户端读到中途事件，但 agent 与测试共享同一事件循环。
    """

    def __init__(self):
        self.started = asyncio.Event()

    async def astream_events(self, inputs, version=None):
        yield {"event": "on_chat_model_stream",
               "data": {"chunk": SimpleNamespace(content="部分产出")}}
        self.started.set()  # 已进入执行中（step_start 之后）
        await asyncio.Event().wait()  # 永久等待，直到任务被 cancel


def _patch_env(monkeypatch, agent, fake_llm=None):
    from app.api import chat as chat_mod
    from conftest import patch_fake_router

    fake_llm = fake_llm or FakeLLM()
    monkeypatch.setattr(chat_mod, "build_llm", lambda usage=None: fake_llm)
    monkeypatch.setattr(chat_mod, "build_agent",
                        lambda user, tools, extra_system="", llm=None: agent)
    patch_fake_router(monkeypatch, fake_llm, chat_mod)
    _patch_env.target_agent = agent  # 供异步测试观测 agent 内部事件
    return fake_llm


@pytest.fixture
def db_session():
    from app.database import SessionLocal

    s = SessionLocal()
    yield s
    s.close()


# ---------------------------------------------------------------------------
# 3. 执行状态持久化：正常完成 → done
# ---------------------------------------------------------------------------

def test_plan_run_persisted_and_finalized(client, seeded, auth_headers,
                                          fake_agent_env, db_session):
    from app.models.plan import PlanRun

    resp = client.post("/api/chat", json={"message": "两步计划任务", "mode": "plan"},
                       headers=auth_headers("student1"))
    events = _sse_data(resp.text)
    assert "error" not in [e["type"] for e in events], resp.text

    rows = db_session.query(PlanRun).all()
    assert len(rows) == 1
    assert rows[0].status == "done"           # 正常完成终态
    state = json.loads(rows[0].plan_json)
    assert len(state["steps"]) == 2
    assert all(s["status"] == "done" for s in state["steps"])
    assert rows[0].user_id == seeded.users.student1.id


# ---------------------------------------------------------------------------
# 4. 取消 → paused + cancelled 事件携带续跑 id
# ---------------------------------------------------------------------------

def test_cancel_mid_plan_marks_paused(seeded, auth_headers, monkeypatch, db_session):
    from app.main import app
    from app.models.plan import PlanRun

    _patch_env(monkeypatch, HangingAgent())

    async def scenario():
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as ac:
            r = await ac.post("/api/auth/login",
                              json={"username": "student1", "password": "pass1234"})
            headers = {"Authorization": f"Bearer {r.json()['access_token']}"}

            events = []
            agent = _patch_env.target_agent

            async def read_stream():
                async with ac.stream("POST", "/api/chat", headers=headers,
                                     json={"message": "长任务", "mode": "plan",
                                           "run_id": "resume-test-run"}) as resp:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        ev = json.loads(line[6:])
                        events.append(ev)
                        if ev["type"] in ("cancelled", "done", "error"):
                            return

            reader = asyncio.create_task(read_stream())
            # ASGITransport 缓冲响应体：无法从客户端观察中途事件，
            # 改用 agent 的进程内 started 事件确认已执行到挂起点
            await asyncio.wait_for(agent.started.wait(), timeout=30)
            cr = await ac.post("/api/chat/cancel", headers=headers,
                               json={"run_id": "resume-test-run"})
            assert cr.status_code == 200
            await asyncio.wait_for(reader, timeout=30)

        cancelled = next(e for e in events if e["type"] == "cancelled")
        assert cancelled.get("plan_run_id"), f"cancelled 应携带 plan_run_id: {cancelled}"

        row = db_session.get(PlanRun, cancelled["plan_run_id"])
        assert row.status == "paused"
        state = json.loads(row.plan_json)
        assert any(s["status"] in ("running", "pending") for s in state["steps"])
        return row

    row = asyncio.run(scenario())
    assert row is not None


def test_cancel_persists_paused_plan_meta_on_message(seeded, auth_headers,
                                                     monkeypatch, db_session):
    """取消后助手消息的 plan 字段含 paused/plan_run_id（历史重开仍有续跑入口）。"""
    from app.main import app
    from app.models.conversation import AiMessage
    from app.models.plan import PlanRun

    _patch_env(monkeypatch, HangingAgent())

    async def scenario():
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as ac:
            r = await ac.post("/api/auth/login",
                              json={"username": "student1", "password": "pass1234"})
            headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
            events = []
            agent = _patch_env.target_agent

            async def read_stream():
                async with ac.stream("POST", "/api/chat", headers=headers,
                                     json={"message": "长任务", "mode": "plan",
                                           "run_id": "resume-meta-run"}) as resp:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        ev = json.loads(line[6:])
                        events.append(ev)
                        if ev["type"] in ("cancelled", "done", "error"):
                            return

            reader = asyncio.create_task(read_stream())
            await asyncio.wait_for(agent.started.wait(), timeout=30)
            await ac.post("/api/chat/cancel", headers=headers,
                          json={"run_id": "resume-meta-run"})
            await asyncio.wait_for(reader, timeout=30)
        cancelled = next(e for e in events if e["type"] == "cancelled")
        return cancelled["conversation_id"]

    cid = asyncio.run(scenario())
    row = db_session.query(PlanRun).filter(
        PlanRun.conversation_id == cid).one()
    assert row.status == "paused"
    assistant = (
        db_session.query(AiMessage)
        .filter(AiMessage.conversation_id == cid, AiMessage.role == "assistant")
        .one()
    )
    meta = json.loads(assistant.plan)
    assert meta["paused"] is True
    assert meta["plan_run_id"] == row.id
    assert any(s["status"] == "done" or s["status"] == "running"
               for s in meta["steps"])


# ---------------------------------------------------------------------------
# 5. 续跑：只补 pending，done 产出注入后续指令
# ---------------------------------------------------------------------------

@pytest.fixture
def resume_env(monkeypatch):
    fake_llm = FakeLLM()
    agent = RecordingAgent()
    _patch_env(monkeypatch, agent, fake_llm)
    return fake_llm, agent


def test_resume_plan_runs_only_pending_and_injects_done_result(
        client, seeded, auth_headers, resume_env, db_session):
    from app.models.plan import PlanRun

    _llm, agent = resume_env
    plan_run = PlanRun(
        user_id=seeded.users.student1.id, conversation_id=None, title="旧任务",
        plan_json=json.dumps({"steps": [
            {"id": 1, "title": "查课表", "goal": "查", "status": "done",
             "result": "课表产出XYZ"},
            {"id": 2, "title": "总结", "goal": "写", "depends_on": [1]},
        ], "revised": 0}),
        status="paused",
    )
    db_session.add(plan_run)
    db_session.commit()

    resp = client.post("/api/chat",
                       json={"message": "继续执行暂停的计划",
                             "mode": "plan", "resume_plan_id": plan_run.id},
                       headers=auth_headers("student1"))
    events = _sse_data(resp.text)
    types = [e["type"] for e in events]
    assert "error" not in types, resp.text
    mode = next(e for e in events if e["type"] == "mode")
    assert mode["reason"] == "续跑暂停的计划"

    # 只有 pending 步骤被执行
    assert len([e for e in events if e["type"] == "step_start"]) == 1
    # done 步骤的产出注入了 pending 步骤的指令（跨步骤上下文传递在续跑时同样生效）
    assert any("课表产出XYZ" in ins for ins in agent.instructions)

    # 计划终态 done
    db_session.expire_all()
    assert db_session.get(PlanRun, plan_run.id).status == "done"


def test_resume_plan_rejects_other_user(client, seeded, auth_headers,
                                        fake_agent_env, db_session):
    from app.models.plan import PlanRun

    plan_run = PlanRun(
        user_id=seeded.users.student2.id, title="他人的计划",
        plan_json=json.dumps({"steps": [], "revised": 0}), status="paused",
    )
    db_session.add(plan_run)
    db_session.commit()
    resp = client.post("/api/chat",
                       json={"message": "偷跑", "resume_plan_id": plan_run.id},
                       headers=auth_headers("student1"))
    events = _sse_data(resp.text)
    assert any(e["type"] == "error" and "不存在或无权访问" in e["message"]
               for e in events)


def test_plan_from_dicts_resets_running_to_pending():
    """中断时正在执行的步骤没有产出，续跑必须复位为 pending 重新执行。"""
    plan = plan_from_dicts([
        {"id": 1, "title": "查课表", "goal": "查", "status": "done", "result": "数据"},
        {"id": 2, "title": "保存计划", "goal": "存", "status": "running",
         "depends_on": [1]},
    ])
    assert plan.steps[0].status == "done"
    assert plan.steps[1].status == "pending"   # running → pending（重新执行）
    assert plan.pending == [plan.steps[1]]

"""本次修复的三个行为：前序结果注入、工具事件请求级去重、生成中断。"""
import asyncio
import json
from types import SimpleNamespace

import pytest
from app.api import chat as chat_mod
from app.api.chat import ACTIVE_RUNS, _register_run, _unregister_run
from fakes import FakeAgent, FakeLLM, _sse_data


# ---------------------------------------------------------------------------
# 1. 取消语义（单元级，不依赖 LLM）
# ---------------------------------------------------------------------------
class FakeTask:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def done(self):
        return False


def test_register_and_unregister():
    task = FakeTask()
    _register_run("run-1", user_id=42, task=task)  # type: ignore[arg-type]
    assert ACTIVE_RUNS["run-1"] == (42, task)
    _unregister_run("run-1")
    assert "run-1" not in ACTIVE_RUNS


def test_unregister_missing_is_noop():
    _unregister_run("never-existed")  # 不应抛异常


def test_cancel_endpoint_cancels_own_task(client, seeded, auth_headers):
    task = FakeTask()
    _register_run("run-owner", user_id=seeded.users.student1.id, task=task)  # type: ignore[arg-type]
    try:
        resp = client.post("/api/chat/cancel", json={"run_id": "run-owner"},
                           headers=auth_headers("student1"))
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert task.cancelled is True
    finally:
        _unregister_run("run-owner")


def test_cancel_endpoint_rejects_other_users_task(client, seeded, auth_headers):
    task = FakeTask()
    _register_run("run-victim", user_id=seeded.users.student1.id, task=task)  # type: ignore[arg-type]
    try:
        # student2 尝试取消 student1 的任务 → 404（不泄露存在性）
        resp = client.post("/api/chat/cancel", json={"run_id": "run-victim"},
                           headers=auth_headers("student2"))
        assert resp.status_code == 404
        assert task.cancelled is False
    finally:
        _unregister_run("run-victim")


def test_cancel_endpoint_unknown_run(client, seeded, auth_headers):
    resp = client.post("/api/chat/cancel", json={"run_id": "nope"},
                       headers=auth_headers("student1"))
    assert resp.status_code == 404


def test_cancel_endpoint_requires_auth(client):
    resp = client.post("/api/chat/cancel", json={"run_id": "any"})
    assert resp.status_code == 401


def test_cancelled_task_completes_as_cancelled():
    """asyncio.Task 被 cancel 后 result() 抛 CancelledError（gen 的收尾路径依赖此语义）。"""

    async def flow():
        task = asyncio.create_task(asyncio.sleep(10))
        await asyncio.sleep(0)  # 让任务先进入运行态
        task.cancel()
        try:
            await task  # 等待取消完成
        except asyncio.CancelledError:
            pass
        assert task.done() and task.cancelled()
        with pytest.raises(asyncio.CancelledError):
            task.result()  # gen() 的 result() 调用即此路径

    asyncio.run(flow())


# ---------------------------------------------------------------------------
# 2. 工具事件请求级去重（集成级：plan 模式两步骤调同一工具 → 只上屏一次）
# ---------------------------------------------------------------------------
class RepeatingToolAgent(FakeAgent):
    """每个步骤都调 get_my_schedule（模拟模型重复查询）。"""

    async def astream_events(self, inputs, config=None, version=None):
        yield {"event": "on_tool_start", "name": "get_my_schedule",
               "data": {"input": {"user_id": 1, "weekday": 1}}}
        yield {"event": "on_tool_end", "name": "get_my_schedule",
               "data": {"input": {"user_id": 1, "weekday": 1},
                        "output": '[{"course_name": "高等数学"}]'}}
        last = inputs["messages"][-1].content
        for ch in f"步骤完成:{last[-12:]}":
            yield {"event": "on_chat_model_stream",
                   "data": {"chunk": SimpleNamespace(content=ch)}}


@pytest.fixture
def repeating_agent_env(monkeypatch):
    fake_llm = FakeLLM()
    monkeypatch.setattr(chat_mod, "build_llm", lambda usage=None: fake_llm)
    monkeypatch.setattr(chat_mod, "build_agent",
                        lambda user, tools, extra_system="", llm=None: RepeatingToolAgent())
    from conftest import patch_fake_router
    patch_fake_router(monkeypatch, fake_llm, chat_mod)
    return fake_llm


def test_tool_events_deduplicated_across_steps(client, seeded, auth_headers,
                                               repeating_agent_env):
    """plan 两步骤都调 get_my_schedule → 前端只应收到一次 tool_start。"""
    resp = client.post("/api/chat",
                       json={"message": "查课表并总结", "mode": "plan"},
                       headers=auth_headers("student1"))
    events = _sse_data(resp.text)
    tool_starts = [e for e in events if e["type"] == "tool_start"]
    tool_ends = [e for e in events if e["type"] == "tool_end"]
    # 请求级去重：整次请求每个工具只上屏一次
    assert len(tool_starts) == 1, f"tool_start 应去重为 1，实为 {len(tool_starts)}"
    assert len(tool_ends) == 1
    # 但两个步骤都真实执行了
    step_ends = [e for e in events if e["type"] == "step_end"]
    assert len(step_ends) == 2
    assert all(e["status"] == "done" for e in step_ends)


# ---------------------------------------------------------------------------
# 3. 前序步骤结果注入指令
# ---------------------------------------------------------------------------
class RecordingAgent(FakeAgent):
    """记录每次收到的 instruction，返回固定文本。"""

    def __init__(self):
        self.instructions: list[str] = []

    async def astream_events(self, inputs, config=None, version=None):
        self.instructions.append(inputs["messages"][-1].content)
        for ch in "产出结果ABC":
            yield {"event": "on_chat_model_stream",
                   "data": {"chunk": SimpleNamespace(content=ch)}}


@pytest.fixture
def recording_agent_env(monkeypatch):
    fake_llm = FakeLLM()
    agent = RecordingAgent()
    monkeypatch.setattr(chat_mod, "build_llm", lambda usage=None: fake_llm)
    monkeypatch.setattr(chat_mod, "build_agent",
                        lambda user, tools, extra_system="", llm=None: agent)
    # run_step 每次尝试直连 get_router 选路——替身固定返回假 LLM
    from conftest import patch_fake_router
    patch_fake_router(monkeypatch, fake_llm, chat_mod)
    return fake_llm, agent


def test_prior_step_results_injected_into_next_instruction(
        client, seeded, auth_headers, recording_agent_env):
    """第二步的指令必须包含第一步的产出（防止重复查询的根因修复）。"""
    _llm, agent = recording_agent_env
    resp = client.post("/api/chat",
                       json={"message": "两步任务", "mode": "plan"},
                       headers=auth_headers("student1"))
    events = _sse_data(resp.text)
    step_ends = [e for e in events if e["type"] == "step_end"]
    assert len(step_ends) == 2, resp.text

    first_instruction, second_instruction = agent.instructions[0], agent.instructions[1]
    # 用块级标记「直接引用」区分前序结果块（护栏句不含此标记）
    assert "直接引用" not in first_instruction              # 第一步没有前序
    assert "直接引用" in second_instruction                 # 第二步有前序
    assert "产出结果ABC" in second_instruction              # 内容是第一步的真实产出
    assert "禁止重复调用相同工具" in second_instruction
    assert "禁止重复保存" in second_instruction              # 防重复落库的指令护栏


def test_first_step_has_no_prior_block(client, seeded, auth_headers, recording_agent_env):
    _llm, agent = recording_agent_env
    client.post("/api/chat", json={"message": "单步任务走 plan", "mode": "plan"},
                headers=auth_headers("student1"))
    assert "直接引用" not in agent.instructions[0]


def test_write_step_instruction_requires_content_before_approval(
        client, seeded, auth_headers, recording_agent_env):
    """写操作授权前必须先流式展示内容：指令层护栏（顺序不能反）。"""
    _llm, agent = recording_agent_env
    client.post("/api/chat", json={"message": "规划任务", "mode": "plan"},
                headers=auth_headers("student1"))
    assert agent.instructions, "步骤指令未被记录"
    for ins in agent.instructions:
        assert "先用可见文本完整展示" in ins
        assert "再调用写工具发起授权" in ins

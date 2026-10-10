"""chat 集成：模式路由、规划执行事件序列、上下文摘要持久化、字段校验。"""
import json
from types import SimpleNamespace

import pytest

from app.config import settings


def _sse_data(resp_text: str) -> list[dict]:
    """解析 SSE 响应体为事件列表。"""
    events = []
    for line in resp_text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


class FakeLLM:
    """按提示词类型返回预设输出，供分类/规划/综合共用。"""

    def __init__(self, *, complex_: bool = True, summary: str = "历史摘要X"):
        self.complex_ = complex_
        self.summary = summary
        self.prompts: list[str] = []

    async def ainvoke(self, messages, **kw):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        if "任务路由器" in prompt:
            c = "complex" if self.complex_ else "simple"
            return SimpleNamespace(content=json.dumps({"complexity": c, "reason": "测试"}))
        if "任务规划器" in prompt:
            return SimpleNamespace(content=json.dumps({"steps": [
                {"title": "查课表", "goal": "查询用户课表", "tool_hint": "get_my_schedule"},
                {"title": "总结", "goal": "汇总结果", "tool_hint": None},
            ]}))
        if "对话历史压缩器" in prompt:
            return SimpleNamespace(content=self.summary)
        if "综合器" in prompt:
            return SimpleNamespace(content="综合后的答案")
        return SimpleNamespace(content="默认回复")


class FakeAgent:
    """伪造 create_agent 返回对象：产出一次工具调用 + 流式文本。"""

    async def astream_events(self, inputs, version=None):
        yield {"event": "on_tool_start", "name": "get_my_schedule",
               "data": {"input": {"user_id": 1, "weekday": 1}}}
        yield {"event": "on_tool_end", "name": "get_my_schedule",
               "data": {"input": {"user_id": 1, "weekday": 1},
                        "output": '[{"course_name": "高等数学"}]'}}
        for ch in "步骤产出文本":
            yield {"event": "on_chat_model_stream",
                   "data": {"chunk": SimpleNamespace(content=ch)}}


@pytest.fixture
def fake_agent_env(monkeypatch):
    """把 LLM 与 Agent 替换为假实现；MCP 工具加载保持真实（无需 Key）。"""
    from app.api import chat as chat_mod

    fake_llm = FakeLLM()
    monkeypatch.setattr(chat_mod, "build_llm", lambda usage=None: fake_llm)
    monkeypatch.setattr(chat_mod, "build_agent",
                        lambda user, tools, extra_system="", llm=None: FakeAgent())
    return fake_llm


# ---------- 字段校验 ----------

def test_mode_field_rejects_invalid(client, make_users, auth_headers):
    resp = client.post("/api/chat", json={"message": "hi", "mode": "banana"},
                       headers=auth_headers("student1"))
    assert resp.status_code == 422


# ---------- direct 模式 ----------

def test_direct_mode_event_sequence(client, seeded, auth_headers, fake_agent_env):
    resp = client.post("/api/chat",
                       json={"message": "我今天有什么课", "mode": "direct"},
                       headers=auth_headers("student1"))
    assert resp.status_code == 200
    events = _sse_data(resp.text)
    types = [e["type"] for e in events]

    assert "mode" in types and "done" in types
    assert "error" not in types
    mode_evt = next(e for e in events if e["type"] == "mode")
    assert mode_evt["mode"] == "direct"
    # 工具事件与流式文本
    assert "tool_start" in types and "tool_end" in types
    assert "text" in types
    text = "".join(e["delta"] for e in events if e["type"] == "text")
    assert text == "步骤产出文本"
    # done 带模式与耗时
    done = next(e for e in events if e["type"] == "done")
    assert done["mode"] == "direct"
    assert done["latency_ms"] >= 0
    assert done["conversation_id"] > 0


def test_auto_mode_classifies_to_direct(client, seeded, auth_headers, fake_agent_env):
    fake_agent_env.complex_ = False
    resp = client.post("/api/chat", json={"message": "我今天有什么课"},
                       headers=auth_headers("student1"))
    events = _sse_data(resp.text)
    mode_evt = next(e for e in events if e["type"] == "mode")
    assert mode_evt["mode"] == "direct"


# ---------- plan 模式 ----------

def test_plan_mode_full_event_sequence(client, seeded, auth_headers, fake_agent_env):
    resp = client.post("/api/chat",
                       json={"message": "分析我的课表并给出建议", "mode": "plan"},
                       headers=auth_headers("student1"))
    assert resp.status_code == 200
    events = _sse_data(resp.text)
    types = [e["type"] for e in events]

    assert "error" not in types, resp.text
    # 模式声明
    assert next(e for e in events if e["type"] == "mode")["mode"] == "plan"
    # 规划事件携带步骤
    plan_evt = next(e for e in events if e["type"] == "plan")
    assert len(plan_evt["steps"]) == 2
    assert plan_evt["steps"][0]["tool_hint"] == "get_my_schedule"
    # 两步各一组 start/end
    assert types.count("step_start") == 2
    assert types.count("step_end") == 2
    step_ends = [e for e in events if e["type"] == "step_end"]
    assert all(e["status"] == "done" for e in step_ends)
    # 综合经流式 text 输出
    assert "text" in types
    # done 携带计划元数据
    done = next(e for e in events if e["type"] == "done")
    assert done["mode"] == "plan"
    assert done["plan"]["done"] == 2
    assert done["plan"]["revised"] == 0
    assert done["plan"]["degraded"] is False


def test_plan_mode_persists_plan_json(client, seeded, auth_headers, fake_agent_env):
    resp = client.post("/api/chat",
                       json={"message": "分析课表", "mode": "plan"},
                       headers=auth_headers("student1"))
    done = next(e for e in _sse_data(resp.text) if e["type"] == "done")
    cid = done["conversation_id"]

    detail = client.get(f"/api/conversations/{cid}", headers=auth_headers("student1")).json()
    assistant = [m for m in detail["messages"] if m["role"] == "assistant"][0]
    plan = assistant["plan"]  # API 已解析为 dict
    assert plan["done"] == 2
    assert [s["title"] for s in plan["steps"]] == ["查课表", "总结"]


def test_auto_mode_classifies_to_plan(client, seeded, auth_headers, fake_agent_env):
    fake_agent_env.complex_ = True
    resp = client.post("/api/chat", json={"message": "多源分析任务"},
                       headers=auth_headers("student1"))
    events = _sse_data(resp.text)
    assert next(e for e in events if e["type"] == "mode")["mode"] == "plan"


def test_plan_creation_failure_falls_back_to_direct(client, seeded, auth_headers, fake_agent_env, monkeypatch):
    """规划 JSON 两次都解析失败 → fallback 事件 + 降级 direct 执行。"""
    from app.api import chat as chat_mod

    async def broken_plan(*a, **kw):
        raise ValueError("计划输出无法解析")

    monkeypatch.setattr(chat_mod, "create_plan", broken_plan)
    resp = client.post("/api/chat",
                       json={"message": "复杂任务", "mode": "plan"},
                       headers=auth_headers("student1"))
    events = _sse_data(resp.text)
    types = [e["type"] for e in events]

    assert "fallback" in types
    fb = next(e for e in events if e["type"] == "fallback")
    assert fb["to"] == "direct"
    assert "规划失败" in fb["reason"]
    # 降级后仍有最终产出
    done = next(e for e in events if e["type"] == "done")
    assert done["mode"] == "direct"
    assert done["degraded"] is True
    assert "text" in types


# ---------- 上下文摘要持久化 ----------

def test_context_compression_emits_event_and_persists_summary(
        client, seeded, auth_headers, fake_agent_env, monkeypatch, db):
    # 压低预算强制触发压缩
    monkeypatch.setattr(settings, "agent_history_max_chars", 10)
    monkeypatch.setattr(settings, "agent_history_keep_recent_chars", 5)

    h1 = auth_headers("student1")
    # 造一个已有历史的会话
    conv = client.post("/api/conversations", json={"title": "旧会话"}, headers=h1).json()
    cid = conv["id"]
    # 手动塞一条长历史（经由 DB）
    from app.models.conversation import AiMessage

    db.add(AiMessage(conversation_id=cid, role="user",
                     content="很久以前我问过一个很长的问题" + "x" * 50))
    db.add(AiMessage(conversation_id=cid, role="assistant",
                     content="很久以前你给过一个很长的回答" + "y" * 50))
    db.commit()

    resp = client.post("/api/chat",
                       json={"message": "现在的问题", "mode": "direct",
                             "conversation_id": cid},
                       headers=h1)
    events = _sse_data(resp.text)
    ctx_events = [e for e in events if e["type"] == "context"]
    assert ctx_events and ctx_events[0]["compressed"] is True
    assert ctx_events[0]["summary_chars"] > 0

    # 摘要已持久化到会话
    detail = client.get(f"/api/conversations/{cid}", headers=h1).json()
    from app.database import SessionLocal

    s = SessionLocal()
    try:
        from app.models.conversation import AiConversation

        conv_row = s.get(AiConversation, cid)
        assert conv_row.summary == "历史摘要X"
    finally:
        s.close()
    # 摘要进了系统提示（extra_system 由 build_agent 假实现接收——校验 LLM 侧无从谈起，
    # 这里断言 build_context 的 prompt 确实包含旧摘要）
    assert any("很久以前" in p for p in fake_agent_env.prompts)

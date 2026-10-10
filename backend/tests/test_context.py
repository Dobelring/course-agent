"""上下文管理：预算切分、超预算压缩、摘要合并、预算内不触发压缩。"""
import asyncio
from types import SimpleNamespace

from app.agent.context import build_context, split_history

MAX, KEEP = 500, 200


def _msgs(*contents: str) -> list[dict]:
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": c}
            for i, c in enumerate(contents)]


class FakeLLM:
    def __init__(self, reply: str = "压缩后的摘要"):
        self.reply = reply
        self.prompts: list[str] = []

    async def ainvoke(self, messages, **kw):
        self.prompts.append(messages[-1]["content"])
        return SimpleNamespace(content=self.reply)


# ---------- split_history ----------

def test_split_under_budget_keeps_all():
    older, recent = split_history(_msgs("短消息1", "短消息2"), max_chars=MAX, keep_recent_chars=KEEP)
    assert older == []
    assert len(recent) == 2


def test_split_over_budget_moves_old_to_older():
    msgs = _msgs("A" * 300, "B" * 300, "C" * 100)
    older, recent = split_history(msgs, max_chars=600, keep_recent_chars=400)
    assert older == [msgs[0]]          # 最旧的进压缩区
    assert [m["content"][:1] for m in recent] == ["B", "C"]  # 最近的保留原文


def test_split_always_keeps_at_least_one_recent():
    msgs = _msgs("X" * 600, "Y" * 600)
    older, recent = split_history(msgs, max_chars=MAX, keep_recent_chars=50)
    assert len(recent) == 1            # 即使单条超预算也保留最新一条
    assert recent[0]["content"].startswith("Y")


# ---------- build_context ----------

def test_build_context_under_budget_no_compression():
    history = _msgs("你好", "你好呀")
    plan = asyncio.run(build_context(FakeLLM(), history, prev_summary="旧摘要",
                                     max_chars=MAX, keep_recent_chars=KEEP))
    assert plan.compressed is False
    assert plan.messages == history     # 原文全量
    assert plan.summary == "旧摘要"      # 既有摘要不动


def test_build_context_over_budget_compresses_older():
    history = _msgs("前情提要很重要" + "x" * 600, "最近的问题", "最近的回答")
    plan = asyncio.run(build_context(FakeLLM("新摘要内容"), history, prev_summary=None,
                                     max_chars=MAX, keep_recent_chars=KEEP))
    assert plan.compressed is True
    assert plan.summary == "新摘要内容"
    # 压缩后只保留最近的原文
    assert plan.messages[-1]["content"] == "最近的回答"
    assert len(plan.messages) < len(history)


def test_build_context_merges_prev_summary_into_prompt():
    llm = FakeLLM("合并后摘要")
    history = _msgs("很早以前的事情" + "y" * 600, "现在的问题")
    plan = asyncio.run(build_context(llm, history, prev_summary="上一轮摘要",
                                     max_chars=MAX, keep_recent_chars=KEEP))
    assert "上一轮摘要" in llm.prompts[0]        # 旧摘要喂给了压缩器
    assert plan.summary == "合并后摘要"
    assert plan.summary_block.startswith("## 此前对话摘要")


def test_summary_block_empty_without_summary():
    plan = asyncio.run(build_context(FakeLLM(), _msgs("hi"), None,
                                     max_chars=MAX, keep_recent_chars=KEEP))
    assert plan.summary_block == ""

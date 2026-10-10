"""跨会话用户画像：节流刷新、提示词注入、best-effort 兜底。"""
import asyncio
from types import SimpleNamespace

from app.agent.memory import maybe_refresh_profile, profile_block, refresh_profile
from app.models.memory import UserMemory


class FakeLLM:
    def __init__(self, reply: str = "大三学生，考研目标985，高数薄弱"):
        self.reply = reply
        self.prompts: list[str] = []

    async def ainvoke(self, messages, **kw):
        self.prompts.append(messages[-1]["content"])
        return SimpleNamespace(content=self.reply)


def _msgs(*contents: str) -> list[dict]:
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": c}
            for i, c in enumerate(contents)]


def test_profile_block_formats_and_empty():
    assert profile_block(None) == ""
    assert profile_block("  ") == ""
    block = profile_block("大三学生")
    assert block.startswith("## 用户画像")
    assert "大三学生" in block


def test_refresh_prompt_contains_old_profile_and_chunk():
    llm = FakeLLM()
    asyncio.run(refresh_profile(llm, "旧画像：喜欢晚上自习", _msgs("我最近在准备考研")))
    prompt = llm.prompts[0]
    assert "旧画像：喜欢晚上自习" in prompt
    assert "我最近在准备考研" in prompt


def test_refresh_merges_profile(db, make_users):
    """第 N 次消息触发刷新，画像落库、计数清零。"""
    user = make_users.student1
    llm = FakeLLM("新画像：目标哈工大")

    # 前 every-1 次不触发
    for _ in range(2):
        result = asyncio.run(maybe_refresh_profile(
            llm, db, user.id, _msgs("第N条消息"), every=3))
        assert result is None
    # 第 3 次触发
    result = asyncio.run(maybe_refresh_profile(
        llm, db, user.id, _msgs("关键信息：跨考计算机"), every=3))
    assert result == "新画像：目标哈工大"

    row = db.query(UserMemory).filter(UserMemory.user_id == user.id).one()
    assert row.profile == "新画像：目标哈工大"
    assert row.hits == 0            # 刷新后清零重新计数


def test_profile_persists_across_conversations(db, make_users):
    """画像跨会话读取：下一次请求读到的是上一次刷新的结果。"""
    user = make_users.student1
    asyncio.run(maybe_refresh_profile(
        FakeLLM("画像内容"), db, user.id, _msgs("重要事实"), every=1))

    # 模拟新会话请求：直接查库即可读到
    row = db.query(UserMemory).filter(UserMemory.user_id == user.id).one()
    assert row.profile == "画像内容"

    # 新一轮对话继续合并（不是覆盖丢失）
    asyncio.run(maybe_refresh_profile(
        FakeLLM("画像内容+新增"), db, user.id, _msgs("新事实"), every=1))
    row2 = db.query(UserMemory).filter(UserMemory.user_id == user.id).one()
    assert row2.profile == "画像内容+新增"
    # 仍是同一行（unique user_id），不重复插入
    assert db.query(UserMemory).filter(UserMemory.user_id == user.id).count() == 1


def test_refresh_failure_is_best_effort(db, make_users):
    """LLM 失败时：不抛异常、不写坏画像、计数保留（下次还会触发）。"""
    user = make_users.student1

    class BrokenLLM:
        async def ainvoke(self, messages, **kw):
            raise ConnectionError("网关超时")

    result = asyncio.run(maybe_refresh_profile(
        BrokenLLM(), db, user.id, _msgs("x"), every=1))
    assert result is None                       # 吞掉异常
    row = db.query(UserMemory).filter(UserMemory.user_id == user.id).one()
    assert row.profile == ""                    # 画像未被写坏
    assert row.hits == 1                        # 计数保留


def test_throttle_never_triggers_below_threshold(db, make_users):
    user = make_users.student1
    llm = FakeLLM()
    for _ in range(9):
        asyncio.run(maybe_refresh_profile(llm, db, user.id, _msgs("x"), every=10))
    assert llm.prompts == []                    # 9 < 10 一次都不调 LLM
    row = db.query(UserMemory).filter(UserMemory.user_id == user.id).one()
    assert row.hits == 9

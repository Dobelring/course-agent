"""反思环节：质量自检解析、fail-open、低分修订替换与落库。"""
import json
from types import SimpleNamespace

from app.agent.reflection import ReflectionResult, reflect


class FakeLLM:
    def __init__(self, content=None, exc: Exception | None = None):
        self.content = content
        self.exc = exc

    async def ainvoke(self, messages, **kw):
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(content=self.content)


def run(coro):
    import asyncio
    return asyncio.run(coro)


def test_reflect_parses_quality_issues_and_revision():
    llm = FakeLLM(json.dumps({"quality": 2, "issues": ["未引用成绩"],
                              "revised": "修订后的完整回答"}))
    r = run(reflect(llm, "问题", "草稿", "（无）", min_quality=3))
    assert isinstance(r, ReflectionResult)
    assert r.quality == 2 and r.issues == ["未引用成绩"]
    assert r.revised == "修订后的完整回答"


def test_reflect_high_quality_keeps_original():
    llm = FakeLLM(json.dumps({"quality": 4, "issues": [], "revised": ""}))
    r = run(reflect(llm, "问题", "草稿"))
    assert r.quality == 4 and r.revised == ""


def test_reflect_low_quality_without_revision_keeps_original():
    """低分但审查器没给修订 → 不替换（没有更好的答案就不动）。"""
    llm = FakeLLM(json.dumps({"quality": 1, "issues": ["差"], "revised": ""}))
    r = run(reflect(llm, "问题", "草稿", min_quality=3))
    assert r.quality == 1 and r.revised == ""


def test_reflect_malformed_json_fails_open():
    """JSON 坏 → fail-open：quality=5 不修订，绝不影响已产出答案。"""
    llm = FakeLLM("这不是JSON{{{")
    r = run(reflect(llm, "问题", "草稿"))
    assert r.quality == 5 and r.revised == "" and r.issues == []


def test_reflect_llm_failure_fails_open():
    llm = FakeLLM(exc=RuntimeError("模型超时"))
    r = run(reflect(llm, "问题", "草稿"))
    assert r.quality == 5 and r.revised == ""


def test_reflect_quality_clamped_to_range():
    llm = FakeLLM(json.dumps({"quality": 99, "issues": [], "revised": ""}))
    assert run(reflect(llm, "q", "a")).quality == 5

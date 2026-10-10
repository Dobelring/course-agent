"""测试共享假件：FakeLLM / FakeAgent / SSE 解析（非测试文件，pytest 不收集）。"""
import json
from types import SimpleNamespace


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

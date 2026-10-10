"""多轮对话上下文管理：预算控制 + 旧历史摘要压缩。

策略（两级）：
1. 预算内：历史消息按原文全部带入；
2. 超预算：仅保留最近 ``keep_recent_chars`` 字符原文，更早的消息用 LLM 压缩成
   一段工作摘要（working summary），与上一轮摘要合并，形成「摘要 + 近期原文」的输入。

摘要持久化在 AiConversation.summary，跨请求累积——这就是本项目的长期记忆机制。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


class LLMProto(Protocol):
    async def ainvoke(self, messages: Sequence[Any], **kwargs: Any) -> Any: ...


SUMMARY_PROMPT = """你是对话历史压缩器。将给定的对话片段压缩为工作摘要，要求：
- 保留：用户目标与偏好、已确认的事实、关键结论、未完成的待办
- 丢弃：寒暄、重复、与后续对话无关的细节
- 输出纯文本摘要，不超过 200 字，不要标题或列表符号

{prev_section}
对话片段：
{chunk}"""


@dataclass
class ContextPlan:
    """一次对话请求的上下文组装结果。"""

    messages: list[dict]          # 传给 LLM 的历史消息（不含本轮用户输入）
    summary: str | None           # 更新后的持久化摘要（None 表示无需更新）
    compressed: bool              # 是否发生了压缩（用于埋点/SSE 事件）

    @property
    def summary_block(self) -> str:
        return f"## 此前对话摘要\n{self.summary}" if self.summary else ""


def _render(msg: dict) -> str:
    role = "用户" if msg.get("role") == "user" else "助手"
    return f"{role}: {msg.get('content', '')}"


def _char_len(msgs: Sequence[dict]) -> int:
    return sum(len(str(m.get("content", ""))) for m in msgs)


def split_history(
    messages: Sequence[dict],
    *,
    max_chars: int,
    keep_recent_chars: int,
) -> tuple[list[dict], list[dict]]:
    """按字符预算切分历史 → (older, recent)。

    - 总长不超预算：older 为空，全部原文保留；
    - 超预算：从最新往回取，recent 尽量占满 keep_recent_chars（至少留 1 条），
      其余归入 older 交给摘要压缩。
    """
    msgs = list(messages)
    if _char_len(msgs) <= max_chars:
        return [], msgs

    recent: list[dict] = []
    used = 0
    for m in reversed(msgs):
        size = len(str(m.get("content", "")))
        if recent and used + size > keep_recent_chars:
            break
        recent.append(m)
        used += size
    recent.reverse()
    older = msgs[: len(msgs) - len(recent)]
    return older, recent


async def build_context(
    llm: LLMProto,
    history: Sequence[dict],
    prev_summary: str | None,
    *,
    max_chars: int,
    keep_recent_chars: int,
) -> ContextPlan:
    """组装上下文：超预算时把旧历史（含旧摘要）压缩进新摘要。"""
    older, recent = split_history(
        history, max_chars=max_chars, keep_recent_chars=keep_recent_chars
    )
    if not older:
        # 未超预算：保留既有摘要，不动历史
        return ContextPlan(messages=list(history), summary=prev_summary, compressed=False)

    prev_section = f"已有摘要（合并进新摘要，不要遗漏未完成事项）：\n{prev_summary}\n" if prev_summary else ""
    chunk = "\n".join(_render(m) for m in older)
    resp = await llm.ainvoke(
        [{"role": "user", "content": SUMMARY_PROMPT.format(prev_section=prev_section, chunk=chunk)}]
    )
    new_summary = str(getattr(resp, "content", resp)).strip()
    return ContextPlan(messages=recent, summary=new_summary, compressed=True)

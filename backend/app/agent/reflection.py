"""反思环节：综合产出后自检质量，低于阈值且给出修订则替换答案。

设计决策（面试可讲）：
- 只在 plan 模式综合之后跑一次（direct 单循环的输出本就短，反思收益低、
  延迟成本占比高）；失败/被拒步骤的报告一并提供，让审查器知道哪些部分
  本就不完整，避免"因失败信息缺失而误判"。
- fail-open：反思调用失败 / JSON 解析失败 → 视为质量 5 分、不修订——
  反思是增益环节，绝不能把已经流式产出的答案搞丢。
- 修订替换走独立的 reflection 事件（前端整体替换正文），不再走 text 增量——
  综合文本已经流式上屏，二次增量会造成内容重复。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from app.agent.planner import _extract_json

REFLECT_PROMPT = """你是质量审查器。审查以下回答是否充分、准确地回应了用户问题。

用户问题：{query}

回答草稿：
{answer}

执行中的失败/未完成部分（这些部分本就不完整，不要因它们的存在而全盘否定）：
{failed_report}

输出 JSON（不要输出其他内容）：
{{"quality": 1到5的整数, "issues": ["问题点，无则空数组"], "revised": "仅当质量低于等于2时给出修订后的完整回答，否则留空字符串"}}

要求：只基于已有信息判断与修订，不要编造执行中未获得的数据。
"""


class LLMProto(Protocol):
    async def ainvoke(self, messages: Sequence[Any], **kwargs: Any) -> Any: ...


@dataclass
class ReflectionResult:
    quality: int = 5
    issues: list[str] = field(default_factory=list)
    revised: str = ""


async def reflect(
    llm: LLMProto,
    query: str,
    answer: str,
    failed_report: str = "",
    *,
    min_quality: int = 3,
) -> ReflectionResult:
    """自检回答质量；任何失败 fail-open（quality=5，不修订）。"""
    prompt = [{
        "role": "user",
        "content": REFLECT_PROMPT.format(
            query=query, answer=answer,
            failed_report=failed_report or "（无）",
        ),
    }]
    try:
        resp = await llm.ainvoke(prompt)
        data = _extract_json(str(getattr(resp, "content", resp)))
        quality = max(1, min(5, int(data.get("quality", 5))))
        issues = [str(i) for i in (data.get("issues") or []) if str(i).strip()]
        revised = str(data.get("revised") or "").strip()
        if quality < min_quality and revised:
            return ReflectionResult(quality=quality, issues=issues, revised=revised)
        return ReflectionResult(quality=quality, issues=issues)
    except Exception:  # noqa: BLE001 — 反思失败绝不影响已产出答案
        return ReflectionResult(quality=5)

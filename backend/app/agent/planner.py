"""任务规划：复杂度分类 → 任务拆解 → 逐步执行 → 失败重规划。

三阶段模型（Plan-and-Execute）：
1. classify:   判断查询是否需要规划（简单问答直接走 create_agent 单循环）
2. plan:       LLM 输出结构化步骤 JSON（每步指定目标/检查点，可提示工具）
3. execute:    逐步执行；单步失败先原地重试（resilience.retry_async），
               仍失败则触发 replan——把失败上下文喂回 LLM 修订剩余步骤

所有 LLM 输出都经 parse_* + 一次修复重试，避免「模型多输出一句话导致 JSON 解析失败」。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


class LLMProto(Protocol):
    async def ainvoke(self, messages: Sequence[Any], **kwargs: Any) -> Any: ...


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class PlanStep:
    """计划中的一步。"""

    id: int
    title: str                 # 步骤标题（给用户看）
    goal: str                  # 执行目标（给执行器看）
    tool_hint: str | None = None   # 建议工具（执行器可参考，不强制）
    depends_on: list[int] = field(default_factory=list)  # 前置步骤 id（空 = 可立即执行/并行）
    status: str = "pending"    # pending / running / done / failed / skipped
    result: str | None = None  # 执行结果摘要
    error: str | None = None   # 失败原因（replan 的输入）

    def to_dict(self) -> dict:
        return {
            "id": self.id, "title": self.title, "goal": self.goal,
            "tool_hint": self.tool_hint, "depends_on": self.depends_on,
            "status": self.status,
            "result": self.result, "error": self.error,
        }


@dataclass
class Plan:
    steps: list[PlanStep] = field(default_factory=list)
    revised: int = 0            # 重规划次数

    @property
    def pending(self) -> list[PlanStep]:
        return [s for s in self.steps if s.status == "pending"]

    @property
    def failed(self) -> list[PlanStep]:
        return [s for s in self.steps if s.status == "failed"]


# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------
CLASSIFY_PROMPT = """你是任务路由器。判断用户请求是否需要多步规划。
复杂任务 = 需要组合 2 个及以上信息源/工具，或含多阶段目标（分析→结论→落地）。
简单任务 = 单次查询、闲聊、一步可答的问题。

只输出 JSON：{{"complexity": "simple" | "complex", "reason": "<10字内>"}}

用户请求：{query}"""

PLAN_PROMPT = """你是任务规划器。把用户目标拆解为 {max_steps} 步以内的可执行计划。

可用工具：{tools}
已有事实（来自历史/摘要，规划时可直接引用）：
{context}

要求：
- 每步是一个独立可验证的执行单元，goal 具体明确
- tool_hint 填最可能用到的工具名（从可用工具选），没有则 null
- depends_on 填该步骤的前置步骤 id 列表（1 起始）；无依赖填 []（可与其他步骤并行）
  只有存在数据依赖时才设依赖——例如「分析成绩」必须等「查成绩」完成；
  「查课表」和「查成绩」互不依赖应填 []
- 最后一步通常是综合产出（回答/保存），依赖所有信息收集步骤

只输出 JSON：
{{"steps": [{{"title": "<短标题>", "goal": "<具体目标>", "tool_hint": "<工具名或null>", "depends_on": [<前置id>...]}}]}}

用户目标：{query}"""

REPLAN_PROMPT = """执行计划时有步骤失败，请修订剩余计划。

原始目标：{query}
可用工具：{tools}
已完成步骤及结果：
{done_report}
失败步骤与原因：
{failed_report}
原计划剩余步骤：
{pending_report}

要求：
- 保留仍然有效的剩余步骤，替换/绕开失败部分
- 若失败源于工具不存在，换用其他工具或降级为直接回答
- depends_on 用**新列表中的 id**（1 起始）表达依赖；互不依赖的步骤填 [] 以并行
- 步骤总数不超过 {max_steps}

只输出 JSON：
{{"steps": [{{"title": "<短标题>", "goal": "<具体目标>", "tool_hint": "<工具名或null>", "depends_on": [<前置id>...]}}]}}"""


# ---------------------------------------------------------------------------
# JSON 解析（含容错与一次修复）
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> dict:
    """从模型输出提取第一个 JSON 对象（容忍代码围栏与前后缀文本）。"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("输出中未找到 JSON 对象")
    return json.loads(text[start : end + 1])


async def _ask_json(llm: LLMProto, messages: list[dict]) -> dict:
    """调用 LLM 并解析 JSON；解析失败时带错误信息重试一次（修复提示）。"""
    async def once(extra: str = "") -> dict:
        msgs = messages + ([{"role": "user", "content": extra}] if extra else [])
        resp = await llm.ainvoke(msgs)
        return _extract_json(str(getattr(resp, "content", resp)))

    try:
        return await once()
    except (ValueError, json.JSONDecodeError):
        try:
            return await once(
                "你上一次的输出无法解析为 JSON。请只输出合法 JSON，不要任何其他文字。"
            )
        except (ValueError, json.JSONDecodeError):
            # 第二次仍失败：把原错误抛给上层做降级（回退单循环）
            raise


def parse_plan(data: dict) -> list[PlanStep]:
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("计划中 steps 缺失或为空")
    steps = []
    for i, s in enumerate(raw_steps, 1):
        if not isinstance(s, dict) or not s.get("goal"):
            raise ValueError(f"第 {i} 步缺少 goal")
        hint = s.get("tool_hint")
        # 依赖语义（兼容旧输出）：
        # - 未提供 depends_on（或 null）→ 默认串行：依赖上一步（保持原行为）
        # - 显式 [] → 无前置，可并行
        # - 显式列表 → 只允许引用更早的 id（di < i），结构上不可能成环（DAG 由构造保证）
        raw_deps = s.get("depends_on")
        if raw_deps is None:
            deps = [i - 1] if i > 1 else []
        elif isinstance(raw_deps, list):
            deps = []
            for d in raw_deps:
                try:
                    di = int(d)
                except (TypeError, ValueError):
                    continue
                if 1 <= di < i and di not in deps:
                    deps.append(di)
        else:
            deps = [i - 1] if i > 1 else []
        steps.append(PlanStep(
            id=i,
            title=str(s.get("title") or f"步骤{i}"),
            goal=str(s["goal"]),
            tool_hint=None if hint in (None, "null", "") else str(hint),
            depends_on=deps,
        ))
    return steps


def renumber_steps(steps: list[PlanStep], max_existing_id: int) -> list[PlanStep]:
    """重规划后把新步骤 id 整体平移，避免与历史步骤 id 冲突（依赖同步平移）。"""
    for i, s in enumerate(steps, 1):
        s.depends_on = [max_existing_id + d for d in s.depends_on]
        s.id = max_existing_id + i
    return steps


def plan_from_dicts(items: list[dict], revised: int = 0) -> Plan:
    """从持久化 JSON（to_dict 列表）还原计划——断点续跑入口。

    保留各步骤的 status/result/error：done 步骤直接复用其产出，
    execute_plan 只会执行仍为 pending 的步骤，天然实现「从断点继续」。
    中断时正在执行的步骤（running）复位为 pending——它没有产出，
    续跑时必须重新执行（写工具由幂等键兜底，不会重复落库）。
    """
    steps = []
    for d in items:
        status = str(d.get("status", "pending"))
        if status == "running":
            status = "pending"
        steps.append(PlanStep(
            id=int(d["id"]),
            title=str(d.get("title", "")),
            goal=str(d.get("goal", "")),
            tool_hint=d.get("tool_hint"),
            depends_on=[int(x) for x in (d.get("depends_on") or [])],
            status=status,
            result=d.get("result"),
            error=d.get("error"),
        ))
    return Plan(steps=steps, revised=int(revised))


# ---------------------------------------------------------------------------
# 三个阶段
# ---------------------------------------------------------------------------
async def classify_complexity(llm: LLMProto, query: str) -> tuple[bool, str]:
    """返回 (是否复杂, 理由)。分类失败保守归为简单（直接走单循环）。"""
    try:
        data = await _ask_json(llm, [{"role": "user", "content": CLASSIFY_PROMPT.format(query=query)}])
        return data.get("complexity") == "complex", str(data.get("reason", ""))
    except Exception:  # noqa: BLE001 — 分类失败不应阻断对话
        return False, "classify_failed"


async def create_plan(
    llm: LLMProto,
    query: str,
    tools: Sequence[str],
    context: str,
    max_steps: int,
) -> list[PlanStep]:
    data = await _ask_json(llm, [{
        "role": "user",
        "content": PLAN_PROMPT.format(
            max_steps=max_steps, tools=", ".join(tools) or "（无）",
            context=context or "（无）", query=query,
        ),
    }])
    return parse_plan(data)[:max_steps]


async def replan(
    llm: LLMProto,
    query: str,
    plan: Plan,
    tools: Sequence[str],
    max_steps: int,
) -> list[PlanStep]:
    """基于失败上下文修订：只返回「剩余步骤」的新列表（含替代步骤）。"""
    done_report = "\n".join(
        f"- [{s.id}] {s.title}: {s.result}" for s in plan.steps if s.status == "done"
    ) or "（无）"
    failed_report = "\n".join(
        f"- [{s.id}] {s.title}: {s.error}" for s in plan.failed
    ) or "（无）"
    pending_report = "\n".join(
        f"- [{s.id}] {s.title}: {s.goal}" for s in plan.pending
    ) or "（无）"
    data = await _ask_json(llm, [{
        "role": "user",
        "content": REPLAN_PROMPT.format(
            query=query, tools=", ".join(tools) or "（无）",
            done_report=done_report, failed_report=failed_report,
            pending_report=pending_report, max_steps=max_steps,
        ),
    }])
    return parse_plan(data)[:max_steps]

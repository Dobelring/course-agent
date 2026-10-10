"""Plan-and-Execute 执行器：逐步执行 → 单步重试 → 失败重规划 → 综合产出。

与 planner.py（纯规划逻辑）分工：
- planner  负责「怎么拆、怎么改」——分类、拆解、重规划的 LLM 交互与 JSON 容错
- 本模块   负责「怎么跑」——执行循环、重试、降级、综合，LLM/工具执行通过
  ``run_step`` 回调注入，便于单测与 SSE 事件转发

失败语义（面试可讲的设计决策）：
- 单步失败：先原地指数退避重试（瞬时故障，如网络抖动）
- 重试耗尽：把失败上下文喂回 LLM 重规划（结构化故障，如工具选错）
- 重规划也失败 / 次数耗尽：停止执行剩余步骤，标记 degraded，已产出的部分结果
  仍会进入综合——保证「有总比没有强」
- 一步都未成功：返回 success=False，上层降级回直接模式（单循环）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol, Sequence

from app.agent.planner import Plan, PlanStep, replan as replan_llm
from app.agent.resilience import retry_async


class LLMProto(Protocol):
    async def ainvoke(self, messages: Sequence[Any], **kwargs: Any) -> Any: ...


EmitFn = Callable[[dict], Awaitable[None]]
StepRunner = Callable[[PlanStep], Awaitable[str]]


SYNTHESIS_PROMPT = """你是综合器。基于以下计划执行结果回答用户问题。

用户问题：{query}

已完成步骤的结果：
{done_report}

未完成/失败的部分（如实说明，不要编造）：
{failed_report}

要求：
- 直接给出完整最终答案，不要复述执行过程
- 失败部分若影响结论完整性，明确指出局限
"""


@dataclass
class ExecutionResult:
    answer: str
    success: bool                    # 是否有至少一个步骤成功产出
    plan: Plan
    degraded: bool = False           # 是否发生了降级（重规划耗尽/综合回退）
    error: str | None = None
    steps_done: int = 0
    steps_failed: int = 0


async def _noop_emit(event: dict) -> None:
    return None


async def synthesize(
    llm: LLMProto,
    query: str,
    plan: Plan,
    emit: EmitFn = _noop_emit,
) -> str:
    """把各步骤结果综合成最终答案；优先流式（边生成边 emit text 事件）。

    失败回退顺序：流式异常且无产出 → 非流式重试 → 拼接步骤结果。
    """
    done = [s for s in plan.steps if s.status == "done"]
    failed = [s for s in plan.steps if s.status in ("failed", "pending", "running")]

    done_report = "\n".join(
        f"- {s.title}：{s.result}" for s in done
    ) or "（无）"
    failed_report = "\n".join(
        f"- {s.title}：{s.error or '未执行'}" for s in failed
    ) or "（无）"
    prompt = [{
        "role": "user",
        "content": SYNTHESIS_PROMPT.format(
            query=query, done_report=done_report, failed_report=failed_report,
        ),
    }]

    # 1) 流式：ChatOpenAI 等实现提供 .stream()，逐 chunk 转发给前端
    if hasattr(llm, "stream"):
        parts: list[str] = []
        try:
            async for chunk in llm.stream(prompt):
                delta = getattr(chunk, "content", "")
                if isinstance(delta, str) and delta:
                    parts.append(delta)
                    await emit({"type": "text", "delta": delta})
            if parts:
                return "".join(parts)
        except Exception:  # noqa: BLE001
            if parts:  # 流到一半断开：已产出部分直接采用，不再重复生成
                return "".join(parts)
            # 未产出任何内容 → 落到非流式重试

    # 2) 非流式重试
    async def _call() -> str:
        resp = await llm.ainvoke(prompt)
        text = str(getattr(resp, "content", resp)).strip()
        if not text:
            raise ValueError("综合结果为空")
        return text

    try:
        return await retry_async(_call, retries=1, base_delay=0.5)
    except Exception:  # noqa: BLE001 — 综合失败回退为拼接，保证有产出
        parts = [s.result for s in done if s.result]
        return "\n\n".join(parts) if parts else ""


async def execute_plan(
    llm: LLMProto,
    query: str,
    plan: Plan,
    run_step: StepRunner,
    *,
    tools: Sequence[str] = (),
    step_retries: int = 2,
    max_replans: int = 2,
    emit: EmitFn = _noop_emit,
) -> ExecutionResult:
    """执行计划直到无 pending 步骤；返回综合后的执行结果。"""
    await emit({"type": "plan", "steps": [s.to_dict() for s in plan.steps]})
    replans_used = 0

    while True:
        step = next((s for s in plan.steps if s.status == "pending"), None)
        if step is None:
            break

        step.status = "running"
        await emit({"type": "step_start", "id": step.id, "title": step.title,
                    "goal": step.goal, "tool_hint": step.tool_hint,
                    "total": sum(1 for s in plan.steps if s.status in
                                 ("pending", "running", "done", "failed"))})

        try:
            result = await retry_async(
                lambda s=step: run_step(s),
                retries=step_retries,
                base_delay=0.5,
                on_retry=lambda n, e, _id=step.id: _emit_retry_sync(emit, _id, n, e),
            )
            if not str(result).strip():
                raise ValueError("步骤产出为空")
            step.status = "done"
            step.result = str(result).strip()
            await emit({"type": "step_end", "id": step.id, "status": "done",
                        "result": step.result[:500]})
        except Exception as exc:  # noqa: BLE001
            step.status = "failed"
            step.error = f"{type(exc).__name__}: {exc}"
            await emit({"type": "step_end", "id": step.id, "status": "failed",
                        "error": step.error})

            # 结构化故障 → 重规划
            if replans_used >= max_replans:
                await emit({"type": "degraded",
                            "reason": f"重规划次数已用尽（{max_replans}），停止执行剩余步骤"})
                _skip_pending(plan)
                break
            replans_used += 1
            try:
                new_steps = await replan_llm(llm, query, plan, tools, max_steps=len(plan.steps))
            except Exception as replan_exc:  # noqa: BLE001 — 重规划失败直接降级
                await emit({"type": "degraded",
                            "reason": f"重规划失败：{type(replan_exc).__name__}: {replan_exc}"})
                _skip_pending(plan)
                break
            # 用新步骤替换全部 pending（失败步骤保留作记录）
            plan.steps = [s for s in plan.steps if s.status != "pending"] + new_steps
            plan.revised += 1
            await emit({"type": "replan", "revised": plan.revised,
                        "steps": [s.to_dict() for s in new_steps]})

    done_count = sum(1 for s in plan.steps if s.status == "done")
    failed_count = sum(1 for s in plan.steps if s.status in ("failed", "skipped"))
    degraded = replans_used >= max_replans or any(
        s.status == "skipped" for s in plan.steps
    )

    if done_count == 0:
        first_error = next((s.error for s in plan.steps if s.error), None)
        return ExecutionResult(
            answer="", success=False, plan=plan, degraded=True,
            error=first_error or "所有步骤均未成功",
            steps_done=0, steps_failed=failed_count,
        )

    answer = await synthesize(llm, query, plan, emit=emit)
    return ExecutionResult(
        answer=answer, success=True, plan=plan, degraded=degraded,
        steps_done=done_count, steps_failed=failed_count,
    )


def _skip_pending(plan: Plan) -> None:
    for s in plan.steps:
        if s.status in ("pending", "running"):
            s.status = "skipped"


def _emit_retry_sync(emit: EmitFn, step_id: int, attempt: int, exc: BaseException) -> None:
    """retry_async 的 on_retry 是同步回调，这里用 asyncio 打包成异步 emit。"""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(emit({"type": "step_retry", "id": step_id, "attempt": attempt,
                           "error": f"{type(exc).__name__}: {exc}"}))

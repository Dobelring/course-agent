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

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol, Sequence

from app.agent.approval import ApprovalDenied
from app.agent.planner import Plan, PlanStep, renumber_steps, replan as replan_llm
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
    user_denied: bool = False        # 全部步骤被用户拒绝（区别于系统降级）
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

    # 1) 流式：ChatOpenAI 等提供异步 .astream()，逐 chunk 转发给前端。
    #    注意必须用 astream——曾经误用同步 .stream() 配 async for，
    #    TypeError 被吞后永远走非流式回退（规划模式不流式的根因）。
    if hasattr(llm, "astream"):
        parts: list[str] = []
        try:
            async for chunk in llm.astream(prompt):
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
    """执行计划：每轮取「就绪批次」（前置全部完成的 pending 步骤）并发执行。

    - 并行度由步骤依赖决定：depends_on 为空即就绪；批次内 asyncio.gather 并发
    - 批次内任一步骤失败 → 批次结束后统一重规划（替代旧的逐失败重规划）
    - 依赖失败/被拒的后续步骤标记 skipped（不无限等待，不死循环）
    - 用户拒绝（ApprovalDenied）→ skipped 且不计入重规划
    """
    await emit({"type": "plan", "steps": [s.to_dict() for s in plan.steps]})
    replans_used = 0

    async def _run_one(step: PlanStep) -> None:
        step.status = "running"
        await emit({"type": "step_start", "id": step.id, "title": step.title,
                    "goal": step.goal, "tool_hint": step.tool_hint,
                    "depends_on": list(step.depends_on),
                    "total": len(plan.steps)})
        try:
            result = await retry_async(
                lambda s=step: run_step(s),
                retries=step_retries,
                base_delay=0.5,
                no_retry=(ApprovalDenied,),
                on_retry=lambda n, e, _id=step.id: _emit_retry_sync(emit, _id, n, e),
            )
            if not str(result).strip():
                raise ValueError("步骤产出为空")
            step.status = "done"
            step.result = str(result).strip()
            await emit({"type": "step_end", "id": step.id, "status": "done",
                        "result": step.result[:500]})
        except ApprovalDenied as exc:
            # 用户主动拒绝：标记 skipped，不算失败、不触发重规划
            step.status = "skipped"
            step.error = str(exc) or "用户拒绝执行"
            await emit({"type": "step_end", "id": step.id, "status": "skipped",
                        "error": step.error})
        except Exception as exc:  # noqa: BLE001
            step.status = "failed"
            step.error = f"{type(exc).__name__}: {exc}"
            await emit({"type": "step_end", "id": step.id, "status": "failed",
                        "error": step.error})

    while True:
        pending = plan.pending
        if not pending:
            break
        by_id = {s.id: s for s in plan.steps}
        ready = [
            s for s in pending
            if all(d in by_id and by_id[d].status == "done" for d in s.depends_on)
        ]

        if not ready:
            # 无就绪步骤 ⟹ 所有 pending 的前置都已失败/被拒 → 阻塞跳过（保证终止）
            for s in pending:
                s.status = "skipped"
                s.error = "前置步骤未完成"
                await emit({"type": "step_end", "id": s.id, "status": "skipped",
                            "error": s.error})
            continue

        # 就绪批次并发执行（批次内步骤互不依赖）
        if len(ready) == 1:
            await _run_one(ready[0])
        else:
            await asyncio.gather(*[_run_one(s) for s in ready])

        # 批次结束统一处理失败 → 重规划（一次批次最多触发一次）
        if not any(s.status == "failed" for s in ready):
            continue
        if replans_used >= max_replans:
            await emit({"type": "degraded",
                        "reason": f"重规划次数已用尽（{max_replans}），停止执行剩余步骤"})
            _skip_pending(plan)
            break
        replans_used += 1
        try:
            new_steps = await replan_llm(llm, query, plan, tools,
                                         max_steps=len(plan.steps))
        except Exception as replan_exc:  # noqa: BLE001 — 重规划失败直接降级
            await emit({"type": "degraded",
                        "reason": f"重规划失败：{type(replan_exc).__name__}: {replan_exc}"})
            _skip_pending(plan)
            break
        # id 整体平移，避免与历史步骤冲突（前端按 id 关联事件）
        max_id = max(s.id for s in plan.steps)
        new_steps = renumber_steps(new_steps, max_id)
        plan.steps = [s for s in plan.steps if s.status != "pending"] + new_steps
        plan.revised += 1
        await emit({"type": "replan", "revised": plan.revised,
                    "steps": [s.to_dict() for s in new_steps]})

    done_count = sum(1 for s in plan.steps if s.status == "done")
    denied = [s for s in plan.steps
              if s.status == "skipped" and s.error and "拒绝" in s.error]
    failed_count = sum(1 for s in plan.steps if s.status in ("failed", "skipped"))
    # 用户拒绝不算降级；降级只看重规划耗尽或非拒绝原因的跳过
    degraded = replans_used >= max_replans or any(
        s.status == "skipped" and s not in denied for s in plan.steps
    )

    if done_count == 0:
        first_error = next((s.error for s in plan.steps if s.error), None)
        user_denied = bool(denied) and len(denied) == sum(
            1 for s in plan.steps if s.status == "skipped")
        return ExecutionResult(
            answer="", success=False, plan=plan,
            degraded=not user_denied,
            user_denied=user_denied,
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

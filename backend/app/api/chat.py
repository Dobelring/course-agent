"""聊天接口：Agent 运行过程通过 SSE 实时推送到前端，结束后落库对话历史。

架构：模式执行（direct/plan）跑在后台任务里，事件统一写入 asyncio.Queue，
生成器只负责 drain 队列转发 SSE。这样写操作审批（ApprovalContext）发生在
工具执行深处时，approval_required 事件仍能实时送达前端，决策由另一条
HTTP 请求（/chat/approvals/{id}）解开 Future。

模式（payload.mode）：
- auto    轻量分类路由：简单走单循环，复杂走 Plan-and-Execute
- direct  强制单循环
- plan    强制规划执行（拆解 → 并发就绪批次 → 重规划 → 综合）

SSE 事件：mode / context / plan / step_start / step_text / step_retry /
step_end / replan / degraded / fallback / approval_required /
approval_resolved / tool_start / tool_end / text / error / done
"""
import asyncio
import json
import time
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from loguru import logger
from pydantic import BaseModel

from app.agent.approval import (
    ApprovalContext,
    ApprovalDenied,
    current_step_id,
    decide,
    gate_write_tools,
)
from app.agent.context import ContextPlan, build_context, split_history
from app.agent.engine import bind_user, build_agent, build_llm, mcp_client_config, pretty_tool_name
from app.agent.planner import Plan, PlanStep, classify_complexity, create_plan
from app.agent.plan_executor import ExecutionResult, execute_plan
from app.agent.resilience import retry_async
from app.agent.usage import UsageCollector
from app.config import settings
from app.core.security import get_current_user
from app.database import SessionLocal
from app.models.conversation import AiConversation, AiMessage
from app.models.user import User

router = APIRouter(prefix="/chat", tags=["chat"])

HISTORY_ROWS_LIMIT = 50


class ChatRequest(BaseModel):
    message: str
    conversation_id: int | None = None
    mode: Literal["auto", "direct", "plan"] = "auto"
    run_id: str | None = None  # 客户端生成，用于中途取消


class CancelRequest(BaseModel):
    run_id: str


# 活跃任务注册表：run_id → (owner_user_id, task)。
# 与审批注册表同局限：单进程内存，多 worker 需换 Redis（见 README 已知局限）。
ACTIVE_RUNS: dict[str, tuple[int, asyncio.Task]] = {}


def _register_run(run_id: str, user_id: int, task: asyncio.Task) -> None:
    ACTIVE_RUNS[run_id] = (user_id, task)


def _unregister_run(run_id: str) -> None:
    ACTIVE_RUNS.pop(run_id, None)


class ApprovalDecision(BaseModel):
    approve: bool


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _extract_sources(raw) -> list | None:
    """从 search_course_documents 的工具输出中尽量提取检索来源（文件名/分数/片段）。"""
    if raw is None:
        return None
    text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str)
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    # MCP 返回 content 块：{"content": [{"type": "text", "text": "..."}]} 或 {"type": "text", "text": "..."}
    if isinstance(data, dict):
        texts = []
        content = data.get("content")
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
        elif isinstance(content, str):
            texts = [content]
        elif data.get("type") == "text" and isinstance(data.get("text"), str):
            texts = [data["text"]]
        if texts:
            try:
                data = json.loads(texts[0])
            except (ValueError, TypeError):
                return None
    if isinstance(data, list):
        items = []
        for it in data:
            if isinstance(it, dict) and ("filename" in it or "content" in it):
                items.append(
                    {k: it.get(k) for k in ("filename", "score", "chunk_index", "content") if k in it}
                )
        return items or None
    return None


def _history_to_dicts(rows) -> list[dict]:
    """DB 行（新→旧）转成上下文模块使用的 dict 列表（旧→新）。"""
    return [
        {"role": "user" if r.role == "user" else "assistant", "content": r.content}
        for r in reversed(rows)
    ]


@router.post("/approvals/{approval_id}")
async def resolve_approval(
    approval_id: str,
    payload: ApprovalDecision,
    current_user: User = Depends(get_current_user),
):
    """写操作审批决策：由 SSE 流的消费方在弹窗确认/拒绝后调用。"""
    err = decide(approval_id, payload.approve, current_user.id)
    if err:
        status = 404 if "不存在" in err else 403
        raise HTTPException(status_code=status, detail=err)
    return {"ok": True, "approved": payload.approve}


@router.post("/cancel")
async def cancel_run(
    payload: CancelRequest,
    current_user: User = Depends(get_current_user),
):
    """中途取消：停掉该 run_id 的后台任务，SSE 流随即收到 cancelled 事件。

    归属校验：只能取消自己的任务；未注册（已结束/不存在）返回 404。
    """
    entry = ACTIVE_RUNS.get(payload.run_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="任务不存在或已结束")
    owner_id, task = entry
    if owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="任务不存在或已结束")
    if not task.done():
        task.cancel()
    return {"ok": True}


@router.post("")
async def chat(payload: ChatRequest, current_user: User = Depends(get_current_user)):
    async def gen():
        started_at = time.perf_counter()
        first_token_ms = None  # 第一个字输出时的耗时
        task: asyncio.Task | None = None
        run_id = payload.run_id or str(uuid.uuid4())  # 提前定义，finally 注销用
        # 用量收集器：挂在 LLM 实例上，覆盖分类/规划/步骤/综合全部调用
        usage = UsageCollector(
            price_input_per_m=settings.llm_price_input_per_m,
            price_output_per_m=settings.llm_price_output_per_m,
        )
        # 提前校验 Key 配置，失败直接返回错误事件
        try:
            llm = build_llm(usage)
        except ValueError as e:
            yield _sse({"type": "error", "message": str(e)})
            return

        db = SessionLocal()
        conversation = None
        collected_text: list[str] = []
        tool_records: list[dict] = []
        retrieval_sources = None
        try:
            # 复用已有会话或新建会话
            if payload.conversation_id:
                conversation = db.get(AiConversation, payload.conversation_id)
                if conversation is None or conversation.user_id != current_user.id:
                    conversation = None
            if conversation is None:
                conversation = AiConversation(
                    user_id=current_user.id,
                    title=payload.message[:20] or "新对话",
                )
                db.add(conversation)
                db.commit()
                db.refresh(conversation)

            # 每次会话拉起独立 MCP 服务器子进程（stdio），工具调用后自动清理
            client = MultiServerMCPClient(mcp_client_config())
            raw_tools = await client.get_tools()
            tool_names = {t.name for t in raw_tools}
            pretty_names = sorted({pretty_tool_name(n) for n in tool_names})
            tools = bind_user(raw_tools, current_user.id)

            # ---------- 上下文组装：预算切分 + 超预算压缩为摘要 ----------
            history_rows = (
                db.query(AiMessage)
                .filter(AiMessage.conversation_id == conversation.id)
                .order_by(AiMessage.id.desc())
                .limit(HISTORY_ROWS_LIMIT)
                .all()
            )
            history = _history_to_dicts(history_rows)
            try:
                ctx: ContextPlan = await build_context(
                    llm, history, conversation.summary,
                    max_chars=settings.agent_history_max_chars,
                    keep_recent_chars=settings.agent_history_keep_recent_chars,
                )
            except Exception as exc:  # noqa: BLE001 — 摘要压缩失败则退化为截断
                _, recent = split_history(
                    history,
                    max_chars=settings.agent_history_max_chars,
                    keep_recent_chars=settings.agent_history_keep_recent_chars,
                )
                ctx = ContextPlan(messages=recent, summary=conversation.summary, compressed=False)
                yield _sse({"type": "context", "compressed": False,
                            "fallback": f"摘要压缩失败：{type(exc).__name__}"})
            if ctx.compressed:
                conversation.summary = ctx.summary
                yield _sse({"type": "context", "compressed": True,
                            "summary_chars": len(ctx.summary or ""),
                            "kept_messages": len(ctx.messages)})

            # ---------- 模式路由 ----------
            mode = payload.mode
            classify_reason = ""
            if mode == "auto":
                is_complex, classify_reason = await classify_complexity(llm, payload.message)
                mode = "plan" if is_complex else "direct"
            yield _sse({"type": "mode", "mode": mode, "reason": classify_reason})

            history_msgs = [
                AIMessage(content=m["content"]) if m["role"] == "assistant"
                else HumanMessage(content=m["content"])
                for m in ctx.messages
            ]
            agent_input = {"messages": history_msgs + [HumanMessage(content=payload.message)]}

            # ---------- 事件队列：模式执行与审批事件的统一出口 ----------
            event_queue: asyncio.Queue = asyncio.Queue()

            async def sink(evt: dict) -> None:
                await event_queue.put(evt)

            approval = ApprovalContext(
                sink, timeout=settings.agent_approval_timeout, user_id=current_user.id,
            )
            approval_names = {
                n.strip() for n in settings.agent_approval_tools.split(",") if n.strip()
            }
            tools = gate_write_tools(tools, approval, approval_names)
            # 三级记忆注入：会话摘要 + 跨会话用户画像
            from app.agent.memory import maybe_refresh_profile, profile_block
            from app.models.memory import UserMemory as _UserMemory

            profile_row = (
                db.query(_UserMemory).filter(_UserMemory.user_id == current_user.id).first()
            )
            extra_system = "\n\n".join(
                part for part in (ctx.summary_block, profile_block(profile_row.profile if profile_row else None))
                if part
            )
            agent = build_agent(current_user, tools, extra_system=extra_system, llm=llm)

            # 请求级共享去重集：同一工具整次请求只上屏一次卡片（跨步骤去重）。
            # tool_records 按 (name, input) 记录——包装层/底层双事件折叠，
            # 但参数不同的真实调用仍各自留痕，统计不失真。
            shown_start: set[str] = set()
            shown_end: set[str] = set()
            recorded: set[tuple[str, str]] = set()

            async def forward_tool_events(events, step_id: int | None = None):
                """从 astream_events 流中提取工具事件并打标（step_id 可空）。"""
                async for event in events:
                    etype = event.get("event")
                    name = event.get("name") or ""
                    if etype == "on_tool_start" and name in tool_names:
                        if name in shown_start:
                            continue  # 同名工具本请求已展示过
                        shown_start.add(name)
                        inp = {
                            k: v for k, v in (event["data"].get("input") or {}).items()
                            if k != "user_id"  # 对前端隐藏内部用户标识
                        }
                        ev = {"type": "tool_start", "name": pretty_tool_name(name), "input": inp}
                        if step_id is not None:
                            ev["step_id"] = step_id
                        yield ev
                    elif etype == "on_tool_end" and name in tool_names:
                        out = event["data"].get("output")
                        end_inp = {
                            k: v for k, v in (event["data"].get("input") or {}).items()
                            if k != "user_id"
                        }
                        key = (name, json.dumps(end_inp, ensure_ascii=False, sort_keys=True,
                                                default=str))
                        if key not in recorded:
                            recorded.add(key)
                            tool_records.append({
                                "name": pretty_tool_name(name),
                                "input": json.dumps(end_inp, ensure_ascii=False, default=str)[:1000],
                                "output": json.dumps(out, ensure_ascii=False, default=str)[:1000],
                            })
                            if pretty_tool_name(name) == "search_course_documents":
                                nonlocal retrieval_sources
                                retrieval_sources = _extract_sources(out)
                        if name in shown_end:
                            continue  # 同名工具本请求已展示过结束态
                        shown_end.add(name)
                        ev = {"type": "tool_end", "name": pretty_tool_name(name),
                              "output": json.dumps(out, ensure_ascii=False, default=str)[:500]}
                        if step_id is not None:
                            ev["step_id"] = step_id
                        yield ev
                    else:
                        yield {"_raw": event}  # 交由调用方处理文本流

            # 规划对象暂存：run_step 读取已完成步骤结果（跨步骤上下文传递）
            plan_holder: dict = {}

            # ---------- direct：单循环（事件入队，由生成器 drain） ----------
            async def run_direct() -> None:
                async for ev in forward_tool_events(
                    agent.astream_events(agent_input, version="v2")
                ):
                    if ev.get("_raw") is not None:
                        event = ev["_raw"]
                        if event.get("event") == "on_chat_model_stream":
                            chunk = event["data"].get("chunk")
                            content = chunk.content if chunk else None
                            if isinstance(content, str) and content:
                                await sink({"type": "text", "delta": content})
                    else:
                        await sink(ev)

            # ---------- plan：规划执行（审批发生在 run_step 深处） ----------
            def _completed_steps_summary(current_id: int) -> str:
                """已完成步骤的结果注入指令——后步复用前步产出，不再重复查询。"""
                plan: Plan | None = plan_holder.get("plan")
                if plan is None:
                    return ""
                parts = []
                for s in plan.steps:
                    if s.id != current_id and s.status == "done" and s.result:
                        parts.append(f"- {s.title}：{s.result[:400]}")
                if not parts:
                    return ""
                return (
                    "\n已完成步骤的结果（直接引用，禁止重复调用相同工具）：\n"
                    + "\n".join(parts)
                )

            async def run_step(step: PlanStep) -> str:
                """执行计划中的单步；返回文本结果，用户拒绝写操作则抛 ApprovalDenied。"""
                token = current_step_id.set(step.id)  # 审批拒绝归属到本步骤
                try:
                    instruction = (
                        f"你正在执行一个多步计划中的一步（总目标：{payload.message}）。\n"
                        f"本步目标：{step.goal}\n"
                        + (f"建议使用的工具：{step.tool_hint}\n" if step.tool_hint else "")
                        + _completed_steps_summary(step.id) + "\n"
                        + "规则：若已完成步骤的结果中已包含保存成功的信息，禁止重复保存；"
                        "请完成本步并输出简明结果（一两句即可），不要重复总目标、不要执行其他步骤。"
                    )
                    chunks: list[str] = []
                    step_input = {"messages": history_msgs + [HumanMessage(content=instruction)]}
                    async for ev in forward_tool_events(
                        agent.astream_events(step_input, version="v2"),
                        step_id=step.id,
                    ):
                        if ev.get("_raw") is not None:
                            event = ev["_raw"]
                            if event.get("event") == "on_chat_model_stream":
                                chunk = event["data"].get("chunk")
                                content = chunk.content if chunk else None
                                if isinstance(content, str) and content:
                                    chunks.append(content)
                                    await sink({"type": "step_text", "id": step.id,
                                                "delta": content})
                        else:
                            await sink(ev)
                    if approval.denials_for(step.id):
                        raise ApprovalDenied("用户拒绝执行该写操作")
                    return "".join(chunks)
                finally:
                    current_step_id.reset(token)

            mode_used = mode
            degraded = False
            plan_meta: dict | None = None

            async def run_plan() -> dict:
                """规划执行主流程；规划失败/无产出时降级 direct（事件仍入同一队列）。"""
                nonlocal mode_used, degraded, plan_meta
                plan: Plan | None = None
                try:
                    steps = await retry_async(
                        lambda: create_plan(
                            llm, payload.message, pretty_names,
                            ctx.summary_block, settings.agent_max_steps,
                        ),
                        retries=1, base_delay=0.5,
                    )
                    plan = Plan(steps=steps)
                    plan_holder["plan"] = plan  # 供 run_step 读取前序步骤结果
                except Exception as exc:  # noqa: BLE001
                    await sink({"type": "fallback", "to": "direct",
                                "reason": f"规划失败：{type(exc).__name__}: {exc}"})
                    mode_used = "direct"
                    degraded = True

                executor: ExecutionResult | None = None
                if plan is not None:
                    text_seen = False

                    async def tracked_emit(evt: dict) -> None:
                        nonlocal text_seen
                        if evt.get("type") == "text":
                            text_seen = True
                        await sink(evt)

                    executor = await execute_plan(
                        llm, payload.message, plan, run_step,
                        tools=pretty_names,
                        step_retries=settings.agent_step_retries,
                        max_replans=settings.agent_max_replans,
                        emit=tracked_emit,
                    )
                    plan_meta = {
                        "steps": [s.to_dict() for s in executor.plan.steps],
                        "revised": executor.plan.revised,
                        "degraded": executor.degraded,
                        "done": executor.steps_done,
                        "failed": executor.steps_failed,
                    }
                    if executor.user_denied and not executor.answer:
                        # 用户主动拒绝：不降级（降级会再次触发审批），直接说明
                        await sink({"type": "text",
                                    "delta": "已按您的选择取消了写入操作，未保存任何数据。"})
                        return {"mode_used": "plan", "plan_meta": plan_meta,
                                "degraded": False}
                    if not executor.success:
                        # 一步都没成功 → 回退单循环，保证有回答
                        await sink({"type": "fallback", "to": "direct",
                                    "reason": executor.error or "规划执行无产出"})
                        mode_used = "direct"
                        degraded = True
                    elif executor.answer and not text_seen:
                        # 综合走了非流式回退 → 整段补发
                        await sink({"type": "text", "delta": executor.answer})

                if mode_used == "direct":
                    await run_direct()
                return {"mode_used": mode_used, "plan_meta": plan_meta,
                        "degraded": degraded}

            # ---------- 启动后台任务 ----------
            if mode == "plan":
                task = asyncio.create_task(run_plan())
            else:
                async def _direct_flow() -> dict:
                    await run_direct()
                    return {"mode_used": "direct", "plan_meta": None, "degraded": False}
                task = asyncio.create_task(_direct_flow())
            _register_run(run_id, current_user.id, task)

            # ---------- drain 队列 → SSE（任务完成且队列排空后退出） ----------
            while True:
                if task.done() and event_queue.empty():
                    break
                try:
                    evt = await asyncio.wait_for(event_queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                etype = evt.get("type")
                if etype in ("text", "step_text") and first_token_ms is None:
                    first_token_ms = int((time.perf_counter() - started_at) * 1000)
                if etype == "text":
                    collected_text.append(evt.get("delta", ""))
                yield _sse(evt)

            # 任务异常 → 错误事件（由外层 except 统一处理）
            # 用户主动取消 → CancelledError（BaseException，不会被外层 except Exception 捕获）
            try:
                result = task.result()
            except asyncio.CancelledError:
                answer = "".join(collected_text)
                latency = int((time.perf_counter() - started_at) * 1000)
                # 保留已生成的部分内容（用户可能需要），标记为已停止
                if answer:
                    answer += "\n\n（已手动停止）"
                db.add(AiMessage(conversation_id=conversation.id, role="user", content=payload.message))
                db.add(
                    AiMessage(
                        conversation_id=conversation.id,
                        role="assistant",
                        content=answer,
                        latency_ms=latency,
                        first_token_ms=first_token_ms,
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        cost_cny=usage.cost_cny,
                    )
                )
                db.commit()
                yield _sse(
                    {
                        "type": "cancelled",
                        "conversation_id": conversation.id,
                        "latency_ms": latency,
                        "partial_chars": len(answer),
                    }
                )
                return

            answer = "".join(collected_text)
            latency = int((time.perf_counter() - started_at) * 1000)
            usage_dict = usage.to_dict()
            # 保存对话历史
            db.add(AiMessage(conversation_id=conversation.id, role="user", content=payload.message))
            db.add(
                AiMessage(
                    conversation_id=conversation.id,
                    role="assistant",
                    content=answer,
                    tool_calls=json.dumps(tool_records, ensure_ascii=False) if tool_records else None,
                    retrieval_sources=(
                        json.dumps(retrieval_sources, ensure_ascii=False) if retrieval_sources else None
                    ),
                    plan=(
                        json.dumps(result["plan_meta"], ensure_ascii=False)
                        if result and result.get("plan_meta") is not None else None
                    ),
                    latency_ms=latency,
                    first_token_ms=first_token_ms,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cost_cny=usage.cost_cny,
                )
            )
            db.commit()
            yield _sse(
                {
                    "type": "done",
                    "conversation_id": conversation.id,
                    "latency_ms": latency,
                    "first_token_ms": first_token_ms,
                    "mode": result["mode_used"] if result else mode_used,
                    "degraded": result["degraded"] if result else degraded,
                    "plan": result["plan_meta"] if result else plan_meta,
                    "usage": usage_dict,
                }
            )
            # 跨会话画像刷新：节流（每 N 条触发）+ best-effort，放在 done 之后不拖尾延迟
            try:
                await maybe_refresh_profile(
                    llm, db, current_user.id,
                    _history_to_dicts(history_rows) + [
                        {"role": "user", "content": payload.message},
                        {"role": "assistant", "content": answer},
                    ],
                    every=settings.memory_refresh_every,
                )
            except Exception:  # noqa: BLE001 — 画像失败绝不影响已完成的对话
                logger.warning("画像刷新异常", exc_info=True)
        except Exception as e:  # noqa: BLE001
            yield _sse({"type": "error", "message": f"服务异常：{e}"})
        finally:
            if task is not None and not task.done():
                task.cancel()  # 客户端断开等中断场景：停止后台执行
            _unregister_run(run_id)
            db.close()

    return StreamingResponse(gen(), media_type="text/event-stream")

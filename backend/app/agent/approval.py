"""写操作审批（human-in-the-loop）：工具执行前经 SSE 征得用户同意。

架构（为什么需要队列）：
    审批事件发生在工具执行内部（agent.astream_events 深处），而 SSE 生成器
    正阻塞等待下一个事件——审批事件无法直接 yield。解法：整个模式执行改为
    「后台任务 + 事件队列」，生成器只负责 drain 队列；工具内的审批经
    ApprovalContext.emit 写入同一队列，决策由另一条 HTTP 请求（审批端点）
    解开 Future，两侧解耦。

失败安全（fail-closed）：
    - 超时未决 → 视为拒绝
    - 决策端点校验归属用户，不能替他人审批
    - 拒绝后工具绝不执行，计划中该步骤标记 skipped（非失败，不触发重规划）
"""
from __future__ import annotations

import asyncio
import json
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from langchain_core.tools import BaseTool
from loguru import logger

from app.core import redisbus, redisc

# 计划步骤归属：run_step 内设置，供审批拒绝回传到具体步骤（并行时各任务独立）
current_step_id: ContextVar[int | None] = ContextVar("current_step_id", default=None)


class ApprovalDenied(Exception):
    """用户拒绝执行该写操作。"""


@dataclass
class PendingApproval:
    approval_id: str
    tool: str
    owner_id: int
    step_id: int | None
    future: asyncio.Future


# 模块级注册表：决策请求与 SSE 流是两条 HTTP 连接，只能经全局表桥接。
# Future 无法跨进程——Redis 启用时归属键 appr:{id}（TTL=审批超时+30s）声明
# 「该审批在哪个 worker 等待」，决策落在别的 worker 时经 Pub/Sub 广播回持有方；
# 未启用 Redis 时保持单进程语义（见 README 已知局限）。
PENDING: dict[str, PendingApproval] = {}


def _approval_key(approval_id: str) -> str:
    return redisc.key("appr", approval_id)


def _set_approval_key(approval_id: str, owner_id: int, ttl: int) -> None:
    redisc.try_op(
        lambda c: c.set(_approval_key(approval_id), owner_id, ex=ttl),
        what="approval key set",
    )


def _clear_approval_key(approval_id: str) -> None:
    redisc.try_op(lambda c: c.delete(_approval_key(approval_id)), what="approval key del")


def brief_args(args: dict, limit: int = 300) -> str:
    try:
        text = json.dumps(args, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(args)
    return text if len(text) <= limit else text[:limit] + "…"


class ApprovalContext:
    """单次请求的审批上下文：发事件、等决策、记录拒绝归属与干预日志。

    决策三态：
    - approve  → True，工具放行
    - reject   → False，denials 计数（步骤标记 skipped，不触发重规划）
    - revise   → 返回用户修改意见（str），由审批门包装成工具结果回给模型——
                 模型在同一轮内调整参数后重新发起审批，不整步重跑
    超时未决按拒绝（fail-closed）；revise 次数超上限也转为拒绝（防无限拉扯）。
    """

    def __init__(self, emit: Callable[[dict], Awaitable[None]], *,
                 timeout: float, user_id: int,
                 max_revises: int = 2,
                 record: Callable[[dict], None] | None = None) -> None:
        self._emit = emit
        self.timeout = timeout
        self.user_id = user_id
        self.max_revises = max_revises
        self._record = record  # 干预日志回调（chat.py 提供，写 approval_logs）
        self.denials: dict[int | None, int] = {}   # step_id -> 硬拒绝次数
        self.revises: dict[int | None, int] = {}   # step_id -> 修改后执行轮数

    def denials_for(self, step_id: int | None) -> int:
        return self.denials.get(step_id, 0)

    async def request(self, tool: str, args: dict) -> bool | str:
        """请求审批；True=批准 / False=拒绝 / str=用户修改意见。"""
        approval_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        step_id = current_step_id.get()
        PENDING[approval_id] = PendingApproval(
            approval_id=approval_id, tool=tool, owner_id=self.user_id,
            step_id=step_id, future=future,
        )
        _set_approval_key(approval_id, self.user_id, int(self.timeout) + 30)
        await self._emit({
            "type": "approval_required",
            "approval_id": approval_id,
            "tool": tool,
            "args": brief_args(args),
            "step_id": step_id,
            "timeout_s": int(self.timeout),
        })
        if self._record is not None:
            self._record({"kind": "request", "approval_id": approval_id,
                          "tool": tool, "args": brief_args(args), "step_id": step_id})
        timed_out = False
        try:
            result = await asyncio.wait_for(future, timeout=self.timeout)
        except asyncio.TimeoutError:
            result = False
            timed_out = True
        finally:
            PENDING.pop(approval_id, None)
            _clear_approval_key(approval_id)

        feedback: str | None = None
        if result is True:
            decision, reason, outcome = "approve", "", True
        elif timed_out:
            decision, reason, outcome = "timeout", "超时未决，按拒绝处理", False
        elif result is False:
            decision, reason, outcome = "reject", "用户拒绝", False
        else:
            feedback = str(result).strip()
            self.revises[step_id] = self.revises.get(step_id, 0) + 1
            if self.revises[step_id] > self.max_revises:
                decision, reason, outcome = "reject", "修改次数已用尽，按拒绝处理", False
            else:
                decision, reason, outcome = "revise", "用户要求修改后重新提交", feedback
        if outcome is False and decision in ("reject", "timeout"):
            self.denials[step_id] = self.denials.get(step_id, 0) + 1

        if self._record is not None:
            self._record({"kind": "decision", "approval_id": approval_id,
                          "decision": decision,
                          "feedback": feedback if decision == "revise" else None})
        await self._emit({
            "type": "approval_resolved",
            "approval_id": approval_id,
            "tool": tool,
            "approved": outcome is True,
            "reason": reason,
        })
        return outcome


def resolve_local(approval_id: str, decision: bool | str, user_id: int) -> str | None:
    """在本进程内解开审批 Future；返回 None 成功，否则返回失败原因。

    线程/循环安全：SSE 流与审批请求可能是两条连接甚至两个事件循环
    （生产 uvicorn 单循环走 set_result；跨循环用 call_soon_threadsafe）。
    """
    pending = PENDING.get(approval_id)
    if pending is None:
        return "审批不存在或已过期"
    if pending.owner_id != user_id:
        return "无权处理他人的审批"
    if pending.future.done():
        return None  # 幂等：重复决策直接成功
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is pending.future.get_loop():
        pending.future.set_result(decision)
    else:
        pending.future.get_loop().call_soon_threadsafe(
            pending.future.set_result, decision
        )
    return None


def decide(approval_id: str, decision: bool | str, user_id: int) -> str | None:
    """决策入口（供 API 调用）：True 批准 / False 拒绝 / str 修改意见。

    返回 None 成功，否则返回失败原因。
    本进程持有审批 → 直接解 Future（单 worker 快路径）；不在本进程 →
    查 Redis 归属键：存在则发布 Pub/Sub 信号给持有 worker（跨进程路径，
    归属键的存在性与 user_id 双重校验），不存在/Redis 未启用 → 与单进程
    一致返回「不存在或已过期」（fail-closed）。
    """
    if PENDING.get(approval_id) is not None:
        return resolve_local(approval_id, decision, user_id)
    owner = redisc.try_op(
        lambda c: c.get(_approval_key(approval_id)),
        default=None,
        what="approval key get",
    )
    if owner is None:
        return "审批不存在或已过期"
    if int(owner) != user_id:
        return "无权处理他人的审批"
    if not redisbus.publish_approval(approval_id, decision, user_id):
        # 归属键在但信号发不出：Redis 在两次调用之间失效，按不存在处理
        return "审批不存在或已过期"
    return None


def _on_approval_signal(data: dict) -> None:
    """Pub/Sub 落地：别的 worker 收到的决策在本进程（持有 Future）执行。"""
    err = resolve_local(
        str(data.get("approval_id", "")), data.get("decision", False),
        int(data.get("user_id", -1)),
    )
    if err is not None:
        # 常见于决策与超时竞态（request 已按超时拒绝）——记 debug 不当错误
        logger.debug("审批信号落地未命中：{}（{}）", err, data.get("approval_id"))


redisbus.set_handler(redisbus.PUBSUB_APPROVAL, _on_approval_signal)


class ApprovalGateTool(BaseTool):
    """包一层：执行前走审批；拒绝时返回拒绝文案给模型（不抛异常）。"""

    underlying: BaseTool
    approval: Any
    pretty_name: str

    async def _arun(self, **kwargs: Any) -> str:
        result = await self.approval.request(self.pretty_name, kwargs)
        if result is True:
            return await self.underlying.ainvoke(kwargs)
        if isinstance(result, str):
            # 修改后执行：意见作为工具结果回给模型，同一轮内调整参数重新发起审批
            return (
                f"【用户未批准当前参数，要求修改后重新提交】用户意见：{result}\n"
                "请根据意见调整工具参数后重新调用本工具；若意见无法满足，如实向用户说明。"
            )
        return (
            f"【操作已被用户拒绝】工具 {self.pretty_name} 未执行。"
            "请告知用户已取消该操作，不要重试。"
        )

    def _run(self, **kwargs: Any) -> str:
        raise RuntimeError("写操作审批依赖异步运行时（_arun），同步路径不应被调用")


def gate_write_tools(tools: list[BaseTool], approval: ApprovalContext,
                     approval_tools: set[str]) -> list[BaseTool]:
    """把配置命中的写工具包上审批门（按去掉命名空间前缀后的名字匹配）。"""
    if not approval_tools:
        return tools
    gated = []
    for t in tools:
        pretty = t.name.split("__")[-1]
        if pretty in approval_tools:
            gated.append(ApprovalGateTool(
                underlying=t, approval=approval, pretty_name=pretty,
                name=t.name, description=t.description, args_schema=t.args_schema,
            ))
        else:
            gated.append(t)
    return gated

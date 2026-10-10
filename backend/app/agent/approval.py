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
# 单进程 asyncio 单线程，无需加锁；多 worker 部署时需换 Redis（见 README 局限）。
PENDING: dict[str, PendingApproval] = {}


def brief_args(args: dict, limit: int = 300) -> str:
    try:
        text = json.dumps(args, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(args)
    return text if len(text) <= limit else text[:limit] + "…"


class ApprovalContext:
    """单次请求的审批上下文：发事件、等决策、记录拒绝归属。"""

    def __init__(self, emit: Callable[[dict], Awaitable[None]], *,
                 timeout: float, user_id: int) -> None:
        self._emit = emit
        self.timeout = timeout
        self.user_id = user_id
        self.denials: dict[int | None, int] = {}  # step_id -> 拒绝次数

    def denials_for(self, step_id: int | None) -> int:
        return self.denials.get(step_id, 0)

    async def request(self, tool: str, args: dict) -> bool:
        """请求审批；返回是否批准（超时视为拒绝）。"""
        approval_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        step_id = current_step_id.get()
        PENDING[approval_id] = PendingApproval(
            approval_id=approval_id, tool=tool, owner_id=self.user_id,
            step_id=step_id, future=future,
        )
        await self._emit({
            "type": "approval_required",
            "approval_id": approval_id,
            "tool": tool,
            "args": brief_args(args),
            "step_id": step_id,
            "timeout_s": int(self.timeout),
        })
        try:
            approved = await asyncio.wait_for(future, timeout=self.timeout)
            reason = "" if approved else "用户拒绝"
        except asyncio.TimeoutError:
            approved = False
            reason = "超时未决，按拒绝处理"
        finally:
            PENDING.pop(approval_id, None)

        await self._emit({
            "type": "approval_resolved",
            "approval_id": approval_id,
            "tool": tool,
            "approved": approved,
            "reason": reason,
        })
        if not approved:
            self.denials[step_id] = self.denials.get(step_id, 0) + 1
        return approved


def decide(approval_id: str, approve: bool, user_id: int) -> str | None:
    """决策入口（供 API 调用）。返回 None 成功，否则返回失败原因。

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
        pending.future.set_result(approve)
    else:
        pending.future.get_loop().call_soon_threadsafe(
            pending.future.set_result, approve
        )
    return None


class ApprovalGateTool(BaseTool):
    """包一层：执行前走审批；拒绝时返回拒绝文案给模型（不抛异常）。"""

    underlying: BaseTool
    approval: Any
    pretty_name: str

    async def _arun(self, **kwargs: Any) -> str:
        approved = await self.approval.request(self.pretty_name, kwargs)
        if not approved:
            return (
                f"【操作已被用户拒绝】工具 {self.pretty_name} 未执行。"
                "请告知用户已取消该操作，不要重试。"
            )
        return await self.underlying.ainvoke(kwargs)

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

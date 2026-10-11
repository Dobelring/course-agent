"""跨进程信号总线：Redis Pub/Sub 承载「取消 run」与「审批决策」。

架构（为什么信号 + 归属键，而不是把状态搬进 Redis）：
- asyncio.Future / Task 句柄无法跨进程传递，所以 ACTIVE_RUNS（run→Task）
  与 PENDING（approval→Future）仍留在持有任务的 worker 进程内存。
- Redis 只补单进程缺的两块：
  1. 归属键（run:xxx / appr:xxx，带 TTL）：请求落在任意 worker 都能判断
     「这个任务/审批在不在这套进程里、属于谁」——没有它就无法区分
     「在别的 worker」与「已结束」，跨进程取消只能一律 404。
  2. 信号广播：决策/取消发布到 channel，持有任务的 worker 订阅后在自己的
     事件循环里落地执行（call local future / cancel local task）。
- 归属校验在两端各做一次：入口端比对归属键的 user_id，落地端比对本地
  注册表的 owner——纵深防御，广播消息本身带 user_id 供落地端复核。

发布用同步客户端（decide/cancel 是毫秒级单次 PUBLISH），监听用异步
客户端常驻订阅；两者共享同一 Redis 实例。监听断线自动重连（1s 间隔）。
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

from loguru import logger

from app.core import redisc

PUBSUB_CANCEL = "course-agent:bus:cancel"
PUBSUB_APPROVAL = "course-agent:bus:approval"

# channel -> 本地落地回调（由持有状态的模块注册：approval.py / chat.py）
_handlers: dict[str, Callable[[dict], None]] = {}
_task: asyncio.Task | None = None
_stopping = False
# 订阅就绪信号：防止「发布早于订阅」的启动竞态。
# 每次 start() 新建——asyncio.Event 会绑定首个等待它的事件循环，
# 而测试各自 asyncio.run 新 loop，复用同一实例会报「bound to a different event loop」
_subscribed: asyncio.Event | None = None


async def wait_subscribed(timeout: float = 5.0) -> bool:
    """等待监听任务完成频道订阅；返回是否就绪（测试/启动同步用）。"""
    ev = _subscribed
    if ev is None:
        return False
    try:
        await asyncio.wait_for(ev.wait(), timeout)
        return True
    except asyncio.TimeoutError:
        return False


def set_handler(channel: str, fn: Callable[[dict], None]) -> None:
    """注册本地落地回调（模块导入时调用，幂等覆盖）。"""
    _handlers[channel] = fn


def publish(channel: str, payload: dict) -> bool:
    """发布一条信号；Redis 未启用或失败返回 False（调用方决定 fail 方向）。"""
    return redisc.try_op(
        lambda c: c.publish(channel, json.dumps(payload, ensure_ascii=False)) is not None,
        default=False,
        what=f"publish {channel}",
    )


def publish_cancel(run_id: str, user_id: int) -> bool:
    return publish(PUBSUB_CANCEL, {"run_id": run_id, "user_id": user_id})


def publish_approval(approval_id: str, decision: bool | str, user_id: int) -> bool:
    return publish(PUBSUB_APPROVAL, {
        "approval_id": approval_id, "decision": decision, "user_id": user_id,
    })


def dispatch(channel: str, data: dict) -> None:
    """把收到的消息交给本地回调（回调异常只记日志，不打断监听循环）。"""
    handler = _handlers.get(channel)
    if handler is None:
        return
    try:
        handler(data)
    except Exception:  # noqa: BLE001 — 落地失败不能影响订阅存活
        logger.warning("Pub/Sub 信号落地失败 channel={} data={}", channel, data, exc_info=True)


async def _listen_loop() -> None:
    while not _stopping:
        client = redisc.get_aioredis()
        if client is None:
            await asyncio.sleep(1)
            continue
        try:
            async with client.pubsub() as pubsub:
                await pubsub.subscribe(PUBSUB_CANCEL, PUBSUB_APPROVAL)
                if _subscribed is not None:
                    _subscribed.set()
                async for msg in pubsub.listen():
                    if _stopping:
                        break
                    if msg.get("type") != "message":
                        continue
                    try:
                        data = json.loads(msg["data"])
                    except (ValueError, TypeError):
                        continue
                    if isinstance(data, dict):
                        dispatch(str(msg["channel"]), data)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — 断线重连
            if _subscribed is not None:
                _subscribed.clear()
            logger.warning("Redis Pub/Sub 监听异常，1s 后重连：{}", exc)
            await asyncio.sleep(1)


async def start() -> bool:
    """启动监听任务（lifespan 调用）；Redis 未启用返回 False。"""
    global _task, _stopping, _subscribed
    if redisc.get_aioredis() is None:
        return False
    if _task is not None and not _task.done():
        return True
    _stopping = False
    _subscribed = asyncio.Event()  # 绑定当前（start 所在）事件循环
    _task = asyncio.create_task(_listen_loop(), name="redisbus-listener")
    logger.info("Redis Pub/Sub 监听已启动（cancel / approval）")
    return True


async def stop() -> None:
    """停止并等待监听任务（幂等）。"""
    global _task, _stopping, _subscribed
    _stopping = True
    _subscribed = None
    task, _task = _task, None
    if task is None:
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001 — 取消/断线收尾
        pass


def is_listening() -> bool:
    """监听任务是否在运行（测试/诊断用）。"""
    return _task is not None and not _task.done()

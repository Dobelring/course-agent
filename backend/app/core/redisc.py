"""Redis 接入层：可选启用、启动期连接、按调用降级。

设计取舍（面试可讲）：
- REDIS_URL 为空 = 完全不启用：所有能力走各自的进程内缺省路径，单 worker
  行为与引入 Redis 之前逐字节一致（252 例既有测试不感知 Redis 存在）。
- 连接失败的降级策略是**本进程停用**：首次 ping 失败或运行期任意一次操作
  抛连接异常，即置位 failed，后续调用直接走本地路径——Redis 挂掉不能拖垮
  API（可用性 > 一致性），恢复需重启进程（运维心智简单，个人规模够用）。
- 降级语义按能力区分：限流退回本地桶（每 worker 各有一份配额，宁可放行
  不可误杀）；审批/取消退回「任务不存在」（fail-closed）；缓存退回 miss。
- 测试用 fakeredis 注入（set_clients），不依赖本机 Redis 服务。
"""
from __future__ import annotations

from typing import Any, Callable, TypeVar

import redis
import redis.asyncio as aioredis
from loguru import logger

from app.config import settings

T = TypeVar("T")

KEY_PREFIX = "course-agent"  # 所有业务键统一前缀，避免与其他应用共库冲突

_sync: redis.Redis | None = None
_async: aioredis.Redis | None = None
_failed = False       # 连接过或运行期失败 → 本进程停用（重启恢复）
_owned = False        # 客户端是否由本模块创建（测试注入的不由我们关闭）


def key(*parts: Any) -> str:
    """业务键拼接：key("rl", "chat:user:1") -> course-agent:rl:chat:user:1。"""
    return ":".join([KEY_PREFIX, *(str(p) for p in parts)])


def set_clients(sync: redis.Redis, async_client: aioredis.Redis | None = None) -> None:
    """测试注入（fakeredis）：显式指定客户端，绕过 REDIS_URL 配置。"""
    global _sync, _async, _failed, _owned
    _sync, _async, _failed, _owned = sync, async_client, False, False


def reset_clients() -> None:
    """断开现有客户端并复位状态（测试 teardown / 连接失败后重试用）。"""
    global _sync, _async, _failed, _owned
    _sync, _async, _failed, _owned = None, None, False, False


def get_redis() -> redis.Redis | None:
    """同步客户端（限流/熔断/审批键/缓存等热路径）；不可用返回 None。"""
    global _sync, _failed
    if _failed:
        return None  # 曾经失败 → 本进程停用（见模块 docstring，重启恢复）
    if _sync is not None:
        return _sync
    if not settings.redis_url.strip():
        return None
    try:
        client = redis.Redis.from_url(
            settings.redis_url.strip(),
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=1,
        )
        client.ping()
    except Exception as exc:  # noqa: BLE001 — 连接失败一律降级
        _failed = True
        logger.warning("Redis 连接失败，本进程降级为进程内实现：{}", exc)
        return None
    _sync, _owned = client, True
    logger.info("Redis 已启用：{}", settings.redis_url.split("@")[-1])
    return _sync


def get_aioredis() -> aioredis.Redis | None:
    """异步客户端（Pub/Sub 监听专用）；不可用返回 None。"""
    global _async, _failed
    if _failed:
        return None
    if _async is not None:
        return _async
    if not settings.redis_url.strip():
        return None
    if get_redis() is None:  # 复用同步侧的连接判定（含 failed 置位）
        return None
    try:
        _async = aioredis.from_url(
            settings.redis_url.strip(),
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=1,
        )
    except Exception as exc:  # noqa: BLE001
        _failed = True
        logger.warning("Redis 异步客户端创建失败，跨进程信号不可用：{}", exc)
        return None
    return _async


def try_op(fn: Callable[[redis.Redis], T], default: T = None, what: str = "") -> T:
    """在同步客户端上执行一次操作；未启用/异常时返回 default 并记日志。

    这是「按调用降级」的统一入口：单次操作失败不改写事实判断之外的状态，
    仅把本次操作当 Redis 未启用处理（同时置位 failed，后续直接走本地）。
    """
    client = get_redis()
    if client is None:
        return default
    global _failed
    try:
        return fn(client)
    except Exception as exc:  # noqa: BLE001 — 运行期异常同样视为不可用
        _failed = True
        logger.warning("Redis 操作失败（{}），本进程降级为进程内实现：{}", what or "?", exc)
        return default


async def init() -> bool:
    """启动期连接检查（lifespan 调用）：返回是否启用 Redis。"""
    return get_redis() is not None


async def shutdown() -> None:
    """关闭本模块创建的客户端（测试注入的原样保留）。"""
    global _sync, _async, _owned
    try:
        if _sync is not None and _owned:
            _sync.close()
        if _async is not None and _owned:
            await _async.aclose()
    except Exception:  # noqa: BLE001 — 关闭失败无需处理
        pass
    finally:
        _sync, _async, _owned = None, None, False

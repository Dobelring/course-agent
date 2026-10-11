"""每用户/IP 令牌桶限流（进程内 + Redis 双后端）。

设计取舍（面试可讲）：
- 令牌桶 vs 固定/滑动窗口：桶允许小突发、长期速率受控；实现只需一个浮点计数，
  不必像滑动窗口那样保存时间片队列。
- 限在依赖层而非全局中间件：用户身份由 JWT 依赖解析，中间件层拿不到 user_id
  （除非再解析一次 token）；按路由声明式挂载——聊天（贵、按用户）与登录
  （防爆破、按 IP）各自配额，互不影响。
- 双后端：配置 REDIS_URL 时走 Redis Lua 令牌桶——补币与扣币在一段脚本内原子
  完成，多 worker 共享同一份配额（进程内 dict 每 worker 各一份，配额会被
  放大 N 倍）；未配置/Redis 异常时退回进程内 dict（单 worker 行为不变，
  Redis 挂掉宁可降配额精度也不误杀请求）。
- Lua 里的时间用 time.time()（墙钟）而非 monotonic：多 worker 必须共享同一
  时间基准，NTP 对齐的墙钟是唯一可选项（单机多 worker 天然同源）。
- 配额为 0 表示关闭（测试与评测脚本用）。
"""
from __future__ import annotations

import time

from fastapi import Depends, HTTPException, Request
from loguru import logger

from app.config import settings
from app.core import redisc
from app.core.security import get_current_user
from app.models.user import User

# key -> bucket。单进程 asyncio 单线程，无锁；量级 = 活跃用户数，无淘汰压力。
_BUCKETS: dict[str, "TokenBucket"] = {}


class TokenBucket:
    """经典令牌桶：capacity 容量、rate 每秒补充速率、惰性按时间差补币。"""

    def __init__(self, per_min: int) -> None:
        self.capacity = float(per_min)
        self.rate = per_min / 60.0  # tokens / second
        self.tokens = self.capacity
        self.updated = time.monotonic()

    def take(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def retry_after(self) -> int:
        """还差多少秒能凑出下一个令牌（向上取整，至少 1）。"""
        return max(1, int((1.0 - self.tokens) / self.rate) + 1)


# Redis 令牌桶：HMGET 读 (tokens, updated) → 按时间差补币 → 扣币 → 回写，
# 全程单脚本原子执行（并发下不存在读改写竞争）；key 过期 = 桶闲置自动清理。
_TOKEN_BUCKET_LUA = """
local k = KEYS[1]
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local data = redis.call('HMGET', k, 'tokens', 'updated')
local tokens = tonumber(data[1])
local updated = tonumber(data[2])
if tokens == nil or updated == nil then
  tokens = capacity
  updated = now
end
tokens = math.min(capacity, tokens + (now - updated) * rate)
local allowed = 0
local retry = 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
else
  retry = math.ceil((1 - tokens) / rate)
  if retry < 1 then retry = 1 end
end
redis.call('HSET', k, 'tokens', tokens, 'updated', now)
redis.call('EXPIRE', k, math.ceil(capacity / rate) + 60)
return {allowed, retry}
"""


def _take(key: str, per_min: int) -> None:
    """消耗一个令牌；配额不足抛 429（带 Retry-After）。"""
    r = redisc.get_redis()
    if r is not None:
        try:
            allowed, retry = r.eval(
                _TOKEN_BUCKET_LUA, 1, redisc.key("rl", key),
                float(per_min), per_min / 60.0, time.time(),
            )
            if int(allowed):
                return
            raise HTTPException(
                status_code=429,
                detail="请求过于频繁，请稍后再试",
                headers={"Retry-After": str(max(1, int(retry)))},
            )
        except HTTPException:
            raise
        except Exception:  # noqa: BLE001 — Redis 异常降级本地桶
            logger.warning("限流 Redis 操作失败，降级为本地桶", exc_info=True)
    # 本地桶（单 worker 缺省路径 / Redis 降级路径）
    bucket = _BUCKETS.get(key)
    if bucket is None:
        bucket = _BUCKETS[key] = TokenBucket(per_min)
    if not bucket.take():
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后再试",
            headers={"Retry-After": str(bucket.retry_after())},
        )


def limit_chat(per_min: int | None = None):
    """聊天接口限流依赖：按登录用户计。per_min=0 关闭。"""

    async def _dep(per_min: int = per_min, current_user: User = Depends(get_current_user)):
        limit = settings.rate_limit_chat_per_min if per_min is None else per_min
        if limit > 0:
            _take(f"chat:user:{current_user.id}", limit)

    return _dep


def limit_login(per_min: int | None = None):
    """登录接口限流依赖：按客户端 IP 计（防爆破），per_min=0 关闭。"""

    def _dep(request: Request, per_min: int = per_min):
        limit = settings.rate_limit_login_per_min if per_min is None else per_min
        if limit > 0:
            ip = request.client.host if request.client else "unknown"
            _take(f"login:ip:{ip}", limit)

    return _dep

"""每用户/IP 令牌桶限流（进程内实现）。

设计取舍（面试可讲）：
- 令牌桶 vs 固定/滑动窗口：桶允许小突发、长期速率受控；实现只需一个浮点计数，
  不必像滑动窗口那样保存时间片队列。
- 限在依赖层而非全局中间件：用户身份由 JWT 依赖解析，中间件层拿不到 user_id
  （除非再解析一次 token）；按路由声明式挂载——聊天（贵、按用户）与登录
  （防爆破、按 IP）各自配额，互不影响。
- 进程内 dict 存桶：与 ACTIVE_RUNS / PENDING 同一单进程前提（见 README 局限），
  多 worker 部署时换 Redis INCR+EXPIRE 或 Lua 令牌桶。
- 配额为 0 表示关闭（测试与评测脚本用）。
"""
from __future__ import annotations

import time

from fastapi import Depends, HTTPException, Request

from app.config import settings
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


def _take(key: str, per_min: int) -> None:
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

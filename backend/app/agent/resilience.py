"""异步重试：指数退避 + 抖动，LLM/工具瞬时故障的统一重试入口。

设计要点：
- 重试的是「调用」而非整个 Agent 循环，避免重复执行已成功的副作用；
- 抖动（jitter）防止多请求同步重试造成尖峰；
- 次数与延迟均可配置（见 Settings.agent_step_retries）。
"""
from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, Sequence, TypeVar

T = TypeVar("T")


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    retries: int = 2,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    jitter: float = 0.3,
    retry_on: Sequence[type[BaseException]] = (Exception,),
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    """执行 ``fn``（无参协程工厂），失败则指数退避后重试。

    :param retries:     首次失败后的额外重试次数（总尝试 = retries + 1）
    :param base_delay:  首次退避秒数，第 n 次重试延迟 = base * 2^(n-1)
    :param max_delay:   单次退避上限（秒）
    :param jitter:      抖动比例，实际延迟在 [d*(1-j), d*(1+j)] 内随机
    :param retry_on:    可重试异常类型；其余异常直接抛出
    :param on_retry:    每次决定重试时的回调（attempt 从 1 计），用于打日志/打点
    """
    attempt = 0
    while True:
        try:
            return await fn()
        except tuple(retry_on) as exc:
            if attempt >= retries:
                raise
            attempt += 1
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            delay *= 1 + random.uniform(-jitter, jitter)
            if on_retry is not None:
                on_retry(attempt, exc)
            await asyncio.sleep(max(delay, 0.0))

"""重试：成功直返、失败退避重试、次数耗尽抛出、可重试类型过滤。"""
import asyncio

import pytest

from app.agent.resilience import retry_async


def test_returns_immediately_on_success():
    calls = 0

    async def ok():
        nonlocal calls
        calls += 1
        return 42

    result = asyncio.run(retry_async(ok, retries=3, base_delay=0.01))
    assert result == 42
    assert calls == 1


def test_retries_then_succeeds():
    calls = 0

    async def flaky():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ConnectionError("瞬时故障")
        return "ok"

    result = asyncio.run(retry_async(flaky, retries=3, base_delay=0.01))
    assert result == "ok"
    assert calls == 3


def test_raises_after_exhausting_retries():
    calls = 0

    async def always_fail():
        nonlocal calls
        calls += 1
        raise TimeoutError("持续失败")

    with pytest.raises(TimeoutError):
        asyncio.run(retry_async(always_fail, retries=2, base_delay=0.01))
    assert calls == 3  # 1 次原始 + 2 次重试


def test_non_retryable_exception_raises_immediately():
    calls = 0

    async def bad_input():
        nonlocal calls
        calls += 1
        raise ValueError("参数错误，不应重试")

    with pytest.raises(ValueError):
        asyncio.run(retry_async(bad_input, retries=5, base_delay=0.01,
                                retry_on=(ConnectionError, TimeoutError)))
    assert calls == 1


def test_on_retry_callback_receives_attempts():
    attempts: list[int] = []

    async def flaky():
        if len(attempts) < 2:
            raise ConnectionError()
        return "done"

    asyncio.run(retry_async(flaky, retries=3, base_delay=0.01,
                            on_retry=lambda n, e: attempts.append(n)))
    assert attempts == [1, 2]


def test_backoff_delays_grow_exponentially():
    """验证退避序列：base * 2^(n-1)（jitter=0 时精确断言）。"""
    delays: list[float] = []
    calls = 0

    async def fail():
        nonlocal calls
        calls += 1
        raise ConnectionError()

    async def fake_sleep(d: float):
        delays.append(d)

    import app.agent.resilience as r

    original = r.asyncio.sleep
    r.asyncio.sleep = fake_sleep  # type: ignore[assignment]
    try:
        with pytest.raises(ConnectionError):
            asyncio.run(retry_async(fail, retries=3, base_delay=1.0, jitter=0.0))
    finally:
        r.asyncio.sleep = original  # type: ignore[assignment]

    assert delays == [1.0, 2.0, 4.0]
    assert calls == 4  # 1 次原始 + 3 次重试

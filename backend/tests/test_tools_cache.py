"""MCP 工具列表缓存：TTL 内复用、并发单飞、TTL=0 关闭、过期重拉。"""
import asyncio

import pytest

from app.agent import engine
from app.config import settings


class FakeMCPClient:
    count = 0

    def __init__(self, config):
        type(self).count += 1

    async def get_tools(self):
        await asyncio.sleep(0.01)
        return [{"tool": type(self).count}]


@pytest.fixture(autouse=True)
def _fake_and_reset(monkeypatch):
    FakeMCPClient.count = 0
    monkeypatch.setattr(engine, "MultiServerMCPClient", FakeMCPClient)
    engine.reset_tools_cache()
    yield
    engine.reset_tools_cache()


def test_cache_reuses_within_ttl(monkeypatch):
    monkeypatch.setattr(settings, "mcp_tools_cache_ttl", 300)
    assert engine.get_tools_cached.__name__  # 存在性
    r1 = asyncio.run(engine.get_tools_cached())
    r2 = asyncio.run(engine.get_tools_cached())
    assert r1 == r2 == [{"tool": 1}]
    assert FakeMCPClient.count == 1            # 只 spawn 一次


def test_concurrent_miss_flies_once(monkeypatch):
    """并发 miss 单飞：两个请求同时到达只 spawn 一个子进程。"""
    monkeypatch.setattr(settings, "mcp_tools_cache_ttl", 300)

    async def flow():
        return await asyncio.gather(engine.get_tools_cached(),
                                    engine.get_tools_cached())

    r1, r2 = asyncio.run(flow())
    assert r1 == r2
    assert FakeMCPClient.count == 1


def test_ttl_zero_disables_cache(monkeypatch):
    monkeypatch.setattr(settings, "mcp_tools_cache_ttl", 0)
    asyncio.run(engine.get_tools_cached())
    asyncio.run(engine.get_tools_cached())
    assert FakeMCPClient.count == 2            # 每请求拉取（旧行为）


def test_expired_ttl_refetches(monkeypatch):
    monkeypatch.setattr(settings, "mcp_tools_cache_ttl", 300)
    asyncio.run(engine.get_tools_cached())
    # 模拟时间流逝：把缓存时间戳拨回 TTL 之前
    engine._tools_cached_at -= settings.mcp_tools_cache_ttl + 1
    asyncio.run(engine.get_tools_cached())
    assert FakeMCPClient.count == 2

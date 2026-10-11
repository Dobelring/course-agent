"""Redis 后端路径（fakeredis 注入，不依赖本机 Redis 服务）。

覆盖六个能力的 Redis 分支：限流 Lua 令牌桶 / 熔断状态共享 / 审批与取消
跨进程信号（Pub/Sub）/ MCP 工具缓存 L2+分布式单飞 / 检索结果缓存 /
降级策略。未注入时（REDIS_URL 为空）全部退回进程内路径——由既有测试保证。
"""
import asyncio
import json
import time
from types import SimpleNamespace

import fakeredis
import pytest
from fastapi import HTTPException

from app.agent import approval as approval_mod
from app.agent import engine
from app.api import chat as chat_mod
from app.config import settings
from app.core import redisbus, redisc
from app.core.ratelimit import _BUCKETS, _take


@pytest.fixture
def fake_redis():
    """共享一个 FakeServer 的同步+异步客户端（模拟同一 Redis 实例）。"""
    server = fakeredis.FakeServer()
    sync_c = fakeredis.FakeRedis(server=server, decode_responses=True)
    async_c = fakeredis.FakeAsyncRedis(server=server, decode_responses=True)
    redisc.set_clients(sync_c, async_c)
    _BUCKETS.clear()
    yield SimpleNamespace(sync=sync_c, async_=async_c, server=server)
    redisc.reset_clients()
    _BUCKETS.clear()


# ---------- 1. 限流：Lua 令牌桶 ----------

def test_lua_bucket_shared_state_not_local(fake_redis):
    """配额状态在 Redis：本地桶表保持为空，多 worker 天然共享同一份配额。"""
    for _ in range(3):
        _take("chat:user:1", 3)          # 容量 3，全部放行
    with pytest.raises(HTTPException) as ei:
        _take("chat:user:1", 3)
    assert ei.value.status_code == 429
    assert int(ei.value.headers["Retry-After"]) >= 1
    assert _BUCKETS == {}                # 本地桶未被创建 → 状态全在 Redis
    # 模拟时间流逝：把 updated 回拨 30s（3/min → 0.05/s × 30 = 1.5 币）
    key = redisc.key("rl", "chat:user:1")
    updated = float(fake_redis.sync.hget(key, "updated"))
    fake_redis.sync.hset(key, "updated", updated - 30)
    _take("chat:user:1", 3)              # 补币后再次放行（桶工作正常）
    with pytest.raises(HTTPException):
        _take("chat:user:1", 3)          # 又打空


def test_lua_bucket_retry_after_matches_refill_rate(fake_redis):
    """Retry-After 按补充速率计算：3/min → 约 20s 出 1 币。"""
    for _ in range(3):
        _take("k", 3)
    with pytest.raises(HTTPException) as ei:
        _take("k", 3)
    retry = int(ei.value.headers["Retry-After"])
    assert 18 <= retry <= 21


def test_ratelimit_redis_failure_falls_back_local(monkeypatch, fake_redis):
    """Redis 运行期异常 → 降级本地桶，请求不被误杀。"""
    class Broken:
        def eval(self, *args, **kwargs):
            raise ConnectionError("redis down")

    monkeypatch.setattr(redisc, "get_redis", lambda: Broken())
    for _ in range(3):
        _take("cb:user:1", 3)
    with pytest.raises(HTTPException):
        _take("cb:user:1", 3)            # 本地桶仍有限流能力
    assert "cb:user:1" in _BUCKETS        # 降级路径生效


def test_try_op_failure_disables_process(fake_redis):
    """按调用降级的统一入口：一次失败即本进程停用（重启恢复）。"""
    class Broken:
        def get(self, k):
            raise ConnectionError("down")

    redisc.set_clients(Broken(), None)
    assert redisc.try_op(lambda c: c.get("x"), default="fb") == "fb"
    assert redisc.get_redis() is None     # 已置位 failed，后续直接跳过


# ---------- 2. 熔断：跨实例共享状态 ----------

def _mk_router():
    from app.agent.router import ModelRouter, RouteConfig

    return ModelRouter(routes=[
        RouteConfig(name="main", base_url="http://x", model="m1",
                    api_key="k", priority=1),
        RouteConfig(name="backup", base_url="http://y", model="m2",
                    api_key="k", priority=2),
    ], threshold=2, cooldown_s=30)


def test_breaker_state_shared_between_instances(fake_redis):
    """两个 ModelRouter 实例 = 两个 worker：一侧熔断，另一侧立即可见。"""
    r1, r2 = _mk_router(), _mk_router()
    r1.report_failure("main")
    assert r2._allow("main")              # 1 次未到阈值
    r1.report_failure("main")
    assert not r2._allow("main")          # 跨实例可见：已熔断
    route, _llm = r2.pick()
    assert route.name == "backup"         # r2 自己没收到 failure 也选备用
    states = {s["name"]: s["state"] for s in r2.snapshot()}
    assert states["main"] == "open"


def test_breaker_half_open_reopen_and_recover(fake_redis):
    r = _mk_router()
    r.report_failure("main")
    r.report_failure("main")
    assert not r._allow("main")
    # 冷却结束 → half-open 放行探测（回拨 opened_at 模拟时间流逝）
    fake_redis.sync.hset(r._brk_key("main"), "opened_at", time.time() - 31)
    assert r._allow("main")
    states = {s["name"]: s["state"] for s in r.snapshot()}
    assert states["main"] == "half-open"
    r.report_failure("main")              # 探测失败 → 重新计时
    assert not r._allow("main")
    r.report_success("main")              # 成功 → 闭合
    assert r._allow("main")
    snap = {s["name"]: s for s in r.snapshot()}
    assert snap["main"]["state"] == "closed" and snap["main"]["failures"] == 0


# ---------- 3. 审批：跨进程决策（Pub/Sub） ----------

def test_approval_decision_cross_worker(fake_redis):
    """决策落在「没有该 PENDING 的 worker」：归属键寻址 + Pub/Sub 回持有方。"""
    async def flow():
        assert await redisbus.start()
        assert await redisbus.wait_subscribed()   # 等订阅就绪，防发布早于订阅
        try:
            events: list[dict] = []

            async def emit(evt):
                events.append(evt)

            ctx = approval_mod.ApprovalContext(emit, timeout=5, user_id=7)
            task = asyncio.create_task(ctx.request("save_study_plan", {"a": 1}))
            for _ in range(200):
                if approval_mod.PENDING and events:
                    break
                await asyncio.sleep(0.01)
            approval_id = next(iter(approval_mod.PENDING))
            assert events[0]["approval_id"] == approval_id
            # 归属键已声明：别的 worker 才能判断「在谁手里」
            assert fake_redis.sync.get(redisc.key("appr", approval_id)) == "7"

            # 模拟决策请求落在另一个 worker：本进程视角查不到 PENDING
            real = approval_mod.PENDING
            approval_mod.PENDING = {}
            try:
                assert approval_mod.decide(approval_id, True, 7) is None
            finally:
                approval_mod.PENDING = real
            return await asyncio.wait_for(task, timeout=3), approval_id
        finally:
            await redisbus.stop()

    result, approval_id = asyncio.run(flow())
    assert result is True                  # 信号落地 → Future 被解开
    assert approval_id not in approval_mod.PENDING
    # 归属键随审批结束清理（不靠 TTL 过期兜底）
    assert fake_redis.sync.get(redisc.key("appr", approval_id)) is None


def test_approval_cross_worker_owner_and_missing(fake_redis):
    """跨进程路径的归属校验与 fail-closed：越权 403 语义、不存在 404 语义。"""
    fake_redis.sync.set(redisc.key("appr", "ax"), 7, ex=60)
    real = approval_mod.PENDING
    approval_mod.PENDING = {}             # 模拟别的 worker 视角
    try:
        assert approval_mod.decide("ax", True, user_id=8) == "无权处理他人的审批"
        assert approval_mod.decide("ghost", True, 7) == "审批不存在或已过期"
    finally:
        approval_mod.PENDING = real


# ---------- 4. 取消：跨进程信号 ----------

def test_cancel_cross_worker(fake_redis):
    """取消请求落在别的 worker：归属键寻址 + Pub/Sub → 持有 Task 的一方执行。"""
    async def flow():
        assert await redisbus.start()
        assert await redisbus.wait_subscribed()   # 等订阅就绪，防发布早于订阅
        try:

            async def run():
                await asyncio.sleep(30)

            task = asyncio.create_task(run())
            chat_mod._register_run("run-x", 42, task)
            assert fake_redis.sync.get(chat_mod._run_key("run-x")) == "42"
            real = chat_mod.ACTIVE_RUNS
            chat_mod.ACTIVE_RUNS = {}     # 模拟请求落在另一个 worker
            try:
                resp = await chat_mod.cancel_run(
                    chat_mod.CancelRequest(run_id="run-x"),
                    current_user=SimpleNamespace(id=42),
                )
            finally:
                chat_mod.ACTIVE_RUNS = real
            assert resp == {"ok": True}
            for _ in range(200):
                if task.done():
                    break
                await asyncio.sleep(0.01)
            assert task.cancelled()
        finally:
            chat_mod._unregister_run("run-x")
            await redisbus.stop()

    asyncio.run(flow())


def test_cancel_404_when_unknown_or_foreign(fake_redis):
    """无归属键 / 归属他人 → 404（与 Redis 未启用时语义一致，fail-closed）。"""
    async def flow():
        with pytest.raises(HTTPException) as e1:
            await chat_mod.cancel_run(
                chat_mod.CancelRequest(run_id="ghost"),
                current_user=SimpleNamespace(id=1),
            )
        assert e1.value.status_code == 404
        fake_redis.sync.set(chat_mod._run_key("other"), 42, ex=60)
        with pytest.raises(HTTPException) as e2:
            await chat_mod.cancel_run(
                chat_mod.CancelRequest(run_id="other"),
                current_user=SimpleNamespace(id=7),   # 非归属者
            )
        assert e2.value.status_code == 404

    asyncio.run(flow())


# ---------- 5. MCP 工具缓存：L2 载荷 + 分布式单飞 ----------

def _mk_tools():
    from langchain_core.tools import StructuredTool

    async def _call(**kwargs):
        return "ok"

    return [StructuredTool(
        name="course__demo", description="演示工具",
        args_schema={"type": "object", "properties": {}},
        coroutine=_call, response_format="content_and_artifact",
    )]


@pytest.fixture
def fake_mcp(monkeypatch):
    class FakeMCPClient:
        count = 0

        def __init__(self, config):
            type(self).count += 1

        async def get_tools(self):
            await asyncio.sleep(0.01)
            return _mk_tools()

    monkeypatch.setattr(engine, "MultiServerMCPClient", FakeMCPClient)
    monkeypatch.setattr(settings, "mcp_tools_cache_ttl", 300)
    engine.reset_tools_cache()
    yield FakeMCPClient
    engine.reset_tools_cache()


def test_tools_l2_payload_rebuild_without_spawn(fake_redis, fake_mcp):
    t1 = asyncio.run(engine.get_tools_cached())
    assert fake_mcp.count == 1                          # 首次 spawn 一次
    assert fake_redis.sync.get(engine._payload_redis_key())  # 载荷已回写 L2

    engine.reset_tools_cache()                          # 清 L1 = 另一个 worker 冷启动
    t2 = asyncio.run(engine.get_tools_cached())
    assert fake_mcp.count == 1                          # 未 spawn，从载荷重建
    assert [t.name for t in t1] == [t.name for t in t2] == ["course__demo"]
    assert t2[0].description == "演示工具"
    assert t2[0].coroutine is not None                  # 行为（coroutine）已重新挂接


def test_tools_lock_waits_for_holder_payload(fake_redis, fake_mcp, monkeypatch):
    """没抢到单飞锁 → 轮询等持锁方的载荷，全程不 spawn。"""
    monkeypatch.setattr(settings, "mcp_tools_lock_wait_s", 3.0)
    payload = engine._dump_tools(_mk_tools())

    async def flow():
        fake_redis.sync.set(engine._lock_redis_key(), "other-worker", nx=True, ex=15)

        async def holder():                 # 模拟持锁方 0.3s 后写出载荷
            await asyncio.sleep(0.3)
            fake_redis.sync.set(engine._payload_redis_key(), payload, ex=300)

        holder_task = asyncio.create_task(holder())
        tools = await engine.get_tools_cached()
        await holder_task
        return tools

    tools = asyncio.run(flow())
    assert fake_mcp.count == 0
    assert tools[0].name == "course__demo"


def test_tools_lock_timeout_falls_back_to_spawn(fake_redis, fake_mcp, monkeypatch):
    """持锁方迟迟不写载荷（可能已崩溃）→ 等待超时兜底自行 spawn。"""
    monkeypatch.setattr(settings, "mcp_tools_lock_wait_s", 0.4)

    async def flow():
        fake_redis.sync.set(engine._lock_redis_key(), "dead-worker", ex=15)
        return await engine.get_tools_cached()

    tools = asyncio.run(flow())
    assert fake_mcp.count == 1
    assert tools[0].name == "course__demo"


def test_tools_unserializable_payload_skips_l2(fake_redis, monkeypatch):
    """载荷不可序列化（如测试桩返回 dict）→ 只走 L1，不写 Redis。"""

    class DictMCPClient:
        count = 0

        def __init__(self, config):
            type(self).count += 1

        async def get_tools(self):
            return [{"tool": 1}]

    monkeypatch.setattr(engine, "MultiServerMCPClient", DictMCPClient)
    monkeypatch.setattr(settings, "mcp_tools_cache_ttl", 300)
    engine.reset_tools_cache()
    r1 = asyncio.run(engine.get_tools_cached())
    r2 = asyncio.run(engine.get_tools_cached())
    assert r1 == r2 == [{"tool": 1}]
    assert DictMCPClient.count == 1
    assert fake_redis.sync.get(engine._payload_redis_key()) is None  # 未写载荷


# ---------- 6. 检索结果缓存 ----------

def test_rag_cache_hit_and_corpus_invalidation(fake_redis, seeded, db, monkeypatch):
    from app.rag import retriever

    calls = {"n": 0}

    def fake_embed(query):
        calls["n"] += 1
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(retriever, "embed_query", fake_embed)
    uid = seeded.users.student1.id

    r1 = retriever.retrieve(db, uid, "什么是装饰器")
    r2 = retriever.retrieve(db, uid, "  什么是装饰器  ")   # 归一化后同键
    assert r1 and r1 == r2
    assert calls["n"] == 1                                 # 第二次命中缓存

    retriever.retrieve(db, uid, "红黑树是什么")            # 不同查询 → miss
    assert calls["n"] == 2

    # 语料变化（ready 切片 +1）→ 语料指纹变 → 缓存自动失效
    from app.models.document import DocumentChunk

    db.add(DocumentChunk(document_id=seeded.docs["ready"].id, chunk_index=9,
                         content="新增切片", embedding=json.dumps([0.0, 0.0, 1.0])))
    db.commit()
    retriever.retrieve(db, uid, "什么是装饰器")
    assert calls["n"] == 3


def test_rag_cache_disabled_by_ttl(fake_redis, seeded, db, monkeypatch):
    from app.rag import retriever

    calls = {"n": 0}

    def fake_embed(query):
        calls["n"] += 1
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(settings, "rag_cache_ttl_s", 0)
    monkeypatch.setattr(retriever, "embed_query", fake_embed)
    uid = seeded.users.student1.id
    retriever.retrieve(db, uid, "q")
    retriever.retrieve(db, uid, "q")
    assert calls["n"] == 2                                 # TTL=0 关闭缓存


# ---------- 7. 生命周期接线 ----------

def test_health_reports_redis_enabled(fake_redis, client):
    """fake_redis 必须先于 client 解析：lifespan 在客户端启动时检查 Redis。"""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["redis"] is True

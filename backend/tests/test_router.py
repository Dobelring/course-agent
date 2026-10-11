"""主备模型路由：优先级选路、同级加权分流、熔断状态机、配置解析。"""
import pytest

from app.agent import router as router_mod
from app.agent.router import CircuitBreaker, ModelRouter, RouteConfig, from_settings, reset_router
from app.config import settings


def _routes(*priorities: int) -> list[RouteConfig]:
    return [
        RouteConfig(name=f"r{p}", base_url="https://x/v1", model="m",
                    api_key="k", priority=p)
        for p in priorities
    ]


@pytest.fixture(autouse=True)
def _reset():
    reset_router()
    yield
    reset_router()


# ---------- 熔断器状态机 ----------

def test_breaker_closed_until_threshold():
    b = CircuitBreaker(threshold=2, cooldown_s=30)
    assert b.allow() is True
    b.record_failure()
    assert b.allow() is True        # 未达阈值仍放行
    b.record_failure()
    assert b.allow() is False       # 达阈值 → open
    assert b.is_open is True


def test_breaker_half_open_after_cooldown():
    b = CircuitBreaker(threshold=1, cooldown_s=30)
    b.record_failure(now=0.0)
    assert b.allow(now=10.0) is False   # 冷却中
    assert b.allow(now=31.0) is True    # half-open 探测


def test_breaker_half_open_success_closes():
    b = CircuitBreaker(threshold=1, cooldown_s=30)
    b.record_failure(now=0.0)
    assert b.allow(now=31.0) is True
    b.record_success()
    assert b.is_open is False
    assert b.allow(now=100.0) is True


def test_breaker_half_open_failure_reopens():
    b = CircuitBreaker(threshold=1, cooldown_s=30)
    b.record_failure(now=0.0)
    b.record_failure(now=40.0)      # half-open 探测失败 → 重新计时
    assert b.allow(now=50.0) is False
    assert b.allow(now=70.1) is True


# ---------- 选路 ----------

def test_pick_prefers_lowest_priority():
    r = ModelRouter(routes=_routes(2, 1), threshold=2, cooldown_s=30)
    route, _ = r.pick()
    assert route.name == "r1"


def test_pick_skips_open_routes():
    r = ModelRouter(routes=_routes(1, 2), threshold=1, cooldown_s=30)
    r.report_failure("r1")
    route, _ = r.pick()
    assert route.name == "r2"


def test_pick_all_open_raises():
    r = ModelRouter(routes=_routes(1, 2), threshold=1, cooldown_s=30)
    r.report_failure("r1")
    r.report_failure("r2")
    with pytest.raises(ValueError, match="熔断"):
        r.pick()


def test_pick_weighted_within_same_priority(monkeypatch):
    routes = [
        RouteConfig(name="a", base_url="u", model="m", api_key="k", priority=1, weight=9),
        RouteConfig(name="b", base_url="u", model="m", api_key="k", priority=1, weight=1),
    ]
    r = ModelRouter(routes=routes, threshold=2, cooldown_s=30)
    monkeypatch.setattr(router_mod.random, "random", lambda: 0.5)  # 累计 0.5/10 → 落在 a
    assert r.pick()[0].name == "a"
    monkeypatch.setattr(router_mod.random, "random", lambda: 0.95)  # 落在 b
    assert r.pick()[0].name == "b"


def test_report_success_resets_failures():
    r = ModelRouter(routes=_routes(1), threshold=2, cooldown_s=30)
    r.report_failure("r1")
    r.report_success("r1")
    r.report_failure("r1")  # 从 0 重新计数
    assert r.breakers["r1"].allow() is True


# ---------- 配置解析 ----------

def test_from_settings_single_route(monkeypatch):
    monkeypatch.setattr(settings, "llm_routes", "")
    monkeypatch.setattr(settings, "llm_api_key", "k")
    r = from_settings()
    assert [x.name for x in r.routes] == ["default"]


def test_from_settings_json_routes(monkeypatch):
    monkeypatch.setattr(settings, "llm_routes",
                        '[{"name":"main","base_url":"u1","model":"m1","priority":1},'
                        ' {"name":"backup","base_url":"u2","model":"m2","priority":2,"weight":3}]')
    monkeypatch.setattr(settings, "llm_api_key", "global-key")
    r = from_settings()
    assert r.routes[0].name == "main" and r.routes[0].api_key == "global-key"
    assert r.routes[1].weight == 3


def test_from_settings_requires_key(monkeypatch):
    monkeypatch.setattr(settings, "llm_routes", "")
    monkeypatch.setattr(settings, "llm_api_key", "")
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        from_settings()


def test_pick_attaches_usage_collector():
    from app.agent.usage import UsageCollector

    r = ModelRouter(routes=_routes(1), threshold=2, cooldown_s=30)
    usage = UsageCollector(price_input_per_m=0, price_output_per_m=0)
    _, llm = r.pick(usage)
    assert usage in llm.callbacks

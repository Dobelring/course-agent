"""主备模型路由：优先级排序 + 同级加权分流 + 连续失败熔断。

选路策略（面试可讲的设计决策）：
- priority 升序为主备顺序（1 最优）；同级多个路由按 weight 加权随机分流（负载均衡）
- 熔断器：同一路由连续失败 N 次 → open（pick 跳过），冷却 cooldown 秒 → half-open
  （放行一个探测请求，成功则闭合、失败则重新熔断）——避免对已故障供应商持续撞墙
- 实例不缓存：每次 pick 新建 ChatOpenAI 并挂本次请求的 UsageCollector——
  成本核算按请求聚合，缓存实例会让回调跨请求串账
- 故障切换粒度：run_step 每次重试前重新 pick，失败路由上报熔断后下一步自动切备用；
  单循环（direct）失败在请求级上报，用户重发即走备用

配置（.env）：
    LLM_ROUTES=[{"name":"qwen-main","base_url":"...","model":"...","api_key":null,
                 "priority":1,"weight":1},{"name":"qwen-backup",...,"priority":2}]
    api_key 为空时回退全局 LLM_API_KEY；LLM_ROUTES 留空则用 LLM_* 构造单路由（向后兼容）。
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI

from app.config import settings


@dataclass
class RouteConfig:
    name: str
    base_url: str
    model: str
    api_key: str
    priority: int = 1
    weight: int = 1


def _weighted_choice(group: list[RouteConfig]) -> RouteConfig:
    """加权轮盘：weight 越大命中概率越高（不用 random.choices——它走
    Random 实例方法，测试桩不好替换；手写轮盘等价且更可测）。"""
    total = sum(max(1, r.weight) for r in group)
    x = random.random() * total
    acc = 0
    for r in group:
        acc += max(1, r.weight)
        if x < acc:
            return r
    return group[-1]


@dataclass
class CircuitBreaker:
    """连续失败熔断：closed →（连续失败≥threshold）open →（冷却结束）half-open。"""

    threshold: int
    cooldown_s: float
    failures: int = 0
    opened_at: float | None = None

    def allow(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if self.opened_at is None:
            return True
        if now - self.opened_at >= self.cooldown_s:
            return True  # half-open：放行探测
        return False

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self.opened_at is not None:
            # half-open 探测仍失败 → 重新计时熔断
            self.opened_at = now
            return
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = now

    @property
    def is_open(self) -> bool:
        return self.opened_at is not None


@dataclass
class ModelRouter:
    routes: list[RouteConfig] = field(default_factory=list)
    threshold: int = 2
    cooldown_s: float = 30.0
    breakers: dict[str, CircuitBreaker] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.routes:
            raise ValueError("未配置任何模型路由（LLM_ROUTES / LLM_API_KEY）")
        self.routes = sorted(self.routes, key=lambda r: r.priority)
        self.breakers = {r.name: CircuitBreaker(self.threshold, self.cooldown_s)
                         for r in self.routes}

    # ---- 选路 ----

    def _available(self) -> list[RouteConfig]:
        return [r for r in self.routes if self.breakers[r.name].allow()]

    def pick(self, usage: Any = None) -> tuple[RouteConfig, ChatOpenAI]:
        """选最优可用路由并构建 LLM 实例；全部熔断时抛 ValueError。"""
        available = self._available()
        if not available:
            raise ValueError(
                "所有模型路由均处于熔断状态，请稍后重试或检查模型服务可用性"
            )
        # 同级加权随机：weight 越大被选中概率越高（负载均衡）；不同级只取最高优先级组
        top_priority = available[0].priority
        group = [r for r in available if r.priority == top_priority]
        route = _weighted_choice(group)
        return route, self._build(route, usage)

    def _build(self, route: RouteConfig, usage: Any) -> ChatOpenAI:
        return ChatOpenAI(
            model=route.model,
            api_key=route.api_key,
            base_url=route.base_url,
            temperature=0.3,
            streaming=True,
            callbacks=[usage] if usage is not None else None,
        )

    # ---- 熔断上报 ----

    def report_success(self, name: str) -> None:
        if name in self.breakers:
            self.breakers[name].record_success()

    def report_failure(self, name: str) -> None:
        if name in self.breakers:
            self.breakers[name].record_failure()

    def snapshot(self) -> list[dict]:
        """当前路由与熔断状态（观测/调试用）。"""
        out = []
        for r in self.routes:
            b = self.breakers[r.name]
            if b.opened_at is None:
                state = "closed"
            elif b.allow():
                state = "half-open"
            else:
                state = "open"
            out.append({"name": r.name, "model": r.model, "priority": r.priority,
                        "weight": r.weight, "state": state, "failures": b.failures})
        return out


# ---- 构建与单例 ----

def from_settings() -> ModelRouter:
    raw = (settings.llm_routes or "").strip()
    if raw:
        items = json.loads(raw)
        routes = [
            RouteConfig(
                name=item["name"],
                base_url=item["base_url"],
                model=item["model"],
                api_key=item.get("api_key") or settings.llm_api_key,
                priority=int(item.get("priority", 1)),
                weight=int(item.get("weight", 1)),
            )
            for item in items
        ]
        if all(not r.api_key for r in routes):
            raise ValueError(
                "LLM_ROUTES 已配置但未提供任何 api_key"
                "（路由项 api_key 或全局 LLM_API_KEY 均为空）"
            )
    else:
        if not settings.llm_api_key:
            raise ValueError("未配置 LLM_API_KEY，请在 backend/.env 中填写")
        routes = [RouteConfig(name="default", base_url=settings.llm_base_url,
                              model=settings.llm_model, api_key=settings.llm_api_key)]
    return ModelRouter(routes=routes, threshold=settings.llm_breaker_threshold,
                       cooldown_s=settings.llm_breaker_cooldown_s)


_router: ModelRouter | None = None


def get_router() -> ModelRouter:
    global _router
    if _router is None:
        _router = from_settings()
    return _router


def reset_router() -> None:
    """测试/配置变更后重建单例。"""
    global _router
    _router = None

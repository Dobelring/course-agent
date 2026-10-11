"""跨进程 Redis 联调（离线，不调 LLM）：四路断言一次验完。

    cd backend && python scripts/e2e_redis_crossproc.py

架构：coordinator 启动一个独立的 worker 子进程（两个真实 OS 进程，共享同一
Redis），worker 侧注册「取消 run + 挂起审批 + 上报熔断失败 + 消耗限流配额」，
coordinator 侧以**空本地注册表**（模拟请求落在另一个 worker）完成四路断言：

    1. 取消   —— 归属键寻址 + Pub/Sub 信号 → worker 的 Task 被 cancel
    2. 审批   —— 归属键寻址 + Pub/Sub 信号 → worker 的 Future 被解开（approve）
    3. 限流   —— Lua 令牌桶配额跨进程共享（3/min：worker 用 2、coordinator
                 第 3 次放行、第 4 次 429 带 Retry-After）
    4. 熔断   —— worker 上报 2 次失败 → coordinator 新建的 ModelRouter 读到
                 共享状态（e2e-main 路由 open，选路跳过）

前置：docker compose up -d redis，backend/.env 配 REDIS_URL。
退出码 0 = 四路全过；键命名带 e2e 前缀，结束自动清理。
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

RUN_ID = "e2e-run"
OWNER = 9527
RATE_KEY = "e2e:rl"
ROUTE = "e2e-main"
READY_KEY = "course-agent:e2e:redis:ready"
RESULT_KEY = "course-agent:e2e:redis:result"
RATE_PER_MIN = 3
BREAKER_COOLDOWN_S = 3600  # 大冷却：测试期间绝不允许半开干扰 open 断言

from app.config import settings  # noqa: E402


def _must_redis():
    from app.core import redisc

    client = redisc.get_redis()
    if client is None:
        sys.exit("REDIS_URL 未配置或 Redis 未启动：先 docker compose up -d redis，"
                 "并在 backend/.env 配置 REDIS_URL=redis://127.0.0.1:6379/0")
    return client


def _mk_router():
    from app.agent.router import ModelRouter, RouteConfig

    return ModelRouter(routes=[
        RouteConfig(name=ROUTE, base_url="http://x", model="m",
                    api_key="k", priority=1),
        RouteConfig(name="e2e-backup", base_url="http://y", model="m",
                    api_key="k", priority=2),
    ], threshold=2, cooldown_s=BREAKER_COOLDOWN_S)


# ---------------------------------------------------------------------------


async def worker() -> int:
    """子进程：注册 run/审批、上报熔断、消耗配额，等信号落地后回报结果。"""
    from app.agent import approval as approval_mod
    from app.agent import router as router_mod
    from app.api import chat as chat_mod
    from app.core import redisbus
    from app.core.ratelimit import _take

    client = _must_redis()
    try:
        assert await redisbus.start(), "Pub/Sub 监听启动失败"
        assert await redisbus.wait_subscribed(), "订阅未就绪"

        # 4) 熔断：本进程（worker）上报 2 次失败 → 共享状态 open
        router = _mk_router()
        router.report_failure(ROUTE)
        router.report_failure(ROUTE)

        # 3) 限流：3/min 配额先用掉 2 个（留给 coordinator 第 3 个）
        _take(RATE_KEY, RATE_PER_MIN)
        _take(RATE_KEY, RATE_PER_MIN)

        # 1) 取消：注册 run（本地 Task + 归属键）
        async def run_forever():
            await asyncio.sleep(300)

        run_task = asyncio.create_task(run_forever())
        chat_mod._register_run(RUN_ID, OWNER, run_task)

        # 2) 审批：真实 ApprovalContext 挂起等待（本地 Future + 归属键）
        events: list[dict] = []

        async def emit(evt):
            events.append(evt)

        ctx = approval_mod.ApprovalContext(emit, timeout=25, user_id=OWNER)
        ap_task = asyncio.create_task(ctx.request("save_study_plan", {"plan": "e2e"}))
        for _ in range(300):
            if approval_mod.PENDING and events:
                break
            await asyncio.sleep(0.02)
        approval_id = events[0]["approval_id"]

        client.set(READY_KEY, json.dumps(
            {"run_id": RUN_ID, "approval_id": approval_id, "owner": OWNER}),
            ex=60)
        print("[worker] ready：run 与审批已挂起，等待跨进程信号", flush=True)

        # 等 coordinator 的取消信号（轮询，避免 CancelledError 传播技巧）
        cancelled = False
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not run_task.done():
            await asyncio.sleep(0.05)
        if run_task.done():
            cancelled = run_task.cancelled()

        # 等 coordinator 的审批决策（Pub/Sub → resolve_local → Future）
        approved = False
        try:
            approved = await asyncio.wait_for(ap_task, timeout=10) is True
        except (asyncio.TimeoutError, asyncio.CancelledError):
            approved = False

        ok = cancelled and approved
        client.set(RESULT_KEY, json.dumps(
            {"cancelled": cancelled, "approved": approved}), ex=60)
        print(f"[worker] 结果：cancelled={cancelled} approved={approved}", flush=True)
        return 0 if ok else 1
    finally:
        chat_mod._unregister_run(RUN_ID)
        await redisbus.stop()


def wait_key(client, key: str, timeout: float = 15.0) -> str | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        val = client.get(key)
        if val:
            return val
        time.sleep(0.05)
    return None


async def coordinator(proc: subprocess.Popen) -> int:
    from app.agent import approval as approval_mod
    from app.api import chat as chat_mod
    from app.core.ratelimit import _take
    from fastapi import HTTPException

    client = _must_redis()
    passed: list[str] = []
    checks = {"n": 0}

    def ok(name: str, cond: bool, detail: str = ""):
        checks["n"] += 1
        if not cond:
            print(f"[FAIL] {name} {detail}", flush=True)
            return False
        passed.append(name)
        print(f"[PASS] {name} {detail}".rstrip(), flush=True)
        return True

    raw = wait_key(client, READY_KEY)
    if raw is None:
        proc.kill()
        print("[FAIL] worker 未就绪（15s 无 ready 键）", flush=True)
        return 1
    ready = json.loads(raw)
    approval_id = ready["approval_id"]

    # 前置：本进程（coordinator）本地注册表必须为空 —— 这正是「另一台 worker」视角
    ok("前置：本进程无 run/审批本地状态",
       not chat_mod.ACTIVE_RUNS and not approval_mod.PENDING)

    # 3) 限流：配额已被 worker 用掉 2/3 → 第 3 次放行、第 4 次 429
    try:
        _take(RATE_KEY, RATE_PER_MIN)
        third_ok = True
    except HTTPException:
        third_ok = False
    ok("限流：共享配额第 3 次放行", third_ok)
    retry_after = None
    try:
        _take(RATE_KEY, RATE_PER_MIN)
        fourth_429 = False
    except HTTPException as e:
        fourth_429 = e.status_code == 429
        retry_after = e.headers.get("Retry-After")
    ok("限流：第 4 次 429 且带 Retry-After", fourth_429 and retry_after,
       f"Retry-After={retry_after}")

    # 1) 取消：越权 404 → 归属本人 → 信号送达 worker
    try:
        await chat_mod.cancel_run(chat_mod.CancelRequest(run_id=RUN_ID),
                                  current_user=SimpleNamespace(id=111))
        foreign_404 = False
    except HTTPException as e:
        foreign_404 = e.status_code == 404
    ok("取消：越权请求 404", foreign_404)

    resp = await chat_mod.cancel_run(chat_mod.CancelRequest(run_id=RUN_ID),
                                     current_user=SimpleNamespace(id=OWNER))
    ok("取消：归属本人返回 ok 并广播", resp == {"ok": True})

    # 2) 审批：越权拒绝 → 归属本人 → 信号送达 worker
    err_foreign = approval_mod.decide(approval_id, True, user_id=111)
    ok("审批：越权决策被拒", err_foreign == "无权处理他人的审批", str(err_foreign))
    err = approval_mod.decide(approval_id, True, OWNER)
    ok("审批：归属本人决策返回 ok 并广播", err is None, str(err))

    # 4) 熔断：worker 上报的失败在本进程的新 router 实例上可见
    fresh = _mk_router()
    states = {s["name"]: s["state"] for s in fresh.snapshot()}
    ok("熔断：跨进程共享 open 状态", not fresh._allow(ROUTE)
       and states.get(ROUTE) == "open", f"snapshot={states.get(ROUTE)}")

    # worker 回报：Task 确实被 cancel、Future 确实被解开
    raw = wait_key(client, RESULT_KEY, timeout=15)
    result = json.loads(raw) if raw else {}
    ok("worker：run 被 cancel 且审批 Future 被解开",
       result.get("cancelled") is True and result.get("approved") is True,
       str(result))

    rc = proc.wait(timeout=15)
    ok("worker 进程退出码为 0", rc == 0, f"rc={rc}")

    # 清理 e2e 键（不留残余影响真实路由/配额）
    client.delete(READY_KEY, RESULT_KEY,
                  f"course-agent:rl:{RATE_KEY}",
                  f"course-agent:brk:{ROUTE}",
                  "course-agent:brk:e2e-backup")
    total = checks["n"]
    print(f"\n跨进程联调：{len(passed)}/{total} 通过"
          f"{'—— 四路（取消/审批/限流/熔断）全部验证 ✓' if len(passed) == total else ''}",
          flush=True)
    return 0 if len(passed) == total else 1


def main() -> int:
    _must_redis()  # 前置校验（同时完成本进程连接）
    env = os.environ.copy()
    env["REDIS_URL"] = settings.redis_url  # 子进程显式继承同一 Redis
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--role", "worker"],
        cwd=str(BACKEND_DIR), env=env,
    )
    try:
        return asyncio.run(coordinator(proc))
    finally:
        if proc.poll() is None:
            proc.kill()


if __name__ == "__main__":
    if "--worker" in sys.argv or (len(sys.argv) > 2 and sys.argv[2] == "worker"):
        sys.exit(asyncio.run(worker()))
    sys.exit(main())

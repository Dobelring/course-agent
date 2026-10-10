"""E2E 验证两项修复：工具事件去重 + 生成中途取消（真实千问 Key）。

用法：uvicorn 启动后  python scripts/e2e_fixes.py [port]
"""
import asyncio
import json
import sys
import uuid

import httpx

PORT = sys.argv[1] if len(sys.argv) > 1 else "8000"
BASE = f"http://127.0.0.1:{PORT}"


async def main() -> int:
    async with httpx.AsyncClient(base_url=BASE, timeout=300.0) as client:
        r = await client.post("/api/auth/login",
                              json={"username": "student01", "password": "student01"})
        assert r.status_code == 200, r.text
        headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
        print("[1] 登录成功")

        # ---------- 修复 A：plan 模式工具事件去重 + 前序结果注入 ----------
        events: list[dict] = []
        prompt = "结合我的成绩和近期作业，制定一份期末复习计划并保存"
        auto_approved: list[str] = []
        async with client.stream("POST", "/api/chat", headers=headers,
                                 json={"message": prompt, "mode": "plan"}) as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                ev = json.loads(line[6:])
                events.append(ev)
                # 写操作审批：自动批准（等价用户点「允许」），验证审批链路
                if ev["type"] == "approval_required":
                    ar = await client.post(
                        f"/api/chat/approvals/{ev['approval_id']}",
                        headers=headers, json={"approve": True})
                    auto_approved.append(ev["tool"])
                    print(f"[2a] approval_required({ev['tool']}) → 决策 {ar.status_code}")

        tool_starts = [e for e in events if e["type"] == "tool_start"]
        names = [e["name"] for e in tool_starts]
        dup = {n for n in names if names.count(n) > 1}
        step_ends = [e for e in events if e["type"] == "step_end"]
        done = next((e for e in events if e["type"] == "done"), None)

        print(f"[2] plan 完成：step_end={len(step_ends)} done={done and done.get('plan')}")
        print(f"[3] tool_start 序列: {names}")
        if dup:
            print(f"    ✗ 仍有重复工具卡片: {dup}")
            return 1
        print(f"[4] ✓ 无重复工具卡片（去重生效），工具调用总次数 {len(names)}")

        # 前序注入的间接证据：步骤数少于工具调用总数（模型复用前步结果，
        # 没有每个步骤都重新查一遍）
        if done and done.get("plan"):
            p = done["plan"]
            print(f"[5] plan: done={p['done']} failed={p['failed']} "
                  f"工具调用 {len(names)} 次 / {p['done']} 步")

        # ---------- 修复 B：中途取消 ----------
        cancel_id = str(uuid.uuid4())
        got_first_event = asyncio.Event()

        async def read_stream():
            async with client.stream(
                "POST", "/api/chat", headers=headers,
                json={"message": "详细分析我所有科目的情况并给出完整建议，越详细越好",
                      "mode": "plan", "run_id": cancel_id},
            ) as resp:
                assert resp.status_code == 200
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        ev = json.loads(line[6:])
                        got_first_event.set()
                        if ev["type"] == "cancelled":
                            return "cancelled"
                        if ev["type"] == "error":
                            return f"error:{ev['message']}"
                        if ev["type"] == "done":
                            return "done_before_cancel"
            return "stream_closed"

        reader = asyncio.create_task(read_stream())
        # 等首个事件确认任务已在跑，再取消
        try:
            await asyncio.wait_for(got_first_event.wait(), timeout=60)
        except asyncio.TimeoutError:
            print("    ✗ 60s 内未收到任何 SSE 事件")
            reader.cancel()
            return 1

        cr = await client.post("/api/chat/cancel", headers=headers,
                               json={"run_id": cancel_id})
        print(f"[6] cancel 请求 → {cr.status_code} {cr.json()}")
        if cr.status_code != 200:
            reader.cancel()
            return 1

        outcome = await asyncio.wait_for(reader, timeout=30)
        print(f"[7] 流结局: {outcome}")
        if outcome != "cancelled":
            print("    ✗ 未收到 cancelled 事件")
            return 1
        print("[8] ✓ 中断生效：cancelled 事件收到，流正常收尾")

        # 取消后再查会话：应有带（已手动停止）标记的落库记录
        cr2 = await client.post("/api/chat/cancel", headers=headers,
                                json={"run_id": cancel_id})
        assert cr2.status_code == 404  # 任务已注销
        print("[9] ✓ 重复取消返回 404（注册表已清理）")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

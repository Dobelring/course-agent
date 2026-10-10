"""E2E 验证：plan 模式 + 写操作审批流 + 并行步骤（真实千问 Key）。

用法：uvicorn 启动后  python scripts/e2e_approval.py [port]
"""
import asyncio
import json
import sys

import httpx

PORT = sys.argv[1] if len(sys.argv) > 1 else "8031"
BASE = f"http://127.0.0.1:{PORT}"


async def main() -> int:
    async with httpx.AsyncClient(base_url=BASE, timeout=300.0) as client:
        # 1. 登录
        r = await client.post("/api/auth/login",
                              json={"username": "student01", "password": "student01"})
        assert r.status_code == 200, r.text
        token = r.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("[1] 登录成功")

        # 2. plan 模式发问（会触发 save_study_plan → 审批）
        events: list[dict] = []
        approved = []
        prompt = "结合我的成绩和近期作业，制定一份期末复习计划并保存"

        async with client.stream("POST", "/api/chat", headers=headers,
                                 json={"message": prompt, "mode": "plan"}) as resp:
            assert resp.status_code == 200, resp.status_code
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                ev = json.loads(line[6:])
                events.append(ev)
                t = ev["type"]
                if t == "mode":
                    print(f"[2] mode={ev['mode']}")
                elif t == "plan":
                    deps = [s.get("depends_on") for s in ev["steps"]]
                    print(f"[3] 计划 {len(ev['steps'])} 步 depends_on={deps}")
                elif t == "step_start":
                    print(f"    step_start #{ev['id']}: {ev['title']} dep={ev.get('depends_on')}")
                elif t == "approval_required":
                    # 3. 收到审批请求 → 立即批准
                    print(f"[4] approval_required: {ev['tool']} (step={ev.get('step_id')})")
                    ar = await client.post(
                        f"/api/chat/approvals/{ev['approval_id']}",
                        headers=headers, json={"approve": True})
                    print(f"    decision -> {ar.status_code} {ar.json()}")
                    approved.append(ar.status_code == 200)
                elif t == "approval_resolved":
                    print(f"    approval_resolved approved={ev['approved']}")
                elif t == "step_end":
                    print(f"    step_end #{ev['id']}: {ev['status']}")
                elif t == "done":
                    print(f"[5] done mode={ev['mode']} plan={ev.get('plan') and {k: v for k, v in ev['plan'].items() if k != 'steps'}}")
                elif t == "error":
                    print(f"    ERROR: {ev['message']}")

        # 4. 断言
        types = [e["type"] for e in events]
        assert "plan" in types, "缺少 plan 事件"
        assert "approval_required" in types, "未触发写操作审批"
        assert approved and all(approved), "审批决策失败"
        assert "approval_resolved" in types
        done = next(e for e in events if e["type"] == "done")
        assert done["mode"] == "plan", f"mode 应为 plan: {done['mode']}"
        assert done["plan"] and done["plan"]["done"] >= 1
        # 审批必须发生在 save 步骤前，且保存成功（plan done 计数）
        print(f"\n[6] 断言通过：审批链路 OK，plan done={done['plan']['done']} "
              f"failed={done['plan']['failed']} latency={done['latency_ms']}ms")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

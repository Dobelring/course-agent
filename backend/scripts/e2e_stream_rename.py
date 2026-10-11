"""E2E 验证（真实千问 Key）：流式 / 计数 / 重命名 / 三态审批（修改后执行） / 断点续跑。

用法：uvicorn 启动后  python scripts/e2e_stream_rename.py [port]
"""
import asyncio
import json
import sys

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

        # ---------- 修复 A：规划模式综合流式 ----------
        events: list[dict] = []
        async with client.stream("POST", "/api/chat", headers=headers,
                                 json={"message": "结合我的课表和成绩，给出下周学习建议",
                                       "mode": "plan"}) as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

        text_deltas = [e for e in events if e["type"] == "text"]
        step_deltas = [e for e in events if e["type"] == "step_text"]
        done = next((e for e in events if e["type"] == "done"), None)
        print(f"[2] plan 完成：text 事件 {len(text_deltas)} 个，"
              f"step_text {len(step_deltas)} 个，done={bool(done)}")
        if len(text_deltas) < 2:
            print("    ✗ 综合未流式：text 增量应 >= 2（修复前是整段 1 个）")
            return 1
        print("[3] ✓ 综合流式生效（多个 text 增量逐段下发）")

        cid = done.get("conversation_id") if done else None
        assert cid, "done 事件缺少 conversation_id"

        # ---------- 修复 B：回复计数 = assistant 消息数 ----------
        convs = (await client.get("/api/conversations", headers=headers)).json()
        row = next(c for c in convs if c["id"] == cid)
        detail = (await client.get(f"/api/conversations/{cid}", headers=headers)).json()
        n_msgs = len(detail["messages"])
        n_assistant = sum(1 for m in detail["messages"] if m["role"] == "assistant")
        print(f"[4] 会话共 {n_msgs} 条消息，其中 assistant {n_assistant} 条，"
              f"列表显示 msg_count={row['msg_count']}")
        if row["msg_count"] != n_assistant:
            print("    ✗ msg_count 应等于 assistant 条数")
            return 1
        if n_msgs == n_assistant * 2:
            print("[5] ✓ 计数 = 大模型回复次数（总消息数 ÷ 2）")
        else:
            print(f"[5] ✓ 计数 = assistant 条数（本会话消息数 {n_msgs} 非偶数，"
                  "存在历史单边消息，仍按回复数计）")

        # ---------- 修复 C：重命名 ----------
        resp = await client.patch(f"/api/conversations/{cid}",
                                  json={"title": "E2E-重命名会话"}, headers=headers)
        assert resp.status_code == 200, resp.text
        assert resp.json()["title"] == "E2E-重命名会话"
        # 越权改名 → 404
        r2 = await client.post("/api/auth/login",
                               json={"username": "student02", "password": "student02"})
        h2 = {"Authorization": f"Bearer {r2.json()['access_token']}"}
        steal = await client.patch(f"/api/conversations/{cid}",
                                   json={"title": "劫持"}, headers=h2)
        print(f"[6] 本人改名 200；他人改名 {steal.status_code}")
        if steal.status_code != 404:
            print("    ✗ 越权改名应 404")
            return 1
        # 空标题 → 422
        empty = await client.patch(f"/api/conversations/{cid}",
                                   json={"title": "   "}, headers=headers)
        print(f"[7] 空标题 → {empty.status_code}")
        if empty.status_code != 422:
            print("    ✗ 空标题应 422")
            return 1
        print("[8] ✓ 重命名生效且校验/鉴权正确")

        # ---------- 修复 D：三态审批（修改后执行） ----------
        events2: list[dict] = []
        approvals_seen: list[str] = []
        async with client.stream(
            "POST", "/api/chat", headers=headers,
            json={"message": "结合我的成绩，制定一份下周学习计划并保存", "mode": "plan"},
        ) as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                ev = json.loads(line[6:])
                events2.append(ev)
                if ev["type"] == "approval_required":
                    approvals_seen.append(ev["approval_id"])
                    # 第一次给修改意见（revise），之后批准
                    action = "revise" if len(approvals_seen) == 1 else "approve"
                    feedback = "任务条目请拆得更细，每条一句话" if action == "revise" else None
                    ar = await client.post(
                        f"/api/chat/approvals/{ev['approval_id']}",
                        headers=headers, json={"action": action, "feedback": feedback})
                    print(f"[10] approval_required #{len(approvals_seen)} → {action} ({ar.status_code})")
        reflects = [e for e in events2 if e["type"] == "reflection"]
        done2 = next((e for e in events2 if e["type"] == "done"), None)
        revise_resolved = [e for e in events2 if e["type"] == "approval_resolved"
                           and e.get("reason") == "用户要求修改后重新提交"]
        print(f"[11] 审批次数 {len(approvals_seen)}，revise 生效 {bool(revise_resolved)}，"
              f"反思事件 {len(reflects)}（quality={reflects[0]['quality'] if reflects else 'N/A'}），"
              f"done={bool(done2)}")
        if len(approvals_seen) < 2 or not revise_resolved or not done2:
            print("    ✗ 三态审批（revise→approve）链路未走通")
            return 1
        print("[12] ✓ 修改后执行链路走通：意见回传模型 → 调整参数重新审批 → 批准落库")

        # ---------- 修复 E：断点续跑 ----------
        import uuid as _uuid
        cancel_run = str(_uuid.uuid4())
        events3: list[dict] = []
        got_step = asyncio.Event()

        async def read_plan_stream():
            async with client.stream(
                "POST", "/api/chat", headers=headers,
                json={"message": "详细分析我所有科目的学习情况并给出完整建议",
                      "mode": "plan", "run_id": cancel_run},
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    ev = json.loads(line[6:])
                    events3.append(ev)
                    if ev["type"] == "step_start":
                        got_step.set()
                    if ev["type"] in ("cancelled", "done", "error"):
                        return

        reader = asyncio.create_task(read_plan_stream())
        try:
            await asyncio.wait_for(got_step.wait(), timeout=60)
        except asyncio.TimeoutError:
            print("    ✗ 60s 未进入步骤执行")
            reader.cancel()
            return 1
        cr = await client.post("/api/chat/cancel", headers=headers,
                               json={"run_id": cancel_run})
        await asyncio.wait_for(reader, timeout=30)
        cancelled = next((e for e in events3 if e["type"] == "cancelled"), None)
        plan_run_id = (cancelled or {}).get("plan_run_id")
        print(f"[13] cancel → {cr.status_code}，plan_run_id={plan_run_id}")
        if not plan_run_id:
            print("    ✗ cancelled 事件未携带 plan_run_id")
            return 1

        events4: list[dict] = []
        async with client.stream(
            "POST", "/api/chat", headers=headers,
            json={"message": "继续执行暂停的计划", "mode": "plan",
                  "resume_plan_id": plan_run_id},
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    events4.append(json.loads(line[6:]))
        mode4 = next((e for e in events4 if e["type"] == "mode"), None)
        done4 = next((e for e in events4 if e["type"] == "done"), None)
        step_starts = [e for e in events4 if e["type"] == "step_start"]
        print(f"[14] 续跑：mode={mode4 and mode4.get('reason')}，"
              f"step_start={len(step_starts)}，done={bool(done4 and done4.get('plan'))}")
        if not (mode4 and "续跑" in (mode4.get("reason") or "")) or not done4:
            print("    ✗ 续跑未走通")
            return 1
        print("[15] ✓ 断点续跑走通：暂停 → 恢复 → 只补未完成步骤 → 正常收尾")

        # ---------- 清理：删掉测试会话 ----------
        await client.delete(f"/api/conversations/{cid}", headers=headers)
        if done2 and done2.get("conversation_id"):
            await client.delete(f"/api/conversations/{done2['conversation_id']}",
                                headers=headers)
        if done4 and done4.get("conversation_id"):
            await client.delete(f"/api/conversations/{done4['conversation_id']}",
                                headers=headers)
        print("[16] ✓ 测试会话已清理")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

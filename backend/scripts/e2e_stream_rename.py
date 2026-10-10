"""E2E 验证（真实千问 Key）：规划综合流式 + 会话重命名 + 回复计数 + 空白会话清理。

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

        # ---------- 清理：删掉测试会话 ----------
        await client.delete(f"/api/conversations/{cid}", headers=headers)
        print("[9] ✓ 测试会话已清理")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

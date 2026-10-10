"""API 集成测试：鉴权、角色隔离、业务接口、/chat 无 Key 优雅降级。"""
import json

from app.api.chat import _extract_sources


# ---------- 健康检查 ----------

def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "app": "course-agent"}


# ---------- 登录 / 注册 ----------

def test_login_success(client, make_users):
    resp = client.post("/api/auth/login",
                       json={"username": "student1", "password": make_users.password})
    assert resp.status_code == 200
    body = resp.json()
    assert body["access_token"]
    assert body["user"]["role"] == "student"


def test_login_wrong_password(client, make_users):
    resp = client.post("/api/auth/login",
                       json={"username": "student1", "password": "nope"})
    assert resp.status_code == 401


def test_register_duplicate_username(client, make_users):
    resp = client.post("/api/auth/register",
                       json={"username": "student1", "password": "x", "name": "重复"})
    assert resp.status_code == 409


def test_me_requires_token(client):
    assert client.get("/api/auth/me").status_code == 401


def test_me_returns_current_user(client, make_users, auth_headers):
    headers = auth_headers("student1")
    resp = client.get("/api/auth/me", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["username"] == "student1"


# ---------- 数据隔离 ----------

def test_schedules_me_isolates_users(client, seeded, auth_headers):
    r1 = client.get("/api/schedules/me", headers=auth_headers("student1"))
    r2 = client.get("/api/schedules/me", headers=auth_headers("student2"))
    assert len(r1.json()) == 2
    assert len(r2.json()) == 1
    assert r2.json()[0]["course_name"] == "大学英语"


def test_admin_endpoints_reject_student(client, seeded, auth_headers):
    resp = client.get("/api/admin/stats", headers=auth_headers("student1"))
    assert resp.status_code == 403
    assert resp.json()["detail"] == "需要管理员权限"


def test_admin_stats_counts(client, seeded, auth_headers):
    resp = client.get("/api/admin/stats", headers=auth_headers("admin1"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["users"] == 3
    assert body["courses"] == 4
    assert body["scores"] == 5


def test_assignments_overdue_flag(client, seeded, auth_headers):
    resp = client.get("/api/assignments", headers=auth_headers("student1"))
    rows = resp.json()
    assert len(rows) == 4
    overdue = [r for r in rows if r["title"] == "程序设计课程设计"]
    assert overdue and overdue[0]["overdue"] is True


def test_toggle_assignment_status_and_isolation(client, seeded, auth_headers):
    rows = client.get("/api/assignments", headers=auth_headers("student1")).json()
    aid = rows[0]["id"]

    # student2 改 student1 的作业 → 404
    resp = client.put(f"/api/assignments/{aid}/status",
                      json={"status": "done"}, headers=auth_headers("student2"))
    assert resp.status_code == 404

    resp = client.put(f"/api/assignments/{aid}/status",
                      json={"status": "done"}, headers=auth_headers("student1"))
    assert resp.status_code == 200
    assert resp.json()["status"] == "done"


def test_scores_me(client, seeded, auth_headers):
    resp = client.get("/api/scores/me", headers=auth_headers("student1"))
    body = resp.json()
    assert body["gpa"] == 3.4
    assert body["average"] == 83.7
    assert body["weakest"]["course_name"] == "高等数学"


def test_stats_me(client, seeded, auth_headers):
    resp = client.get("/api/stats/me", headers=auth_headers("student1"))
    body = resp.json()
    assert body["schedules"] == 2
    assert body["pending_assignments"] == 3
    assert body["documents"] == 2
    assert body["next_due_at"]  # 有逾期作业，应给出最近截止时间


# ---------- 学习计划 ----------

def test_plans_crud_and_isolation(client, seeded, auth_headers):
    h1 = auth_headers("student1")
    h2 = auth_headers("student2")

    # 建计划 + 两个条目（走 API，模拟 Agent 落库后的前端操作）
    plan_id = None
    # 计划由 MCP 工具落库，这里直接经由列表接口验证隔离：
    # student2 看不到 student1 的计划（seed 里没有计划，先造一条）
    from app.database import SessionLocal
    from app.models.plan import StudyPlan, StudyPlanItem

    db = SessionLocal()
    try:
        plan = StudyPlan(user_id=seeded.users.student1.id, title="p1",
                         content="c", status="active")
        db.add(plan)
        db.commit()
        db.refresh(plan)
        plan_id = plan.id
        db.add(StudyPlanItem(plan_id=plan.id, item_order=0, title="任务A", is_done=True))
        db.add(StudyPlanItem(plan_id=plan.id, item_order=1, title="任务B", is_done=False))
        db.commit()
    finally:
        db.close()

    # student1 可见，progress 正确
    got = client.get(f"/api/plans/{plan_id}", headers=h1)
    assert got.status_code == 200
    assert got.json()["progress"] == {"done": 1, "total": 2}

    # student2 不可见 → 404
    assert client.get(f"/api/plans/{plan_id}", headers=h2).status_code == 404

    # student2 列表为空；student1 有一条
    assert client.get("/api/plans", headers=h2).json() == []
    assert len(client.get("/api/plans", headers=h1).json()) == 1

    # 条目打卡
    items = client.get(f"/api/plans/{plan_id}", headers=h1).json()["items"]
    done_item = next(i for i in items if i["title"] == "任务B")
    resp = client.put(f"/api/plans/{plan_id}/items/{done_item['id']}",
                      json={"is_done": True}, headers=h1)
    assert resp.json()["is_done"] is True

    # student2 删计划 → 404
    assert client.delete(f"/api/plans/{plan_id}", headers=h2).status_code == 404


# ---------- 对话历史 ----------

def test_conversations_isolation(client, seeded, auth_headers):
    h1 = auth_headers("student1")
    h2 = auth_headers("student2")

    created = client.post("/api/conversations", json={"title": "会话A"}, headers=h1)
    cid = created.json()["id"]

    assert client.get(f"/api/conversations/{cid}", headers=h2).status_code == 404
    assert client.delete(f"/api/conversations/{cid}", headers=h2).status_code == 404

    titles = [c["title"] for c in client.get("/api/conversations", headers=h1).json()]
    assert "会话A" in titles
    assert client.get("/api/conversations", headers=h2).json() == []

    assert client.delete(f"/api/conversations/{cid}", headers=h1).json()["ok"] is True


# ---------- /chat 无 Key 优雅降级 ----------

def test_chat_without_api_key_returns_sse_error(client, seeded, auth_headers):
    resp = client.post("/api/chat", json={"message": "你好"},
                       headers=auth_headers("student1"))
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    # 流的第一段应是结构化错误事件，而不是 500 / 崩溃
    assert '"type": "error"' in resp.text
    assert "LLM_API_KEY" in resp.text


# ---------- SSE 检索溯源解析 ----------

def test_extract_sources_from_tool_output_list():
    raw = json.dumps([
        {"filename": "讲义.md", "score": 0.91, "chunk_index": 0, "content": "……", "extra": 1},
        {"filename": "课件.md", "score": 0.72, "chunk_index": 2, "content": "……"},
    ], ensure_ascii=False)
    sources = _extract_sources(raw)
    assert len(sources) == 2
    assert sources[0] == {"filename": "讲义.md", "score": 0.91, "chunk_index": 0, "content": "……"}
    assert "extra" not in sources[0]


def test_extract_sources_from_mcp_content_block():
    payload = {"content": [{"type": "text", "text": json.dumps(
        [{"filename": "a.md", "score": 0.5, "content": "x"}], ensure_ascii=False)}]}
    sources = _extract_sources(payload)
    assert sources[0]["filename"] == "a.md"


def test_extract_sources_unparseable_returns_none():
    assert _extract_sources(None) is None
    assert _extract_sources("not json at all") is None
    assert _extract_sources({"foo": "bar"}) is None

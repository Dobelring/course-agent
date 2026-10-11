"""MCP 工具直调：课表/成绩/作业/计划/文档检索 + user_id 数据隔离。"""
import json
from datetime import datetime, timedelta

from app.mcp.server import (
    get_my_schedule,
    get_my_scores,
    get_upcoming_assignments,
    list_my_documents,
    save_study_plan,
    search_course_documents,
)
from app.models.plan import StudyPlan, StudyPlanItem


def test_get_my_schedule_returns_only_own_rows(seeded):
    rows = get_my_schedule(seeded.users.student1.id)
    assert len(rows) == 2
    assert all(r["course_name"] in ("程序设计基础", "高等数学") for r in rows)
    assert {"weekday", "start_period", "location", "teacher"} <= set(rows[0])


def test_get_my_schedule_isolates_users(seeded):
    assert get_my_schedule(seeded.users.student2.id)[0]["course_name"] == "大学英语"
    assert get_my_schedule(99999) == []


def test_get_my_scores_gpa_math(seeded):
    # student1: 92(4.0×3) 85(3.7×3.5) 76(2.7×5) 88(3.7×2)
    # gpa = (12 + 12.95 + 13.5 + 7.4) / 13.5 = 45.85 / 13.5 = 3.4
    result = get_my_scores(seeded.users.student1.id)
    assert result["gpa"] == 3.4
    assert result["average"] == 83.7
    assert result["weakest"]["course_name"] == "高等数学"
    assert len(result["courses"]) == 4


def test_get_my_scores_empty_user(make_users):
    assert get_my_scores(make_users.student2.id)["gpa"] is None


def test_get_upcoming_assignments_includes_overdue_and_window(seeded):
    rows = get_upcoming_assignments(seeded.users.student1.id, days=7)
    titles = [r["title"] for r in rows]
    assert "数据结构实验三" in titles      # +2 天
    assert "高数第4章作业" in titles        # +5 天
    assert "程序设计课程设计" in titles      # 逾期（-1 天）也应提示
    assert "已交的作业" not in titles        # done 不提示
    overdue = [r for r in rows if r["title"] == "程序设计课程设计"]
    assert overdue[0]["overdue"] is True


def test_get_upcoming_assignments_excludes_far_deadline(seeded):
    rows = get_upcoming_assignments(seeded.users.student1.id, days=1)
    titles = [r["title"] for r in rows]
    assert "高数第4章作业" not in titles     # +5 天超出窗口
    assert "程序设计课程设计" in titles       # 逾期无论窗口多小都提示


def test_get_upcoming_assignments_isolates_users(seeded):
    titles = [r["title"] for r in get_upcoming_assignments(seeded.users.student2.id)]
    assert titles == ["student2的作业"]


def test_save_study_plan_persists_plan_and_items(db, seeded):
    result = save_study_plan(
        seeded.users.student1.id,
        title="期末复习计划",
        goal="各科不挂科",
        content="# 计划全文",
        items=["复习第1章", "复习第2章", "", "  "],
    )
    assert result["items"] == 2  # 空白条目被丢弃

    plan = db.get(StudyPlan, result["id"])
    assert plan.user_id == seeded.users.student1.id
    items = db.query(StudyPlanItem).filter(StudyPlanItem.plan_id == plan.id).all()
    assert [i.title for i in items] == ["复习第1章", "复习第2章"]


def test_search_course_documents_isolates_users(db, seeded, monkeypatch):
    monkeypatch.setattr("app.rag.retriever.embed_query", lambda q: [1.0, 0.0, 0.0])
    mine = search_course_documents(seeded.users.student1.id, "装饰器", top_k=5)
    assert all(r["filename"] != "u2_notes.md" for r in mine)

    theirs = search_course_documents(seeded.users.student2.id, "装饰器", top_k=5)
    assert all(r["filename"] == "u2_notes.md" for r in theirs)


def test_list_my_documents_isolates_users(seeded):
    names1 = [d["filename"] for d in list_my_documents(seeded.users.student1.id)]
    assert set(names1) == {"u1_notes.md", "u1_draft.md"}
    assert [d["filename"] for d in list_my_documents(seeded.users.student2.id)] == ["u2_notes.md"]


def test_save_study_plan_idempotent_within_window(db, seeded):
    """幂等：窗口期内同 (user, 内容指纹) 重复保存返回既有计划，不产生重复行。"""
    uid = seeded.users.student1.id
    args = dict(title="期末复习计划", goal="不挂科", content="# 全文",
                items=["任务1", "任务2"])
    first = save_study_plan(uid, **args)
    second = save_study_plan(uid, **args)
    assert second["id"] == first["id"]
    assert second["deduplicated"] is True
    assert db.query(StudyPlan).filter(StudyPlan.user_id == uid).count() == 1

    # 内容不同（items 变化）→ 指纹不同 → 新计划
    third = save_study_plan(uid, title="期末复习计划", goal="不挂科",
                            content="# 全文", items=["任务1", "任务2改"])
    assert third["id"] != first["id"]
    assert "deduplicated" not in third


def test_save_study_plan_idempotency_window_expires(db, seeded, monkeypatch):
    """窗口过期后同内容再次保存 → 允许新建（窗口由配置控制）。"""
    from app.config import settings

    uid = seeded.users.student1.id
    args = dict(title="短期计划", goal="g", content="# c", items=["x"])
    first = save_study_plan(uid, **args)
    monkeypatch.setattr(settings, "tool_idempotency_window_s", 0)
    second = save_study_plan(uid, **args)
    assert second["id"] != first["id"]
    assert "deduplicated" not in second

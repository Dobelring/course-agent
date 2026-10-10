"""pytest 全局夹具：独立测试库 + 演示数据，全程无需真实 API Key。

重要：DATABASE_URL / LLM_API_KEY 必须在导入任何 app.* 模块之前设置，
因此环境变量赋值放在本文件最顶部执行。
"""
import os
from pathlib import Path
from types import SimpleNamespace

_TEST_DB = Path(__file__).resolve().parent / "test_course_agent.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB.as_posix()}"
# 单元测试不依赖模型服务；/chat 的“未配置 Key”错误分支也依赖此状态
os.environ["LLM_API_KEY"] = ""
# 单测关闭交叉编码器重排（避免下载模型）；混合召回保持开启以覆盖 RRF 逻辑
os.environ["RAG_RERANK"] = "false"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import bcrypt  # noqa: E402
import pytest  # noqa: E402

# 所有测试账号统一密码；rounds=4 加速测试（与生产无关）
PASSWORD = "pass1234"
_PASSWORD_HASH = bcrypt.hashpw(PASSWORD.encode("utf-8"), bcrypt.gensalt(4)).decode("utf-8")


@pytest.fixture(scope="session", autouse=True)
def _test_db_file():
    _TEST_DB.unlink(missing_ok=True)
    yield
    # 先释放连接再删文件（Windows 下文件被占用会 unlink 失败），失败也不影响结果
    try:
        from app.database import engine

        engine.dispose()
    finally:
        try:
            _TEST_DB.unlink(missing_ok=True)
        except PermissionError:
            pass


@pytest.fixture(autouse=True)
def fresh_tables(_test_db_file):
    """每个用例前后重建表，保证用例之间完全隔离。"""
    from app import models  # noqa: F401  触发 ORM 模型注册
    from app.database import Base, engine

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def db():
    from app.database import SessionLocal

    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def make_users(db):
    """创建三个账号：student1 / student2 / admin。"""

    def _make(username: str, role: str = "student", name: str | None = None):
        from app.models.user import User

        user = User(
            username=username,
            password_hash=_PASSWORD_HASH,
            name=name or username,
            role=role,
            student_no=username.upper() if role == "student" else None,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    student1 = _make("student1", "student", "张三")
    student2 = _make("student2", "student", "李四")
    admin = _make("admin1", "admin", "王老师")
    return SimpleNamespace(student1=student1, student2=student2, admin=admin, password=PASSWORD)


@pytest.fixture
def seeded(make_users, db):
    """在三个账号之上补齐课程、课表、成绩、作业与文档切片（已提交，MCP 子进程可见）。"""
    import json
    from datetime import datetime, timedelta

    from app.models.assignment import Assignment
    from app.models.course import Course, Schedule
    from app.models.document import CourseDocument, DocumentChunk
    from app.models.score import Score

    users = make_users

    courses = {}
    for code, name, teacher, credit in [
        ("CS101", "程序设计基础", "李老师", 3),
        ("CS201", "数据结构", "王老师", 3.5),
        ("MATH101", "高等数学", "赵老师", 5),
        ("ENG101", "大学英语", "孙老师", 2),
    ]:
        course = Course(code=code, name=name, teacher=teacher, credit=credit)
        db.add(course)
        courses[code] = course
    db.flush()

    # 课表：student1 两节、student2 一节
    db.add(Schedule(user_id=users.student1.id, course_id=courses["CS101"].id,
                    weekday=1, start_period=1, end_period=2, week_start=1, week_end=16,
                    location="教学楼A301"))
    db.add(Schedule(user_id=users.student1.id, course_id=courses["MATH101"].id,
                    weekday=1, start_period=3, end_period=4, week_start=1, week_end=16,
                    location="教学楼A201"))
    db.add(Schedule(user_id=users.student2.id, course_id=courses["ENG101"].id,
                    weekday=2, start_period=5, end_period=6, week_start=1, week_end=16,
                    location="外语楼B102"))

    # 成绩：student1 四门，期望 GPA=3.4 / average=83.7 / weakest=高等数学
    for code, score in [("CS101", 92), ("CS201", 85), ("MATH101", 76), ("ENG101", 88)]:
        db.add(Score(user_id=users.student1.id, course_id=courses[code].id, score=score))
    db.add(Score(user_id=users.student2.id, course_id=courses["MATH101"].id, score=81))

    def _due(days: int) -> datetime:
        return (datetime.now() + timedelta(days=days)).replace(hour=23, minute=0, second=0, microsecond=0)

    # 作业：3 条 pending（含 1 条逾期）+ 1 条 done
    for code, title, days, status in [
        ("CS201", "数据结构实验三", 2, "pending"),
        ("MATH101", "高数第4章作业", 5, "pending"),
        ("CS101", "程序设计课程设计", -1, "pending"),
        ("CS101", "已交的作业", 1, "done"),
    ]:
        db.add(Assignment(user_id=users.student1.id, course_id=courses[code].id,
                          title=title, due_at=_due(days), status=status))
    db.add(Assignment(user_id=users.student2.id, course_id=courses["MATH101"].id,
                      title="student2的作业", due_at=_due(1), status="pending"))

    # 文档切片：student1 ready + pending 各一，student2 ready 一份
    doc1 = CourseDocument(user_id=users.student1.id, filename="u1_notes.md",
                          file_path="/tmp/u1.md", file_type="md", status="ready")
    doc2 = CourseDocument(user_id=users.student1.id, filename="u1_draft.md",
                          file_path="/tmp/u1d.md", file_type="md", status="pending")
    doc3 = CourseDocument(user_id=users.student2.id, filename="u2_notes.md",
                          file_path="/tmp/u2.md", file_type="md", status="ready")
    db.add_all([doc1, doc2, doc3])
    db.flush()
    db.add(DocumentChunk(document_id=doc1.id, chunk_index=0,
                         content="装饰器是一种为函数增加额外功能的语法糖。",
                         embedding=json.dumps([1.0, 0.0, 0.0])))
    db.add(DocumentChunk(document_id=doc1.id, chunk_index=1,
                         content="红黑树是一种自平衡二叉查找树。",
                         embedding=json.dumps([0.0, 1.0, 0.0])))
    db.add(DocumentChunk(document_id=doc2.id, chunk_index=0,
                         content="pending 文档不应被检索到。",
                         embedding=json.dumps([1.0, 0.0, 0.0])))
    db.add(DocumentChunk(document_id=doc3.id, chunk_index=0,
                         content="student2 的私有资料内容。",
                         embedding=json.dumps([1.0, 0.0, 0.0])))
    db.commit()

    return SimpleNamespace(
        users=users,
        courses=courses,
        docs={"ready": doc1, "pending": doc2, "other": doc3},
    )


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers(client):
    """工厂夹具：auth_headers('student1') -> Authorization 头。"""

    def _login(username: str, password: str = PASSWORD) -> dict:
        resp = client.post("/api/auth/login", json={"username": username, "password": password})
        assert resp.status_code == 200, resp.text
        return {"Authorization": f"Bearer {resp.json()['access_token']}"}

    return _login


@pytest.fixture
def fake_agent_env(monkeypatch):
    """把 LLM 与 Agent 替换为假实现；MCP 工具加载保持真实（无需 Key）。

    原定义在 test_chat_modes.py，因 test_api.py 也要复用而上移到公共夹具。
    """
    from app.api import chat as chat_mod
    from fakes import FakeAgent, FakeLLM

    fake_llm = FakeLLM()
    monkeypatch.setattr(chat_mod, "build_llm", lambda usage=None: fake_llm)
    monkeypatch.setattr(chat_mod, "build_agent",
                        lambda user, tools, extra_system="", llm=None: FakeAgent())
    return fake_llm

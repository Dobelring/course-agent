"""限流：令牌桶单元行为、聊天按用户、登录按 IP、配额关闭直通。"""
import pytest

from app.core import ratelimit
from app.core.ratelimit import TokenBucket, _BUCKETS, limit_chat, limit_login
from app.config import settings


@pytest.fixture(autouse=True)
def clean_buckets():
    _BUCKETS.clear()
    yield
    _BUCKETS.clear()


# ---------- TokenBucket 单元 ----------

def test_bucket_allows_burst_then_rejects():
    bucket = TokenBucket(per_min=3)  # 容量 3，速率 3/60 每秒
    assert [bucket.take() for _ in range(3)] == [True, True, True]
    assert bucket.take() is False
    assert bucket.retry_after() >= 1


def test_bucket_refills_over_time(monkeypatch):
    clock = {"t": 100.0}
    monkeypatch.setattr(ratelimit.time, "monotonic", lambda: clock["t"])
    bucket = TokenBucket(per_min=60)  # 容量 60，速率 1 token/秒
    for _ in range(60):
        bucket.take()
    assert bucket.take() is False     # 打空
    clock["t"] += 2.0                 # 过 2 秒 → 补 2 个
    assert bucket.take() is True
    assert bucket.take() is True
    assert bucket.take() is False


def test_bucket_capacity_caps_refill(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(ratelimit.time, "monotonic", lambda: clock["t"])
    bucket = TokenBucket(per_min=60)
    clock["t"] += 3600  # 补充量远超容量
    for _ in range(60):
        assert bucket.take() is True
    assert bucket.take() is False  # 不超过容量 60


# ---------- 聊天按用户 ----------

def test_chat_limit_per_user(client, make_users, auth_headers, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_chat_per_min", 2)
    h1 = auth_headers("student1")
    h2 = auth_headers("student2")
    # 无 Key 时 /chat 返回 SSE 错误流（200），但限流依赖在鉴权之后先计数
    assert client.post("/api/chat", json={"message": "a"}, headers=h1).status_code == 200
    assert client.post("/api/chat", json={"message": "b"}, headers=h1).status_code == 200
    resp = client.post("/api/chat", json={"message": "c"}, headers=h1)
    assert resp.status_code == 429
    assert resp.headers["Retry-After"]
    # 另一个用户不受影响
    assert client.post("/api/chat", json={"message": "a"}, headers=h2).status_code == 200


def test_chat_limit_zero_disables(client, seeded, auth_headers, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_chat_per_min", 0)
    for _ in range(15):
        assert client.post("/api/chat", json={"message": "x"},
                           headers=auth_headers("student1")).status_code == 200


def test_chat_limit_requires_auth(client, make_users, auth_headers, monkeypatch):
    """未登录请求先 401（限流依赖声明在鉴权之后，鉴权失败不会消耗配额）。"""
    monkeypatch.setattr(settings, "rate_limit_chat_per_min", 2)
    assert client.post("/api/chat", json={"message": "x"}).status_code == 401
    assert _BUCKETS == {}  # 未生成任何桶


# ---------- 登录按 IP ----------

def test_login_limit_per_ip(client, seeded, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_login_per_min", 2)
    ok = {"username": "student1", "password": "pass1234"}
    bad = {"username": "student1", "password": "wrong"}
    assert client.post("/api/auth/login", json=ok).status_code == 200
    assert client.post("/api/auth/login", json=bad).status_code == 401  # 失败也计数
    resp = client.post("/api/auth/login", json=ok)
    assert resp.status_code == 429

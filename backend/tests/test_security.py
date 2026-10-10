"""安全：密码哈希、JWT 签发/校验、角色守卫、GPA 换算。"""
from datetime import timedelta

import jwt
import pytest
from fastapi import HTTPException

from app.config import settings
from app.core.security import (
    create_access_token,
    get_current_admin,
    get_current_user,
    hash_password,
    verify_password,
)
from app.models.score import score_to_gpapoint


# ---------- 密码 ----------

def test_password_hash_roundtrip():
    h = hash_password("s3cret")
    assert h != "s3cret"
    assert verify_password("s3cret", h) is True
    assert verify_password("wrong", h) is False


def test_password_hash_unique_per_call():
    assert hash_password("same") != hash_password("same")  # bcrypt 盐值随机


# ---------- JWT ----------

def test_create_access_token_payload():
    token = create_access_token(42, "alice", "admin")
    payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    assert payload["sub"] == "42"
    assert payload["username"] == "alice"
    assert payload["role"] == "admin"
    assert "exp" in payload


def test_get_current_user_rejects_wrong_secret():
    token = jwt.encode(
        {"sub": "1", "exp": 9999999999},
        "attacker-secret",
        algorithm=settings.jwt_algorithm,
    )

    class Cred:
        credentials = token

    with pytest.raises(HTTPException) as e:
        get_current_user(credentials=Cred(), db=None)
    assert e.value.status_code == 401


def test_get_current_user_rejects_expired_token(db):
    from fastapi.security import HTTPAuthorizationCredentials

    expire = __import__("datetime").datetime.now(__import__("datetime").timezone.utc) - timedelta(minutes=1)
    token = jwt.encode(
        {"sub": "1", "exp": expire},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    cred = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    with pytest.raises(HTTPException) as e:
        get_current_user(credentials=cred, db=db)
    assert e.value.status_code == 401


def test_get_current_user_returns_user(db, make_users):
    from fastapi.security import HTTPAuthorizationCredentials

    u = make_users.student1
    token = create_access_token(u.id, u.username, u.role)
    cred = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    assert get_current_user(credentials=cred, db=db).id == u.id


# ---------- 角色守卫 ----------

def test_get_current_admin_allows_admin(make_users):
    assert get_current_admin(make_users.admin).role == "admin"


def test_get_current_admin_rejects_student(make_users):
    with pytest.raises(HTTPException) as e:
        get_current_admin(make_users.student1)
    assert e.value.status_code == 403


# ---------- GPA 换算（边界值） ----------

@pytest.mark.parametrize("score,expected", [
    (95, 4.0), (90, 4.0), (89, 3.7), (85, 3.7),
    (84, 3.3), (82, 3.3), (81, 3.0), (78, 3.0),
    (77, 2.7), (75, 2.7), (74, 2.3), (72, 2.3),
    (71, 2.0), (68, 2.0), (67, 1.5), (64, 1.5),
    (63, 1.0), (60, 1.0), (59, 0.0), (0, 0.0),
])
def test_score_to_gpapoint_boundaries(score, expected):
    assert score_to_gpapoint(score) == expected

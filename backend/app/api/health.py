from fastapi import APIRouter

from app.core import redisc

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check():
    # redis：本进程 Redis 是否已启用（REDIS_URL 配置且连接成功）——演示/排障用
    return {"status": "ok", "app": "course-agent", "redis": redisc.get_redis() is not None}

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text
from loguru import logger

from app import models  # noqa: F401  # 导入模型以注册到 Base.metadata
from app.api import (
    academic,
    admin,
    auth,
    chat,
    conversations,
    documents,
    flashcards,
    health,
    plans,
    schedules,
)
from app.database import Base, engine
from app.config import settings


def _ensure_columns() -> None:
    """轻量列迁移（项目未引入 Alembic）：create_all 不会给已有表补列。

    新列均为可空，重复执行安全（列存在则跳过）。
    """
    inspector = inspect(engine)
    migrations = [
        ("ai_conversations", "summary",
         "ALTER TABLE ai_conversations ADD COLUMN summary TEXT"),
        ("ai_messages", "plan",
         "ALTER TABLE ai_messages ADD COLUMN plan TEXT"),
        ("ai_messages", "input_tokens",
         "ALTER TABLE ai_messages ADD COLUMN input_tokens INTEGER"),
        ("ai_messages", "output_tokens",
         "ALTER TABLE ai_messages ADD COLUMN output_tokens INTEGER"),
        ("ai_messages", "cost_cny",
         "ALTER TABLE ai_messages ADD COLUMN cost_cny DOUBLE"),
        ("study_plans", "fingerprint",
         "ALTER TABLE study_plans ADD COLUMN fingerprint VARCHAR(64)"),
        ("document_chunks", "embedding_blob",
         "ALTER TABLE document_chunks ADD COLUMN embedding_blob BLOB"),
    ]
    for table, column, ddl in migrations:
        try:
            columns = {c["name"] for c in inspector.get_columns(table)}
        except Exception:  # noqa: BLE001 — 表还不存在时由 create_all 创建
            continue
        if column not in columns:
            with engine.begin() as conn:
                conn.execute(text(ddl))
            logger.info("列迁移：{}.{} 已添加", table, column)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 开发阶段：启动时自动建表（正式项目改用迁移工具 Alembic）
    Base.metadata.create_all(bind=engine)
    _ensure_columns()
    # 模型预热：把 embedding/重排的首次加载与推理开销移出请求路径（best-effort）
    if settings.rag_warmup:
        import threading

        def _warm() -> None:
            from app.rag import reranker as rag_reranker
            from app.rag.embedder import warmup as warm_embedder

            warm_embedder()
            if settings.rag_rerank:
                rag_reranker.warmup()

        # 后台线程预热：不阻塞服务启动，失败仅记日志
        threading.Thread(target=_warm, name="rag-warmup", daemon=True).start()
    yield


app = FastAPI(title="course-agent", version="0.1.0", lifespan=lifespan)

# 开发环境：允许本地 Vite 前端跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(auth.router, prefix="/api")
app.include_router(schedules.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(plans.router, prefix="/api")
app.include_router(academic.router, prefix="/api")
app.include_router(flashcards.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(conversations.router, prefix="/api")
app.include_router(admin.router, prefix="/api")

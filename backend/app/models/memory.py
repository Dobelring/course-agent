"""跨会话用户画像（UserMemory）。"""
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class UserMemory(Base):
    """用户级长期记忆：一行一用户，profile 为滚动更新的画像文本。

    与 AiConversation.summary 的区别：summary 是会话级（对话结束即固化），
    profile 是用户级、跨所有会话持续合并。
    """

    __tablename__ = "user_memories"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True, nullable=False)
    profile: Mapped[str] = mapped_column(Text, nullable=False, default="")
    hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # 距上次刷新的消息计数
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=func.now(), onupdate=func.now()
    )

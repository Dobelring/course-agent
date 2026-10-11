"""审批日志：每次写操作审批的请求与决策落库，支撑「人工干预率」度量。

与 PENDING 内存注册表分离：注册表只活到决策完成（桥接两条 HTTP 连接），
本表是持久审计记录——评测脚本据其计算干预率/拒绝率，管理后台可查总量。
"""
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class ApprovalLog(Base):
    __tablename__ = "approval_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    approval_id: Mapped[str] = mapped_column(String(36), index=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    # 冗余会话 id 便于追溯；不加 FK——会话删除后日志仍保留（审计记录不随数据删除）
    conversation_id: Mapped[int | None] = mapped_column(Integer)
    tool: Mapped[str] = mapped_column(String(100))
    args: Mapped[str | None] = mapped_column(Text)  # brief_args JSON 串
    step_id: Mapped[int | None] = mapped_column(Integer)
    # approve / reject / revise / timeout；未决策前为 NULL
    decision: Mapped[str | None] = mapped_column(String(20))
    feedback: Mapped[str | None] = mapped_column(Text)  # revise 的用户修改意见
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    decided_at: Mapped[datetime | None] = mapped_column(DateTime)

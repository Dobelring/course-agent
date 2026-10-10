"""对话历史：会话列表 / 详情 / 重命名 / 删除（仅限当前用户自己的会话）。"""
import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.database import get_db
from app.models.conversation import AiConversation, AiMessage
from app.models.user import User

router = APIRouter(prefix="/conversations", tags=["conversations"])


class ConversationCreate(BaseModel):
    title: str = "新对话"


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)


def _get_own_conversation(db: Session, user: User, cid: int) -> AiConversation:
    conv = db.get(AiConversation, cid)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(status_code=404, detail="会话不存在")
    return conv


@router.post("")
def create_conversation(
    payload: ConversationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conv = AiConversation(user_id=current_user.id, title=payload.title)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return {"id": conv.id, "title": conv.title, "created_at": str(conv.created_at)}


@router.get("")
def list_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # 计数只统计 assistant 消息 = 大模型回复次数（不是 user+assistant 总条数）
    rows = (
        db.query(
            AiConversation,
            func.count(AiMessage.id).label("msg_count"),
        )
        .outerjoin(
            AiMessage,
            (AiMessage.conversation_id == AiConversation.id)
            & (AiMessage.role == "assistant"),
        )
        .filter(AiConversation.user_id == current_user.id)
        .group_by(AiConversation.id)
        .order_by(AiConversation.updated_at.desc())
        .all()
    )
    return [
        {
            "id": c.id,
            "title": c.title,
            "msg_count": n,
            "created_at": str(c.created_at),
            "updated_at": str(c.updated_at),
        }
        for c, n in rows
    ]


@router.get("/{cid}")
def get_conversation(
    cid: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conv = _get_own_conversation(db, current_user, cid)
    msgs = (
        db.query(AiMessage)
        .filter(AiMessage.conversation_id == conv.id)
        .order_by(AiMessage.id)
        .all()
    )
    return {
        "id": conv.id,
        "title": conv.title,
        "created_at": str(conv.created_at),
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "tool_calls": json.loads(m.tool_calls) if m.tool_calls else [],
                "retrieval_sources": json.loads(m.retrieval_sources) if m.retrieval_sources else None,
                "plan": json.loads(m.plan) if m.plan else None,
                "latency_ms": m.latency_ms,
                "input_tokens": m.input_tokens,
                "output_tokens": m.output_tokens,
                "cost_cny": m.cost_cny,
                "created_at": str(m.created_at),
            }
            for m in msgs
        ],
    }


@router.patch("/{cid}")
def rename_conversation(
    cid: int,
    payload: ConversationUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conv = _get_own_conversation(db, current_user, cid)
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="标题不能为空")
    conv.title = title
    db.commit()
    return {"id": conv.id, "title": conv.title}


@router.delete("/{cid}")
def delete_conversation(
    cid: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conv = _get_own_conversation(db, current_user, cid)
    db.query(AiMessage).filter(AiMessage.conversation_id == conv.id).delete()
    db.delete(conv)
    db.commit()
    return {"ok": True}

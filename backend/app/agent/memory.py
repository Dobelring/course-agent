"""跨会话用户记忆：user 级画像，随对话演进刷新。

三级记忆架构（对应 README 核心设计）：
1. 近期原文   —— 本轮请求的最近 N 条消息（context.split_history）
2. 会话摘要   —— AiConversation.summary，超预算时压缩，跨请求累积
3. 用户画像   ——— 本模块 UserMemory.profile，跨会话累积（当前模块）

刷新时机：会话内第 N 条助手消息触发一次（memory_refresh_every），
把「旧画像 + 本会话近期消息」交给 LLM 合并——只保留跨会话稳定信息
（身份/偏好/长期目标/薄弱科目），丢弃一次性细节。失败不影响主流程。
"""
from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

PROFILE_PROMPT = """你是用户画像维护器。将「已有画像」与「本次对话片段」合并为更新后的画像。

要求：
- 只保留跨会话稳定的信息：身份角色、长期目标、偏好、薄弱科目、习惯约束
- 丢弃一次性细节（某次作业、某天安排）与寒暄
- 与已有画像矛盾时以新信息为准
- 输出纯文本画像，不超过 200 字，不要标题和列表符号

已有画像：
{existing}

本次对话片段：
{chunk}"""


def profile_block(profile: str | None) -> str:
    """画像转系统提示片段（空画像返回空串，不占提示词预算）。"""
    profile = (profile or "").strip()
    if not profile:
        return ""
    return f"## 用户画像（跨会话记忆）\n{profile}"


def _render_chunk(messages: list[dict], limit: int = 6, max_chars: int = 2000) -> str:
    """取最近消息拼为对话片段（限条数与字符数，控制摘要成本）。"""
    lines = [
        f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('content', '')}"
        for m in messages[-limit:]
    ]
    return "\n".join(lines)[:max_chars]


async def refresh_profile(llm, existing: str | None, messages: list[dict]) -> str:
    """LLM 合并画像；调用失败向上抛出（由 maybe_refresh_profile 兜底）。"""
    resp = await llm.ainvoke([{
        "role": "user",
        "content": PROFILE_PROMPT.format(
            existing=existing or "（无）",
            chunk=_render_chunk(messages),
        ),
    }])
    return str(getattr(resp, "content", resp)).strip()


async def maybe_refresh_profile(
    llm,
    db: Session,
    user_id: int,
    messages: list[dict],
    *,
    every: int,
    force: bool = False,
) -> str | None:
    """按节流阈值刷新画像；始终 best-effort（异常返回 None，不中断主流程）。

    返回最新画像文本；未到刷新点或失败返回 None。
    计数器存内存（进程级），重启后从 0 重新计——画像持久化不受影响。
    """
    from app.models.memory import UserMemory

    row = db.query(UserMemory).filter(UserMemory.user_id == user_id).first()
    if row is None:
        row = UserMemory(user_id=user_id, profile="", hits=0)
        db.add(row)
        db.commit()
        db.refresh(row)

    row.hits += 1
    if not force and row.hits < every:
        db.commit()
        return None

    try:
        new_profile = await refresh_profile(llm, row.profile, messages)
        if new_profile:
            row.profile = new_profile
            row.updated_at = datetime.utcnow()
        row.hits = 0
        db.commit()
        logger.info("用户画像已刷新 user_id=%s chars=%s", user_id, len(row.profile))
        return row.profile or None
    except Exception:  # noqa: BLE001 — 画像刷新失败绝不影响对话主流程
        logger.warning("画像刷新失败 user_id=%s", user_id, exc_info=True)
        db.commit()  # hits 已 +1，提交计数
        return None

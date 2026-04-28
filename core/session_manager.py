from __future__ import annotations

import json
import re

from core.config import get_settings
from core.router import ChatMessage
from db.redis_client import RedisClient


def state_key(session_id: str) -> str:
    """Redis key storing per-session state (e.g. active problem id)."""
    settings = get_settings()
    return f"{settings.redis_prefix}:chat:{session_id}:state"


def history_key(session_id: str) -> str:
    """Redis key storing compact conversation history for one session."""
    settings = get_settings()
    return f"{settings.redis_prefix}:chat:{session_id}:history"


def is_finish_confirmation(text: str) -> bool:
    """
    Heuristic completion detector.

    We intentionally use a score-based rule instead of one keyword, so that
    accidental words are less likely to unlock a new problem unexpectedly.
    """
    lowered = text.lower().strip()
    strong_positive = (
        "已完成",
        "完成了",
        "结束当前问题",
        "结束这个问题",
        "进入下一题",
        "下一个问题",
        "新问题",
        "切换问题",
        "换题",
        "ac了",
        "通过了",
        "懂了",
        "明白了",
    )
    weak_positive = ("谢谢", "ok", "好的", "继续", "next")
    negative = ("没懂", "不会", "卡住", "不明白", "再讲", "继续这个题", "还没")

    score = 0
    if any(token in lowered for token in strong_positive):
        score += 2
    if any(token in lowered for token in weak_positive):
        score += 1
    if any(token in lowered for token in negative):
        score -= 2
    if re.search(r"(finish|complete)\s*(current|this)?\s*(question|problem)?", lowered):
        score += 2
    return score >= 2


async def load_history(session_id: str, limit: int = 8) -> list[ChatMessage]:
    """Load recent chat history used for multi-turn context."""
    redis = RedisClient.get_client()
    items = await redis.lrange(history_key(session_id), max(-2 * limit, -100), -1)
    history: list[ChatMessage] = []
    for item in items:
        try:
            parsed = json.loads(item)
        except json.JSONDecodeError:
            continue
        role = parsed.get("role")
        content = parsed.get("content")
        if role in {"system", "user", "assistant"} and isinstance(content, str):
            history.append(ChatMessage(role=role, content=content))
    return history


async def append_history(session_id: str, role: str, content: str) -> None:
    """Append one turn to history and keep only the last 20 entries."""
    redis = RedisClient.get_client()
    key = history_key(session_id)
    await redis.rpush(key, json.dumps({"role": role, "content": content}, ensure_ascii=False))
    await redis.ltrim(key, -20, -1)


async def get_active_problem_id(session_id: str) -> str | None:
    """Get current active problem id for the session."""
    redis = RedisClient.get_client()
    return await redis.hget(state_key(session_id), "active_problem_id")


async def set_active_problem_id(session_id: str, problem_id: str) -> None:
    """Set current active problem id for the session."""
    redis = RedisClient.get_client()
    await redis.hset(state_key(session_id), mapping={"active_problem_id": problem_id})


async def clear_active_problem_id(session_id: str) -> None:
    """Clear current active problem id for the session."""
    redis = RedisClient.get_client()
    await redis.hdel(state_key(session_id), "active_problem_id")

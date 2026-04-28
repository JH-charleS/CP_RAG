from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from core.config import get_settings
from core.llm_client import call_llm_api
from core.router import (
    ChatMessage,
    cache_query_answer_background,
    extract_problem_id,
    run_query_pipeline,
)
from db.redis_client import RedisClient

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, description="User natural language query.")
    session_id: str = Field(default="default", description="Conversation session id.")
    finish_current: bool = Field(
        default=False,
        description="Mark current problem as completed, allowing switch to a new problem.",
    )


class QueryResponse(BaseModel):
    answer: str
    from_cache: bool
    cache_similarity: float | None = None
    matched_problem_id: str | None = None
    rag_hits: int = 0
    session_id: str = "default"
    waiting_for_completion: bool = False
    active_problem_id: str | None = None


def _state_key(session_id: str) -> str:
    settings = get_settings()
    return f"{settings.redis_prefix}:chat:{session_id}:state"


def _history_key(session_id: str) -> str:
    settings = get_settings()
    return f"{settings.redis_prefix}:chat:{session_id}:history"


def _is_finish_confirmation(text: str) -> bool:
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


async def _load_history(session_id: str, limit: int = 8) -> list[ChatMessage]:
    redis = RedisClient.get_client()
    items = await redis.lrange(_history_key(session_id), max(-2 * limit, -100), -1)
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


async def _append_history(session_id: str, role: str, content: str) -> None:
    redis = RedisClient.get_client()
    key = _history_key(session_id)
    await redis.rpush(key, json.dumps({"role": role, "content": content}, ensure_ascii=False))
    await redis.ltrim(key, -20, -1)


@router.post("", response_model=QueryResponse, summary="CP-RAG query endpoint")
async def query_route(payload: QueryRequest) -> QueryResponse:
    try:
        session_id = payload.session_id.strip() or "default"
        redis = RedisClient.get_client()
        state_key = _state_key(session_id)

        finish_confirmed = payload.finish_current or _is_finish_confirmation(payload.query)
        if finish_confirmed:
            await redis.hdel(state_key, "active_problem_id")

        active_problem_id = await redis.hget(state_key, "active_problem_id")
        incoming_problem = extract_problem_id(payload.query)
        if (
            active_problem_id
            and incoming_problem is not None
            and incoming_problem.normalized_id != active_problem_id
            and not finish_confirmed
        ):
            return QueryResponse(
                answer=(
                    f"当前会话仍在讨论题目 {active_problem_id}。"
                    "若你确认已完成当前题，请在下一条请求设置 finish_current=true，"
                    "或在文本中输入“下一个问题/新问题/结束当前问题”。"
                ),
                from_cache=True,
                cache_similarity=None,
                matched_problem_id=active_problem_id,
                rag_hits=0,
                session_id=session_id,
                waiting_for_completion=True,
                active_problem_id=active_problem_id,
            )

        pipeline_result = await run_query_pipeline(payload.query)
        if pipeline_result.from_cache and pipeline_result.cached_answer is not None:
            if incoming_problem is not None:
                active_problem_id = incoming_problem.normalized_id
                await redis.hset(state_key, mapping={"active_problem_id": active_problem_id})
            await _append_history(session_id, "user", payload.query)
            await _append_history(session_id, "assistant", pipeline_result.cached_answer)
            return QueryResponse(
                answer=pipeline_result.cached_answer,
                from_cache=True,
                cache_similarity=pipeline_result.cache_similarity,
                matched_problem_id=active_problem_id,
                rag_hits=0,
                session_id=session_id,
                waiting_for_completion=bool(active_problem_id),
                active_problem_id=active_problem_id,
            )

        if pipeline_result.payload is None:
            raise RuntimeError("Pipeline returned no LLM payload on cache miss.")

        history = await _load_history(session_id)
        messages = pipeline_result.payload.get("messages", [])
        if len(messages) >= 2 and history:
            system_message = messages[0]
            latest_user_message = messages[-1]
            pipeline_result.payload["messages"] = [system_message, *history, latest_user_message]

        answer = await call_llm_api(pipeline_result.payload)
        cache_query_answer_background(
            query=payload.query,
            query_vector=pipeline_result.query_vector,
            answer=answer,
        )
        await _append_history(session_id, "user", payload.query)
        await _append_history(session_id, "assistant", answer)

        matched_problem_id = None
        rag_hits = 0
        if pipeline_result.route_result is not None:
            rag_hits = len(pipeline_result.route_result.rag_hits)
            if pipeline_result.route_result.matched_problem is not None:
                matched_problem_id = pipeline_result.route_result.matched_problem.normalized_id
                await redis.hset(state_key, mapping={"active_problem_id": matched_problem_id})
                active_problem_id = matched_problem_id

        return QueryResponse(
            answer=answer,
            from_cache=False,
            cache_similarity=None,
            matched_problem_id=matched_problem_id,
            rag_hits=rag_hits,
            session_id=session_id,
            waiting_for_completion=bool(active_problem_id),
            active_problem_id=active_problem_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from core.llm_client import call_llm_api
from core.router import (
    cache_query_answer_background,
    extract_problem_id,
    run_query_pipeline,
)
from core.session_manager import (
    append_history,
    clear_active_problem_id,
    get_active_problem_id,
    is_finish_confirmation,
    load_history,
    set_active_problem_id,
)

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


@router.post("", response_model=QueryResponse, summary="CP-RAG query endpoint")
async def query_route(payload: QueryRequest) -> QueryResponse:
    try:
        session_id = payload.session_id.strip() or "default"

        finish_confirmed = payload.finish_current or is_finish_confirmation(payload.query)
        if finish_confirmed:
            await clear_active_problem_id(session_id)

        active_problem_id = await get_active_problem_id(session_id)
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
                await set_active_problem_id(session_id, active_problem_id)
            await append_history(session_id, "user", payload.query)
            await append_history(session_id, "assistant", pipeline_result.cached_answer)
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

        history = await load_history(session_id)
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
        await append_history(session_id, "user", payload.query)
        await append_history(session_id, "assistant", answer)

        matched_problem_id = None
        rag_hits = 0
        if pipeline_result.route_result is not None:
            rag_hits = len(pipeline_result.route_result.rag_hits)
            if pipeline_result.route_result.matched_problem is not None:
                matched_problem_id = pipeline_result.route_result.matched_problem.normalized_id
                await set_active_problem_id(session_id, matched_problem_id)
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

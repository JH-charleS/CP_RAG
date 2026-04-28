from __future__ import annotations

import re
import asyncio
import json
import math
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, TypedDict

import aiomysql
from sentence_transformers import SentenceTransformer

from core.config import get_settings
from db.milvus_client import MilvusClient
from db.mysql_pool import MySQLPool
from db.redis_client import RedisClient


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMPayload(TypedDict, total=False):
    model: str
    messages: list[ChatMessage]
    temperature: float
    max_tokens: int
    metadata: dict[str, str]


@dataclass(slots=True, frozen=True)
class ProblemMatch:
    platform: str
    normalized_id: str
    raw_id: str


@dataclass(slots=True)
class SolutionRecord:
    problem_id: str
    title: str
    context_markdown: str
    ac_code: str
    source: str
    created_at: str | None


@dataclass(slots=True)
class QueryRouteResult:
    matched_problem: ProblemMatch | None
    solution: SolutionRecord | None
    rag_hits: list["RAGHit"]


@dataclass(slots=True)
class RAGHit:
    score: float
    payload: dict[str, Any]


@dataclass(slots=True)
class QueryPipelineResult:
    from_cache: bool
    query_vector: list[float]
    cached_answer: str | None
    route_result: QueryRouteResult | None
    payload: LLMPayload | None
    cache_similarity: float | None = None


_EMBEDDER: SentenceTransformer | None = None


_PROBLEM_PATTERNS: list[tuple[str, re.Pattern[str], Any]] = [
    (
        "luogu",
        re.compile(r"(?<![A-Za-z0-9])(?:luogu\s*)?(P[1-9]\d{3})(?!\d)", flags=re.IGNORECASE),
        lambda m: m.group(1).upper(),
    ),
    (
        "codeforces",
        re.compile(r"(?<![A-Za-z0-9])(?:codeforces\s*)?(\d{1,4}[A-Za-z]{1,2})(?![A-Za-z0-9])", flags=re.IGNORECASE),
        lambda m: f"CF{m.group(1).upper()}",
    ),
    (
        "leetcode",
        re.compile(
            r"(?<![A-Za-z0-9])(?:leetcode\s*)?(?:LC\s*[-_ ]*)?0*(\d{1,4})(?!\d)",
            flags=re.IGNORECASE,
        ),
        lambda m: f"LC{int(m.group(1)):04d}",
    ),
    (
        "atcoder",
        re.compile(
            r"(?<![A-Za-z0-9])(?:atcoder\s*)?([A-Za-z]{2,3}\d{3,4}[A-Za-z]?)(?![A-Za-z0-9])",
            flags=re.IGNORECASE,
        ),
        lambda m: f"AT{m.group(1).upper()}",
    ),
]


SOCRATIC_SYSTEM_PROMPT: str = (
    "你是 CP-RAG 的苏格拉底式算法助教。你的目标不是直接给答案，而是通过提问和启发帮助学生理解思路。\n"
    "教学原则：\n"
    "1) 先确认题意和约束，再引导学生提出可行方向。\n"
    "2) 每次只给一个关键提示，优先问问题，避免一次性泄露完整解法。\n"
    "3) 结合题解上下文时，要明确引用“思路/复杂度/边界条件/实现细节”。\n"
    "4) 若学生明确要求完整答案，可先给分步骤框架，再给完整解答。\n"
    "5) 代码讲解要解释变量含义、状态转移或贪心依据，并提醒常见坑点。\n"
    "6) 输出简洁、结构清晰，优先中文；必要时保留英文术语。"
)


def extract_problem_id(query: str) -> ProblemMatch | None:
    """Extract the first recognized problem ID from user query."""
    for platform, pattern, normalizer in _PROBLEM_PATTERNS:
        match = pattern.search(query)
        if match is None:
            continue
        normalized_id = str(normalizer(match))
        return ProblemMatch(platform=platform, normalized_id=normalized_id, raw_id=match.group(0).strip())
    return None


def _get_embedder() -> SentenceTransformer:
    """Lazily load local sentence-transformers model."""
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    settings = get_settings()
    _EMBEDDER = SentenceTransformer(settings.embedding_model_name)
    return _EMBEDDER


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _cache_index_key() -> str:
    settings = get_settings()
    return f"{settings.redis_prefix}:query_cache:index"


def _cache_entry_key(cache_id: str) -> str:
    settings = get_settings()
    return f"{settings.redis_prefix}:query_cache:{cache_id}"


async def embed_query(text: str) -> list[float]:
    """Generate embedding vector for query text using BGE-M3."""
    embedder = _get_embedder()
    vector = await asyncio.to_thread(
        embedder.encode,
        text,
        normalize_embeddings=True,
    )
    return [float(x) for x in vector.tolist()]


async def get_cached_answer_by_similarity(
    query_vector: list[float],
    *,
    threshold: float | None = None,
) -> tuple[str, float] | None:
    """Tier 1: semantic cache lookup in Redis using cosine similarity."""
    settings = get_settings()
    similarity_threshold = threshold if threshold is not None else settings.redis_cache_similarity_threshold
    redis = RedisClient.get_client()
    index_key = _cache_index_key()
    entry_ids = await redis.zrevrange(index_key, 0, settings.redis_cache_max_entries - 1)
    if not entry_ids:
        return None

    best_answer: str | None = None
    best_score: float = -1.0
    for cache_id in entry_ids:
        entry_key = _cache_entry_key(cache_id)
        entry_data = await redis.hgetall(entry_key)
        if not entry_data:
            continue
        vector_raw = entry_data.get("query_vector")
        answer = entry_data.get("answer")
        if not vector_raw or not answer:
            continue
        try:
            cached_vector = json.loads(vector_raw)
            if not isinstance(cached_vector, list):
                continue
            cached_vector_f = [float(x) for x in cached_vector]
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

        score = cosine_similarity(query_vector, cached_vector_f)
        if score > best_score:
            best_score = score
            best_answer = answer

    if best_answer is not None and best_score >= similarity_threshold:
        return best_answer, best_score
    return None


async def cache_query_answer(
    query: str,
    query_vector: list[float],
    answer: str,
) -> None:
    """Persist query embedding and answer into Redis cache."""
    settings = get_settings()
    redis = RedisClient.get_client()
    cache_id = str(uuid.uuid4())
    now_ts = time.time()
    entry_key = _cache_entry_key(cache_id)
    index_key = _cache_index_key()
    payload: dict[str, str] = {
        "query": query,
        "query_vector": json.dumps(query_vector, ensure_ascii=False),
        "answer": answer,
        "created_at": str(now_ts),
    }
    await redis.hset(entry_key, mapping=payload)
    await redis.zadd(index_key, {cache_id: now_ts})

    # Retain only the latest N entries to bound memory.
    max_entries = settings.redis_cache_max_entries
    size = await redis.zcard(index_key)
    overflow = size - max_entries
    if overflow > 0:
        stale_ids = await redis.zrange(index_key, 0, overflow - 1)
        if stale_ids:
            stale_keys = [_cache_entry_key(item_id) for item_id in stale_ids]
            await redis.delete(*stale_keys)
            await redis.zrem(index_key, *stale_ids)


def cache_query_answer_background(
    query: str,
    query_vector: list[float],
    answer: str,
) -> asyncio.Task[None]:
    """Schedule non-blocking cache write after LLM response."""
    return asyncio.create_task(cache_query_answer(query=query, query_vector=query_vector, answer=answer))


async def fetch_solution_by_id(problem_id: str) -> SolutionRecord | None:
    """Tier 2: fetch markdown solution from MySQL by problem ID."""
    pool = MySQLPool.get_pool()
    async with pool.acquire() as connection:
        async with connection.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(
                """
                SELECT id, title, context, ac_code, source, created_at
                FROM solutions
                WHERE UPPER(id) = UPPER(%s)
                LIMIT 1
                """,
                (problem_id,),
            )
            row: dict[str, Any] | None = await cursor.fetchone()
            if row is None:
                return None
            context = row.get("context")
            # Backward compatibility for older schema/scripts that used `content`.
            if not context:
                context = row.get("content", "")
            return SolutionRecord(
                problem_id=str(row.get("id", "")),
                title=str(row.get("title", "")),
                context_markdown=str(context or ""),
                ac_code=str(row.get("ac_code", "") or ""),
                source=str(row.get("source", "") or ""),
                created_at=str(row.get("created_at")) if row.get("created_at") is not None else None,
            )


async def rag_search(query: str, *, top_k: int | None = None) -> list[RAGHit]:
    """Tier 3: semantic retrieval from Milvus using BGE-M3 embeddings."""
    settings = get_settings()
    if not await MilvusClient.has_collection(settings.milvus_collection):
        return []

    vector = await embed_query(query)
    output_fields = [field.strip() for field in settings.milvus_output_fields.split(",") if field.strip()]
    hits_raw = await MilvusClient.search(
        collection_name=settings.milvus_collection,
        query_vector=vector,
        vector_field=settings.milvus_vector_field,
        output_fields=output_fields,
        limit=top_k or settings.rag_top_k,
    )
    hits: list[RAGHit] = []
    for item in hits_raw:
        score = float(item.get("score", 0.0))
        payload = {k: v for k, v in item.items() if k != "score"}
        hits.append(RAGHit(score=score, payload=payload))
    return hits


async def route_query(query: str) -> QueryRouteResult:
    """Run query router with Tier 1 + Tier 2 + Tier 3 retrieval."""
    matched = extract_problem_id(query)
    rag_hits = await rag_search(query)
    if matched is None:
        return QueryRouteResult(matched_problem=None, solution=None, rag_hits=rag_hits)
    solution = await fetch_solution_by_id(matched.normalized_id)
    return QueryRouteResult(matched_problem=matched, solution=solution, rag_hits=rag_hits)


def _format_rag_context(hits: list[RAGHit]) -> str:
    if not hits:
        return "未命中 Milvus 语义检索结果。"
    lines: list[str] = []
    for index, hit in enumerate(hits, start=1):
        payload = hit.payload
        text = payload.get("context") or payload.get("content") or payload.get("text") or ""
        title = payload.get("title") or "unknown"
        source = payload.get("source") or "unknown"
        doc_id = payload.get("id") or payload.get("doc_id") or f"doc_{index}"
        lines.append(
            f"[RAG#{index}] score={hit.score:.4f} id={doc_id} title={title} source={source}\n"
            f"{text}"
        )
    return "\n\n".join(lines)


async def run_query_pipeline(user_query: str) -> QueryPipelineResult:
    """
    Execute full pipeline:
    1) Tier 1 semantic cache hit in Redis.
    2) If miss, continue Tier 2/Tier 3 router + payload build.
    """
    query_vector = await embed_query(user_query)
    cache_hit = await get_cached_answer_by_similarity(query_vector)
    if cache_hit is not None:
        answer, similarity = cache_hit
        return QueryPipelineResult(
            from_cache=True,
            query_vector=query_vector,
            cached_answer=answer,
            route_result=None,
            payload=None,
            cache_similarity=similarity,
        )

    route_result = await route_query(user_query)
    payload = build_llm_payload(user_query=user_query, route_result=route_result)
    return QueryPipelineResult(
        from_cache=False,
        query_vector=query_vector,
        cached_answer=None,
        route_result=route_result,
        payload=payload,
    )


async def run_query_pipeline_with_llm(
    user_query: str,
    llm_call: Callable[[LLMPayload], Awaitable[str]],
) -> str:
    """
    Orchestrate Tier1/Tier2/Tier3 and return final answer string.
    If Tier1 cache hit, return directly; otherwise call LLM and async-write cache.
    """
    result = await run_query_pipeline(user_query)
    if result.from_cache and result.cached_answer is not None:
        return result.cached_answer

    if result.payload is None:
        raise RuntimeError("LLM payload is missing on cache miss path.")

    answer = await llm_call(result.payload)
    cache_query_answer_background(
        query=user_query,
        query_vector=result.query_vector,
        answer=answer,
    )
    return answer


def build_llm_payload(
    user_query: str,
    route_result: QueryRouteResult,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> LLMPayload:
    """Assemble final payload for upstream LLM API."""
    retrieval_context = "未识别到题号，或知识库中暂无对应题解。"
    rag_context = _format_rag_context(route_result.rag_hits)
    metadata: dict[str, str] = {"tier": "tier1_only", "problem_id": ""}

    if route_result.matched_problem is not None:
        metadata["problem_id"] = route_result.matched_problem.normalized_id
        metadata["platform"] = route_result.matched_problem.platform
        metadata["tier"] = "tier2_hit" if route_result.solution is not None else "tier2_miss"
    metadata["rag_hits"] = str(len(route_result.rag_hits))

    if route_result.solution is not None:
        retrieval_context = (
            f"题号: {route_result.solution.problem_id}\n"
            f"标题: {route_result.solution.title}\n"
            f"来源: {route_result.solution.source or 'unknown'}\n"
            f"题解Markdown:\n{route_result.solution.context_markdown}\n\n"
            f"参考代码:\n{route_result.solution.ac_code}"
        )

    user_content = (
        f"[用户问题]\n{user_query}\n\n"
        f"[Tier2 精确检索上下文]\n{retrieval_context}\n\n"
        f"[Tier3 RAG 语义检索上下文]\n{rag_context}\n\n"
        "请按苏格拉底助教风格回答。"
    )

    return LLMPayload(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        metadata=metadata,
        messages=[
            ChatMessage(role="system", content=SOCRATIC_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_content),
        ],
    )

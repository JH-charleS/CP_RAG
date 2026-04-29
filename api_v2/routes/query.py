from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from core_v2.pipeline import run_v2_pipeline

router = APIRouter(prefix="/query", tags=["query-v2"])


class QueryV2Response(BaseModel):
    answer: str
    intent: str = Field(description="Predicted query intent label.")
    rewritten_query: str
    image_summary: str | None = None
    rag_hits: int = 0


@router.post("", response_model=QueryV2Response, summary="CP-RAG v2 query endpoint")
async def query_route_v2(
    query: str = Form(..., min_length=1),
    image: UploadFile | None = File(default=None),
) -> QueryV2Response:
    try:
        result = await run_v2_pipeline(query, image)
        return QueryV2Response(
            answer=result.answer,
            intent=result.intent,
            rewritten_query=result.rewritten_query,
            image_summary=result.image_summary or None,
            rag_hits=len(result.rag_hits),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"V2 query failed: {exc}") from exc

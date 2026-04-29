from __future__ import annotations

from fastapi import UploadFile
from langchain_core.runnables import RunnableLambda

from core_v2.intent_classifier import classify_query_intent
from core_v2.models import QueryArtifacts
from core_v2.qwen_client import qwen_answer, qwen_rewrite_query, qwen_vision_describe_image
from core_v2.retriever import retrieve_with_parent

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png"}


async def _multimodal_enrich(art: QueryArtifacts, image: UploadFile | None) -> QueryArtifacts:
    if image is None:
        art.merged_query = art.raw_query
        return art
    if image.content_type not in SUPPORTED_IMAGE_TYPES:
        raise ValueError("Only jpg/png images are supported.")
    image_bytes = await image.read()
    art.image_summary = await qwen_vision_describe_image(image_bytes, image.content_type or "image/jpeg")
    art.merged_query = (
        f"[用户原始问题]\n{art.raw_query}\n\n"
        f"[图片识别内容]\n{art.image_summary}"
    )
    return art


def _classify(art: QueryArtifacts) -> QueryArtifacts:
    art.intent = classify_query_intent(art.merged_query or art.raw_query)
    return art


async def _rewrite(art: QueryArtifacts) -> QueryArtifacts:
    art.rewritten_query = await qwen_rewrite_query(art.merged_query or art.raw_query, art.intent)
    return art


def _retrieve(art: QueryArtifacts) -> QueryArtifacts:
    hits, context = retrieve_with_parent(art.rewritten_query or art.merged_query or art.raw_query)
    art.rag_hits = hits
    art.rag_context = context
    return art


async def _answer(art: QueryArtifacts) -> QueryArtifacts:
    art.answer = await qwen_answer(
        user_query=art.raw_query,
        rag_context=art.rag_context,
        intent=art.intent,
    )
    return art


async def run_v2_pipeline(query: str, image: UploadFile | None) -> QueryArtifacts:
    artifacts = QueryArtifacts(raw_query=query)

    artifacts = await _multimodal_enrich(artifacts, image)
    artifacts = RunnableLambda(lambda x: _classify(x)).invoke(artifacts)
    artifacts = await _rewrite(artifacts)
    artifacts = RunnableLambda(lambda x: _retrieve(x)).invoke(artifacts)
    artifacts = await _answer(artifacts)
    return artifacts

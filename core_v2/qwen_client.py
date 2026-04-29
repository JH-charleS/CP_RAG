from __future__ import annotations

import base64
from typing import Any

import httpx

from core.config import get_settings


def _base_url() -> str:
    return get_settings().v2_qwen_api_base_url.rstrip("/")


def _headers() -> dict[str, str]:
    settings = get_settings()
    if not settings.v2_qwen_api_key:
        raise RuntimeError("V2_QWEN_API_KEY is empty. Please configure it in .env.")
    return {
        "Authorization": f"Bearer {settings.v2_qwen_api_key}",
        "Content-Type": "application/json",
    }


async def _chat_completion(payload: dict[str, Any]) -> str:
    url = f"{_base_url()}/chat/completions"
    timeout = httpx.Timeout(get_settings().llm_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, headers=_headers(), json=payload)
        response.raise_for_status()
    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("Qwen response has no choices.")
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
        content = "\n".join([item for item in text_parts if item.strip()])
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Qwen response content is empty.")
    return content.strip()


async def qwen_vision_describe_image(image_bytes: bytes, mime_type: str) -> str:
    settings = get_settings()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{encoded}"
    payload = {
        "model": settings.v2_qwen_vision_model,
        "messages": [
            {
                "role": "system",
                "content": "你是算法题图片理解助手，提取题目关键信息，输出简明中文要点。",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请识别图片中的题意、输入输出、约束、关键公式。"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": 500,
    }
    return await _chat_completion(payload)


async def qwen_rewrite_query(query: str, intent: str) -> str:
    settings = get_settings()
    prompt = (
        "你是检索查询优化器。请根据意图标签将用户问题改写成适合向量检索的短查询，"
        "保留算法术语、题目约束和关键词；只输出改写结果，不要解释。\n"
        f"意图标签: {intent}\n"
        f"用户问题: {query}"
    )
    payload = {
        "model": settings.v2_qwen_text_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 200,
    }
    return await _chat_completion(payload)


async def qwen_answer(user_query: str, rag_context: str, intent: str) -> str:
    settings = get_settings()
    payload = {
        "model": settings.v2_qwen_text_model,
        "messages": [
            {
                "role": "system",
                "content": "你是 CP-RAG v2 助教。请结合上下文给出准确、简洁、可执行的解题引导。",
            },
            {
                "role": "user",
                "content": (
                    f"[意图类别]\n{intent}\n\n"
                    f"[用户问题]\n{user_query}\n\n"
                    f"[检索上下文]\n{rag_context}\n\n"
                    "请给出分步骤指导。"
                ),
            },
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
    }
    return await _chat_completion(payload)

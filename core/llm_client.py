from __future__ import annotations

from typing import Any

import httpx

from core.config import get_settings
from core.router import LLMPayload


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


async def call_llm_api(payload: LLMPayload) -> str:
    """
    Call an OpenAI-compatible chat completions API.

    This works with providers such as DeepSeek compatible endpoints when
    `LLM_API_BASE_URL` and `LLM_API_KEY` are configured in `.env`.
    """
    settings = get_settings()
    if not settings.llm_api_key:
        raise RuntimeError("LLM_API_KEY is empty. Please set it in .env.")

    base_url = _normalize_base_url(settings.llm_api_base_url)
    url = f"{base_url}/chat/completions"
    body: dict[str, Any] = {
        "model": payload.get("model", settings.llm_model),
        "messages": payload.get("messages", []),
        "temperature": payload.get("temperature", 0.2),
        "max_tokens": payload.get("max_tokens", 1200),
    }

    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(settings.llm_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, headers=headers, json=body)
        response.raise_for_status()
        data: dict[str, Any] = response.json()

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("LLM response has no choices field.")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("LLM response content is empty.")
    return content

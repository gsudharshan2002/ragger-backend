import json
from typing import AsyncGenerator, Optional

import httpx

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question based on the provided context. "
    "If the context doesn't contain enough information, say so clearly. "
    "Always cite your sources when possible."
)


def _get_api_key(provider: str) -> Optional[str]:
    if provider == "gemini":
        return settings.GEMINI_API_KEY
    return settings.GROQ_API_KEY


def _get_api_url(provider: str) -> str:
    if provider == "gemini":
        return GEMINI_API_URL
    return GROQ_API_URL


async def is_llm_configured() -> bool:
    api_key = _get_api_key(settings.LLM_PROVIDER)
    return bool(api_key)


async def generate_completion_stream(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> AsyncGenerator[dict, None]:
    provider = settings.LLM_PROVIDER
    api_key = _get_api_key(provider)

    if not api_key:
        env_var = "GEMINI_API_KEY" if provider == "gemini" else "GROQ_API_KEY"
        yield {
            "type": "error",
            "error": f"{env_var} is not set. Please add it in Settings or set it in your .env file.",
        }
        return

    model = model or settings.GROQ_MODEL
    messages = [
        {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        async with httpx.AsyncClient(timeout=settings.LLM_STREAM_TIMEOUT) as client:
            async with client.stream(
                "POST",
                _get_api_url(provider),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True,
                },
            ) as response:
                if not response.is_success:
                    error_body = await response.aread()
                    yield {
                        "type": "error",
                        "error": f"{provider} API error ({response.status_code}): {error_body.decode()}",
                    }
                    return

                buffer = ""
                output_token_count = 0

                async for chunk in response.aiter_text():
                    buffer += chunk
                    lines = buffer.split("\n")
                    buffer = lines.pop()

                    for line in lines:
                        trimmed = line.strip()
                        if not trimmed or not trimmed.startswith("data: "):
                            continue

                        payload = trimmed[6:]
                        if payload == "[DONE]":
                            yield {
                                "type": "done",
                                "tokens": {"input": None, "output": output_token_count, "total": None},
                            }
                            return

                        try:
                            parsed = json.loads(payload)
                            choices = parsed.get("choices", [])
                            if not choices:
                                continue

                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                output_token_count += 1
                                yield {"type": "token", "content": content}
                        except json.JSONDecodeError:
                            continue

                yield {
                    "type": "done",
                    "tokens": {"input": None, "output": output_token_count, "total": None},
                }
    except Exception as e:
        yield {
            "type": "error",
            "error": f"{provider} streaming request failed: {e}",
        }

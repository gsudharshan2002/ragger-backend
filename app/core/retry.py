import asyncio
from typing import Awaitable, Callable, TypeVar

import httpx

from app.core.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


async def retry_on_rate_limit(
    fn: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> T:
    """Retry an async provider call with exponential backoff when it fails
    with 429 (rate limited). Any other error is re-raised immediately;
    a 429 is re-raised once retries are exhausted. Honors the provider's
    Retry-After header when present instead of guessing the wait time."""
    attempt = 0
    while True:
        try:
            return await fn()
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 429 or attempt >= max_retries:
                raise
            retry_after = e.response.headers.get("retry-after")
            delay = float(retry_after) if retry_after else base_delay * (2 ** attempt)
            logger.warning(
                f"Rate limited (429) calling {e.request.url.host}; "
                f"retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})..."
            )
            await asyncio.sleep(delay)
            attempt += 1

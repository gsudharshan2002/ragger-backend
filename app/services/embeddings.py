import asyncio
from typing import TYPE_CHECKING, Any, Literal, Optional, List

import httpx

from app.core.config import settings
from app.core.logging_config import get_logger
from app.core.retry import retry_on_rate_limit

logger = get_logger(__name__)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

COHERE_EMBED_URL = "https://api.cohere.com/v2/embed"

_model: Optional[Any] = None


def _get_model(model_name: str) -> Any:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        try:
            _model = SentenceTransformer(model_name)
            logger.info(f"Loaded local embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{model_name}': {e}")
            if model_name == settings.EMBEDDING_MODEL:
                raise
            # Persisted settings pointed at a model this backend can't load
            # (e.g. a provider-specific name like an OpenAI/Cohere model id,
            # since only local sentence-transformers models are supported
            # here). Fall back to the configured default instead of failing
            # every embedding call.
            logger.warning(
                f"Falling back to default embedding model: {settings.EMBEDDING_MODEL}"
            )
            _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


async def _resolve_embedding_provider() -> str:
    from app.services.storage import get_settings

    persisted = await get_settings()
    provider = persisted.get("embeddingProvider") or settings.EMBEDDING_PROVIDER
    return provider if provider in ("local", "cohere") else "local"


async def is_embedding_configured() -> bool:
    # Local always works. Cohere falls back to local when unconfigured (see
    # get_embeddings_for_texts), so this stays True either way.
    return True


async def generate_query_embedding(text: str) -> Optional[List[float]]:
    embeddings = await get_embeddings_for_texts([text], input_type="search_query")
    return embeddings[0] if embeddings else None


async def _cohere_embed(
    texts: List[str],
    model: str,
    input_type: Literal["search_document", "search_query"],
) -> List[List[float]]:
    async def call() -> httpx.Response:
        async with httpx.AsyncClient(timeout=settings.EMBEDDING_TIMEOUT) as client:
            response = await client.post(
                COHERE_EMBED_URL,
                headers={
                    "Authorization": f"Bearer {settings.COHERE_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "texts": texts,
                    "input_type": input_type,
                    "embedding_types": ["float"],
                },
            )
            response.raise_for_status()
            return response

    response = await retry_on_rate_limit(call)
    return response.json()["embeddings"]["float"]


async def get_embeddings_for_texts(
    texts: List[str],
    input_type: Literal["search_document", "search_query"] = "search_document",
) -> Optional[List[List[float]]]:
    if not texts:
        return None

    try:
        from app.services.storage import get_settings

        persisted = await get_settings()
        provider = await _resolve_embedding_provider()

        if provider == "cohere" and settings.COHERE_API_KEY:
            model_name = persisted.get("cohereEmbedModel") or settings.COHERE_EMBED_MODEL
            batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)
            all_embeddings: List[List[float]] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                all_embeddings.extend(await _cohere_embed(batch, model_name, input_type))
            return all_embeddings

        if provider == "cohere":
            logger.warning(
                "Embedding provider is 'cohere' but COHERE_API_KEY is not set; "
                "falling back to the local model instead."
            )

        model_name = persisted.get("embeddingModel") or settings.EMBEDDING_MODEL
        model = _get_model(model_name)
        # Use batch size from settings, but ensure at least 1
        batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)
        all_embeddings: List[List[float]] = []

        # Process in batches to avoid excessive memory usage
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Encode in a thread to avoid blocking the event loop
            batch_embeddings = await asyncio.to_thread(
                model.encode, batch, normalize_embeddings=False
            )
            # Convert numpy arrays to lists
            all_embeddings.extend([emb.tolist() for emb in batch_embeddings])

        return all_embeddings
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None

import asyncio
from typing import TYPE_CHECKING, Any, Optional, List

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

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


async def is_embedding_configured() -> bool:
    # Always configured for local model
    return True


async def generate_query_embedding(text: str) -> Optional[List[float]]:
    embeddings = await get_embeddings_for_texts([text])
    return embeddings[0] if embeddings else None


async def get_embeddings_for_texts(texts: List[str]) -> Optional[List[List[float]]]:
    if not texts:
        return None

    try:
        from app.services.storage import get_settings

        persisted = await get_settings()
        provider = persisted.get("embeddingProvider") or settings.EMBEDDING_PROVIDER
        if provider != "local":
            logger.warning(
                f"Embedding provider '{provider}' is not supported by this backend "
                "(only local sentence-transformers models are supported); "
                "using the default local model instead."
            )
            model_name = settings.EMBEDDING_MODEL
        else:
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

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
        try:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(model_name)
            logger.info(f"Loaded local embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
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

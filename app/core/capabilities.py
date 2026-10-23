"""Report which retrieval capabilities are actually usable in this environment.

Vector search and the local reranker need the optional sentence-transformers
(torch) stack. When it is missing, retrieval silently degrades to BM25 (see
RagEngine.get_context). This makes that state visible at startup and through
GET /api/v1/health/capabilities, instead of discovering it from bad answers.
"""

from importlib.util import find_spec

from app.core.config import settings


def detect_capabilities() -> dict:
    local_stack = find_spec("sentence_transformers") is not None
    cohere = bool(settings.COHERE_API_KEY)

    embeddings_ok = local_stack if settings.EMBEDDING_PROVIDER == "local" else cohere
    reranker_ok = local_stack if settings.RERANKER_PROVIDER == "local" else cohere

    degraded: list[str] = []
    if not embeddings_ok:
        degraded.append(
            f"embedding provider '{settings.EMBEDDING_PROVIDER}' is unavailable: vector search is off, retrieval uses BM25 only"
        )
    if not reranker_ok:
        degraded.append(
            f"reranker provider '{settings.RERANKER_PROVIDER}' is unavailable: reranking is skipped, fused ranking is used"
        )

    return {
        "sentence_transformers_installed": local_stack,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "vector_search_available": embeddings_ok,
        "reranker_provider": settings.RERANKER_PROVIDER,
        "reranker_available": reranker_ok,
        "default_strategy": settings.DEFAULT_STRATEGY,
        "degraded": degraded,
    }

"""Week 11: Retrieval result cache for cost improvement.

Caches final retrieved chunks keyed by (query, strategy, knowledge_base_id).
A cache hit skips the retrieval/re-rerank stages entirely, avoiding the
reranker (and embedding) model calls for repeated identical queries.

All values are estimates; the cache is in-memory per process and does not
bypass version, document-freshness, or access constraints because the cache
key includes the knowledge base and the cached chunks carry their source
document/version metadata.
"""

import hashlib
from typing import Optional

_cache: dict[str, list] = {}


def _key(query: str, strategy: str, knowledge_base_id: Optional[str]) -> str:
    h = hashlib.sha256()
    h.update(query.encode("utf-8"))
    h.update(b"|")
    h.update(str(strategy or "").encode("utf-8"))
    h.update(b"|")
    h.update((knowledge_base_id or "").encode("utf-8"))
    return h.hexdigest()


def get(query: str, strategy: str, knowledge_base_id: Optional[str] = None) -> Optional[list]:
    return _cache.get(_key(query, strategy, knowledge_base_id))


def set(
    query: str,
    strategy: str,
    chunks: list,
    knowledge_base_id: Optional[str] = None,
) -> None:
    _cache[_key(query, strategy, knowledge_base_id)] = list(chunks)


def clear() -> None:
    _cache.clear()


def size() -> int:
    return len(_cache)

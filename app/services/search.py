import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import httpx
import numpy as np

from app.core.config import settings
from app.core.retry import retry_on_rate_limit
from app.models.schemas import StoredChunk
import asyncio
from app.core.logging_config import get_logger
logger = get_logger(__name__)

COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"

# Keyword hits in BM25 score as if they appeared KEYWORD_BOOST times in the
# chunk's character text, so a chunk whose stored metadata keywords match the
# query outranks plain content-only matches.
KEYWORD_BOOST = 3

# English stopwords dropped from BM25 *query* terms so the terms shown alongside
# a question actually reflect what the question asks ("does the publisher..." ->
# ["publisher", "protocol", "subscriber", "value", "events"] instead of
# ["does", "the", "publisher", ...]). They contribute only noise to scoring
# anyway (near-universal document frequency -> tiny IDF).
_QUERY_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can't cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he he'd he'll he's
    her here here's hers herself him himself his how how's i i'd i'll i'm i've
    if in into is isn't it it's its itself let's me more most mustn't my myself
    no nor not of off on once only or other ought our ours ourselves out over
    own same shan't she she'd she'll she's should shouldn't so some such than
    that that's the their theirs them themselves then there there's these they
    they'd they'll they're they've this those through to too under until up
    very was wasn't we we'd we'll we're we've were weren't what what's when
    when's where where's which while who who's whom why why's with won't would
    wouldn't you you'd you'll you're you've your yours yourself yourselves
    """.split()
)

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

_cross_encoder: Optional[Any] = None
_cross_encoder_model_name: Optional[str] = None

def _get_cross_encoder(model_name: str) -> Any:
    global _cross_encoder, _cross_encoder_model_name
    if _cross_encoder is None or _cross_encoder_model_name != model_name:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(model_name)
        _cross_encoder_model_name = model_name
    return _cross_encoder


@dataclass
class VectorResult:
    chunk: StoredChunk
    chunk_id: str
    score: float
    rank: int = 0
    method: str = "vector"


@dataclass
class BM25Result:
    chunk: StoredChunk
    chunk_id: str
    score: float
    rank: int = 0
    query_terms: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    method: str = "bm25"


@dataclass
class RRFResult:
    chunk: StoredChunk
    chunk_id: str
    rrf_score: float
    rank: int = 0
    method: str = "rrf"


@dataclass
class RerankResult:
    chunk: StoredChunk
    chunk_id: str
    rerank_score: float
    rank: int = 0
    method: str = "reranker"


@dataclass
class MMRResult:
    chunk_id: str
    chunk: StoredChunk
    mmr_score: float
    relevance_score: float
    max_similarity: float
    selected: bool
    rank: int = 0
    method: str = "mmr"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    arr_a = np.array(a)
    arr_b = np.array(b)
    norm_a = np.linalg.norm(arr_a)
    norm_b = np.linalg.norm(arr_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(arr_a, arr_b) / (norm_a * norm_b))


def dot_product(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(np.dot(np.array(a), np.array(b)))


def l2_distance(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return float("inf")
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def vector_search(
    query_embedding: list[float],
    chunks: list[StoredChunk],
    top_k: int,
    similarity_threshold: float = 0.0,
    similarity: str = "cosine",
) -> list[VectorResult]:
    if not query_embedding or not chunks:
        return []

    scored: list[VectorResult] = []
    for chunk in chunks:
        if not chunk.embedding:
            continue

        if similarity == "cosine":
            score = cosine_similarity(query_embedding, chunk.embedding)
        elif similarity == "dot_product":
            score = dot_product(query_embedding, chunk.embedding)
        elif similarity == "l2":
            score = -l2_distance(query_embedding, chunk.embedding)
        else:
            score = cosine_similarity(query_embedding, chunk.embedding)

        passes_threshold = (
            score <= similarity_threshold
            if similarity == "l2"
            else score >= similarity_threshold
        )
        if passes_threshold:
            scored.append(
                VectorResult(
                    chunk=chunk,
                    chunk_id=chunk.id,
                    score=score,
                    method="vector",
                )
            )

    scored.sort(key=lambda x: x.score, reverse=True)
    for i, r in enumerate(scored):
        r.rank = i + 1

    return scored[:top_k]


def _tokenize(text: str, tokenizer: str = "standard") -> list[str]:
    import re
    tokens = re.findall(r"\b\w+\b", text.lower())
    if tokenizer == "whitespace":
        tokens = text.lower().split()
    elif tokenizer == "porter":
        try:
            from nltk.stem import PorterStemmer
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(t) for t in tokens]
        except ImportError:
            pass
    return tokens


def _query_terms(query: str, tokenizer: str = "standard") -> list[str]:
    """Tokenize a search query and drop stopwords, so the terms reported and
    scored for a question are its meaningful content words. Falls back to the
    raw tokens if every token is a stopword (e.g. a query like "what is")."""
    tokens = _tokenize(query, tokenizer)
    meaningful = [t for t in tokens if t not in _QUERY_STOPWORDS]
    return meaningful if meaningful else tokens


def bm25_search(
    query: str,
    chunks: list[StoredChunk],
    top_k: int,
    language: str = "english",
    tokenizer: str = "standard",
) -> list[BM25Result]:
    if not chunks:
        return []

    query_tokens = _query_terms(query, tokenizer)

    # Build corpus - each chunk scores against its full text PLUS its stored
    # metadata keywords, so keyword matches get extra weight (KEYWORD_BOOST).
    corpus = []
    chunk_keywords = []
    for chunk in chunks:
        keywords = list(chunk.metadata.get("keywords") or [])
        keyword_tokens = _tokenize(" ".join(keywords), tokenizer) if keywords else []
        content_tokens = _tokenize(chunk.content, tokenizer)
        corpus.append(content_tokens + keyword_tokens * KEYWORD_BOOST)
        chunk_keywords.append(keywords)

    # Calculate IDF
    n_docs = len(corpus)
    df = {}
    for tokens in corpus:
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1

    idf = {}
    for token, count in df.items():
        idf[token] = math.log((n_docs - count + 0.5) / (count + 0.5)) + 1.0

    # BM25 parameters
    k1 = 1.5
    b = 0.75

    # Calculate avg doc length
    avgdl = sum(len(tokens) for tokens in corpus) / n_docs if n_docs > 0 else 0

    scored: list[BM25Result] = []
    for i, chunk in enumerate(chunks):
        doc_tokens = corpus[i]
        doc_len = len(doc_tokens)
        if doc_len == 0:
            continue

        score = 0.0
        for token in query_tokens:
            if token not in idf:
                continue
            # Count term frequency in doc
            tf = doc_tokens.count(token)
            if tf == 0:
                continue
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
            score += idf[token] * (numerator / denominator)

        matched_keywords = []
        if chunk_keywords[i]:
            query_set = set(query_tokens)
            for kw in chunk_keywords[i]:
                kw_tokens = set(_tokenize(kw, tokenizer))
                if query_set & kw_tokens:
                    matched_keywords.append(kw)

        if score > 0:
            scored.append(
                BM25Result(
                    chunk=chunk,
                    chunk_id=chunk.id,
                    score=score,
                    query_terms=query_tokens,
                    matched_keywords=matched_keywords,
                    method="bm25",
                )
            )

    scored.sort(key=lambda x: x.score, reverse=True)
    for i, r in enumerate(scored):
        r.rank = i + 1

    return scored[:top_k]


def rrf_fusion(
    vector_results: list[VectorResult],
    bm25_results: list[BM25Result],
    k: int = 60,
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[RRFResult]:
    scores: dict[str, float] = {}
    chunk_map: dict[str, StoredChunk] = {}

    for r in vector_results:
        rrf_score = vector_weight / (k + r.rank)
        scores[r.chunk_id] = scores.get(r.chunk_id, 0) + rrf_score
        chunk_map[r.chunk_id] = r.chunk

    for r in bm25_results:
        rrf_score = bm25_weight / (k + r.rank)
        scores[r.chunk_id] = scores.get(r.chunk_id, 0) + rrf_score
        chunk_map[r.chunk_id] = r.chunk

    results = [
        RRFResult(chunk=chunk_map[cid], chunk_id=cid, rrf_score=score, method="rrf")
        for cid, score in scores.items()
    ]
    results.sort(key=lambda x: x.rrf_score, reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1

    return results


async def _cohere_rerank(query: str, documents: list[str], model: str, top_n: int) -> list[tuple[int, float]]:
    async def call() -> httpx.Response:
        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT) as client:
            response = await client.post(
                COHERE_RERANK_URL,
                headers={
                    "Authorization": f"Bearer {settings.COHERE_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                },
            )
            response.raise_for_status()
            return response

    response = await retry_on_rate_limit(call)
    return [(r["index"], r["relevance_score"]) for r in response.json()["results"]]


async def rerank_documents(
    query: str,
    chunks: list[StoredChunk],
    model: Optional[str] = None,
    candidate_count: int = 20,
    top_n: int = 10,
) -> list[RerankResult]:
    candidates = chunks[:candidate_count] if len(chunks) > candidate_count else chunks
    if not candidates:
        return []

    from app.services.storage import get_settings

    persisted = await get_settings()
    provider = persisted.get("rerankerProvider") or "local"

    if provider == "cohere" and settings.COHERE_API_KEY:
        cohere_model = persisted.get("cohereRerankModel") or settings.COHERE_RERANK_MODEL
        ranked = await _cohere_rerank(query, [c.content for c in candidates], cohere_model, top_n)
        scored = [
            RerankResult(
                chunk=candidates[index],
                chunk_id=candidates[index].id,
                rerank_score=float(score),
                method="reranker",
            )
            for index, score in ranked
        ]
        for i, r in enumerate(scored):
            r.rank = i + 1
        return scored

    if provider == "cohere":
        logger.warning(
            "Reranker provider is 'cohere' but COHERE_API_KEY is not set; "
            "falling back to the local cross-encoder instead."
        )

    cross_encoder = _get_cross_encoder(model or "cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(query, chunk.content) for chunk in candidates]
    scores = await asyncio.to_thread(cross_encoder.predict, pairs)

    scored: list[RerankResult] = []
    for chunk, score in zip(candidates, scores):
        scored.append(
            RerankResult(
                chunk=chunk,
                chunk_id=chunk.id,
                rerank_score=float(score),
                method="reranker",
            )
        )

    scored.sort(key=lambda x: x.rerank_score, reverse=True)
    for i, r in enumerate(scored):
        r.rank = i + 1

    return scored[:top_n]


def mmr_selection(
    chunks: list[StoredChunk],
    scores: list[float],
    lambda_: float = 0.7,
    candidate_count: int = 15,
    final_count: int = 8,
) -> list[MMRResult]:
    candidates = chunks[:candidate_count]
    candidate_scores = scores[:candidate_count]

    if not candidates:
        return []

    # Normalize scores
    max_score = max(candidate_scores) if candidate_scores else 1.0
    min_score = min(candidate_scores) if candidate_scores else 0.0
    score_range = max_score - min_score

    normalized_scores = [
        (s - min_score) / score_range if score_range > 0 else 0.0
        for s in candidate_scores
    ]

    selected: list[int] = []
    selected_embeddings = []
    # The actual mmr value (relevance minus diversity penalty) and the
    # max-similarity-to-already-selected that produced it, for whichever
    # candidate wins each round - both are otherwise local to this loop
    # and would be lost instead of ending up on the returned MMRResult.
    mmr_by_idx: dict[int, float] = {}
    max_sim_by_idx: dict[int, float] = {}

    remaining = list(range(len(candidates)))

    while len(selected) < final_count and remaining:
        best_idx = -1
        best_mmr = float("-inf")
        best_max_sim = 0.0

        for idx in remaining:
            relevance = normalized_scores[idx]

            # Max similarity to already selected
            max_sim = 0.0
            if selected_embeddings:
                for sel_emb in selected_embeddings:
                    if candidates[idx].embedding and sel_emb:
                        sim = cosine_similarity(candidates[idx].embedding, sel_emb)
                        max_sim = max(max_sim, sim)

            mmr = lambda_ * relevance - (1 - lambda_) * max_sim

            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = idx
                best_max_sim = max_sim

        if best_idx == -1:
            break

        selected.append(best_idx)
        mmr_by_idx[best_idx] = best_mmr
        max_sim_by_idx[best_idx] = best_max_sim
        if candidates[best_idx].embedding:
            selected_embeddings.append(candidates[best_idx].embedding)
        remaining.remove(best_idx)

    results: list[MMRResult] = []
    for rank, idx in enumerate(selected):
        results.append(
            MMRResult(
                chunk_id=candidates[idx].id,
                chunk=candidates[idx],
                mmr_score=mmr_by_idx[idx],
                relevance_score=normalized_scores[idx],
                max_similarity=max_sim_by_idx[idx],
                selected=True,
                rank=rank + 1,
                method="mmr",
            )
        )

    # Candidates never chosen - reported too (selected=False) so
    # "rejected_count" telemetry downstream reflects reality instead of
    # always reading zero. Scored against the FINAL selected set, since
    # that's the only single well-defined comparison point for a chunk
    # that was never itself added to the growing selected set.
    for idx in remaining:
        relevance = normalized_scores[idx]
        max_sim = 0.0
        for sel_emb in selected_embeddings:
            if candidates[idx].embedding and sel_emb:
                max_sim = max(max_sim, cosine_similarity(candidates[idx].embedding, sel_emb))
        mmr = lambda_ * relevance - (1 - lambda_) * max_sim
        results.append(
            MMRResult(
                chunk_id=candidates[idx].id,
                chunk=candidates[idx],
                mmr_score=mmr,
                relevance_score=relevance,
                max_similarity=max_sim,
                selected=False,
                rank=0,
                method="mmr",
            )
        )

    return results

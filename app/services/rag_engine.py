from datetime import datetime, timezone
from typing import AsyncGenerator, Optional
from uuid import uuid4

from app.core.config import settings
from app.core.logging_config import get_logger
from app.models.schemas import (
    RagEngineConfig,
    RagStrategy,
    FullTrace,
    VectorSearchConfig,
    BM25Config,
    RRFConfig,
    RerankerConfig,
    MMRConfig,
    LLMConfig,
)
from app.services.embeddings import (
    is_embedding_configured,
    generate_query_embedding,
)
from app.services.search import (
    VectorResult,
    BM25Result,
    RRFResult,
    RerankResult,
    MMRResult,
    vector_search,
    bm25_search,
    rrf_fusion,
    rerank_documents,
    mmr_selection,
)
from app.services.llm import (
    is_llm_configured,
    generate_completion_stream,
)
from app.services.storage import (
    get_all_chunks,
    get_all_documents,
)

logger = get_logger(__name__)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0

    return max(1, int(len(text.split()) / 0.75))


def should_run_vector(strategy: RagStrategy) -> bool:
    return strategy in [
        RagStrategy.VECTOR,
        RagStrategy.HYBRID,
        RagStrategy.HYBRID_RRF,
        RagStrategy.HYBRID_RERANK,
        RagStrategy.HYBRID_RERANK_MMR,
    ]


def should_run_bm25(strategy: RagStrategy) -> bool:
    return strategy in [
        RagStrategy.BM25,
        RagStrategy.HYBRID,
        RagStrategy.HYBRID_RRF,
        RagStrategy.HYBRID_RERANK,
        RagStrategy.HYBRID_RERANK_MMR,
    ]


def should_run_rrf(strategy: RagStrategy) -> bool:
    return strategy in [
        RagStrategy.HYBRID_RRF,
        RagStrategy.HYBRID_RERANK_MMR,
    ]


def should_run_reranker(
    strategy: RagStrategy,
    config: RagEngineConfig,
) -> bool:
    return (
        strategy
        in [
            RagStrategy.HYBRID_RERANK,
            RagStrategy.HYBRID_RERANK_MMR,
        ]
        and config.reranker.enabled
    )


def should_run_mmr(
    strategy: RagStrategy,
    config: RagEngineConfig,
) -> bool:
    return (
        strategy == RagStrategy.HYBRID_RERANK_MMR
        and config.mmr.enabled
    )


MAX_CONTEXT_TOKENS = 4000
MAX_CONTEXT_CHUNKS = 5


async def get_default_rag_config(
    strategy: RagStrategy,
) -> RagEngineConfig:
    return RagEngineConfig(
        strategy=strategy,
        vector=VectorSearchConfig(
            embedding_model=settings.EMBEDDING_MODEL,
            top_k=settings.DEFAULT_TOP_K,
            similarity="cosine",
            similarity_threshold=0.0,
        ),
        bm25=BM25Config(
            top_k=settings.DEFAULT_TOP_K,
            language="english",
            tokenizer="standard",
        ),
        rrf=RRFConfig(
            k=60,
            vector_weight=1.0,
            bm25_weight=1.0,
        ),
        reranker=RerankerConfig(
            enabled=False,
            model=settings.RERANKER_MODEL,
            candidate_count=settings.DEFAULT_TOP_K,
            top_n=10,
        ),
        mmr=MMRConfig(
            enabled=True,
            lambda_=settings.MMR_LAMBDA,
            candidate_count=15,
            final_count=8,
        ),
        llm=LLMConfig(
            model=settings.GROQ_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        ),
    )


def _truncate_to_tokens(
    text: str,
    max_tokens: int,
) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text

    words = text.split()
    truncated = []
    count = 0

    for word in words:
        count += 1

        if count / 0.75 > max_tokens:
            break

        truncated.append(word)

    return (
        " ".join(truncated)
        + "\n\n[Context truncated due to token limit]"
    )


def build_prompt(
    query: str,
    context_chunks: list,
    config: RagEngineConfig,
) -> dict:
    system = (
        config.llm_model_prompt
        if hasattr(config, "llm_model_prompt")
        else settings.SYSTEM_PROMPT
    )

    system_tokens = estimate_tokens(system)
    user_tokens = estimate_tokens(query)

    available = max(
        500,
        MAX_CONTEXT_TOKENS - system_tokens - user_tokens,
    )

    limited_chunks = context_chunks[:MAX_CONTEXT_CHUNKS]

    context_parts = []

    for i, chunk in enumerate(limited_chunks):
        header = (
            f"[Source {i + 1}] "
            f"Document: {chunk.document_name}, "
            f"Page: {chunk.page}"
        )

        if chunk.section:
            header += f", Section: {chunk.section}"

        context_parts.append(
            f"{header}\n{chunk.content}"
        )

    context = "\n\n".join(context_parts)

    context = _truncate_to_tokens(
        context,
        available,
    )

    context_tokens = estimate_tokens(context)

    return {
        "system": system,
        "context": context,
        "user": query,
        "system_tokens": system_tokens,
        "context_tokens": context_tokens,
        "user_tokens": user_tokens,
        "total_tokens": (
            system_tokens
            + context_tokens
            + user_tokens
        ),
    }


async def execute_rag(
    query: str,
    strategy: Optional[RagStrategy] = None,
    config_overrides: Optional[dict] = None,
    knowledge_base_id: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    trace_id = str(uuid4())
    run_id = str(uuid4())
    request_id = str(uuid4())

    start_time = datetime.now(timezone.utc)

    # ---------------------------------------------------------
    # Build configuration
    # ---------------------------------------------------------

    settings_strategy = (
        strategy or settings.DEFAULT_STRATEGY
    )

    try:
        config = await get_default_rag_config(
            settings_strategy
        )
    except Exception:
        config = await get_default_rag_config(
            RagStrategy.HYBRID_RRF
        )

    if config_overrides:
        config = RagEngineConfig(
            **{
                **config.model_dump(),
                **config_overrides,
            }
        )

    config.strategy = settings_strategy

    events: list[dict] = []

    def emit(
        stage: str,
        event: str,
        data: dict,
    ) -> dict:
        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        trace_event = {
            "stage": stage,
            "event": event,
            "data": data,
            "timestamp": timestamp,
        }

        events.append(trace_event)

        return {
            "type": event,
            "stage": stage,
            "data": data,
            "timestamp": timestamp,
        }

    # ---------------------------------------------------------
    # Query started
    # ---------------------------------------------------------

    yield emit(
        "query",
        "query.started",
        {
            "query": query,
            "strategy": config.strategy.value,
            "request_id": request_id,
            "knowledge_base_id": knowledge_base_id,
        },
    )

    # ---------------------------------------------------------
    # Load chunks
    # ---------------------------------------------------------

    try:
        if knowledge_base_id:
            all_docs = await get_all_documents()

            kb_doc_ids = {
                d.id
                for d in all_docs
                if getattr(
                    d,
                    "knowledge_base_id",
                    None,
                )
                == knowledge_base_id
            }

            all_chunks_raw = await get_all_chunks()

            all_chunks = [
                c
                for c in all_chunks_raw
                if c.document_id in kb_doc_ids
            ]
        else:
            all_chunks = await get_all_chunks()

    except Exception as e:
        error_msg = f"Failed to load chunks: {e}"

        yield emit(
            "query",
            "trace.failed",
            {
                "error": error_msg,
            },
        )

        yield {
            "type": "error",
            "error": error_msg,
        }

        return

    if not all_chunks:
        error_msg = (
            "No document chunks found. "
            "Please upload and process documents first."
        )

        yield emit(
            "query",
            "trace.failed",
            {
                "error": error_msg,
            },
        )

        yield {
            "type": "error",
            "error": error_msg,
        }

        return

    yield emit(
        "query",
        "query.processed",
        {
            "chunk_count": len(all_chunks),
        },
    )

    # ---------------------------------------------------------
    # Stage variables
    # ---------------------------------------------------------

    vector_results: list[VectorResult] = []
    bm25_results: list[BM25Result] = []
    fused_results: list[RRFResult] = []
    rerank_results: list[RerankResult] = []
    mmr_results: list[MMRResult] = []

    final_chunks: list = []

    answer = ""

    llm_input_tokens: Optional[int] = None
    llm_output_tokens: Optional[int] = None
    llm_total_tokens: Optional[int] = None

    llm_latency_ms = 0
    llm_status = "streaming"

    # ---------------------------------------------------------
    # Vector Search
    # ---------------------------------------------------------

    if should_run_vector(config.strategy):
        yield emit(
            "vector",
            "vector.started",
            {
                "top_k": config.vector.top_k,
                "embedding_model": (
                    config.vector.embedding_model
                ),
            },
        )

        embedding_configured = (
            await is_embedding_configured()
        )

        if embedding_configured:
            try:
                query_embedding = (
                    await generate_query_embedding(
                        query
                    )
                )

                if query_embedding:
                    vector_results = vector_search(
                        query_embedding,
                        all_chunks,
                        config.vector.top_k,
                        config.vector.similarity_threshold,
                        config.vector.similarity,
                    )

                    for r in vector_results:
                        yield emit(
                            "vector",
                            "vector.chunk.retrieved",
                            {
                                "chunk_id": r.chunk_id,
                                "rank": r.rank,
                                "score": r.score,
                                "document": (
                                    r.chunk.document_name
                                ),
                                "page": r.chunk.page,
                            },
                        )

                    yield emit(
                        "vector",
                        "vector.completed",
                        {
                            "result_count": len(
                                vector_results
                            )
                        },
                    )

            except Exception as e:
                logger.error(
                    f"Vector search failed: {e}"
                )

                yield emit(
                    "vector",
                    "vector.completed",
                    {
                        "result_count": 0,
                        "error": str(e),
                    },
                )
        else:
            yield emit(
                "vector",
                "vector.completed",
                {
                    "result_count": 0,
                    "skipped": (
                        "Embeddings not configured"
                    ),
                },
            )

    # ---------------------------------------------------------
    # BM25 Search
    # ---------------------------------------------------------

    if should_run_bm25(config.strategy):
        yield emit(
            "bm25",
            "bm25.started",
            {
                "top_k": config.bm25.top_k,
                "language": config.bm25.language,
            },
        )

        bm25_results = bm25_search(
            query,
            all_chunks,
            config.bm25.top_k,
            config.bm25.language,
            config.bm25.tokenizer,
        )

        for r in bm25_results:
            yield emit(
                "bm25",
                "bm25.chunk.retrieved",
                {
                    "chunk_id": r.chunk_id,
                    "rank": r.rank,
                    "score": r.score,
                    "document": (
                        r.chunk.document_name
                    ),
                    "page": r.chunk.page,
                    "query_terms": (
                        r.query_terms[:5]
                    ),
                },
            )

        yield emit(
            "bm25",
            "bm25.completed",
            {
                "result_count": len(
                    bm25_results
                )
            },
        )

    # ---------------------------------------------------------
    # RRF Fusion
    # ---------------------------------------------------------

    if should_run_rrf(config.strategy):
        yield emit(
            "rrf",
            "rrf.started",
            {
                "k": config.rrf.k,
            },
        )

        fused_results = rrf_fusion(
            vector_results,
            bm25_results,
            config.rrf.k,
            config.rrf.vector_weight,
            config.rrf.bm25_weight,
        )

        for r in fused_results:
            yield emit(
                "rrf",
                "rrf.result.retrieved",
                {
                    "chunk_id": r.chunk_id,
                    "rank": r.rank,
                    "score": r.rrf_score,
                },
            )

        yield emit(
            "rrf",
            "rrf.completed",
            {
                "result_count": len(
                    fused_results
                ),
            },
        )

    # ---------------------------------------------------------
    # Reranker
    # ---------------------------------------------------------

    if (
        should_run_reranker(
            config.strategy,
            config,
        )
        and fused_results
    ):
        yield emit(
            "reranker",
            "reranker.started",
            {
                "model": config.reranker.model,
            },
        )

        rerank_chunks = [
            r.chunk
            for r in fused_results[
                : config.reranker.candidate_count
            ]
        ]

        rerank_results = await rerank_documents(
            query,
            rerank_chunks,
            config.reranker.model,
            config.reranker.candidate_count,
            config.reranker.top_n,
        )

        for r in rerank_results:
            yield emit(
                "rerank",
                "rerank.score.updated",
                {
                    "chunk_id": r.chunk_id,
                    "rank": r.rank,
                    "score": r.rerank_score,
                },
            )

        yield emit(
            "reranker",
            "reranker.completed",
            {
                "result_count": len(
                    rerank_results
                ),
            },
        )

    # ---------------------------------------------------------
    # MMR
    # ---------------------------------------------------------

    if (
        should_run_mmr(
            config.strategy,
            config,
        )
        and rerank_results
    ):
        yield emit(
            "mmr",
            "mmr.started",
            {
                "lambda": config.mmr.lambda_,
            },
        )

        mmr_results = mmr_selection(
            [r.chunk for r in rerank_results],
            [r.rerank_score for r in rerank_results],
            config.mmr.lambda_,
            config.mmr.candidate_count,
            config.mmr.final_count,
        )

        selected = [
            r
            for r in mmr_results
            if r.selected
        ]

        for r in selected:
            yield emit(
                "mmr",
                "mmr.selection.updated",
                {
                    "chunk_id": r.chunk_id,
                    "rank": r.rank,
                    "mmr_score": r.mmr_score,
                    "relevance_score": (
                        r.relevance_score
                    ),
                    "selected": True,
                },
            )

        yield emit(
            "mmr",
            "mmr.completed",
            {
                "selected_count": len(selected),
                "rejected_count": (
                    len(mmr_results)
                    - len(selected)
                ),
            },
        )

    # ---------------------------------------------------------
    # Build final context
    # ---------------------------------------------------------

    final_chunks = _build_final_context(
        config.strategy,
        vector_results,
        bm25_results,
        fused_results,
        rerank_results,
        mmr_results,
        config,
    )

    yield emit(
        "context",
        "context.built",
        {
            "chunk_count": len(final_chunks),
            "document_count": len(
                {
                    c.document_id
                    for c in final_chunks
                }
            ),
            "total_tokens": sum(
                c.token_count
                for c in final_chunks
            ),
        },
    )

    # ---------------------------------------------------------
    # Build prompt
    # ---------------------------------------------------------

    prompt = build_prompt(
        query,
        final_chunks,
        config,
    )

    yield emit(
        "prompt",
        "prompt.built",
        {
            "system_tokens": (
                prompt["system_tokens"]
            ),
            "context_tokens": (
                prompt["context_tokens"]
            ),
            "user_tokens": (
                prompt["user_tokens"]
            ),
            "total_tokens": (
                prompt["total_tokens"]
            ),
        },
    )

    # ---------------------------------------------------------
    # LLM Configuration Check
    # ---------------------------------------------------------

    if not await is_llm_configured():
        error_msg = (
            "LLM API key not configured. "
            "Cannot generate LLM response."
        )

        yield emit(
            "llm",
            "trace.failed",
            {
                "error": error_msg,
            },
        )

        yield {
            "type": "error",
            "error": error_msg,
        }

        await _finalize_trace(
            llm_status,
            trace_id,
            run_id,
            request_id,
            query,
            config,
            events,
            vector_results,
            bm25_results,
            fused_results,
            rerank_results,
            mmr_results,
            final_chunks,
            prompt,
            answer,
            llm_input_tokens,
            llm_output_tokens,
            llm_total_tokens,
            llm_latency_ms,
            "failed",
            error_msg,
            knowledge_base_id,
            start_time,
        )

        return

    # ---------------------------------------------------------
    # LLM Generation
    # ---------------------------------------------------------

    yield emit(
        "llm",
        "llm.started",
        {
            "model": config.llm.model,
            "temperature": config.llm.temperature,
            "max_tokens": config.llm.max_tokens,
        },
    )

    llm_start = datetime.now(timezone.utc)

    try:
        async for chunk in generate_completion_stream(
            prompt["system"]
            + "\n\n--- Context ---\n\n"
            + prompt["context"],
            prompt["user"],
            config.llm.model,
            config.llm.temperature,
            config.llm.max_tokens,
        ):
            if (
                chunk["type"] == "token"
                and chunk.get("content")
            ):
                answer += chunk["content"]

                yield {
                    "type": "llm.token",
                    "content": chunk["content"],
                }

            elif chunk["type"] == "done":
                tokens = chunk.get(
                    "tokens",
                    {},
                )

                llm_input_tokens = tokens.get(
                    "input"
                )
                llm_output_tokens = tokens.get(
                    "output"
                )
                llm_total_tokens = tokens.get(
                    "total"
                )

                llm_status = "completed"

            elif chunk["type"] == "error":
                llm_status = "failed"

                error_msg = chunk.get(
                    "error",
                    "LLM generation failed",
                )

                yield emit(
                    "llm",
                    "trace.failed",
                    {
                        "error": error_msg,
                    },
                )

                yield {
                    "type": "error",
                    "error": error_msg,
                }

                llm_latency_ms = int(
                    (
                        datetime.now(timezone.utc)
                        - llm_start
                    ).total_seconds()
                    * 1000
                )

                yield emit(
                    "llm",
                    "llm.completed",
                    {
                        "model": config.llm.model,
                        "latency_ms": (
                            llm_latency_ms
                        ),
                        "status": "failed",
                        "error": error_msg,
                    },
                )

                await _finalize_trace(
                    llm_status,
                    trace_id,
                    run_id,
                    request_id,
                    query,
                    config,
                    events,
                    vector_results,
                    bm25_results,
                    fused_results,
                    rerank_results,
                    mmr_results,
                    final_chunks,
                    prompt,
                    answer,
                    llm_input_tokens,
                    llm_output_tokens,
                    llm_total_tokens,
                    llm_latency_ms,
                    "failed",
                    error_msg,
                    knowledge_base_id,
                    start_time,
                )

                return

        llm_latency_ms = int(
            (
                datetime.now(timezone.utc)
                - llm_start
            ).total_seconds()
            * 1000
        )

        yield {
            "type": "llm.done",
            "data": {
                "model": config.llm.model,
                "latency_ms": llm_latency_ms,
            },
        }

        yield emit(
            "llm",
            "llm.completed",
            {
                "model": config.llm.model,
                "latency_ms": llm_latency_ms,
                "status": "completed",
            },
        )

    except Exception as e:
        llm_latency_ms = int(
            (
                datetime.now(timezone.utc)
                - llm_start
            ).total_seconds()
            * 1000
        )

        llm_status = "failed"

        error_msg = (
            f"LLM streaming error: {e}"
        )

        yield emit(
            "llm",
            "trace.failed",
            {
                "error": error_msg,
            },
        )

        yield {
            "type": "error",
            "error": error_msg,
        }

        yield emit(
            "llm",
            "llm.completed",
            {
                "model": config.llm.model,
                "latency_ms": llm_latency_ms,
                "status": "failed",
                "error": error_msg,
            },
        )

        await _finalize_trace(
            llm_status,
            trace_id,
            run_id,
            request_id,
            query,
            config,
            events,
            vector_results,
            bm25_results,
            fused_results,
            rerank_results,
            mmr_results,
            final_chunks,
            prompt,
            answer,
            llm_input_tokens,
            llm_output_tokens,
            llm_total_tokens,
            llm_latency_ms,
            "failed",
            error_msg,
            knowledge_base_id,
            start_time,
        )

        return

    # ---------------------------------------------------------
    # Trace Completed
    # ---------------------------------------------------------

    trace = await _finalize_trace(
        llm_status,
        trace_id,
        run_id,
        request_id,
        query,
        config,
        events,
        vector_results,
        bm25_results,
        fused_results,
        rerank_results,
        mmr_results,
        final_chunks,
        prompt,
        answer,
        llm_input_tokens,
        llm_output_tokens,
        llm_total_tokens,
        llm_latency_ms,
        "completed",
        None,
        knowledge_base_id,
        start_time,
    )

    yield emit(
        "trace",
        "trace.completed",
        trace.model_dump(),
    )


def _build_final_context(
    strategy: RagStrategy,
    vector_results: list[VectorResult],
    bm25_results: list[BM25Result],
    fused_results: list[RRFResult],
    rerank_results: list[RerankResult],
    mmr_results: list[MMRResult],
    config: RagEngineConfig,
) -> list:
    chunks = []

    if (
        should_run_mmr(
            strategy,
            config,
        )
        and mmr_results
    ):
        selected = [
            r
            for r in mmr_results
            if r.selected
        ]

        for r in selected:
            chunks.append(r.chunk)

        return chunks

    if (
        should_run_reranker(
            strategy,
            config,
        )
        and rerank_results
    ):
        return [
            r.chunk
            for r in rerank_results
        ]

    if (
        should_run_rrf(strategy)
        and fused_results
    ):
        return [
            r.chunk
            for r in fused_results
        ]

    if (
        should_run_vector(strategy)
        and should_run_bm25(strategy)
    ):
        seen = set()

        for r in vector_results:
            if r.chunk_id not in seen:
                seen.add(r.chunk_id)
                chunks.append(r.chunk)

        for r in bm25_results:
            if r.chunk_id not in seen:
                seen.add(r.chunk_id)
                chunks.append(r.chunk)

        return chunks

    if should_run_vector(strategy):
        return [
            r.chunk
            for r in vector_results
        ]

    if should_run_bm25(strategy):
        return [
            r.chunk
            for r in bm25_results
        ]

    return chunks


async def _finalize_trace(
    llm_status: str,
    trace_id: str,
    run_id: str,
    request_id: str,
    query: str,
    config: RagEngineConfig,
    events: list[dict],
    vector_results: list[VectorResult],
    bm25_results: list[BM25Result],
    fused_results: list[RRFResult],
    rerank_results: list[RerankResult],
    mmr_results: list[MMRResult],
    final_chunks: list,
    prompt: dict,
    answer: str,
    llm_input_tokens: Optional[int],
    llm_output_tokens: Optional[int],
    llm_total_tokens: Optional[int],
    llm_latency_ms: int,
    overall_status: str,
    error: Optional[str],
    knowledge_base_id: Optional[str],
    start_time: datetime,
) -> FullTrace:

    # ---------------------------------------------------------
    # Total latency
    # ---------------------------------------------------------

    total_latency_ms = int(
        (
            datetime.now(timezone.utc)
            - start_time
        ).total_seconds()
        * 1000
    )

    # ---------------------------------------------------------
    # Context data
    # ---------------------------------------------------------

    context_data = {
        "chunks": [
            {
                "chunk_id": c.id,
                "document_id": c.document_id,
                "document_name": c.document_name,
                "page": c.page,
                "section": c.section,
                "content": c.content,
                "token_count": c.token_count,
                "score": 0.0,
                "method": "final",
            }
            for c in final_chunks
        ],
        "total_tokens": sum(
            c.token_count
            for c in final_chunks
        ),
        "chunk_count": len(final_chunks),
        "document_count": len(
            {
                c.document_id
                for c in final_chunks
            }
        ),
    }

    # ---------------------------------------------------------
    # LLM data
    # ---------------------------------------------------------

    llm_data = {
        "provider": settings.LLM_PROVIDER,
        "model": config.llm.model,
        "latency_ms": llm_latency_ms,
        "input_tokens": llm_input_tokens,
        "output_tokens": llm_output_tokens,
        "total_tokens": llm_total_tokens,
        "answer": answer,
        "status": llm_status,
    }

    if error:
        llm_data["error"] = error

    # ---------------------------------------------------------
    # Create trace
    # ---------------------------------------------------------

    trace = FullTrace(
        id=trace_id,
        run_id=run_id,
        session_id=request_id,
        request_id=request_id,
        timestamp=datetime.now(timezone.utc),
        query=query,
        strategy=config.strategy,
        config=config,
        query_processing={
            "original_query": query,
            "token_count": estimate_tokens(query),
            "processing_duration_ms": max(
                0,
                total_latency_ms
                - llm_latency_ms,
            ),
        },
        context=context_data,
        prompt=prompt,
        llm=llm_data,
        sources=[
            {
                "document_id": c.document_id,
                "document_name": c.document_name,
                "page": c.page,
                "section": c.section,
                "chunk_id": c.id,
                "retrieval_method": "final",
                "score": 0.0,
            }
            for c in final_chunks
        ],
        total_latency_ms=total_latency_ms,
        status=overall_status,
    )

    if error:
        trace.error = error

    # ---------------------------------------------------------
    # Optional Vector Search trace
    # ---------------------------------------------------------

    if vector_results:
        trace.vector_search = {
            "latency_ms": 0,
            "chunk_count": len(
                vector_results
            ),
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "document_name": (
                        r.chunk.document_name
                    ),
                    "page": r.chunk.page,
                    "score": r.score,
                    "rank": r.rank,
                }
                for r in vector_results
            ],
        }

    # ---------------------------------------------------------
    # Optional BM25 trace
    # ---------------------------------------------------------

    if bm25_results:
        trace.bm25 = {
            "latency_ms": 0,
            "chunk_count": len(
                bm25_results
            ),
            "query_terms": (
                bm25_results[0].query_terms
                if bm25_results
                else []
            ),
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "document_name": (
                        r.chunk.document_name
                    ),
                    "page": r.chunk.page,
                    "score": r.score,
                    "rank": r.rank,
                }
                for r in bm25_results
            ],
        }

    # ---------------------------------------------------------
    # Optional RRF trace
    # ---------------------------------------------------------

    if fused_results:
        trace.rrf = {
            "latency_ms": 0,
            "input_vector_chunks": len(
                vector_results
            ),
            "input_bm25_chunks": len(
                bm25_results
            ),
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "document_name": (
                        r.chunk.document_name
                    ),
                    "page": r.chunk.page,
                    "score": r.rrf_score,
                    "rank": r.rank,
                }
                for r in fused_results
            ],
        }

    # ---------------------------------------------------------
    # Optional Reranker trace
    # ---------------------------------------------------------

    if rerank_results:
        trace.reranker = {
            "latency_ms": 0,
            "model": config.reranker.model,
            "input_chunk_count": len(
                rerank_results
            ),
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "document_name": (
                        r.chunk.document_name
                    ),
                    "page": r.chunk.page,
                    "score": r.rerank_score,
                    "rank": r.rank,
                }
                for r in rerank_results
            ],
        }

    # ---------------------------------------------------------
    # Optional MMR trace
    # ---------------------------------------------------------

    if mmr_results:
        selected = [
            r
            for r in mmr_results
            if r.selected
        ]

        trace.mmr = {
            "latency_ms": 0,
            "selected_count": len(selected),
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "document_name": (
                        r.chunk.document_name
                    ),
                    "page": r.chunk.page,
                    "score": r.mmr_score,
                    "rank": r.rank,
                    "selected": r.selected,
                }
                for r in mmr_results
            ],
        }

    # ---------------------------------------------------------
    # Persist COMPLETE trace
    # ---------------------------------------------------------

    try:
        from app.services.storage import add_trace

        await add_trace(
            trace.model_dump()
        )

    except Exception as e:
        logger.error(
            f"Failed to persist trace: {e}"
        )

    return trace
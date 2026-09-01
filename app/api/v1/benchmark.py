import math

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from uuid import uuid4

from app.services.storage import get_all_datasets

router = APIRouter()

_ALL_METRIC_KEYS = [
    "hitRate", "recall", "precision", "mrr", "ndcg",
    "latencyMs", "inputTokens", "outputTokens", "totalTokens", "cost",
]

_ALL_FAILURE_CATEGORIES = [
    "retrieval_failure", "missing_source", "wrong_source", "poor_ranking",
    "poor_context", "poor_answer", "citation_failure", "latency_failure",
    "token_limit_failure", "llm_failure", "prompt_issue",
]


_DOC_EXTENSIONS = (".pdf", ".md", ".txt", ".docx", ".doc")


def _source_keys(source: dict) -> set[str]:
    """Identify a source by every signal available: chunk id (when it
    actually corresponds to this backend's own ids) and normalized
    document+page. Golden datasets are often authored independently of any
    particular backend and carry synthetic/placeholder chunk ids (e.g.
    "P1-C1") that will never match real generated ids, and document names
    may or may not include the file extension - so a source counts as the
    same source if ANY identity signal overlaps, not just chunk id."""
    keys: set[str] = set()
    chunk_id = source.get("chunkId") or source.get("chunk_id")
    if chunk_id:
        keys.add(f"chunk:{chunk_id}")
    document = str(source.get("document") or "").strip().lower()
    for ext in _DOC_EXTENSIONS:
        if document.endswith(ext):
            document = document[: -len(ext)]
            break
    page = source.get("page")
    if document:
        keys.add(f"doc:{document}|{page}")
    return keys


def _is_match(a: set[str], b: set[str]) -> bool:
    return bool(a & b)


def _hit_rate(retrieved: list[set[str]], expected: list[set[str]]) -> float:
    if not expected:
        return 0.0
    return 1.0 if any(_is_match(r, e) for r in retrieved for e in expected) else 0.0


def _recall(retrieved: list[set[str]], expected: list[set[str]]) -> float:
    if not expected:
        return 0.0
    matched = sum(1 for e in expected if any(_is_match(r, e) for r in retrieved))
    return matched / len(expected)


def _precision(retrieved: list[set[str]], expected: list[set[str]]) -> float:
    if not retrieved:
        return 0.0
    matched = sum(1 for r in retrieved if any(_is_match(r, e) for e in expected))
    return matched / len(retrieved)


def _mrr(retrieved: list[set[str]], expected: list[set[str]]) -> float:
    for i, r in enumerate(retrieved):
        if any(_is_match(r, e) for e in expected):
            return 1.0 / (i + 1)
    return 0.0


def _ndcg(retrieved: list[set[str]], expected: list[set[str]], k: int = 10) -> float:
    if not expected or not retrieved:
        return 0.0

    def dcg(items: list[set[str]]) -> float:
        score = 0.0
        for i, item in enumerate(items[:k]):
            rel = 1 if any(_is_match(item, e) for e in expected) else 0
            score += (2 ** rel - 1) / math.log2(i + 2)
        return score

    ideal_count = min(len(expected), k)
    idcg = sum((2 ** 1 - 1) / math.log2(i + 2) for i in range(ideal_count))
    if idcg == 0:
        return 0.0
    return dcg(retrieved) / idcg


_REFUSAL_PHRASES = (
    "could not find", "couldn't find", "cannot find", "can't find",
    "do not have enough information", "don't have enough information",
    "no relevant information", "not contain", "does not contain",
    "unable to answer", "i don't know", "i do not know",
    "no information available", "not able to find",
)


def _looks_like_refusal(answer: str) -> bool:
    """A refusal-shaped answer despite the expected source having been
    retrieved points at the LLM/prompt, not retrieval - the model had the
    right context in front of it and still didn't use it."""
    lowered = (answer or "").strip().lower()
    if not lowered:
        return True
    return any(phrase in lowered for phrase in _REFUSAL_PHRASES)


_METRIC_DECIMALS = 4


def _round_metric(value: float) -> float:
    return round(value, _METRIC_DECIMALS)


def _average_metrics(metric_dicts: list[dict]) -> dict:
    if not metric_dicts:
        return {key: 0.0 for key in _ALL_METRIC_KEYS}
    return {
        key: _round_metric(sum(m.get(key, 0.0) for m in metric_dicts) / len(metric_dicts))
        for key in _ALL_METRIC_KEYS
    }


@router.get("/runs")
async def list_runs() -> dict:
    """List persisted benchmark runs, most recent first."""
    from app.services.storage import list_benchmark_results

    results = await list_benchmark_results()
    results = sorted(results, key=lambda r: r.get("startedAt", ""), reverse=True)
    return {"success": True, "data": results}


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    from app.services.storage import get_benchmark_result

    run = await get_benchmark_result(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Benchmark run not found")
    return {"success": True, "data": run}


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str) -> dict:
    """Permanently delete one persisted benchmark run. Only removes the
    matching entry from the results store - every other run is untouched."""
    from app.services.storage import delete_benchmark_result

    deleted = await delete_benchmark_result(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Benchmark run not found")
    return {"success": True}


class BenchmarkRunRequest(BaseModel):
    datasetId: str
    strategy: str = "hybrid-rerank-mmr"
    ragConfig: Optional[dict] = None


_EMPTY_METRICS = {
    "hitRate": 0,
    "recall": 0,
    "precision": 0,
    "mrr": 0,
    "ndcg": 0,
    "faithfulness": 0,
    "answerRelevance": 0,
    "contextPrecision": 0,
    "contextRecall": 0,
    "latencyMs": 0,
    "inputTokens": 0,
    "outputTokens": 0,
    "totalTokens": 0,
    "cost": 0,
}


async def _execute_case(test_case, strategy, rag_config) -> tuple[dict, Optional[dict]]:
    """Run a single golden case through the RAG pipeline and score it.

    Returns (case_result, breakdown_metrics) where breakdown_metrics is None
    when the case has no expected sources and should be excluded from the
    difficulty/tag aggregate breakdowns.
    """
    from datetime import datetime, timezone
    from app.models.schemas import RagStrategy
    from app.services.rag_engine import execute_rag
    from app.services.storage import get_settings

    case_started = datetime.now(timezone.utc)
    persisted_settings = await get_settings()
    cost_per_token = persisted_settings.get("costPerToken")
    if not isinstance(cost_per_token, (int, float)):
        cost_per_token = 0.0

    # The benchmark UI's config has no "enabled" toggle for reranker/mmr -
    # it expects strategy alone to decide which stages run. Since
    # execute_rag() does a wholesale replace of the reranker/mmr sub-config
    # (wiping any enabled flag back to its pydantic default of False),
    # derive and inject the correct flag here so the configured
    # reranker/mmr settings actually take effect for this strategy.
    effective_rag_config = dict(rag_config) if rag_config else {}
    if isinstance(effective_rag_config.get("reranker"), dict):
        effective_rag_config["reranker"] = {
            **effective_rag_config["reranker"],
            "enabled": strategy in (RagStrategy.HYBRID_RERANK, RagStrategy.HYBRID_RERANK_MMR),
        }
    if isinstance(effective_rag_config.get("mmr"), dict):
        effective_rag_config["mmr"] = {
            **effective_rag_config["mmr"],
            "enabled": strategy == RagStrategy.HYBRID_RERANK_MMR,
        }

    try:
        answer = ""
        trace = None
        pipeline_error = None
        async for event in execute_rag(
            test_case.query, strategy, effective_rag_config, None
        ):
            if event.get("type") == "llm.token" and "content" in event:
                answer += event["content"]
            if event.get("type") == "trace.completed":
                trace = event.get("data")
            if event.get("type") in ("error", "trace.failed"):
                pipeline_error = event.get("data", {}).get("error") or event.get("error")
        actual_sources = []
        if trace:
            actual_sources = [
                {
                    "document": source.get("document_name", ""),
                    "page": source.get("page", 0),
                    "section": source.get("section", ""),
                    "chunkId": source.get("chunk_id", ""),
                    "score": source.get("score", 0),
                }
                for source in trace.get("sources", [])
            ]

        expected_sources = test_case.expected_sources or []
        expected_keys = [_source_keys(s) for s in expected_sources]
        retrieved_keys = [_source_keys(s) for s in actual_sources]

        # System metrics the pipeline already computes but previously never
        # surfaced here: real end-to-end latency and real LLM output token
        # count come straight from the trace. Input tokens aren't returned
        # by the streaming providers used here, so we fall back to the same
        # word-count estimate the pipeline itself uses for prompt building
        # (trace["prompt"]["total_tokens"]) rather than leaving it at 0.
        # There's no per-model pricing table in this codebase, so cost is
        # only ever an approximation: total tokens * the user's own
        # approximate cost-per-token setting (Settings > Cost), defaulting
        # to 0 if they haven't set one - never a fabricated real price.
        llm_data = (trace or {}).get("llm") or {}
        output_tokens = llm_data.get("output_tokens") or 0
        input_tokens = llm_data.get("input_tokens")
        if input_tokens is None:
            input_tokens = (trace or {}).get("prompt", {}).get("total_tokens") or 0
        real_latency_ms = (trace or {}).get("total_latency_ms")
        if real_latency_ms is None:
            real_latency_ms = int((datetime.now(timezone.utc) - case_started).total_seconds() * 1000)
        total_tokens = input_tokens + output_tokens
        system_metrics = {
            "latencyMs": real_latency_ms,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": total_tokens,
            "cost": _round_metric(total_tokens * cost_per_token),
        }

        breakdown_metrics = None
        if not expected_keys:
            status = "not_run"
            failure_categories: list[str] = []
            failure_explanation = ""
            metrics = {**_EMPTY_METRICS, **system_metrics}
        else:
            metrics = {
                **_EMPTY_METRICS,
                **system_metrics,
                "hitRate": _round_metric(_hit_rate(retrieved_keys, expected_keys)),
                "recall": _round_metric(_recall(retrieved_keys, expected_keys)),
                "precision": _round_metric(_precision(retrieved_keys, expected_keys)),
                "mrr": _round_metric(_mrr(retrieved_keys, expected_keys)),
                "ndcg": _round_metric(_ndcg(retrieved_keys, expected_keys)),
            }
            if pipeline_error:
                status = "failed"
                failure_categories = ["llm_failure"]
                failure_explanation = (
                    f"LLM failure: the pipeline could not generate an answer ({pipeline_error}). "
                    "This is an LLM/model configuration issue, not a retrieval problem."
                )
            elif metrics["hitRate"] == 0:
                status = "failed"
                if actual_sources:
                    failure_categories = ["missing_source"]
                    failure_explanation = "Retrieved sources did not include any expected source."
                else:
                    failure_categories = ["retrieval_failure"]
                    failure_explanation = "No sources were retrieved."
            elif _looks_like_refusal(answer):
                status = "failed"
                failure_categories = ["prompt_issue"]
                failure_explanation = (
                    "Retrieval succeeded (the expected source was found), but the model's answer "
                    "indicates it could not use the retrieved context. This is likely an LLM/system "
                    "prompt issue, not a retrieval problem - consider reviewing the prompt template "
                    "or the model/config being used."
                )
            elif metrics["recall"] >= 0.999:
                status = "passed"
                failure_categories = []
                failure_explanation = ""
            else:
                status = "partial"
                failure_categories = ["poor_ranking"]
                failure_explanation = "Some expected sources were not retrieved."

            breakdown_metrics = metrics

        case_result = {
            "caseId": test_case.id,
            "status": status,
            "query": test_case.query,
            "difficulty": test_case.difficulty or "medium",
            "expectedAnswer": test_case.expectedAnswer or "",
            "expectedSources": expected_sources,
            "actualAnswer": answer,
            "actualSources": actual_sources,
            "metrics": metrics,
            "failureCategories": failure_categories,
            "failureExplanation": failure_explanation,
            "latencyMs": int((datetime.now(timezone.utc) - case_started).total_seconds() * 1000),
            "durationMs": int((datetime.now(timezone.utc) - case_started).total_seconds() * 1000),
            "tokenCount": output_tokens,
            "actualPages": sorted({source["page"] for source in actual_sources}),
            "traceId": trace.get("id", "") if trace else "",
            "runId": "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return case_result, breakdown_metrics
    except Exception as e:
        metrics = {**_EMPTY_METRICS}
        case_result = {
            "caseId": getattr(test_case, "id", "unknown"),
            "status": "failed",
            "query": getattr(test_case, "query", ""),
            "difficulty": getattr(test_case, "difficulty", None) or "medium",
            "expectedAnswer": getattr(test_case, "expectedAnswer", None) or "",
            "expectedSources": getattr(test_case, "expected_sources", None) or [],
            "actualAnswer": "",
            "actualSources": [],
            "metrics": metrics,
            "failureCategories": ["retrieval_failure"],
            "failureExplanation": str(e),
            "latencyMs": int((datetime.now(timezone.utc) - case_started).total_seconds() * 1000),
            "durationMs": int((datetime.now(timezone.utc) - case_started).total_seconds() * 1000),
            "tokenCount": 0,
            "actualPages": [],
            "traceId": "",
            "runId": "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        }
        breakdown_metrics = metrics if getattr(test_case, "expected_sources", None) else None
        return case_result, breakdown_metrics


def _finalize_run(payload: "BenchmarkRunRequest", dataset, started_at: str, case_results: list[dict], by_difficulty: dict, by_tag: dict) -> dict:
    from datetime import datetime, timezone

    evaluated_metrics = [r["metrics"] for r in case_results if r["status"] != "not_run"]
    aggregate_metrics = _average_metrics(evaluated_metrics)
    difficulty_breakdown = {level: _average_metrics(vals) for level, vals in by_difficulty.items()}
    tag_breakdown = {tag: _average_metrics(vals) for tag, vals in by_tag.items()}

    failure_categories_summary = {cat: 0 for cat in _ALL_FAILURE_CATEGORIES}
    for r in case_results:
        for cat in r["failureCategories"]:
            if cat in failure_categories_summary:
                failure_categories_summary[cat] += 1

    return {
        "id": str(uuid4()),
        "datasetId": payload.datasetId,
        "strategy": payload.strategy,
        "startedAt": started_at,
        "completedAt": datetime.now(timezone.utc).isoformat(),
        "totalTests": len(case_results),
        "completedTests": len(case_results),
        "passedTests": sum(1 for r in case_results if r["status"] == "passed"),
        "partialTests": sum(1 for r in case_results if r["status"] == "partial"),
        "failedTests": sum(1 for r in case_results if r["status"] == "failed"),
        "status": "completed",
        "datasetName": dataset.name,
        "datasetVersion": dataset.current_version,
        "config": payload.ragConfig or {},
        "aggregateMetrics": aggregate_metrics,
        "difficultyBreakdown": difficulty_breakdown,
        "tagBreakdown": tag_breakdown,
        "failureCategories": failure_categories_summary,
        "results": case_results,
    }


async def _load_dataset_version(dataset_id: str):
    from app.services.storage import get_dataset

    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    current_version = next(
        (v for v in dataset.versions if v.version == dataset.current_version), None
    )
    if not current_version:
        raise HTTPException(status_code=404, detail="Dataset version not found")

    return dataset, current_version


@router.post("/run")
async def run_benchmark(payload: BenchmarkRunRequest) -> dict:
    """Run a benchmark synchronously over a dataset's current version cases."""
    from datetime import datetime, timezone

    from app.services.storage import add_benchmark_result
    from app.models.schemas import RagStrategy

    dataset, current_version = await _load_dataset_version(payload.datasetId)

    try:
        strategy = RagStrategy(payload.strategy)
    except ValueError:
        strategy = RagStrategy.HYBRID_RERANK_MMR

    started_at = datetime.now(timezone.utc).isoformat()
    case_results = []
    by_difficulty: dict[str, list[dict]] = {}
    by_tag: dict[str, list[dict]] = {}

    for test_case in getattr(current_version, "cases", []):
        case_result, breakdown_metrics = await _execute_case(test_case, strategy, payload.ragConfig)
        case_results.append(case_result)
        if breakdown_metrics is not None:
            by_difficulty.setdefault(case_result["difficulty"], []).append(breakdown_metrics)
            for tag in getattr(test_case, "tags", []):
                by_tag.setdefault(tag, []).append(breakdown_metrics)

    result = _finalize_run(payload, dataset, started_at, case_results, by_difficulty, by_tag)
    await add_benchmark_result(result)
    return {"success": True, "data": result}


@router.post("/run-stream")
async def run_benchmark_stream(payload: BenchmarkRunRequest):
    """Run a benchmark one test case at a time, streaming progress as SSE
    so the UI can show live "N / total" progress instead of waiting for the
    whole suite to finish before rendering anything."""
    import json as json_lib
    from datetime import datetime, timezone

    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import StreamingResponse

    from app.services.storage import add_benchmark_result
    from app.models.schemas import RagStrategy

    dataset, current_version = await _load_dataset_version(payload.datasetId)

    try:
        strategy = RagStrategy(payload.strategy)
    except ValueError:
        strategy = RagStrategy.HYBRID_RERANK_MMR

    cases = getattr(current_version, "cases", [])

    async def event_generator():
        started_at = datetime.now(timezone.utc).isoformat()
        case_results: list[dict] = []
        by_difficulty: dict[str, list[dict]] = {}
        by_tag: dict[str, list[dict]] = {}
        total = len(cases)

        yield f"data: {json_lib.dumps({'type': 'benchmark.started', 'total': total})}\n\n"

        for index, test_case in enumerate(cases, start=1):
            yield f"data: {json_lib.dumps({'type': 'case.started', 'index': index, 'total': total, 'caseId': test_case.id, 'query': test_case.query})}\n\n"

            case_result, breakdown_metrics = await _execute_case(test_case, strategy, payload.ragConfig)
            case_results.append(case_result)
            if breakdown_metrics is not None:
                by_difficulty.setdefault(case_result["difficulty"], []).append(breakdown_metrics)
                for tag in getattr(test_case, "tags", []):
                    by_tag.setdefault(tag, []).append(breakdown_metrics)

            yield f"data: {json_lib.dumps(jsonable_encoder({'type': 'case.completed', 'index': index, 'total': total, 'result': case_result}))}\n\n"

        result = _finalize_run(payload, dataset, started_at, case_results, by_difficulty, by_tag)
        await add_benchmark_result(result)

        yield f"data: {json_lib.dumps(jsonable_encoder({'type': 'benchmark.completed', 'data': result}))}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

import math

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from uuid import uuid4

from app.services.storage import get_all_datasets
from app.models.schemas import BenchmarkConfig, BenchmarkRun, BenchmarkRunStatus

router = APIRouter()

# In-memory store for benchmark runs
_benchmark_runs: dict[str, BenchmarkRun] = {}

_ALL_METRIC_KEYS = [
    "hitRate", "recall", "precision", "mrr", "ndcg",
    "faithfulness", "answerRelevance", "contextPrecision", "contextRecall",
]

_ALL_FAILURE_CATEGORIES = [
    "retrieval_failure", "missing_source", "wrong_source", "poor_ranking",
    "poor_context", "poor_answer", "citation_failure", "latency_failure",
    "token_limit_failure",
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


def _average_metrics(metric_dicts: list[dict]) -> dict:
    if not metric_dicts:
        return {key: 0.0 for key in _ALL_METRIC_KEYS}
    return {
        key: sum(m.get(key, 0.0) for m in metric_dicts) / len(metric_dicts)
        for key in _ALL_METRIC_KEYS
    }


class BenchmarkStartRequest(BaseModel):
    config: BenchmarkConfig


class BenchmarkRunResponse(BaseModel):
    id: str
    status: str
    aggregate_metrics: dict = {}
    total_tests: int = 0
    passed_tests: int = 0
    partial_tests: int = 0
    failed_tests: int = 0


@router.get("/runs")
async def list_runs() -> dict:
    runs = list(_benchmark_runs.values())
    return {"success": True, "data": [r.model_dump(by_alias=True) for r in runs]}


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    run = _benchmark_runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Benchmark run not found")
    return {"success": True, "data": run.model_dump(by_alias=True)}


@router.post("/start")
async def start_benchmark(payload: BenchmarkStartRequest) -> dict:
    from datetime import datetime
    from app.services.rag_engine import execute_rag

    run = BenchmarkRun(
        id=str(uuid4()),
        config=payload.config,
        status=BenchmarkRunStatus.RUNNING,
        started_at=datetime.utcnow(),
    )
    _benchmark_runs[run.id] = run

    # Note: In a real implementation, this would run the benchmark pipeline
    # asynchronously for each test case in the dataset.
    # For now, we return the run ID and mark it as completed with empty metrics.
    run.status = BenchmarkRunStatus.COMPLETED
    run.completed_at = datetime.utcnow()

    return {"success": True, "data": run.model_dump(by_alias=True)}


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str) -> dict:
    if run_id in _benchmark_runs:
        del _benchmark_runs[run_id]
        return {"success": True}
    raise HTTPException(status_code=404, detail="Benchmark run not found")


class BenchmarkRunRequest(BaseModel):
    datasetId: str
    strategy: str = "hybrid-rerank-mmr"
    ragConfig: Optional[dict] = None


@router.post("/run")
async def run_benchmark(payload: BenchmarkRunRequest) -> dict:
    """Run a benchmark synchronously over a dataset's current version cases."""
    from datetime import datetime, timezone
    from uuid import uuid4

    from app.services.storage import get_dataset, add_benchmark_result
    from app.services.rag_engine import execute_rag
    from app.models.schemas import RagStrategy

    dataset = await get_dataset(payload.datasetId)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    current_version = next(
        (v for v in dataset.versions if v.version == dataset.current_version), None
    )
    if not current_version:
        raise HTTPException(status_code=404, detail="Dataset version not found")

    try:
        strategy = RagStrategy(payload.strategy)
    except ValueError:
        strategy = RagStrategy.HYBRID_RERANK_MMR

    started_at = datetime.now(timezone.utc).isoformat()
    case_results = []
    empty_metrics = {
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

    by_difficulty: dict[str, list[dict]] = {}
    by_tag: dict[str, list[dict]] = {}

    for test_case in getattr(current_version, "cases", []):
        case_started = datetime.now(timezone.utc)
        try:
            answer = ""
            trace = None
            async for event in execute_rag(
                test_case.query, strategy, payload.ragConfig, None
            ):
                if event.get("type") == "llm.token" and "content" in event:
                    answer += event["content"]
                if event.get("type") == "trace.completed":
                    trace = event.get("data")
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

            if not expected_keys:
                status = "not_run"
                failure_categories: list[str] = []
                failure_explanation = ""
                metrics = {**empty_metrics}
            else:
                metrics = {
                    **empty_metrics,
                    "hitRate": _hit_rate(retrieved_keys, expected_keys),
                    "recall": _recall(retrieved_keys, expected_keys),
                    "precision": _precision(retrieved_keys, expected_keys),
                    "mrr": _mrr(retrieved_keys, expected_keys),
                    "ndcg": _ndcg(retrieved_keys, expected_keys),
                }
                if metrics["hitRate"] == 0:
                    status = "failed"
                    failure_categories = ["missing_source"] if actual_sources else ["retrieval_failure"]
                    failure_explanation = (
                        "Retrieved sources did not include any expected source."
                        if actual_sources else "No sources were retrieved."
                    )
                elif metrics["recall"] >= 0.999:
                    status = "passed"
                    failure_categories = []
                    failure_explanation = ""
                else:
                    status = "partial"
                    failure_categories = ["poor_ranking"]
                    failure_explanation = "Some expected sources were not retrieved."

                by_difficulty.setdefault(test_case.difficulty or "medium", []).append(metrics)
                for tag in test_case.tags:
                    by_tag.setdefault(tag, []).append(metrics)

            case_results.append({
                "caseId": test_case.id,
                "status": status,
                "query": test_case.query,
                "actualAnswer": answer,
                "actualSources": actual_sources,
                "metrics": metrics,
                "failureCategories": failure_categories,
                "failureExplanation": failure_explanation,
                "latencyMs": int((datetime.now(timezone.utc) - case_started).total_seconds() * 1000),
                "durationMs": int((datetime.now(timezone.utc) - case_started).total_seconds() * 1000),
                "tokenCount": 0,
                "actualPages": sorted({source["page"] for source in actual_sources}),
                "traceId": trace.get("id", "") if trace else "",
                "runId": "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            metrics = {**empty_metrics}
            case_results.append({
                "caseId": getattr(test_case, "id", "unknown"),
                "status": "failed",
                "query": getattr(test_case, "query", ""),
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
            })
            if getattr(test_case, "expected_sources", None):
                by_difficulty.setdefault(getattr(test_case, "difficulty", None) or "medium", []).append(metrics)
                for tag in getattr(test_case, "tags", []):
                    by_tag.setdefault(tag, []).append(metrics)

    evaluated_metrics = [r["metrics"] for r in case_results if r["status"] != "not_run"]
    aggregate_metrics = _average_metrics(evaluated_metrics)
    difficulty_breakdown = {level: _average_metrics(vals) for level, vals in by_difficulty.items()}
    tag_breakdown = {tag: _average_metrics(vals) for tag, vals in by_tag.items()}

    failure_categories_summary = {cat: 0 for cat in _ALL_FAILURE_CATEGORIES}
    for r in case_results:
        for cat in r["failureCategories"]:
            if cat in failure_categories_summary:
                failure_categories_summary[cat] += 1

    result = {
        "id": str(uuid4()),
        "datasetId": payload.datasetId,
        "strategy": payload.strategy,
        "startedAt": started_at,
        "completedAt": datetime.now(timezone.utc).isoformat(),
        "totalTests": len(getattr(current_version, "cases", [])),
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
    await add_benchmark_result(result)
    return {"success": True, "data": result}

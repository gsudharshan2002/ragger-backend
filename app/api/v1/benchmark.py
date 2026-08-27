from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from uuid import uuid4

from app.services.storage import get_all_datasets
from app.models.schemas import BenchmarkConfig, BenchmarkRun, BenchmarkRunStatus

router = APIRouter()

# In-memory store for benchmark runs
_benchmark_runs: dict[str, BenchmarkRun] = {}


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
            case_results.append({
                "caseId": test_case.id,
                "status": "passed",
                "query": test_case.query,
                "actualAnswer": answer,
                "actualSources": actual_sources,
                "metrics": {**empty_metrics},
                "failureCategories": [],
                "failureExplanation": "",
                "latencyMs": int((datetime.now(timezone.utc) - case_started).total_seconds() * 1000),
                "durationMs": int((datetime.now(timezone.utc) - case_started).total_seconds() * 1000),
                "tokenCount": 0,
                "actualPages": sorted({source["page"] for source in actual_sources}),
                "traceId": trace.get("id", "") if trace else "",
                "runId": "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            case_results.append({
                "caseId": getattr(test_case, "id", "unknown"),
                "status": "failed",
                "query": getattr(test_case, "query", ""),
                "actualAnswer": "",
                "actualSources": [],
                "metrics": {**empty_metrics},
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

    result = {
        "id": str(uuid4()),
        "datasetId": payload.datasetId,
        "strategy": payload.strategy,
        "startedAt": started_at,
        "completedAt": datetime.now(timezone.utc).isoformat(),
        "totalTests": len(getattr(current_version, "cases", [])),
        "completedTests": len(case_results),
        "passedTests": 0,
        "partialTests": 0,
        "failedTests": sum(1 for result in case_results if result["status"] == "failed"),
        "status": "completed",
        "datasetName": dataset.name,
        "datasetVersion": dataset.current_version,
        "config": payload.ragConfig or {},
        "aggregateMetrics": {**empty_metrics},
        "difficultyBreakdown": {},
        "tagBreakdown": {},
        "failureCategories": {},
        "results": case_results,
    }
    await add_benchmark_result(result)
    return {"success": True, "data": result}

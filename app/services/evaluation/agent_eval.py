"""Agent vs. baseline-RAG evaluation harness.

Runs a dataset's golden cases through both the ReAct AgentEngine and the
baseline RAG pipeline (execute_rag) for the same strategy, scores both on
retrieval hit rate, answer-quality heuristics, latency, and step count, and
persists the comparison to the existing trace storage.

No LLM-judge is wired up anywhere in this codebase yet - benchmark.py's
EvaluationMetrics.faithfulness/answer_relevance fields exist in the schema
but are never actually populated by its own harness either. The heuristic
scorers here (word-overlap with the query / grounding in retrieved context)
are a stand-in for a real judge model, not a claim of judge-quality scoring.
"""
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from app.api.v1.benchmark import _hit_rate, _looks_like_refusal, _round_metric, _source_keys
from app.core.config import settings
from app.core.logging_config import get_logger
from app.models.schemas import AgentRunRequest, RagStrategy
from app.services.agent_engine import AgentEngine
from app.services.rag_engine import execute_rag
from app.services.storage import add_trace, get_dataset

logger = get_logger(__name__)


def _answer_relevance(answer: str, query: str) -> float:
    if _looks_like_refusal(answer):
        return 0.0
    query_words = {w for w in query.lower().split() if len(w) > 3}
    if not query_words:
        return 0.0
    answer_lower = answer.lower()
    matched = sum(1 for w in query_words if w in answer_lower)
    return _round_metric(matched / len(query_words))


def _faithfulness(answer: str, context_chunks: list[str]) -> float:
    if _looks_like_refusal(answer) or not context_chunks:
        return 0.0
    context_text = " ".join(context_chunks).lower()
    answer_words = {w for w in answer.lower().split() if len(w) > 4}
    if not answer_words:
        return 0.0
    grounded = sum(1 for w in answer_words if w in context_text)
    return _round_metric(grounded / len(answer_words))


async def _run_baseline(query: str, strategy: RagStrategy, knowledge_base_id: Optional[str]) -> dict[str, Any]:
    answer = ""
    trace: Optional[dict] = None

    async for event in execute_rag(query, strategy, None, knowledge_base_id):
        if event.get("type") == "llm.token" and "content" in event:
            answer += event["content"]
        if event.get("type") == "trace.completed":
            trace = event.get("data")

    trace = trace or {}
    return {
        "answer": answer,
        "sources": trace.get("sources", []),
        "context_chunks": [c.get("content", "") for c in trace.get("context", {}).get("chunks", [])],
        "latency_ms": trace.get("total_latency_ms", 0),
    }


async def _run_agent(
    query: str,
    strategy: Optional[RagStrategy],
    knowledge_base_id: Optional[str],
    max_steps: int,
) -> dict[str, Any]:
    engine = AgentEngine()
    request = AgentRunRequest(
        query=query, strategy=strategy, knowledge_base_id=knowledge_base_id, max_steps=max_steps
    )

    response: dict = {}
    async for event in engine.run_stream(request):
        if event.get("type") == "trace.completed":
            response = event.get("data") or {}

    steps = response.get("steps", [])
    context_chunks = [
        chunk.get("excerpt", "")
        for step in steps
        if step.get("action", {}).get("tool") == "retrieve"
        for chunk in step.get("observation", {}).get("output", {}).get("chunks", [])
    ]

    return {
        "answer": response.get("answer", ""),
        "sources": response.get("sources", []),
        "context_chunks": context_chunks,
        "latency_ms": response.get("totalLatencyMs", 0),
        "step_count": len(steps),
    }


async def run_agent_evaluation(
    dataset_id: str,
    strategy: RagStrategy = RagStrategy.HYBRID_RERANK_MMR,
    knowledge_base_id: Optional[str] = None,
    max_steps: Optional[int] = None,
) -> dict[str, Any]:
    """Run every case in a dataset's current version through both the agent
    and the baseline RAG pipeline for comparison, and persist the result."""
    max_steps = max_steps or settings.AGENT_MAX_STEPS

    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise ValueError(f"Dataset not found: {dataset_id}")

    current_version = next((v for v in dataset.versions if v.version == dataset.current_version), None)
    if not current_version:
        raise ValueError(f"Dataset version not found: {dataset.current_version}")

    case_results = []
    for case in current_version.cases:
        expected_keys = [_source_keys(s) for s in (case.expected_sources or [])]

        baseline = await _run_baseline(case.query, strategy, knowledge_base_id)
        agent = await _run_agent(case.query, strategy, knowledge_base_id, max_steps)

        baseline_keys = [_source_keys(s) for s in baseline["sources"]]
        agent_keys = [_source_keys(s) for s in agent["sources"]]

        case_results.append(
            {
                "caseId": case.id,
                "query": case.query,
                "baseline": {
                    "answer": baseline["answer"],
                    "latencyMs": baseline["latency_ms"],
                    "hitRate": _round_metric(_hit_rate(baseline_keys, expected_keys)) if expected_keys else None,
                    "answerRelevance": _answer_relevance(baseline["answer"], case.query),
                    "faithfulness": _faithfulness(baseline["answer"], baseline["context_chunks"]),
                },
                "agent": {
                    "answer": agent["answer"],
                    "latencyMs": agent["latency_ms"],
                    "stepCount": agent["step_count"],
                    "hitRate": _round_metric(_hit_rate(agent_keys, expected_keys)) if expected_keys else None,
                    "answerRelevance": _answer_relevance(agent["answer"], case.query),
                    "faithfulness": _faithfulness(agent["answer"], agent["context_chunks"]),
                },
            }
        )

    def _avg(side: str, metric: str) -> float:
        values = [r[side][metric] for r in case_results if r[side][metric] is not None]
        return _round_metric(sum(values) / len(values)) if values else 0.0

    summary = {
        "id": str(uuid4()),
        "kind": "agent_eval",
        "datasetId": dataset_id,
        "datasetName": dataset.name,
        "strategy": strategy.value,
        "maxSteps": max_steps,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "caseCount": len(case_results),
        "aggregate": {
            "baseline": {
                "hitRate": _avg("baseline", "hitRate"),
                "answerRelevance": _avg("baseline", "answerRelevance"),
                "faithfulness": _avg("baseline", "faithfulness"),
                "latencyMs": _avg("baseline", "latencyMs"),
            },
            "agent": {
                "hitRate": _avg("agent", "hitRate"),
                "answerRelevance": _avg("agent", "answerRelevance"),
                "faithfulness": _avg("agent", "faithfulness"),
                "latencyMs": _avg("agent", "latencyMs"),
                "avgStepCount": (
                    _round_metric(sum(r["agent"]["stepCount"] for r in case_results) / len(case_results))
                    if case_results
                    else 0.0
                ),
            },
        },
        "results": case_results,
    }

    try:
        await add_trace(summary)
    except Exception as e:
        logger.error(f"Failed to persist agent evaluation: {e}")

    return summary

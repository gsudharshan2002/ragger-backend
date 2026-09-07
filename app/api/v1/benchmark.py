import asyncio
import json
import math
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from uuid import uuid4

from app.core.logging_config import get_logger
from app.services.storage import get_all_datasets

router = APIRouter()
logger = get_logger(__name__)

_run_lock = asyncio.Lock()

_DEVELOPER_DOC_RESULTS_DIR = Path(__file__).resolve().parents[3] / "evals" / "results"
# Points at the two report files from the most recent POST /run, so GET
# /results can return that exact pair instead of guessing from directory
# contents (which breaks under concurrent runs or a same-second CLI write
# into the same directory).
_LATEST_PAIR_PATH = _DEVELOPER_DOC_RESULTS_DIR / "_latest_pair.json"


def _write_pair_pointer(baseline_file: str, improved_file: str) -> None:
    _DEVELOPER_DOC_RESULTS_DIR.mkdir(exist_ok=True)
    # Atomic write (temp file + os.replace) so a reader never sees a
    # half-written pointer file - a plain write_text() offers no such
    # guarantee, and would corrupt the pointer if two runs ever wrote it
    # at the same time.
    tmp_path = _LATEST_PAIR_PATH.with_suffix(f".{uuid4().hex[:8]}.tmp")
    tmp_path.write_text(
        json.dumps({"baseline": baseline_file, "improved": improved_file}), encoding="utf-8"
    )
    tmp_path.replace(_LATEST_PAIR_PATH)


def _read_pair_pointer() -> Optional[dict]:
    if not _LATEST_PAIR_PATH.exists():
        return None
    try:
        pointer = json.loads(_LATEST_PAIR_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    data: dict[str, dict] = {}
    for key in ("baseline", "improved"):
        filename = pointer.get(key)
        if not filename:
            continue
        try:
            data[key] = json.loads((_DEVELOPER_DOC_RESULTS_DIR / filename).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None  # pointer stale/broken - fall back to the directory heuristic
    return data or None


@router.get("/developer-docs/results")
async def get_developer_docs_results() -> dict:
    """Return the Week 6 developer-documentation evaluation comparison.

    Prefers the explicit pointer written by the most recent POST /run (see
    _write_pair_pointer) so the pairing is never guessed. Falls back to the
    "two most recent files" heuristic only for reports written before the
    pointer existed, or if the pointer file is missing/corrupt.
    """
    if not _DEVELOPER_DOC_RESULTS_DIR.exists():
        return {"success": True, "data": {}}

    pair = _read_pair_pointer()
    if pair:
        return {"success": True, "data": pair}

    reports = []
    for report_path in _DEVELOPER_DOC_RESULTS_DIR.glob("*.json"):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "created_at" in report and "summary" in report:
            reports.append(report)
    reports.sort(key=lambda r: r["created_at"])

    # Pair the latest report with the most recent PRIOR report of the SAME
    # strategy - a "before" from a different retrieval method isn't a
    # meaningful comparison, it's just a different report.
    data: dict[str, dict] = {}
    if reports:
        latest = reports[-1]
        data["improved"] = latest
        same_strategy = [r for r in reports[:-1] if r.get("strategy") == latest.get("strategy")]
        if same_strategy:
            data["baseline"] = same_strategy[-1]
    return {"success": True, "data": data}


@router.get("/developer-docs/reports")
async def list_developer_docs_reports(strategy: Optional[str] = None) -> dict:
    """List saved developer-docs eval reports, optionally filtered to one
    strategy, newest first - real run history instead of guessing which
    prior run to compare against."""
    reports = []
    for path in _DEVELOPER_DOC_RESULTS_DIR.glob("*.json"):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "created_at" not in report or "summary" not in report:
            continue
        if strategy and report.get("strategy") != strategy:
            continue
        reports.append({
            "filename": path.name,
            "strategy": report.get("strategy"),
            "created_at": report["created_at"],
            "combined_score": report["summary"].get("combined_score"),
            "label": report.get("label"),
        })
    reports.sort(key=lambda r: r["created_at"], reverse=True)
    return {"success": True, "data": reports}


class DeveloperDocCase(BaseModel):
    """A single developer-documentation eval case. `id`/`question`/`problem_type`
    are accessed unconditionally by evals/run_developer_docs.py's _run_case
    (after it has already spent a real RAG pipeline call), so validating them
    here rejects a malformed case with a clear 422 up front instead of a
    cryptic KeyError string buried in that case's "error" field.
    """

    id: str
    question: str
    problem_type: str
    mode: str = "unknown"
    tags: list[str] = []
    regression: bool = False
    regression_evidence: Optional[dict] = None
    expected_sources: list[dict] = []
    expected_answer_keywords: list[str] = []


class DeveloperDocsRunRequest(BaseModel):
    cases: list[DeveloperDocCase]
    strategy: str = "hybrid-rerank-mmr"
    baselineStrategy: Optional[str] = None
    noJudge: bool = False
    casesFileName: Optional[str] = None
    useRagas: bool = False
    knowledgeBaseId: Optional[str] = None


@router.post("/developer-docs/run")
async def run_developer_docs(payload: DeveloperDocsRunRequest) -> dict:
    """Run the Week 6 developer-documentation eval and persist a fresh
    report so the frontend comparison updates on Refresh.

    The frontend sends:
      - cases: the test-case JSON array (from the uploaded file)
      - strategy: the RagStrategy slug to run now (defaults to
        hybrid-rerank-mmr, the one actual improved pipeline)
      - baselineStrategy: optional - only set this to explicitly compare
        against a DIFFERENT strategy run fresh right now. Left unset (the
        normal case), "baseline" instead means "the most recent PRIOR report
        for this SAME strategy", loaded from disk rather than re-executed -
        i.e. did changing something about THIS strategy actually help,
        compared to what it scored before you changed it. That's a same-
        strategy before/after, not a different-retrieval-method comparison.
      - noJudge: skip LLM-as-judge (faster, keyword-only scoring)
      - casesFileName: the uploaded file's name, stored as report metadata
      - useRagas: also compute the bonus faithfulness/context-precision metrics
        (2 extra LLM calls per case - off by default)
      - knowledgeBaseId: optional - scope retrieval to one knowledge base, same
        as Chat does. Omit to search every chunk in the store (prior behavior).
    """
    async with _run_lock:
        from evals.run_developer_docs import RESULTS_DIR, _latest_previous_run_for_strategy, run_strategy

        cases_file = payload.casesFileName or "developer_docs_cases.json"
        label = Path(cases_file).stem
        cases = [c.model_dump() for c in payload.cases]

        improved_report, improved_path = await run_strategy(
            cases, payload.strategy, use_judge=not payload.noJudge,
            label=label, cases_file=cases_file, use_ragas=payload.useRagas,
            knowledge_base_id=payload.knowledgeBaseId,
        )

        baseline_report: Optional[dict] = None
        baseline_filename: Optional[str] = None
        if payload.baselineStrategy:
            # Explicit override: compare against a genuinely different strategy,
            # run fresh right now (the old behavior).
            baseline_report, baseline_path = await run_strategy(
                cases, payload.baselineStrategy, use_judge=not payload.noJudge,
                label=label, cases_file=cases_file, use_ragas=payload.useRagas,
                knowledge_base_id=payload.knowledgeBaseId,
            )
            baseline_filename = baseline_path.name
        else:
            # Default: same-strategy before/after - the most recent PRIOR report
            # for this exact strategy, not re-executed.
            previous = _latest_previous_run_for_strategy(RESULTS_DIR, payload.strategy, exclude_path=improved_path)
            if previous:
                baseline_filename = previous.pop("_source_path")
                baseline_report = previous

        data: dict[str, dict] = {"improved": improved_report}
        if baseline_report:
            data["baseline"] = baseline_report
        # Always write the pointer so GET /results reflects the latest run even when no baseline exists
        _write_pair_pointer(baseline_filename or "", improved_path.name)
        return {"success": True, "data": data}


class GenerateForLabelingRequest(BaseModel):
    cases: list[DeveloperDocCase]
    strategy: str = "bm25"
    casesFileName: Optional[str] = None
    knowledgeBaseId: Optional[str] = None


@router.post("/developer-docs/generate-for-labeling")
async def generate_for_labeling(payload: GenerateForLabelingRequest) -> dict:
    """Run the eval once with the judge OFF, producing a frozen answer set
    with no judge_verdict anywhere in it - the only kind of report safe to
    hand-label blind from (see evals/labeling.py). A single-strategy run,
    not a baseline/improved comparison pair - labeling needs one frozen
    answer set, not two.
    """
    from evals.labeling import get_label_session
    from evals.run_developer_docs import run_strategy

    cases_file = payload.casesFileName or "developer_docs_cases.json"
    label = Path(cases_file).stem
    await run_strategy(
        [c.model_dump() for c in payload.cases], payload.strategy, use_judge=False, label=label,
        cases_file=cases_file, knowledge_base_id=payload.knowledgeBaseId,
    )
    return {"success": True, "data": get_label_session()}


@router.get("/developer-docs/default-cases")
async def get_default_cases() -> dict:
    """Return the built-in developer_docs_cases.json so the frontend
    can offer a one-click 'Load default cases' button."""
    from pathlib import Path

    cases_path = Path(__file__).resolve().parents[3] / "evals" / "developer_docs_cases.json"
    if not cases_path.exists():
        return {"success": False, "detail": "Default cases file not found"}
    import json
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    return {"success": True, "data": cases}


@router.get("/developer-docs/label-session")
async def get_label_session_endpoint() -> dict:
    """Everything the labeling UI needs: the latest judge-free report's
    cases, any labels already saved, and whether those saved labels
    conflict with the current report (labeled against an older run)."""
    from evals.labeling import get_label_session

    return {"success": True, "data": get_label_session()}


@router.delete("/developer-docs/label-session")
async def clear_label_session_endpoint() -> dict:
    """Delete labels_25.json so the user can start fresh with a new
    generate-for-labeling run against a different report or strategy."""
    from evals.labeling import LABELS_PATH

    if LABELS_PATH.exists():
        LABELS_PATH.unlink()
    return {"success": True}


@router.post("/developer-docs/cases/{case_id}/clear-regression")
async def clear_regression_endpoint(case_id: str) -> dict:
    """Demote a regression case back to a normal case in
    developer_docs_cases.json once its hand label turns to pass - strips
    regression/regression_evidence so the fix is reflected in the source
    file, not just the current session's view of it."""
    from evals.labeling import clear_regression_flag

    case = await clear_regression_flag(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"Case {case_id!r} not found in developer_docs_cases.json")
    return {"success": True, "data": case}


class LabelRequest(BaseModel):
    case_id: str
    label: str  # "pass" | "fail"


@router.post("/developer-docs/label")
async def save_label_endpoint(payload: LabelRequest) -> dict:
    """Save one blind label. Same guarantee as the CLI tool: refuses if
    labels_25.json already exists against a different report than the
    current judge-free one, so labels from two different answer sets can
    never get silently mixed together."""
    from evals.labeling import find_latest_no_judge_report, save_label

    report = find_latest_no_judge_report()
    if not report:
        raise HTTPException(status_code=404, detail="No --no-judge report found to label against.")
    try:
        data = await save_label(report["_source_path"], report.get("created_at", ""), payload.case_id, payload.label)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"success": True, "data": {"labeled_count": len(data["labels"])}}


@router.post("/developer-docs/validate-judge")
async def validate_judge_endpoint() -> dict:
    """Run the CURRENTLY active judge prompt against the exact frozen
    answers already hand-labeled in labels_25.json, and compute agreement.
    Save judge.py's prompt as judge_v1.txt / judge_v2.txt around whichever
    call of this you treat as the before/after milestone."""
    from evals.labeling import run_judge_validation

    try:
        result = await run_judge_validation()
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"success": True, "data": result}


@router.get("/developer-docs/judge-runs")
async def list_judge_runs() -> dict:
    """Return every judge-validation run saved under evals/results/, newest
    first, so the frontend can render prior agreement results on a fresh page
    load without re-running the judge."""
    runs = []
    for path in _DEVELOPER_DOC_RESULTS_DIR.glob("*_judge_validation.json"):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        run["_filename"] = path.name
        runs.append(run)
    runs.sort(key=lambda r: r.get("validated_at", ""), reverse=True)
    return {"success": True, "data": runs}


@router.delete("/developer-docs/judge-runs/{filename}")
async def delete_judge_run(filename: str) -> dict:
    """Delete one saved judge-validation run by filename, or all of them with
    the special value 'all' (e.g. to reset the agreement panel)."""
    if filename == "all":
        deleted = 0
        for path in _DEVELOPER_DOC_RESULTS_DIR.glob("*_judge_validation.json"):
            path.unlink(missing_ok=True)
            deleted += 1
        return {"success": True, "data": {"deleted": deleted}}

    path = _DEVELOPER_DOC_RESULTS_DIR / filename
    if not path.name.endswith("_judge_validation.json") or not path.exists():
        raise HTTPException(status_code=404, detail=f"Judge run {filename!r} not found")
    path.unlink(missing_ok=True)
    return {"success": True, "data": {"deleted": 1}}


_PREDICTION_PATH = Path(__file__).resolve().parents[3] / "evals" / "prediction.txt"


class PredictionRequest(BaseModel):
    text: str


@router.post("/developer-docs/prediction")
async def save_prediction(payload: PredictionRequest) -> dict:
    """Save the required one-sentence prediction of what a judge-prompt
    iteration will fix - written BEFORE the iteration, so it can be checked
    against what the iteration actually changed afterward."""
    _PREDICTION_PATH.write_text(payload.text.strip() + "\n", encoding="utf-8")
    return {"success": True}


@router.get("/developer-docs/prediction")
async def get_prediction() -> dict:
    if not _PREDICTION_PATH.exists():
        return {"success": True, "data": {"text": "", "exists": False}}
    return {"success": True, "data": {"text": _PREDICTION_PATH.read_text(encoding="utf-8").strip(), "exists": True}}


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
        # Each expected source can only award relevance once, at the first
        # (highest-ranked) retrieved item that matches it - otherwise several
        # retrieved chunks landing on the same expected page (common, since a
        # page is usually split into multiple chunks) would each score full
        # relevance while idcg still assumes only len(expected) relevant
        # items exist, letting dcg exceed idcg and ndcg exceed 1.0.
        score = 0.0
        claimed: set[int] = set()
        for i, item in enumerate(items[:k]):
            rel = 0
            for idx, e in enumerate(expected):
                if idx not in claimed and _is_match(item, e):
                    rel = 1
                    claimed.add(idx)
                    break
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


async def _execute_case(test_case, strategy, rag_config, knowledge_base_id: Optional[str] = None) -> tuple[dict, Optional[dict]]:
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
            test_case.query, strategy, effective_rag_config, knowledge_base_id
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
                    "document": source.get("documentName") or source.get("document_name", ""),
                    "page": source.get("page", 0),
                    "section": source.get("section", ""),
                    "chunkId": source.get("chunkId") or source.get("chunk_id", ""),
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
            "durationMs": int((datetime.now(timezone.utc) - case_started).total_seconds() * 1000),
            "tokenCount": 0,
            "actualPages": [],
            "traceId": "",
            "runId": "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        }
        # Never contribute to difficulty/tag breakdown averages here - an
        # exception means the pipeline crashed (LLM error, bug, timeout),
        # not that retrieval genuinely found nothing. Folding this all-zero
        # `metrics` dict into by_difficulty/by_tag would conflate an
        # infrastructure failure with a real retrieval-quality failure and
        # unfairly drag down that bucket's average.
        return case_result, None


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
        case_result, breakdown_metrics = await _execute_case(
            test_case, strategy, payload.ragConfig, dataset.knowledge_base_id
        )
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

        try:
            yield f"data: {json_lib.dumps({'type': 'benchmark.started', 'total': total})}\n\n"

            for index, test_case in enumerate(cases, start=1):
                yield f"data: {json_lib.dumps({'type': 'case.started', 'index': index, 'total': total, 'caseId': test_case.id, 'query': test_case.query})}\n\n"

                case_result, breakdown_metrics = await _execute_case(
                    test_case, strategy, payload.ragConfig, dataset.knowledge_base_id
                )
                case_results.append(case_result)
                if breakdown_metrics is not None:
                    by_difficulty.setdefault(case_result["difficulty"], []).append(breakdown_metrics)
                    for tag in getattr(test_case, "tags", []):
                        by_tag.setdefault(tag, []).append(breakdown_metrics)

                yield f"data: {json_lib.dumps(jsonable_encoder({'type': 'case.completed', 'index': index, 'total': total, 'result': case_result}))}\n\n"

            result = _finalize_run(payload, dataset, started_at, case_results, by_difficulty, by_tag)
            await add_benchmark_result(result)

            yield f"data: {json_lib.dumps(jsonable_encoder({'type': 'benchmark.completed', 'data': result}))}\n\n"
        except Exception as e:
            # _execute_case already catches errors within its own scope, but
            # _finalize_run, add_benchmark_result, and JSON serialization
            # itself can all still raise here. Without this, an unhandled
            # exception crashes the generator and the SSE connection just
            # drops - no benchmark.completed, no error event - leaving the
            # frontend stuck showing "running" forever with nothing to
            # signal that it should stop waiting.
            logger.error(f"Benchmark stream failed: {e}")
            yield f"data: {json_lib.dumps({'type': 'error', 'error': str(e)})}\n\n"

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

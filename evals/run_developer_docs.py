"""Run the Week 6 developer-documentation evaluation set."""

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.models.schemas import RagStrategy
from app.services.rag_engine import execute_rag
from evals.assertions import (
    check_deprecated_without_migration_note,
    check_unknown_endpoints,
    load_deprecations,
    load_openapi_paths,
)
from evals.judge import judge_answer
from evals.ragas_metrics import compute_context_precision, compute_faithfulness

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "developer_docs_cases.json"
RESULTS_DIR = ROOT / "results"

# Delay (seconds) inserted between consecutive eval cases. The eval fires one
# RAG generation call per case (plus a judge call, and 2 more with --ragas),
# all sequentially with zero spacing by default. On rate-limited providers
# (Groq's free tier caps requests/minute quite low) this burst trips the 429
# immediately, so we throttle calls to stay under the limit. Override via the
# EVAL_CASE_DELAY_MS env var, or 0 to disable throttling entirely.
DEFAULT_CASE_DELAY_S = float(os.environ.get("EVAL_CASE_DELAY_MS", "2000")) / 1000.0

# Loaded once at import time, not per-case - these fixtures don't change
# mid-run. If they're ever missing (e.g. this eval script points at a
# different, non-GitHub docs set), the assertions just find nothing to
# flag rather than crashing the whole eval.
try:
    _OPENAPI_PATHS = load_openapi_paths()
    _DEPRECATIONS = load_deprecations()
except OSError:
    _OPENAPI_PATHS = []
    _DEPRECATIONS = []

TAXONOMY_MODES = [
    "retrieval_failure",
    "missing_source",
    "wrong_source",
    "poor_ranking",
    "poor_context",
    "poor_answer",
    "citation_failure",
    "latency_failure",
    "token_limit_failure",
    "llm_failure",
    "prompt_issue",
]

def _value(item: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if item.get(key) is not None:
            return item[key]
    return default


def _source_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    actual_document = str(_value(actual, "document", "documentName", "document_name")).lower()
    expected_document = str(expected.get("document", "")).lower()
    actual_stem = Path(actual_document).stem
    expected_stem = Path(expected_document).stem
    return actual_stem == expected_stem and int(actual.get("page", 0)) == int(expected.get("page", 0))


def _answer_score(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    lowered = answer.lower()
    return sum(keyword.lower() in lowered for keyword in keywords) / len(keywords)


async def _run_case(
    case: dict[str, Any],
    strategy: str,
    use_judge: bool,
    use_ragas: bool = False,
    knowledge_base_id: str | None = None,
) -> dict[str, Any]:
    try:
        answer = ""
        trace: dict[str, Any] | None = None
        error = ""

        async for event in execute_rag(case["question"], RagStrategy(strategy), None, knowledge_base_id):
            event_type = event.get("type")
            if event_type == "llm.token":
                answer += event.get("content", "")
            elif event_type == "trace.completed":
                trace = event.get("data") or {}
            elif event_type in ("error", "trace.failed"):
                error = str(_value(event, "error", default="RAG pipeline failed"))

        actual_sources = (trace or {}).get("sources", [])
        expected_sources = case.get("expected_sources", [])
        # Cheap rule-based checks first: does retrieval find the right source?
        retrieval_score = (
            sum(any(_source_matches(actual, expected) for actual in actual_sources) for expected in expected_sources)
            / len(expected_sources)
            if expected_sources
            else 0.0
        )
        actual_answer = (trace or {}).get("llm", {}).get("answer") or answer
        expected_keywords = case.get("expected_answer_keywords", [])
        keyword_answer_score = _answer_score(actual_answer, expected_keywords)

        # Harder to check with a rule: does the answer actually convey those facts,
        # correctly and helpfully? An LLM judge grades that; its number is only
        # trustworthy once validated against human grading (see validate_judge.py).
        judge_verdict: str | None = None
        judge_reasoning = ""
        if use_judge:
            judge_result = await judge_answer(case["question"], actual_answer, expected_keywords)
            judge_verdict = judge_result["verdict"]
            judge_reasoning = judge_result["reasoning"]

        answer_score = (1.0 if judge_verdict == "pass" else 0.0) if judge_verdict is not None else keyword_answer_score
        combined_score = (retrieval_score + answer_score) / 2

        # Deterministic checks the judge is explicitly told not to grade
        # (see judge.py's system prompt) - a parser and a spec/list lookup,
        # not an LLM guess. A violation fails the case regardless of what
        # the judge or keyword score said, because these are hard facts,
        # not matters of judgment.
        unknown_endpoints = check_unknown_endpoints(actual_answer, _OPENAPI_PATHS)
        deprecation_violations = check_deprecated_without_migration_note(actual_answer, _DEPRECATIONS)
        assertions_passed = not unknown_endpoints and not deprecation_violations

        # Bonus: RAGAS-style faithfulness and context precision. Opt-in only
        # (use_ragas) - each is its own LLM call, and neither feeds status/
        # combined_score, since they measure something the pass/fail scores
        # above don't: whether the answer is grounded in *some* context
        # (faithfulness) and whether the *right* context was ranked highly
        # (context precision) - a case can be 100% faithful while confidently
        # grounded in the wrong document version.
        faithfulness = None
        context_precision = None
        if use_ragas:
            prompt_context = (trace or {}).get("prompt", {}).get("context", "")
            faithfulness = (await compute_faithfulness(actual_answer, prompt_context))["score"]
            context_precision = (await compute_context_precision(case["question"], prompt_context))["score"]

        status = "passed" if combined_score == 1.0 and not error and assertions_passed else "failed"
        return {
            "id": case["id"],
            "question": case["question"],
            "problem_type": case["problem_type"],
            "mode": case.get("mode", "unknown"),
            "tags": case.get("tags", []),
            "regression": bool(case.get("regression", False)),
            "regression_evidence": case.get("regression_evidence"),
            "status": status,
            "retrieval_score": round(retrieval_score, 4),
            "answer_score": round(answer_score, 4),
            "keyword_answer_score": round(keyword_answer_score, 4),
            "judge_verdict": judge_verdict,
            "judge_reasoning": judge_reasoning,
            "combined_score": round(combined_score, 4),
            "assertions": {
                "unknown_endpoints": unknown_endpoints,
                "deprecated_without_migration_note": [v["symbols"] for v in deprecation_violations],
            },
            "faithfulness": faithfulness,
            "context_precision": context_precision,
            "expected_sources": expected_sources,
            "actual_sources": actual_sources,
            "expected_answer_keywords": expected_keywords,
            "answer": actual_answer,
            "error": error,
        }
    except Exception as exc:
        # A single malformed case or a pipeline exception must not discard
        # every other case's results for this strategy run.
        return {
            "id": case.get("id", "unknown"),
            "question": case.get("question", ""),
            "problem_type": case.get("problem_type", ""),
            "mode": case.get("mode", "unknown"),
            "tags": case.get("tags", []),
            "regression": bool(case.get("regression", False)),
            "regression_evidence": case.get("regression_evidence"),
            "status": "failed",
            "retrieval_score": 0.0,
            "answer_score": 0.0,
            "keyword_answer_score": 0.0,
            "judge_verdict": None,
            "judge_reasoning": "",
            "combined_score": 0.0,
            "assertions": {"unknown_endpoints": [], "deprecated_without_migration_note": []},
            "faithfulness": None,
            "context_precision": None,
            "expected_sources": case.get("expected_sources", []),
            "actual_sources": [],
            "expected_answer_keywords": case.get("expected_answer_keywords", []),
            "answer": "",
            "error": str(exc),
        }


def _average(results: list[dict[str, Any]], key: str) -> float:
    return round(sum(result[key] for result in results) / len(results), 4) if results else 0.0


def _average_present(results: list[dict[str, Any]], key: str) -> float | None:
    """Like _average, but for optional (possibly-None) fields such as the
    RAGAS bonus metrics, which are only populated when use_ragas=True.
    Cases where the metric wasn't computed are excluded rather than
    counted as 0 - otherwise an average across a mostly-uncomputed set
    would understate the real score."""
    values = [r[key] for r in results if r.get(key) is not None]
    return round(sum(values) / len(values), 4) if values else None


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_type.setdefault(result["problem_type"], []).append(result)
        by_mode.setdefault(result.get("mode", "unknown"), []).append(result)
    return {
        "retrieval_score": _average(results, "retrieval_score"),
        "answer_score": _average(results, "answer_score"),
        "combined_score": _average(results, "combined_score"),
        "faithfulness": _average_present(results, "faithfulness"),
        "context_precision": _average_present(results, "context_precision"),
        "problem_type_scores": {
            problem_type: {
                "cases": len(group),
                "combined_score": _average(group, "combined_score"),
            }
            for problem_type, group in sorted(by_type.items())
        },
        "mode_scores": {
            mode: {
                "cases": len(group),
                "passed": sum(1 for r in group if r["status"] == "passed"),
                "pass_rate": round(sum(1 for r in group if r["status"] == "passed") / len(group), 4),
            }
            for mode, group in sorted(by_mode.items())
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default=str(CASES_PATH),
        help="Path to the eval cases JSON file (defaults to developer_docs_cases.json). "
        "Point this at a different file to evaluate a different document.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional free-form tag stored with the run (e.g. a RAG config name). "
        "Purely descriptive - it no longer selects which file gets overwritten.",
    )
    parser.add_argument("--strategy", default="bm25")
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the LLM judge and score answers with the keyword-matching rule "
        "only (no GROQ_API_KEY required, faster, but cruder).",
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="Also compute the bonus RAGAS-style faithfulness and context precision "
        "metrics for every case (2 extra LLM calls per case - off by default).",
    )
    parser.add_argument(
        "--knowledge-base-id",
        default=None,
        help="Optional - scope retrieval to one knowledge base, same as Chat does. "
        "Omit to search every chunk in the store (prior behavior, unchanged).",
    )
    return parser.parse_args()


def _latest_previous_run(results_dir: Path) -> dict[str, Any] | None:
    """Return the most recent existing eval run report, or None if there isn't
    one. Requires "summary" (not just "created_at") so judge-validation
    reports - which are also timestamped JSON in this same directory but
    aren't eval runs - are never mistaken for one."""
    reports = []
    for path in results_dir.glob("*.json"):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "created_at" in report and "summary" in report:
            reports.append(report)
    if not reports:
        return None
    return max(reports, key=lambda r: r["created_at"])


async def run_strategy(
    cases: list[dict[str, Any]],
    strategy: str,
    use_judge: bool = True,
    label: str | None = None,
    cases_file: str | None = None,
    use_ragas: bool = False,
    knowledge_base_id: str | None = None,
    case_delay: float = DEFAULT_CASE_DELAY_S,
) -> tuple[dict[str, Any], Path]:
    """Run the developer-documentation eval set through a single retrieval
    strategy and persist a timestamped report to evals/results/.

    `cases_file` should be the name of the file the `cases` argument actually
    came from (e.g. the uploaded file's name) - it is stored as report
    metadata verbatim and must not be assumed to be developer_docs_cases.json.

    `use_ragas` opts into the bonus faithfulness/context-precision metrics
    (evals/ragas_metrics.py) - off by default since each case costs 2 extra
    LLM calls.

    `knowledge_base_id` is optional - when omitted (the default, matching
    prior behavior), retrieval searches every chunk in the store. Set it to
    scope retrieval to one knowledge base, same as Chat already does.

    Returns (report dict, path the report was written to). Reused by the CLI
    (main) and the API endpoint that powers the Week 6 Refresh button so both
    produce identical reports.
    """
    cases_file_name = cases_file or CASES_PATH.name
    labels = label or Path(cases_file_name).stem
    RESULTS_DIR.mkdir(exist_ok=True)
    results: list[dict[str, Any]] = []
    for i, case in enumerate(cases):
        results.append(await _run_case(case, strategy, use_judge, use_ragas, knowledge_base_id))
        # Throttle between cases to avoid tripping the provider's per-minute
        # rate limit with a burst of back-to-back LLM calls. print the progress
        # so a long (multi-minute) eval isn't mistaken for a hang.
        if i < len(cases) - 1 and case_delay > 0:
            await asyncio.sleep(case_delay)
    created_at = datetime.now(UTC)
    report = {
        "track": "developer-documentation",
        "label": labels,
        "cases_file": cases_file_name,
        "strategy": strategy,
        "created_at": created_at.isoformat(),
        "summary": _summary(results),
        "results": results,
    }
    # strategy + a short random suffix keep filenames unique even when two
    # runs (e.g. baseline and improved) complete within the same second.
    output_path = RESULTS_DIR / f"{created_at.strftime('%Y%m%dT%H%M%S')}_{strategy}_{uuid4().hex[:8]}_{labels}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report, output_path


async def main() -> None:
    args = _parse_args()
    cases_path = Path(args.cases)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    label = args.label or cases_path.stem

    previous_run = _latest_previous_run(RESULTS_DIR)

    use_judge = not args.no_judge
    report, _ = await run_strategy(
        cases, args.strategy, use_judge, label, cases_file=cases_path.name,
        use_ragas=args.ragas, knowledge_base_id=args.knowledge_base_id,
    )

    if previous_run:
        before = previous_run["summary"]["combined_score"]
        after = report["summary"]["combined_score"]
        print(f"previous_run_label={previous_run.get('label', 'unknown')}")
        print(f"before_combined_score={before:.4f}")
        print(f"after_combined_score={after:.4f}")
        print(f"delta={after - before:+.4f}")
        before_types = previous_run["summary"].get("problem_type_scores", {})
        for problem_type, current in report["summary"]["problem_type_scores"].items():
            previous = before_types.get(problem_type, {}).get("combined_score", 0.0)
            print(f"{problem_type}_before={previous:.4f}")
            print(f"{problem_type}_after={current['combined_score']:.4f}")
            print(f"{problem_type}_delta={current['combined_score'] - previous:+.4f}")
    print("\nPass rate by mode:")
    for mode, stats in report["summary"]["mode_scores"].items():
        print(f"  {mode}: {stats['passed']}/{stats['cases']} passed ({stats['pass_rate']:.2%})")

    if args.ragas:
        faithfulness = report["summary"].get("faithfulness")
        context_precision = report["summary"].get("context_precision")
        print(f"\nAverage faithfulness: {faithfulness}")
        print(f"Average context precision: {context_precision}")
        # The bonus finding: an answer that's fully grounded in *some*
        # context (faithfulness >= 0.9) while that context was the wrong
        # document version for the question asked (mode == wrong_source).
        # The averages above can look fine while hiding exactly this.
        confidently_wrong = [
            r for r in report["results"]
            if r.get("mode") == "wrong_source" and (r.get("faithfulness") or 0) >= 0.9
        ]
        if confidently_wrong:
            print("\nConfidently-wrong candidates (high faithfulness, wrong-version mode):")
            for r in confidently_wrong:
                print(
                    f"  [{r['id']}] faithfulness={r['faithfulness']} "
                    f"context_precision={r.get('context_precision')} retrieval_score={r['retrieval_score']}"
                )

    print(json.dumps(report["summary"], indent=2))
    print(f"Saved report for strategy={report['strategy']}, label={report['label']}")


if __name__ == "__main__":
    asyncio.run(main())

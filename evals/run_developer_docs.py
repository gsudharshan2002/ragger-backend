"""Run the Week 6 developer-documentation evaluation set."""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.models.schemas import RagStrategy
from app.services.rag_engine import execute_rag
from evals.judge import judge_answer

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "developer_docs_cases.json"
RESULTS_DIR = ROOT / "results"


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


async def _run_case(case: dict[str, Any], strategy: str, use_judge: bool) -> dict[str, Any]:
    try:
        answer = ""
        trace: dict[str, Any] | None = None
        error = ""

        async for event in execute_rag(case["question"], RagStrategy(strategy), None, None):
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

        status = "passed" if combined_score == 1.0 and not error else "failed"
        return {
            "id": case["id"],
            "question": case["question"],
            "problem_type": case["problem_type"],
            "tags": case.get("tags", []),
            "status": status,
            "retrieval_score": round(retrieval_score, 4),
            "answer_score": round(answer_score, 4),
            "keyword_answer_score": round(keyword_answer_score, 4),
            "judge_verdict": judge_verdict,
            "judge_reasoning": judge_reasoning,
            "combined_score": round(combined_score, 4),
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
            "tags": case.get("tags", []),
            "status": "failed",
            "retrieval_score": 0.0,
            "answer_score": 0.0,
            "keyword_answer_score": 0.0,
            "judge_verdict": None,
            "judge_reasoning": "",
            "combined_score": 0.0,
            "expected_sources": case.get("expected_sources", []),
            "actual_sources": [],
            "expected_answer_keywords": case.get("expected_answer_keywords", []),
            "answer": "",
            "error": str(exc),
        }


def _average(results: list[dict[str, Any]], key: str) -> float:
    return round(sum(result[key] for result in results) / len(results), 4) if results else 0.0


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_type.setdefault(result["problem_type"], []).append(result)
    return {
        "retrieval_score": _average(results, "retrieval_score"),
        "answer_score": _average(results, "answer_score"),
        "combined_score": _average(results, "combined_score"),
        "problem_type_scores": {
            problem_type: {
                "cases": len(group),
                "combined_score": _average(group, "combined_score"),
            }
            for problem_type, group in sorted(by_type.items())
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
) -> tuple[dict[str, Any], Path]:
    """Run the developer-documentation eval set through a single retrieval
    strategy and persist a timestamped report to evals/results/.

    `cases_file` should be the name of the file the `cases` argument actually
    came from (e.g. the uploaded file's name) - it is stored as report
    metadata verbatim and must not be assumed to be developer_docs_cases.json.

    Returns (report dict, path the report was written to). Reused by the CLI
    (main) and the API endpoint that powers the Week 6 Refresh button so both
    produce identical reports.
    """
    cases_file_name = cases_file or CASES_PATH.name
    labels = label or Path(cases_file_name).stem
    RESULTS_DIR.mkdir(exist_ok=True)
    results = [await _run_case(case, strategy, use_judge) for case in cases]
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
    report, _ = await run_strategy(cases, args.strategy, use_judge, label, cases_file=cases_path.name)

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

    print(json.dumps(report["summary"], indent=2))
    print(f"Saved report for strategy={report['strategy']}, label={report['label']}")


if __name__ == "__main__":
    asyncio.run(main())

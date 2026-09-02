"""Run the Week 6 developer-documentation evaluation set."""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.models.schemas import RagStrategy
from app.services.rag_engine import execute_rag

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


async def _run_case(case: dict[str, Any], strategy: str) -> dict[str, Any]:
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
    retrieval_score = (
        sum(any(_source_matches(actual, expected) for actual in actual_sources) for expected in expected_sources)
        / len(expected_sources)
        if expected_sources
        else 0.0
    )
    actual_answer = (trace or {}).get("llm", {}).get("answer") or answer
    answer_score = _answer_score(actual_answer, case.get("expected_answer_keywords", []))
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
        "combined_score": round(combined_score, 4),
        "expected_sources": expected_sources,
        "actual_sources": actual_sources,
        "expected_answer_keywords": case.get("expected_answer_keywords", []),
        "answer": actual_answer,
        "error": error,
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
    parser.add_argument("--label", default="latest", choices=("baseline", "improved", "latest"))
    parser.add_argument("--strategy", default="bm25")
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    results = [await _run_case(case, args.strategy) for case in cases]
    report = {
        "track": "developer-documentation",
        "label": args.label,
        "strategy": args.strategy,
        "created_at": datetime.now(UTC).isoformat(),
        "summary": _summary(results),
        "results": results,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / f"{args.label}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.label == "latest":
        (RESULTS_DIR / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    baseline_path = RESULTS_DIR / "baseline.json"
    if args.label == "improved" and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        before = baseline["summary"]["combined_score"]
        after = report["summary"]["combined_score"]
        print(f"before_combined_score={before:.4f}")
        print(f"after_combined_score={after:.4f}")
        print(f"delta={after - before:+.4f}")
        before_types = baseline["summary"].get("problem_type_scores", {})
        for problem_type, current in report["summary"]["problem_type_scores"].items():
            previous = before_types.get(problem_type, {}).get("combined_score", 0.0)
            print(f"{problem_type}_before={previous:.4f}")
            print(f"{problem_type}_after={current['combined_score']:.4f}")
            print(f"{problem_type}_delta={current['combined_score'] - previous:+.4f}")

    print(json.dumps(report["summary"], indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())

"""Shared logic for blind hand-labeling of the developer-docs judge
validation - used by both the CLI (label_answers.py) and the /week-6
frontend labeling UI.

Keeping this in one place means both interfaces enforce the exact same
blind-by-construction guarantee: labels can only be written against a report
where the judge was never called, so there is nothing judge-derived to leak
into a human label.
"""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parent / "results"
LABELS_PATH = Path(__file__).resolve().parent / "labels_25.json"
CASES_PATH = Path(__file__).resolve().parent / "developer_docs_cases.json"

CRITERION = "correct and helpful (the judge's single binary pass/fail criterion)"

# Prevent concurrent label writes from clobbering each other when multiple
# browser tabs or users save labels at the same time.
_label_lock = asyncio.Lock()
# Same guard, for writes to developer_docs_cases.json (see clear_regression_flag).
_cases_lock = asyncio.Lock()


def find_latest_no_judge_report() -> dict[str, Any] | None:
    """Most recent report where the judge was never called - every result's
    judge_verdict is None. The only kind of report safe to label blind from."""
    candidates = []
    for path in RESULTS_DIR.glob("*.json"):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "created_at" not in report or not report.get("results"):
            continue
        if all(r.get("judge_verdict") is None for r in report["results"]):
            candidates.append((path, report))
    if not candidates:
        return None
    path, report = max(candidates, key=lambda pair: pair[1]["created_at"])
    report["_source_path"] = path.name
    return report


def load_labels_file() -> dict[str, Any] | None:
    if not LABELS_PATH.exists():
        return None
    return json.loads(LABELS_PATH.read_text(encoding="utf-8"))


def get_label_session() -> dict[str, Any]:
    """Everything a labeling UI (CLI or web) needs to render: which report is
    being labeled, its cases, any labels already saved, and whether the saved
    labels file conflicts with the current report (labeled against a
    different, older no-judge run)."""
    report = find_latest_no_judge_report()
    if not report:
        return {"available": False, "reason": "no_no_judge_report"}

    existing = load_labels_file()
    conflict = bool(existing) and existing.get("source_report") != report["_source_path"]

    cases = [
        {
            "id": c["id"],
            "question": c["question"],
            "answer": c.get("answer", ""),
            "mode": c.get("mode", "unknown"),
            "regression": bool(c.get("regression", False)),
            "regression_evidence": c.get("regression_evidence"),
        }
        for c in report["results"]
    ]
    return {
        "available": True,
        "source_report": report["_source_path"],
        "source_report_created_at": report.get("created_at", ""),
        "criterion": CRITERION,
        "cases": cases,
        "labels": {} if conflict else (existing or {}).get("labels", {}),
        "conflict": conflict,
        "conflict_report": (existing or {}).get("source_report") if conflict else None,
    }


async def run_judge_validation() -> dict[str, Any]:
    """Run the CURRENTLY active judge prompt against the exact frozen
    answers already hand-labeled in labels_25.json, and compute agreement.
    Whichever judge.py prompt is live when this is called determines
    whether the result is a "before" or "after" measurement - snapshot
    judge.py's prompt to judge_v1.txt/judge_v2.txt around whichever call you
    treat as that milestone.
    """
    from evals.judge import judge_answer

    labels_data = load_labels_file()
    if not labels_data:
        raise ValueError("No labels_25.json found - label answers before validating the judge.")

    report_path = RESULTS_DIR / labels_data["source_report"]
    if not report_path.exists():
        raise ValueError(f"Labeled report {labels_data['source_report']} no longer exists under evals/results/.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cases_by_id = {c["id"]: c for c in report["results"]}

    comparisons = []
    for case_id, human_label in labels_data["labels"].items():
        case = cases_by_id.get(case_id)
        if not case:
            continue
        expected_keywords = case.get("expected_answer_keywords", [])
        judge_result = await judge_answer(case["question"], case.get("answer", ""), expected_keywords)
        verdict = judge_result["verdict"]
        comparisons.append({
            "case_id": case_id,
            "question": case["question"],
            "answer": case.get("answer", ""),
            "mode": case.get("mode", "unknown"),
            "human_label": human_label,
            "judge_verdict": verdict,
            "judge_reasoning": judge_result["reasoning"],
            "agree": verdict == human_label,
            # No keyword checklist to grade against (retrieval-type cases never
            # get one - see developer_docs_cases.json) means the judge graded
            # "correct and helpful" with zero grounding. That comparison can't
            # validate the judge's use of expected_answer_keywords, so it's
            # tracked separately and excluded from agreement_rate below rather
            # than silently counted as if it were a grounded comparison.
            "has_keywords": bool(expected_keywords),
        })

    gradeable = [c for c in comparisons if c["has_keywords"]]
    ungrounded_count = len(comparisons) - len(gradeable)
    graded = [c for c in gradeable if c["judge_verdict"] is not None]
    agreement_rate = round(sum(1 for c in graded if c["agree"]) / len(graded), 4) if graded else 0.0

    validation = {
        "labels_source_report": labels_data["source_report"],
        "labeled_at": labels_data["labeled_at"],
        "validated_at": datetime.now(UTC).isoformat(),
        "criterion": labels_data["criterion"],
        "agreement_rate": agreement_rate,
        "graded_count": len(graded),
        "ungraded_count": len(gradeable) - len(graded),
        "ungrounded_count": ungrounded_count,
        "total_labels": len(comparisons),
        "comparisons": comparisons,
    }
    output_path = RESULTS_DIR / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}_judge_validation.json"
    output_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return validation


async def clear_regression_flag(case_id: str) -> dict[str, Any] | None:
    """Demote a case in developer_docs_cases.json back to a normal case by
    stripping its regression/regression_evidence fields. Called once a
    regression case's hand label turns to "pass", confirming the fix holds -
    the case keeps being tracked, it's just no longer an active known
    failure. Returns the (possibly unchanged) case, or None if case_id
    doesn't exist in the file.

    Uses an async lock for the same reason save_label does: prevent
    concurrent writes to the file from clobbering each other.
    """
    async with _cases_lock:
        cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
        for case in cases:
            if case.get("id") == case_id:
                case.pop("regression", None)
                case.pop("regression_evidence", None)
                CASES_PATH.write_text(json.dumps(cases, indent=2), encoding="utf-8")
                return case
        return None


async def save_label(source_report: str, source_report_created_at: str, case_id: str, label: str) -> dict[str, Any]:
    """Add or update one label. Refuses if labels_25.json already exists
    against a DIFFERENT report than `source_report`, to avoid silently
    mixing labels from two different answer sets.

    Uses an async lock to prevent concurrent writes from clobbering each
    other when multiple browser tabs or users save labels simultaneously."""
    if label not in ("pass", "fail"):
        raise ValueError(f"label must be 'pass' or 'fail', got {label!r}")

    async with _label_lock:
        existing = load_labels_file()
        if existing and existing.get("source_report") != source_report:
            raise ValueError(
                f"labels_25.json already exists against a different report "
                f"({existing.get('source_report')}) than {source_report}. "
                f"Delete labels_25.json first to relabel against a new report."
            )

        labels = dict((existing or {}).get("labels", {}))
        labels[case_id] = label

        data = {
            "source_report": source_report,
            "source_report_created_at": source_report_created_at,
            "criterion": CRITERION,
            "labeled_at": datetime.now(UTC).isoformat(),
            "labels": labels,
        }
        LABELS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data

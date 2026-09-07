"""Validate the developer-docs LLM judge against your own human grading.

An AI judge you never checked is just a confident number nobody trusts. Run
this after evals.run_developer_docs to grade the same answers yourself and
see how often the judge agrees with you, before relying on its score.

Usage:
    uv run python -m evals.validate_judge
    uv run python -m evals.validate_judge --run evals/results/<file>.json
    uv run python -m evals.validate_judge --human-grades evals/my_grades.json
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _latest_run(results_dir: Path) -> dict[str, Any] | None:
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


def _prompt_human_grade(question: str, answer: str) -> str:
    print("\n" + "=" * 60)
    print(f"Q: {question}")
    print(f"A: {answer or '(empty)'}")
    while True:
        choice = input("Your grade - correct and helpful? [p]ass / [f]ail: ").strip().lower()
        if choice in ("p", "pass"):
            return "pass"
        if choice in ("f", "fail"):
            return "fail"
        print("Please enter 'p' or 'f'.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", default=None,
        help="Path to a specific run report to validate (defaults to the most recent run under evals/results/).",
    )
    parser.add_argument(
        "--human-grades", default=None,
        help="Path to a JSON file mapping case id -> 'pass'/'fail', to skip interactive prompting.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.run:
        report = json.loads(Path(args.run).read_text(encoding="utf-8"))
    else:
        report = _latest_run(RESULTS_DIR)
        if not report:
            raise SystemExit("No eval runs found under evals/results/. Run `evals.run_developer_docs` first.")

    cases = [r for r in report["results"] if r.get("judge_verdict") is not None]
    if not cases:
        raise SystemExit(
            "This run has no judge verdicts to validate - re-run "
            "`evals.run_developer_docs` without --no-judge first."
        )

    if args.human_grades:
        human_grades: dict[str, str] = json.loads(Path(args.human_grades).read_text(encoding="utf-8"))
    else:
        print(f"Grading {len(cases)} answers from run '{report.get('label', 'unknown')}' ({report['created_at']}).")
        print("Grade each answer yourself first - the judge's verdict is hidden until the end, to avoid bias.")
        human_grades = {case["id"]: _prompt_human_grade(case["question"], case["answer"]) for case in cases}

    comparisons = []
    agreements = 0
    for case in cases:
        human = human_grades.get(case["id"])
        if human not in ("pass", "fail"):
            continue
        judge = case["judge_verdict"]
        agree = human == judge
        agreements += agree
        comparisons.append({
            "id": case["id"],
            "question": case["question"],
            "human_grade": human,
            "judge_verdict": judge,
            "judge_reasoning": case.get("judge_reasoning", ""),
            "agree": agree,
        })

    if not comparisons:
        raise SystemExit("No cases had a human grade to compare against the judge.")

    agreement_rate = round(agreements / len(comparisons), 4)
    validation_report = {
        "validated_run_label": report.get("label", "unknown"),
        "validated_run_created_at": report.get("created_at", ""),
        "created_at": datetime.now(UTC).isoformat(),
        "agreement_rate": agreement_rate,
        "graded_cases": len(comparisons),
        "comparisons": comparisons,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / f"judge_validation_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}.json"
    output_path.write_text(json.dumps(validation_report, indent=2), encoding="utf-8")

    print(f"\nAgreement rate: {agreement_rate:.2%} ({agreements}/{len(comparisons)} cases)")
    if agreement_rate >= 0.8:
        print("The judge agrees with your grading often enough to trust its score.")
    else:
        print("The judge disagrees with your grading too often - don't trust its score yet. Mismatches:")
        for c in comparisons:
            if not c["agree"]:
                print(f"  - [{c['id']}] human={c['human_grade']} judge={c['judge_verdict']} ({c['judge_reasoning']})")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

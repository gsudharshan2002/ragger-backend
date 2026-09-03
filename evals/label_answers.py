"""Blind hand-labeling tool for validating the developer-docs judge.

Task Set E requires 25+ hand labels that PROVABLY predate any judge run on
the same answers. The only way to guarantee that structurally - not just by
discipline - is to label answers from a report generated with --no-judge:
such a report has no judge_verdict anywhere in it to accidentally see,
because the judge was never called to produce it. See evals/labeling.py for
the shared logic this also backs the /week-6 frontend labeling UI with.

Usage:
    uv run python -m evals.run_developer_docs --no-judge   # freeze answers first
    uv run python -m evals.label_answers                   # then label them blind
"""

from evals.labeling import LABELS_PATH, get_label_session, save_label


def _prompt_label(index: int, total: int, question: str, answer: str) -> str:
    print("\n" + "=" * 60)
    print(f"[{index}/{total}]")
    print(f"Q: {question}")
    print(f"A: {answer or '(empty)'}")
    while True:
        choice = input("Correct and helpful? [p]ass / [f]ail / [s]kip: ").strip().lower()
        if choice in ("p", "pass"):
            return "pass"
        if choice in ("f", "fail"):
            return "fail"
        if choice in ("s", "skip"):
            return "skip"
        print("Please enter 'p', 'f', or 's'.")


def main() -> None:
    session = get_label_session()
    if not session["available"]:
        raise SystemExit(
            "No --no-judge report found under evals/results/. Run:\n"
            "  uv run python -m evals.run_developer_docs --no-judge\n"
            "first, so there is a judge-free report to label blind from."
        )
    if session["conflict"]:
        raise SystemExit(
            f"{LABELS_PATH.name} already exists and was labeled against a DIFFERENT "
            f"report ({session['conflict_report']}) than the current one "
            f"({session['source_report']}). Delete {LABELS_PATH.name} first if you "
            f"really want to restart against the new report - mixing labels from "
            f"two different answer sets isn't valid."
        )

    cases = session["cases"]
    labels = dict(session["labels"])
    if labels:
        print(f"Resuming: {len(labels)} labels already saved from a previous session.")

    print(f"Labeling {len(cases)} answers from report '{session['source_report']}' "
          f"(generated {session['source_report_created_at']}).")
    print("This report was generated with --no-judge, so there is no judge verdict "
          "anywhere in it to see - you are grading blind by construction, not just "
          "by discipline.")
    print(f"Criterion: {session['criterion']}")

    for i, case in enumerate(cases, start=1):
        if case["id"] in labels:
            continue
        result = _prompt_label(i, len(cases), case["question"], case["answer"])
        if result == "skip":
            continue
        data = save_label(session["source_report"], session["source_report_created_at"], case["id"], result)
        labels = data["labels"]

    print(f"\nSaved {len(labels)} labels to {LABELS_PATH}")
    if len(labels) < 25:
        print(f"Note: only {len(labels)} labels saved so far - the rubric wants 25+.")


if __name__ == "__main__":
    main()

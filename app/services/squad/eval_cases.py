"""Week 10: 10 evaluation cases selected from evals/developer_docs_cases.json.

Selected deterministically to preserve existing case identifiers and include
all wrong_source cases (gh-002, gh-005, gh-010). All cases are GitHub API
version-specific questions that exercise the version/deprecation worker and
code-sample worker paths.
"""

from pathlib import Path
import json

DATASET_PATH = Path(__file__).resolve().parents[3] / "evals" / "developer_docs_cases.json"

SELECTED_CASE_IDS = [
    "gh-001",
    "gh-002",
    "gh-003",
    "gh-004",
    "gh-005",
    "gh-006",
    "gh-007",
    "gh-008",
    "gh-009",
    "gh-010",
]


def load_selected_cases() -> list[dict]:
    """Load the 10 selected evaluation cases from the dataset."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    by_id = {c["id"]: c for c in all_cases}
    return [by_id[cid] for cid in SELECTED_CASE_IDS if cid in by_id]


WRONG_SOURCE_CASES = {"gh-002", "gh-005", "gh-010"}

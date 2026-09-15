"""Task Set E, Step 3: the 10-question race set.

Pulled from evals/developer_docs_cases.json - real indexed content, real
expected_answer_keywords, not invented. 5 straightforward (single retrieve
suffices) + 5 requires_deprecation_check=True (well above the rubric's
"at least 3" floor), so the race isn't decided by the input mix - see
that file's own warning about single-page-lookup-only test sets.
"""

RACE_QUESTIONS = [
    # --- Straightforward: single retrieve suffices, no version/deprecation angle ---
    {
        "id": "combine-001",
        "question": "What does the Combine framework provide?",
        "requires_deprecation_check": False,
        "expected_keywords": ["declarative", "swift", "over time"],
    },
    {
        "id": "combine-005",
        "question": "What is AnyPublisher in Combine?",
        "requires_deprecation_check": False,
        "expected_keywords": ["type erasure", "wrap"],
    },
    {
        "id": "gh-001",
        "question": "In the current GitHub API version, what field should you read to get your core rate limit information?",
        "requires_deprecation_check": False,
        "expected_keywords": ["resources.core", "core"],
    },
    {
        "id": "gh-004",
        "question": "On the current API version, what value does a submodule entry report for its type field when listing repository contents?",
        "requires_deprecation_check": False,
        "expected_keywords": ["submodule"],
    },
    {
        "id": "gh-009",
        "question": "What HTTP status code does the current API version return on a successful workflow dispatch request, and what is in the response body?",
        "requires_deprecation_check": False,
        "expected_keywords": ["200", "workflow run"],
    },
    # --- Cross-dependent: correct answer depends on what check_deprecation reveals ---
    {
        "id": "gh-002",
        "question": "Using API Version 2026-03-10, does the GET /rate_limit response still include a top-level rate field?",
        "requires_deprecation_check": True,
        "expected_keywords": ["removed", "resources.core"],
    },
    {
        "id": "gh-003",
        "question": "If my integration currently reads the top-level rate field from /rate_limit, what should I change to keep working on the current API version?",
        "requires_deprecation_check": True,
        "expected_keywords": ["resources.core", "migrat"],
    },
    {
        "id": "gh-006",
        "question": "Does the current version of GET /repos/{owner}/{repo} still include a has_downloads field?",
        "requires_deprecation_check": True,
        "expected_keywords": ["removed"],
    },
    {
        "id": "gh-007",
        "question": "My code reads pull_request.merge_commit_sha after a merge completes. Will that still work on the current API version, and if not what should I do instead?",
        "requires_deprecation_check": True,
        "expected_keywords": ["removed", "commit history"],
    },
    {
        "id": "gh-013",
        "question": "What changed about attestation list responses between API Version 2022-11-28 and 2026-03-10?",
        "requires_deprecation_check": True,
        "expected_keywords": ["bundle", "bundle_url"],
    },
]
"""Task Set E, Step 3: the race question set.

Kept identical (same 10 ids, same question text) to evals/trajectory_eval.py's
TRAJECTORY_CASES, so the race harness (Week 7) and the trajectory eval (Week 8)
grade the agent/workflow against the exact same 10 questions - the two eval
frameworks measure different things (pass/fail vs. tool-path/step-efficiency)
but should never disagree about WHICH questions are in scope. If you add,
remove, or reword a case in one file, mirror it in the other.

Covers three reference docs (github_combined_reference.pdf,
swagger_v2_reference.pdf, swagger_v3_reference.pdf) plus one deliberate
negative/refusal case. Kept the one case that caught a real product bug during
a live run (11's hallucination-off-a-decoy), so the set still has diagnostic
value, not just easy wins.

expected_keywords notes (see race_harness.py's _passed - plain, case-
insensitive, ALL-must-match substring check, no OR/negation support):
  - Where the literal answer is a generic-looking token (true/false/a bare
    version number), the keyword pairs the FIELD NAME with the VALUE (e.g.
    case 11's ["allow", "unicode", "identifiers", "false"]) instead of just
    the value alone - a bare "true" or "false" would pass on nearly any
    unrelated answer.
  - A camelCase field name is only kept as ONE merged token when it reads as
    an unmistakable code identifier the model is likely to quote verbatim.
    When it reads as two or more ordinary English words squashed together
    (packageVersion, allowUnicodeIdentifiers), split it into separate word
    keywords instead (["package", "version"], ["allow", "unicode",
    "identifiers"]) - a real run showed the model naturally writing "package
    version" (with a space) in prose, which a merged "packageversion" token
    never matches, wrongly failing an otherwise-correct answer.
  - Case 07's expected answer is "'public_repo' OR 'repo'"; since the
    checker is AND-only, the keyword is just "repo" (a substring of
    "public_repo" too, so either valid answer passes).
  - Case 11 (allowUnicodeIdentifiers) is a deliberate trap: the doc has a
    decoy chunk mentioning "(Default: true)" in an unrelated v2-vs-v3
    formatting comparison, right next to the real options table stating
    the actual default is false. A real run caught the workflow
    hallucinating "true" from the decoy - this case is intentionally kept
    hard, not something to loosen.
  - Case 13 (negative/refusal) is "expected_behavior: refusal_or_fallback"
    in the source data - EITHER the LLM's crafted refusal ("I could not
    find that information in the provided documents.", from
    settings.SYSTEM_PROMPT in app/core/config.py) OR the agent's generic
    budget-exhaustion fallback ("I could not find a sufficient answer
    within the allotted budget.") should count as correct - the point is
    not hallucinating, not the exact wording. A real run showed the agent
    hit the second form, so the keyword is the substring both messages
    share: "could not find" (not the fuller phrase - that only matched the
    first form and wrongly failed the second).
"""

RACE_QUESTIONS = [
    # --- github_combined_reference.pdf ---
    {
        "id": "01",
        "question": "What is the default value for the mode parameter when rendering a Markdown document in GitHub API version 2025-06-01?",
        "requires_deprecation_check": False,
        "expected_keywords": ["mode", "markdown"],
    },
    {
        "id": "07",
        "question": "Which OAuth scopes are required to create a repository for an authenticated user using the GitHub REST API?",
        "requires_deprecation_check": False,
        "expected_keywords": ["scope", "repo"],
    },
    {
        "id": "12",
        "question": "What is the HTTP success response status code returned when creating a pull request in the GitHub REST API?",
        "requires_deprecation_check": False,
        "expected_keywords": ["201", "created"],
    },
    {
        "id": "04",
        "question": "What is the maximum file size limit supported when rendering Markdown content via the GitHub REST API?",
        "requires_deprecation_check": False,
        "expected_keywords": ["400", "kb"],
    },
    {
        "id": "09",
        "question": "What is the default value of the draft parameter when creating a pull request via the GitHub REST API?",
        "requires_deprecation_check": False,
        "expected_keywords": ["draft", "false"],
    },
    # --- swagger_v2_reference.pdf ---
    {
        "id": "10",
        "question": "What default value does Swagger Codegen 2.x assign to the packageName option in the Python generator?",
        "requires_deprecation_check": False,
        "expected_keywords": ["swagger_client"],
    },
    # --- swagger_v3_reference.pdf ---
    {
        "id": "02",
        "question": "What Maven group id is used to download the Swagger Codegen CLI jar for version 3.x?",
        "requires_deprecation_check": False,
        "expected_keywords": ["io.swagger.codegen.v3"],
    },
    {
        "id": "05",
        "question": "What is the default package version string generated by the Python generator in Swagger Codegen v3?",
        "requires_deprecation_check": False,
        "expected_keywords": ["package", "version", "1.0.0"],
    },
    {
        "id": "11",
        "question": "What is the default value of allowUnicodeIdentifiers in the Swagger Codegen 3.x Java generator?",
        "requires_deprecation_check": False,
        "expected_keywords": ["allow", "unicode", "identifiers", "false"],
    },
    # --- Negative case: nothing in any indexed doc covers this; correct
    # behavior is an explicit refusal, not a guess. ---
    {
        "id": "13",
        "question": "What is the rate limit for unauthenticated GraphQL queries in GitHub API version 2025-06-01?",
        "requires_deprecation_check": False,
        "expected_keywords": ["could not find"],
    },
]

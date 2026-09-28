"""Task Set E, Step 3: the race question set.

A curated 10-question subset of the original hand-authored 15 (see git
history / prior conversation for the full set), covering three reference
docs (github_combined_reference.pdf, swagger_v2_reference.pdf,
swagger_v3_reference.pdf). Trimmed for diversity, not just size: dropped
near-duplicate "default value" lookups that exercised the same question
shape as one already kept, and 2 of 3 negative/refusal cases that all
tested identical refusal behavior. Kept both cases that caught real product
bugs during live runs (03's retrieval gap, 11's hallucination-off-a-decoy),
so the set still has diagnostic value, not just easy wins.

expected_keywords notes (see race_harness.py's _passed - plain, case-
insensitive, ALL-must-match substring check, no OR/negation support):
  - Where the literal answer is a generic-looking token (true/false/a bare
    version number), the keyword pairs the FIELD NAME with the VALUE (e.g.
    ["sortparamsbyrequiredflag", "true"]) instead of just the value alone -
    a bare "true" or "1.0.0" would pass on nearly any unrelated answer.
  - A camelCase field name is only kept as ONE merged token (e.g.
    "sortparamsbyrequiredflag") when it reads as an unmistakable code
    identifier the model is likely to quote verbatim. When it reads as two
    or more ordinary English words squashed together (packageVersion,
    allowUnicodeIdentifiers), split it into separate word keywords instead
    (["package", "version"], ["allow", "unicode", "identifiers"]) - a real
    run showed the model naturally writing "package version" (with a
    space) in prose, which a merged "packageversion" token never matches,
    wrongly failing an otherwise-correct answer.
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
    # --- swagger_v2_reference.pdf ---
    {
        "id": "03",
        "question": "What is the default library template used by the Java client generator in Swagger Codegen 2.x?",
        "requires_deprecation_check": False,
        "expected_keywords": ["okhttp-gson"],
    },
    {
        "id": "06",
        "question": "What is the default value of sortParamsByRequiredFlag in the Swagger Codegen 2.x Java client generator?",
        "requires_deprecation_check": False,
        "expected_keywords": ["sortparamsbyrequiredflag", "true"],
    },
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

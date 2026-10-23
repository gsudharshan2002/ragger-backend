# Week 11: Support Drill — Developer Documentation Track

## Scenario

Find the code sample that used a deprecated method.

## Incident (synthetic fixture — explicitly labeled)

This is a **synthetic, developer-constructed incident fixture**, NOT an actual
customer incident. It was created to exercise the trace lookup and debugging
workflow and is labeled `synthetic: true` in the trace record.

- Target incident identifier: `week11-incident-deprecated-assignee-001`
- Case: `gh-deprecated-001`
- Incident type: `deprecated_method`
- Question: "When creating a GitHub issue via POST /repos/{owner}/{repo}/issues,
  which field assigns users?"
- Bad answer: "Use the `assignee` field (string) to assign a user when creating
  an issue via POST /repos/{owner}/{repo}/issues."
- Correct replacement: `assignees` (array of strings). `assignee` is deprecated.
- Deprecated method: `assignee`
- Trace file: `artifacts/week11/trace.json` (complete exported trace)

## Timed incident search

| Field | Value |
|-------|-------|
| Search start | 2026-10-16T09:05:16.423801+00:00 |
| Search finish | 2026-10-16T09:05:16.578+00:00 |
| Elapsed time | 00:00 (mm:ss) — ~0.15s measured |
| Five-minute target met | Yes (well under 5 minutes) |

## Exact search method / log slice

Used the implemented trace search mechanism
(`app/services.storage.search_traces`, exposed via `GET /api/v1/traces/search`
and `scripts/trace_search.py`).

Filter applied:
- `trace_id = week11-incident-deprecated-assignee-001`

The trace was located immediately by its unique trace ID. Alternative
successful lookups that also return the incident:
- `chunk_id = 4704` (the retrieved chunk containing the deprecated `assignee`
  reference) — returns the incident plus other traces referencing that chunk.
- `prompt_version = 1` (returns many traces; incident among them).
- `error = false` + status `completed` (broad).

## Fields that enabled the lookup

- `trace_id` (unique, primary key) — instant lookup.
- `chunk_id` within `sources[*].chunk_id` — content-based lookup.
- `prompt_version` — version-based filtering.
- `status` / `error` — outcome-based filtering.

## Root-cause evidence

Inspecting the complete trace (`artifacts/week11/trace.json`):

1. **Retrieval** surfaced chunk `4704` (old-version 2022-11-28 documentation)
   describing the deprecated `assignee` field, plus chunk `5262` noting the
   prior version "only accepted a single assignee (string)".
2. **Prompt version 1** had no instruction prohibiting deprecated-method
   recommendations.
3. **Model generation** (span `model.generation`, 8,420 ms) produced an answer
   recommending `assignee` as the current implementation instead of the
   `assignees` replacement.
4. **Root cause**: outdated retrieval surfaced pre-deprecation documentation
   and the prompt (v1) lacked a deprecation-guard rule, so the model recommended
   the deprecated method as the current implementation.

## Result

The bad answer was located and confirmed via the implemented search mechanism
in ~0.15s, meeting the five-minute target. The incident is recorded as
synthetic (not an actual customer incident).

## Human verification status

Pending. A human squadmate should verify the fixture and the search procedure
by running:

```
python scripts/trace_search.py --trace-id week11-incident-deprecated-assignee-001
```

and confirming the returned trace's `answer` recommends the deprecated
`assignee` field and its `deprecated_method`/`correct_replacement` fields match.

## Deployment and rollback plan (two-line)

Canary: deploy prompt v2 (deprecation guard) to a small traffic percentage and
compare the deprecated-method error rate, correctness, latency, and cost
against the v1 baseline before broader rollout.
Rollback: if regression thresholds are breached, revert `SYSTEM_PROMPT` and
`PROMPT_VERSION` to v1 while preserving all traces for investigation.

Regression thresholds (to be set once baseline measurements are stable):
- Deprecated-method recommendation rate must not increase.
- Correctness (expected-keyword pass rate) must not decrease by >2 points.
- p95 latency and cost per request must not increase by >10%.

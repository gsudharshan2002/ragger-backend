# Week 11: Support drill, regression result and deployment plan

**Everything below comes from real runs on 2026-10-23** (live model `openai/gpt-oss-20b` via NVIDIA, BM25 retrieval,
real knowledge base: `current-version.md`, `old-version.md`, Apple Combine PDF, Swagger PDFs). The earlier
synthetic-fixture version of these files is preserved in `synthetic_fixture/` and is **not** used here.
This is a developer-run experiment, not a customer incident, and no human has verified the drill yet.

## 1. The incident (real, reproduced live)

- Question (case `gh-deprecated-001`): *"When creating a GitHub issue via POST /repos/{owner}/{repo}/issues, which field assigns users instead of the deprecated assignee field?"*
- Bad answer: **"I could not find that information in the provided documents."**
- Trace: `c25a04e0-c4c1-49f0-9e57-37f3e54f3722` (prompt v1, 147 s, 5 model calls, 3 retrievals). Full trace: `trace.json`.

## 2. Search (measured)

| | Filters | Candidates returned | Wall time (CLI) |
|---|---|---:|---:|
| Old fields only | `--prompt-version 1 --mode agent` | 5 traces to open and read | 0.53 s |
| New text filter | `--query-contains assignee --answer-contains "could not find" --prompt-version 1` | **1 trace (the bad answer)** | 0.47 s |

Command: `python scripts/trace_search.py --query-contains assignee --answer-contains "could not find" --prompt-version 1`

- **What was missing before:** the search could not filter on the question or answer text, so "find the bad answer" meant opening every candidate. Added `query_contains` / `answer_contains` (storage, `GET /api/v1/traces/search`, CLI) with tests.
- **What else was missing:** agent runs saved no `id`, `timestamp`, `status`, `llm`, `cost_usd` or spans, so they could not be found by trace id at all. Fixed; every agent and squad run is now one stored trace (`tests/test_agent_tracing.py`).
- Wall time includes Python start-up; the lookup itself is a scan of about 1,030 JSONL records. Elapsed time is under 1 s, far inside the 5-minute target. **Human-run timed drill: pending.** A teammate should run the command above and confirm the returned trace's answer is the refusal.

## 3. Root cause (from the trace)

- Retrieval worked: the same chunks (467, 1121, 5463, 2612) were retrieved in the failing and the passing run.
- Docs mismatch: `current-version.md` documents `assignees` under **GET** issue ("read and write assignees exclusively through the `assignees` array"). The question asks about **POST** create. The evidence is only partly on point.
- The answer step is non-deterministic: same context, sometimes a correct answer and sometimes the refusal.
- **The expected failure (recommending the deprecated field) did not occur.**

## 4. RED → GREEN, honestly (`live_regression.json`, `live_regression.log`)

Live runs of the real agent, prompt v1 (no deprecation guard) vs v2 (guard added), n = 5 each:

| | Correct | Recommended deprecated `assignee` | Refused ("could not find") |
|---|---:|---:|---:|
| v1 baseline | 4 / 5 | 0 | 1 |
| v2 (with guard) | 3 / 5 | 0 | 2 |

- The prompt-v2 change is **not shown to help**: the live model never recommended the deprecated field under v1, so there was no live RED to fix. The 4/5 vs 3/5 gap is within noise at n = 5.
- `tests/test_week11_regression.py` is a **deterministic prompt-presence check** (does the prompt contain the guard, does the context hold the replacement). It passes, but it does not prove the model behaves better.
- Fix applied for a failure that *was* observed live (trace `e3892bc8…`, `trace_wasted_vector_step.json`): the agent chose the `vector` strategy, which returns nothing when no embedding provider works, wasting a step. Pure `vector` now falls back to BM25 when vector search is unavailable (`tests/test_agent_tracing.py`). Its effect on the live numbers is not measured.

## 5. Deployment and rollback plan

- **Canary:** serve prompt v2 to a small traffic share and compare against v1 on refusal rate, deprecated-field recommendation rate, correctness, p99 latency and cost per request.
- **Rollback:** if any threshold below is breached, set `SYSTEM_PROMPT`/`PROMPT_VERSION` back to v1 and keep all traces.
- Thresholds (baseline is small, n = 5, so treat as provisional): refusal rate must not exceed baseline by more than 20 points; deprecated-field recommendations must stay at 0; cost and p99 must not rise more than 10%.
- No canary has been run.

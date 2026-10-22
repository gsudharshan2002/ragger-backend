# Week 10 & Week 11 — Interview Q&A (Concepts)

Multi-Agent RAG Squad, observability, cost attribution, and failure-to-test loops.

---

## Week 10 — Multi-Agent Squad vs Single Agent

### Q: What architecture did you implement for the multi-agent squad?
**A:** A Manager-Plus-Two-Specialists architecture:
- **Orchestrator (Manager):** analyzes the question, delegates to specialists, collects results, combines evidence into a final answer, and handles specialist errors without inventing facts. It tracks delegation hops, latency, token usage, and failures, and enforces timeouts/retry limits.
- **Version / Deprecation Worker:** resolves SDK/package versions, checks deprecation via existing `check_deprecation` retrieval tools, returns evidence + source references + an explicit uncertainty/error state.
- **Code-Sample Worker:** retrieves documentation and finds code examples compatible with the requested SDK version, returning samples with source evidence and reporting unverified assumptions.

Coordination is simple and explicit — no autonomous planning framework.

### Q: How is the squad integrated without breaking the existing single-agent baseline?
**A:** The existing `/api/v1/agent/run` single-agent endpoint and `AgentEngine` are preserved unchanged. The squad is added as a separate, additive endpoint `POST /api/v1/squad/run` (wired into `app/api/v1/router.py`). Both arms receive identical questions and input context for a fair comparison.

### Q: Why must the orchestrator not claim version/deprecation verification unless evidence exists?
**A:** To prevent hallucination. If the version/deprecation worker fails or returns no evidence, the orchestrator must disclose missing verification rather than stating a version is current/deprecated without a source. This is a correctness/safety guard.

### Q: What telemetry did you collect for both arms?
**A:** Per case and aggregate:
- **Pass rate** = passed / 10 × 100 (keyword-based rubric matching expected answer keywords).
- **p50 / p99 latency** (wall-clock per question, nearest-rank percentile; p99 of 10 samples is the max).
- **Total token consumption** (input + output), distinguished as estimated (word-count heuristic) vs actual provider-reported.
- **Cost per question** (tokens × `settings.COST_PER_TOKEN`, labeled estimated).
- **Delegation hops** for the squad, recorded in `handoffs.log` (JSON Lines) with timestamp, case ID, source/destination agent, purpose, input/output/total tokens, latency, status, retry count, error.

### Q: What is the Context Re-Send Multiplier?
**A:** Multi-Agent Total Tokens / Single-Agent Total Tokens for the same 10 cases, rounded to one decimal. It quantifies how much extra token traffic the squad generates versus the single agent. If the single-agent denominator is 0, it's reported as unavailable.

### Q: What is the Largest Handoff Bottleneck?
**A:** The single inter-agent hop with the highest token consumption, identified from the actual `handoffs.log` (not hardcoded). Reported as source → destination, absolute tokens, % of squad total tokens, measured vs estimated.

### Q: How does controlled HTTP 500 fault injection work?
**A:** Env-gated (`RAGGER_FAULT_INJECTION=1` + `RAGGER_FAULT_CASE_ID=<case>`). It's off by default (never affects production) and forces the version/deprecation worker to return an HTTP 500 for exactly one case. The orchestrator must catch it, disclose the failed verification, and degrade gracefully (partial answer + disclosure, or safe failure) — never fabricate version claims or code samples.

### Q: What are the Week 10 artifacts?
**A:** `artifacts/week10/race_table.md`, `handoffs.log`, `failure_case.md`, `verdict.md`, `agentcard.json`, `benchmark_details.csv`. Reproduced via `scripts/run_week10_experiment.py`.

### Q: What is the verdict rule?
**A:** KEEP if the squad meaningfully improves pass rate; KILL if it lowers pass rate or adds cost/latency without measurable correctness gain. Sunk-cost bias is explicitly rejected — the decision rests only on measured results.

### Q: What is the AgentCard bonus?
**A:** A machine-readable `agentcard.json` describing the orchestrator's identity, capabilities, input/output interfaces, worker delegation, authentication (none claimed), endpoint, and MCP servers. Plus an A2A-style lifecycle mapping for the HTTP 500 failure and a two-line protocol-value comparison.

---

## Week 11 — Production Observability, Cost & Failure-to-Test

### Q: What request-level fields does Week 11 observability capture?
**A:** Request ID, trace ID, timestamp, duration, endpoint, execution mode, sanitized input, outcome status (success/error/timeout/partial), prompt version, model provider + model ID, retrieved document/chunk IDs, retrieval config, total input/output tokens, total cost USD, error type + sanitized details, agent/worker identity, cache hit/miss. Every request has a trace ID connecting it to its execution steps.

### Q: What span-level observability is recorded?
**A:** Distinct spans for retrieval, reranking, tool execution, model generation, agent handoffs, and response assembly. Each span records trace ID, span ID, parent span ID, stage, start time, duration, success/failure, input/output tokens, cost USD, context IDs, tool name/outcome, retry count, and error.

### Q: Where are traces stored and how are they made searchable?
**A:** Reused `app/data/traces.jsonl` with additive fields (`spans`, `prompt_version`, `cost_usd`, `cache_hit`, `input_tokens`, `output_tokens`) so existing consumers don't break. `app/services/storage.py` provides `search_traces(filters)` supporting trace_id, request_id, status, prompt_version, model, mode, chunk_id, error, latency/cost outliers, and time range. Exposed via `GET /api/v1/traces/search` and `GET /{trace_id}/export`, plus `scripts/trace_search.py` and `scripts/trace_export.py`.

### Q: How is logging failure isolated from user requests?
**A:** Trace persistence in `_finalize_trace` is wrapped in try/except; a logging failure never breaks a successful user request. Structured JSON logs; no API keys, auth headers, or personal info stored.

### Q: What was the support-drill incident?
**A:** A labeled **synthetic** fixture: trace `week11-incident-deprecated-assignee-001`, question about assigning users to a GitHub issue. The model answered with the deprecated `assignee` field instead of the current `assignees` replacement. Root cause: retrieval surfaced outdated 2022-11-28 docs and prompt v1 had no deprecation-guard rule.

### Q: How was the drill timed and did it meet the target?
**A:** Search by `trace_id` located the incident in ~0.15s (well under the 5-minute target). The trace was exported to `artifacts/week11/trace.json`. Human verification is pending.

### Q: Describe the RED → GREEN failure-to-test loop.
**A:**
1. **Case:** Added permanent regression case `gh-deprecated-001` to `evals/developer_docs_cases.json` + `tests/test_week11_regression.py`.
2. **RED:** Ran pre-fix — 1 failed (`system_prompt_prohibits_deprecated_methods`: prompt v1 lacked deprecation guard), 3 passed. Output saved to `artifacts/week11/red_output.log`.
3. **Fix:** Prompt v2 added deprecation-guard rule ("Do not recommend deprecated methods… use current replacement") and `PROMPT_VERSION` bumped to "2".
4. **GREEN:** Post-fix — 4 passed, 0 failed. Output saved to `artifacts/week11/green_output.log`. No regressions in existing cases.

### Q: What is the cost attribution by stage?
**A:** For the incident trace:
- **Retrieval** (reranker): $0.000080 (120 in / 40 out tokens)
- **Generation** (LLM): $0.000575 (940 in / 210 out tokens)
- **Tool executions:** $0 (none distinct from retrieval/generation)
- **Total:** $0.000655 estimated (no provider-reported usage available; labeled estimated)
Full detail in `artifacts/week11/cost_by_stage.md`.

### Q: What cost improvement was implemented and measured?
**A:** Retrieval caching (`app/services/retrieval_cache.py`) — caches final chunks keyed by (query, strategy, kb_id). Cache hits skip the reranker/embedding calls. Baseline $0.000655 → cache-hit $0.000575 (−12.2%). Cached chunks retain source/version metadata so caching cannot bypass version/freshness/access constraints.

### Q: What is the 10× scaling bottleneck?
**A:** LLM provider API rate limits (requests-per-minute / concurrent connections / TPM), not cost ($0.000655/request is negligible) or p99 latency (moderate p50; p99 skewed by outliers). Evidence: per-request cost is tiny, every request makes ≥1 LLM call, so provider rate caps saturate first under 10× load. Mitigations: retrieval caching, rate-limit-aware backoff, concurrency caps, cheaper-model routing. Documented in `artifacts/week11/tenx.md` (no uncontrolled production load test executed).

### Q: What is the deployment and rollback plan?
**A:** Two-line plan in `drill.md`:
- **Canary:** deploy prompt v2 to a small traffic %, compare deprecated-method error rate, correctness, latency, cost vs v1 baseline.
- **Rollback:** revert `SYSTEM_PROMPT` / `PROMPT_VERSION` to v1 if regression thresholds breached, preserving traces.
Thresholds: deprecated-method rate must not rise; correctness must not drop >2 pts; p95 latency/cost must not rise >10%.

### Q: What are all Week 11 artifacts?
**A:** `artifacts/week11/drill.md`, `trace.json`, `red_output.log`, `green_output.log`, `cost_by_stage.md`, `tenx.md`, plus scripts `trace_search.py`, `trace_export.py`, `seed_incident.py`, and the permanent regression case + test.

### Q: What limitations should an interviewer know about?
**A:** No live LLM API key was configured, so pass rates and costs are estimates/0, not provider-reported. The incident is explicitly synthetic (not a real customer incident). Human drill verification is pending. The 23 errors in `test_api.py` are pre-existing MCP session-manager setup issues unrelated to Week 10/11 changes.

---

## Key Concepts Summary

| Concept | Week | One-line definition |
|---------|------|---------------------|
| Manager-Plus-Two-Specialists | 10 | Orchestrator delegates to version/deprecation and code-sample workers, then combines evidence. |
| Context Re-Send Multiplier | 10 | Squad total tokens ÷ single-agent total tokens (same 10 cases). |
| Controlled Fault Injection | 10 | Env-gated HTTP 500 into one worker for one case; never on by default. |
| Pass Rate | 10 | passed cases / 10 × 100, using a consistent rubric for both arms. |
| Span | 11 | Timed instrumented operation (trace_id, span_id, parent, stage, tokens, cost). |
| Trace ID | 11 | Unique ID correlating a request to all its spans/steps. |
| RED → GREEN | 11 | Capture failing output before the fix, apply fix, capture passing output after. |
| Cost Attribution | 11 | Retrieval / Generation / Tool executions; each billable op counted once. |
| Retrieval Cache | 11 | Cache keyed by (query, strategy, kb) to skip retrieval on repeat; version-safe. |
| First Bottleneck (10×) | 11 | LLM provider rate limits, based on measured per-request cost/latency. |

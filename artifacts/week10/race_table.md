# Week 10: Single-Agent vs Multi-Agent Squad Comparison

Dataset: `evals/developer_docs_cases.json` (10 GitHub API version-specific cases)
Case IDs: gh-001, gh-002, gh-003, gh-004, gh-005, gh-006, gh-007, gh-008, gh-009, gh-010
Evaluator: keyword-presence (all expected keywords must appear in answer)
Model: `nvidia` / `meta-llama/llama-3.3-70b-instruct`
Cost per token: `5e-07` (estimated, not provider-reported)
Token method: estimated (word-count heuristic); not provider-reported usage

## Summary

| Metric | Single Agent | Squad |
|--------|-------------|-----|
| Pass rate | 0.0% (0/10) | 10.0% (1/10) |
| p50 latency (ms) | 10932 | 3534 |
| p99 latency (ms) | 24082 | 12118 |
| Total input tokens | 249 | 1566 |
| Total output tokens | 130 | 344 |
| Total tokens | 379 | 1910 |
| Total cost | $0.000189 | $0.000955 |
| Avg cost/question | $0.000019 | $0.000096 |

| Context Re-Send Multiplier | 5.0 |
| Notes | Multi-Agent Total Tokens / Single-Agent Total Tokens = 1910 / 379 |

## Per-Case Results

| Case | Single Pass | Single Latency (ms) | Single Tokens | Squad Pass | Squad Latency (ms) | Squad Tokens | Squad Hops |
|------|-------------|---------------------|---------------|------------|--------------------|--------------|------------|
| gh-001 | N | 13225 | 37 | N | 2487 | 160 | 3 |
| gh-002 | N | 22972 | 33 | N | 5549 | 176 | 3 |
| gh-003 | N | 7310 | 43 | N | 3116 | 173 | 3 |
| gh-004 | N | 9746 | 39 | Y | 12118 | 218 | 4 |
| gh-005 | N | 24082 | 34 | N | 3534 | 183 | 3 |
| gh-006 | N | 10932 | 29 | N | 5767 | 156 | 3 |
| gh-007 | N | 11741 | 46 | N | 7111 | 261 | 4 |
| gh-008 | N | 6457 | 42 | N | 3357 | 205 | 3 |
| gh-009 | N | 5734 | 43 | N | 2080 | 181 | 3 |
| gh-010 | N | 12241 | 33 | N | 4813 | 197 | 4 |

## Methodology Notes
- Token counts are estimated via word-count heuristic (same as `rag_engine.estimate_tokens`), not provider-reported.
- Costs are estimates based on `settings.COST_PER_TOKEN`; actual provider pricing may differ.
- p50/p99 use nearest-rank percentile; p99 of 10 samples is the max value.
- Squad token counts include orchestrator + both specialist workers + final answer generation.
- All handoffs are logged in `artifacts/week10/handoffs.log` (JSON Lines).

## Largest Handoff Bottleneck
- Source: `orchestrator`
- Destination: `llm_answer`
- Purpose: final_answer_generation
- Tokens: 161 (estimated)
- % of squad total tokens: 8.4%
- Case: gh-007
- Status: success
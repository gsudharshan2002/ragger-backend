# Week 10: Single-Agent vs Multi-Agent Squad Comparison

Dataset: `evals/developer_docs_cases.json` (10 GitHub API version-specific cases)
Case IDs: gh-001, gh-002, gh-003, gh-004, gh-005, gh-006, gh-007, gh-008, gh-009, gh-010
Evaluator: keyword-presence (all expected keywords must appear in answer)
Model: `groq` / `openai/gpt-oss-20b`
Cost per token: `5e-07` (estimated, not provider-reported)
Token method: estimated (word-count heuristic); not provider-reported usage

## Summary

| Metric | Single Agent | Squad |
|--------|-------------|-----|
| Pass rate | 0.0% (0/10) | 10.0% (1/10) |
| p50 latency (ms) | 287 | 4 |
| p99 latency (ms) | 287 | 16 |
| Total input tokens | 0 | 1729 |
| Total output tokens | 0 | 260 |
| Total tokens | 0 | 1989 |
| Total cost | $0.000000 | $0.000994 |
| Avg cost/question | $0.000000 | $0.000099 |

| Context Re-Send Multiplier | unavailable |

## Per-Case Results

| Case | Single Pass | Single Latency (ms) | Single Tokens | Squad Pass | Squad Latency (ms) | Squad Tokens | Squad Hops |
|------|-------------|---------------------|---------------|------------|--------------------|--------------|------------|
| gh-001 | N | 287 | 0 | N | 16 | 195 | 4 |
| gh-002 | N | 0 | 0 | N | 4 | 179 | 4 |
| gh-003 | N | 0 | 0 | N | 4 | 220 | 4 |
| gh-004 | N | 0 | 0 | N | 4 | 204 | 4 |
| gh-005 | N | 0 | 0 | Y | 4 | 183 | 4 |
| gh-006 | N | 0 | 0 | N | 4 | 163 | 4 |
| gh-007 | N | 0 | 0 | N | 4 | 231 | 4 |
| gh-008 | N | 0 | 0 | N | 4 | 215 | 4 |
| gh-009 | N | 0 | 0 | N | 4 | 220 | 4 |
| gh-010 | N | 0 | 0 | N | 5 | 179 | 4 |

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
- Tokens: 131 (estimated)
- % of squad total tokens: 6.6%
- Case: gh-007
- Status: failure
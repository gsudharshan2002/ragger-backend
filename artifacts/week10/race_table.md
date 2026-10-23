# Week 10: Single-Agent vs Multi-Agent Squad Comparison

Dataset: `evals/developer_docs_cases.json` (10 GitHub API version-specific cases)
Case IDs: gh-001, gh-002, gh-003, gh-004, gh-005, gh-006, gh-007, gh-008, gh-009, gh-010
Evaluator: strict keyword scorer (`app/services/evaluation/scoring.py`): whole-word match, all expected keywords required, refusals never pass
Model: `nvidia` / `openai/gpt-oss-20b` (same model and evaluator for both arms)
Retrieval strategy: `bm25` (same for both arms)
Cost per token: `5e-07` (flat configured rate, not provider billing)
Token method: single agent = estimated, squad = estimated (both measured at the LLM layer, every model call counted once)

## Summary

| Metric | Single Agent | Squad |
|--------|-------------|-----|
| Pass rate | 60.0% (6/10) | 70.0% (7/10) |
| p50 latency (ms) | 186746 | 50631 |
| p99 latency (ms) | 225394 | 134449 |
| Total input tokens | 23628 | 11623 |
| Total output tokens | 4262 | 3451 |
| Total tokens | 27890 | 15074 |
| Total cost | $0.013945 | $0.007537 |
| Avg cost/question | $0.001394 | $0.000754 |

## Context Re-Send Multiplier

**0.5** = squad total tokens / single-agent total tokens = 15074 / 27890 (same 10 cases)

## Per-Case Results

Result: Y = pass, N = wrong/refused answer, ERR = no answer produced (provider/LLM error).

| Case | Single | Single ms | Single tokens | Squad | Squad ms | Squad tokens | Squad hops |
|------|--------|-----------|---------------|-------|----------|--------------|------------|
| gh-001 | Y | 186746 | 2918 | Y | 126411 | 1651 | 4 |
| gh-002 | ERR | 225394 | 1831 | Y | 19425 | 1257 | 3 |
| gh-003 | N | 195482 | 3181 | N | 87381 | 1419 | 3 |
| gh-004 | ERR | 221956 | 1741 | Y | 107644 | 1198 | 3 |
| gh-005 | Y | 204208 | 3421 | N | 50631 | 1400 | 3 |
| gh-006 | Y | 211046 | 2882 | ERR | 134449 | 734 | 3 |
| gh-007 | Y | 134658 | 4354 | Y | 25881 | 2008 | 4 |
| gh-008 | Y | 175013 | 2803 | Y | 66169 | 1638 | 3 |
| gh-009 | Y | 35373 | 2799 | Y | 33315 | 2314 | 4 |
| gh-010 | ERR | 155295 | 1960 | Y | 47209 | 1455 | 3 |

Errored cases (no answer produced): single 3, squad 1. These are provider/LLM failures, not wrong answers.

## Known scorer weaknesses (scores above are NOT adjusted)
- `gh-003`: keyword `migrat` is a false negative. On 2026-10-23 re-run, both arms answered 'stop reading the top-level rate field and read resources.core instead; rate was removed'. That is the migration advice, but neither answer contains the stem 'migrat'.

## Methodology Notes
- Tokens for both arms are measured at the LLM layer (`app/services/llm.py` usage tracker): every model call is counted once, with its full prompt (retrieved context included).
- A call's tokens are provider-reported when the stream carries a usage chunk; otherwise estimated by word count and flagged above.
- Squad totals include only the two orchestrator model calls (routing, final answer). Worker hops move retrieved text and make no model call, so their payload size is logged in `handoffs.log` (`model_call: false`) but not added to totals (it is already inside the final-answer prompt).
- Cost = total tokens x `settings.COST_PER_TOKEN` (one flat configured rate for input and output; not provider billing).
- Latency is wall-clock per question, measured identically for both arms. p50/p99 use nearest-rank percentile; with 10 samples p99 is the maximum, so it is a weak estimate.
- Scoring is a keyword check (see Evaluator). It cannot judge correctness beyond keyword presence; read the answers for borderline cases.
- All handoffs, including every retry attempt, are in `artifacts/week10/handoffs.log` (JSON Lines).

## Largest Handoff Bottleneck
- Source: `orchestrator`
- Destination: `llm_answer`
- Purpose: final_answer_generation
- Tokens: 1692 (provider-reported)
- % of squad total tokens: 11.2%
- Case: gh-007
- Status: success
- Largest worker payload (not a model call, not in totals): `version_deprecation` 467 tokens (estimated) in gh-007
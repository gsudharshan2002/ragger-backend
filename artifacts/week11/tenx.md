# Week 11: 10x Scaling Analysis

Basis: actual trace data in `app/data/traces.jsonl` (985 completed traces)
plus the synthetic incident trace for full span-level cost/token detail
(`artifacts/week11/trace.json`). All per-request cost/token figures are
estimates derived from span token counts x `settings.COST_PER_TOKEN`; no
provider-reported usage metadata is available.

## Current measured baseline (from stored traces)

| Metric | Value |
|--------|-------|
| Traces stored | 985 |
| Completed | 857 |
| Failed | 43 |
| Latency p50 | 3,361 ms |
| Latency p99 | 461,947 ms (outlier-skewed; several long-running/hung requests) |
| Representative per-request cost | $0.000655 (estimated, incident trace) |
| Representative per-request tokens | 1,310 max / ~5 avg (most historical traces predate token tracking) |

Historical traces were recorded before Week 11 cost/token instrumentation, so
most carry $0 / 0 tokens. The incident trace is the only request with
complete per-span cost data and is used as the representative request.

## Expected 10x workload

- Request volume: ~9,850 equivalent requests (10 x 985).
- LLM generation calls: ~10x (one per RAG request).
- Retrieval reranker calls: ~10x (unless cache hit).
- Aggregate estimated cost: ~$0.00655 for the same mix (still small in
  absolute dollars, but 10x the provider RPM/token consumption).

## First bottleneck: API rate limits (LLM provider)

The component most likely to saturate first under 10x load is the **LLM
provider's rate limit (requests-per-minute / concurrent-requests / TPM)**,
not raw cost and not p99 latency alone.

Numeric evidence supporting this conclusion:

- Per-request cost is negligible ($0.000655 estimated). Even at 10x volume
  (~$0.00655 per batch of 985), cost would not be the binding constraint
  unless volume grows to hundreds of thousands of requests.
- Latency p50 (3,361 ms) is moderate; p99 is inflated by a few outliers
  (hung/slow requests), which indicates a reliability/timeout issue rather
  than a throughput ceiling under normal load.
- Every RAG request makes at least one LLM generation call plus potentially
  an embedding/reranker call. Under 10x concurrent load, the LLM provider's
  requests-per-minute and concurrent-connection caps are reached well before
  cost or per-request latency become dominant.

Assumptions: request mix stays similar (mostly RAG, similar token counts);
no major increase in per-request token volume; provider rate limits are the
dominant external ceiling.

## Practical mitigation

1. **Retrieval caching** (implemented, `app/services/retrieval_cache.py`):
   cache-hit requests skip the reranker + embedding calls, cutting provider
   token/RPM consumption by up to ~12% per cached request (measured:
   $0.000080 saved per request).
2. **Rate-limit-aware backoff**: `app/core/retry.py` already retries on HTTP
   429 honoring `Retry-After`; keep and tune `max_retries`/`base_delay`.
3. **Request queuing / concurrency cap** at the gateway so bursts don't
   overwhelm the provider; return 429 with a sensible retry-after rather
   than failing open.
4. **Route simple/short questions to a cheaper, faster model** where the
   deployment supports multiple models.

If evidence later shows cost or p99 latency dominates instead, revisit; this
assessment is based on the measured baseline above, not an uncontrolled
10x production load test (which was deliberately not executed).

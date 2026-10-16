# Week 11: Cost Attribution by Stage

Isolated request: the synthetic deprecated-method incident trace
(`week11-incident-deprecated-assignee-001`), question: "When creating a
GitHub issue via POST /repos/{owner}/{repo}/issues, which field assigns
users?". Full trace: `artifacts/week11/trace.json`.

All dollar values are **estimates** derived from span-level token counts
multiplied by `settings.COST_PER_TOKEN` (5e-07 per token, flat rate). No
provider-reported usage metadata is available (no live LLM API key was
configured during this run), so every figure is labeled estimated, not
actual provider billing.

## Attribution rule

Each billable operation is counted exactly once. Spans are categorized by
stage:

- **Retrieval** — vector search and reranker model calls.
- **Generation** — the LLM answer-generation model call.
- **Tool executions** — separately billed tool calls that are neither
  retrieval nor generation. In this trace there are none distinct from the
  retrieval/generation spans above (vector_search and reranker are counted
  under retrieval; llm under generation), so this stage is $0.

## Per-stage breakdown (incident trace)

| Stage | Input tokens | Output tokens | Latency | Cost (USD) | Basis |
|-------|-------------:|--------------:|---------|------------|-------|
| Retrieval (reranker) | 120 | 40 | 312 ms | $0.000080 | estimated |
| Retrieval (vector search) | 0 | 0 | 14 ms | $0.000000 | local BM25/embedding, no direct API charge |
| Generation (LLM) | 940 | 210 | 8420 ms | $0.000575 | estimated |
| Tool executions (separate) | 0 | 0 | — | $0.000000 | none distinct from retrieval/generation |
| Response assembly | 0 | 0 | 8 ms | $0.000000 | no model call |
| **Total** | **1,060** | **250** | **8,754 ms** | **$0.000655** | estimated |

Span-level detail is in `artifacts/week11/trace.json` (`spans` array). Token
counts and costs come from the span records, which are populated by the
Week 11 span instrumentation (`app/services/observability.py`).

## Cost improvement: retrieval caching

**Baseline** (no cache, one representative request): **$0.000655** per request
(estimated).

**Improvement implemented**: `app/services/retrieval_cache.py` caches the
final retrieved chunks keyed by `(query, strategy, knowledge_base_id)`. On a
cache hit, the retrieval/re-rerank stages are skipped entirely, avoiding the
reranker model call.

**Post-improvement** (cache hit, equivalent input): retrieval stage cost drops
from $0.000080 to $0.000000.

| Metric | Baseline | After (cache hit) | Change |
|--------|----------|-------------------|--------|
| Retrieval cost | $0.000080 | $0.000000 | -$0.000080 |
| Total request cost | $0.000655 | $0.000575 | -$0.000080 |
| % cost reduction | — | — | **-12.2%** |
| Latency (retrieval skipped) | 8,754 ms | ~8,442 ms | ~-3.6% |

The cache key includes the knowledge base id and the cached chunks retain
their source document/version metadata, so cached results cannot silently
bypass version, document-freshness, or access constraints.

**Limitation**: this is a deterministic estimate computed from the incident
trace's span data, not a live A/B measurement with repeated requests and an
API key. Live measurement requires a configured LLM provider key and
repeated equivalent requests to observe real cache-hit rates.

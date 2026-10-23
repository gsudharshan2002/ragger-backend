# Week 11: 10x scaling analysis (from real traces)

Basis: the 10 real agent requests in `live_regression.json` (live model `openai/gpt-oss-20b` via NVIDIA, BM25, run
one at a time on 2026-10-23), read back from their stored traces. The old analysis used a synthetic fixture and the
historical trace file, which mostly predates token tracking; it is archived in `synthetic_fixture/tenx.md`.

## Measured baseline (n = 10 requests, 40 model calls)

| Metric | Value |
|---|---|
| Request latency | p50 157 s, mean 182 s, max 316 s (with n = 10, p99 is just the max) |
| Model calls per request | 4.0 on average (5 at most); retrievals per request 2.0 |
| Tokens per request | 3,115 input + 571 output (provider-reported counts) |
| Cost per request | mean $0.00184 (range $0.00105 to $0.00295), at the flat configured rate of 5e-07 USD/token |
| Single model-call latency | p50 19.5 s, p90 120 s |
| Model-call failures | **8 of 40 calls (20%) hit the 120 s stream timeout** |

Cost is the flat `COST_PER_TOKEN` estimate, not provider billing. Retrieval is local BM25: $0 per request.

## 10x workload

Current measured workload = 1 request at a time. 10x = 10 requests in flight.

- Model calls in flight: about 10 (40 per 10 requests).
- Provider throughput needed at the observed 157 s per request: 10 / 157 s = 0.064 requests/s, or 0.25 model calls/s (15 per minute).
- Cost for 10 simultaneous requests: about $0.018 per batch. Not a constraint.

## Realistic 9,000-developer model (assumptions, not measurements)

Assume 20% of 9,000 developers are active on a given day, 10 questions each, over an 8-hour day, and a 5x burst:

| | Average | Burst (5x) |
|---|---:|---:|
| Requests / second | 0.63 | 3.1 |
| Requests in flight (rate x 157 s, Little's law) | about 98 | about 490 |
| Model calls / minute (4.0 per request) | about 150 | about 750 |
| Cost / day at $0.00184 per request | about $33 | n/a |

## First bottleneck: p99 latency and provider throughput, not cost

- Cost: about $33/day under these assumptions. Not the first limit.
- Latency: the single-call timeout rate is already 20% at **one** concurrent request. A request needs about 4 sequential model calls, so a timeout in any one stalls the request (p50 157 s, max 316 s). This breaks an IDE use case long before any budget does.
- Rate limits: at 10x the load is about 15 calls/minute and at the 9,000-developer burst about 750 calls/minute. **The provider's actual request limit was not measured or verified here**, so I cannot say at what load a 429 appears. The leading risk is that the free-tier endpoint cannot sustain the burst; this is an assumption to confirm with the provider's published limits.

## Mitigations (not yet implemented unless stated)

1. Lower the stream timeout and retry once with backoff (today a 120 s timeout silently becomes an empty reasoning step).
2. Cache retrieval and final answers keyed by (question, KB version, prompt version).
3. Cut calls per request: the live traces repeat the same retrieval 2 to 3 times with different wording and the same chunks (e.g. 467, 1121, 5463).
4. Concurrency cap plus queue in front of the provider, returning 429 with Retry-After when full.
5. Route simple questions to a smaller or faster model; use a paid tier with published limits.

Not done: no load test was run (an uncontrolled 10x test was out of scope), and no canary.

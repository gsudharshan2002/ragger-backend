# Cost by stage

Trace: `bb46791a-84cf-49de-b752-65ec23b8e15f` (kind `agent`, mode `agent`, prompt v2, model `openai/gpt-oss-20b`)
Question: When creating a GitHub issue via POST /repos/{owner}/{repo}/issues, which field assigns users instead of the deprecated assignee field?

| Stage | Calls | Input tokens | Output tokens | Latency (ms) | Cost (USD) |
|-------|------:|-------------:|--------------:|-------------:|-----------:|
| retrieval | 1 | 0 | 0 | 5 | $0.000000 |
| generation | 3 | 866 | 139 | 275606 | $0.000503 |
| tools | 0 | 0 | 0 | 0 | $0.000000 |
| **total** | | **866** | **139** | | **$0.000503** |

- Token source: provider-reported (read from each `llm.generate` span).
- Price: flat `settings.COST_PER_TOKEN` = 5e-07 USD/token for input and output; this is a configured estimate, not provider billing.
- Retrieval cost is $0: BM25, local embeddings and the local reranker have no per-request API charge here.
- Latency per stage is the sum of its spans; stages run sequentially in the agent, so they add up to about the request latency.
- Request latency (from trace): 275632 ms.

## Notes on this trace

- This is a **degraded run**: 2 of its 3 model calls hit the 120 s stream timeout and returned no tokens, so only one call's tokens are counted (`trace_clean_metadata.json`). Per-request cost across the 10 earlier live runs averaged $0.00184 (`tenx.md`).
- Failed calls are not billed in this estimate because no tokens were reported for them; a provider may bill differently.
- Cache-hit saving: none measured. The retrieval cost is already $0 (local BM25), so a retrieval cache would save latency but no dollars here; the old "12.2% saving" claim came from a synthetic fixture and is archived in `synthetic_fixture/`.
- Reproduce: `python scripts/cost_by_stage.py --trace-id <id>`.

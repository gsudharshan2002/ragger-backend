# Week 12: Integration seam list

**Reconstructed on 2026-10-23.** No original "Tuesday" seam list exists in the repository (searched the tree and git
history), so this list is rebuilt from the integration problems actually found while inspecting and running the
working tree. It was not written before the work it describes. Status values: **fixed**, **accepted**, **ran-out-of-time**.

| # | Seam (where two parts meet) | Problem observed | Status | Evidence |
|---|---|---|---|---|
| 1 | retrieval -> reranker | A reranker import failure aborted retrieval; agents swallowed it and answered from no context | fixed | `rag_engine.get_context` falls back to fused ranking; capability check at startup |
| 2 | retrieval -> `vector` strategy | Agent picked `vector`; with no embedding provider it returned nothing and wasted a step | fixed | falls back to BM25; `tests/test_agent_tracing.py` |
| 3 | agent -> trace store | Agent traces had no spans, `id`, `timestamp`, `status`; unfindable by trace id | fixed | `agent_engine` begins a trace; `test_agent_run_is_one_searchable_trace` |
| 4 | LLM layer -> token accounting | Single-agent tokens counted only query + answer; squad counted worker payloads again | fixed | `llm.track_usage`, `llm.generate` spans, provider-reported usage |
| 5 | redaction -> span metadata | Redaction pattern `token` also masked `max_tokens` / `tokens_estimated` | fixed | found in review; `test_redact_keeps_token_count_fields_but_masks_credentials` |
| 6 | squad -> workers | No retries, retry limit unused; "no evidence" logged as an error without text | fixed | `SquadOrchestrator._delegate`, `tests/test_squad_orchestrator.py` |
| 7 | dataset -> knowledge base | Cases expect `current-version.md` etc.; KB held a different, fabricated doc | fixed | `scripts/seed_github_kb.py` |
| 8 | scoring -> answers | Refusal repeating the keyword scored as a pass; short keywords matched inside words | fixed | `evaluation/scoring.py`, `tests/test_scoring.py` |
| 9 | app lifespan -> tests | MCP session manager can run once per process; every test client restarted it | fixed | `tests/conftest.py` |
| 10 | request -> SDK version | No version resolver; no `sdk_version` metadata on documents or filter in retrieval | ran-out-of-time | not started |
| 11 | agent -> symbol check / changelog tools | Only `retrieve`, `check_deprecation`, `answer` exist; no symbol checker | ran-out-of-time | not started |
| 12 | agent -> MCP -> server (trace id) | MCP tools take no trace id; no span for MCP calls; version filter not carried | ran-out-of-time | not started |
| 13 | answer -> code validation | No validation of generated code samples | ran-out-of-time | not started |
| 14 | LLM provider -> agent loop | 8 of 40 model calls hit the 120 s timeout; a timeout becomes an empty reasoning step | ran-out-of-time | `artifacts/week11/tenx.md` |
| 15 | environment -> vector search | `torch`/`sentence-transformers` have no build for Python 3.15; vector search and local reranker unavailable | accepted | `GET /api/v1/health/capabilities`; BM25 used |
| 16 | environment -> LangGraph agent | `langgraph` is not installed here; `/agent/run-langgraph` cannot run | accepted | module import fails; not exercised by tests |
| 17 | trace writes -> storage | Whole JSONL file is rewritten per trace; concurrent writers can lose records | accepted | `storage.add_trace`; fine for single-process use |
| 18 | KB content -> SDK v2/v3 questions | KB documents GitHub API versions (2022-11-28, 2026-03-10), not an SDK with v2.x/v3.0 | ran-out-of-time | need grounded SDK docs (or confirm the Swagger PDFs are the intended ones) |

Peer review of this list: pending (see the review template once `artifacts/week12/` is complete).

# Next.js Frontend Integration Instructions

Use these instructions when updating the Next.js frontend against the current `ragger-backend` API.

## API client

- Use `NEXT_PUBLIC_API_URL` as the API base URL. The backend default is `http://localhost:8000`.
- Prefix every endpoint with `/api/v1`.
- Successful JSON responses use `{ success: true, data: ... }`.
- Handle non-2xx responses and `{ success: false, error: ... }` as user-visible errors.
- Backend fields are camelCase (`datasetId`, `startedAt`, `aggregateMetrics`, `actualSources`).

## Chat

`POST /api/v1/chat/send`

```json
{
  "query": "How does retrieval work?",
  "strategy": "hybrid-rerank-mmr",
  "knowledgeBaseId": null
}
```

The response data contains `answer`, `sources`, and a full `trace`. Use `sources` for citations and keep the `trace.id` for trace/details views.

For streaming chat, use `POST /api/v1/chat/stream` with the same body. Parse Server-Sent Events line by line. Each event is JSON after `data: `; append `llm.token.content` to the answer, and stop on `data: [DONE]`. Do not parse the stream as one JSON response.

Supported strategies are `vector`, `bm25`, `hybrid`, `hybrid-rrf`, `hybrid-rerank`, and `hybrid-rerank-mmr`.

## Datasets and benchmarks

- `GET /api/v1/datasets` lists datasets.
- `GET /api/v1/datasets/{datasetId}` returns one dataset and its versions/cases.
- `POST /api/v1/datasets` creates a dataset with `{ name, description?, tags? }`.
- `POST /api/v1/datasets/{datasetId}` creates a version with `{ version?, casesCount?, changeNote? }`.
- `POST /api/v1/benchmark/run` runs the current dataset version synchronously.
- `POST /api/v1/benchmark/run-stream` runs it with progress events.
- `GET /api/v1/benchmark/runs` lists saved runs; `GET /api/v1/benchmark/runs/{runId}` gets one; `DELETE /api/v1/benchmark/runs/{runId}` deletes one.

Benchmark request body:

```json
{
  "datasetId": "dataset-id",
  "strategy": "hybrid-rerank-mmr",
  "ragConfig": {}
}
```

For `run-stream`, handle these SSE events:

- `benchmark.started`: initialize progress from `total`.
- `case.started`: show the current `index`, `total`, `caseId`, and `query`.
- `case.completed`: add `result` to the case table.
- `benchmark.completed`: replace local state with `data`, the persisted final run.
- `[DONE]`: close the stream.

Run results expose `passedTests`, `partialTests`, `failedTests`, `aggregateMetrics`, `difficultyBreakdown`, `tagBreakdown`, `failureCategories`, and per-case `results`. Display `status` and `failureExplanation`; do not infer failure type from the score alone.

## Week 6 developer-documentation evaluation

The backend includes the deterministic Week 6 developer-documentation evaluation set and reports:

`GET /api/v1/benchmark/developer-docs/results`

The eval isn't tied to a fixed document or a fixed pair of labels - each `uv run python -m evals.run_developer_docs` run is saved as its own timestamped file, and this endpoint always returns the two most recent runs, whichever cases/document they used. The response data may contain `baseline` (the older of the two) and `improved` (the newer), each with `summary` and per-case `results`. Render the summary and comparison without assuming both labels exist - there may be only one run so far, or none.

Treat the scores as a relative before/after comparison, not a pass/fail rubric: an `improved` report scoring higher than `baseline` is an improvement, not necessarily a 100% pass. The frontend should make partial/failed cases visible and should not label Week 6 as complete solely because an `improved` report exists.

## RAG settings

- `GET /api/v1/rag/config` loads settings and document/chunk counts.
- `PUT /api/v1/rag/config` updates only supplied camelCase fields.
- `POST /api/v1/rag/reindex-embeddings` starts embedding regeneration and returns `{ reindexed, provider }` on success.

Keep loading, empty, streaming, error, and completed states distinct. Avoid hardcoding benchmark results when the API is available.

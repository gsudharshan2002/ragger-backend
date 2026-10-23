# Ragger Backend

FastAPI backend for the RAG (Retrieval-Augmented Generation) application. This replaces the Node.js/TypeScript backend with a Python implementation while keeping the exact same API contract so the Next.js frontend works unchanged.

## Features

- **RAG Pipeline**: Vector search, BM25, RRF fusion, reranking, MMR selection
- **LLM Streaming**: Groq and Gemini providers with SSE streaming
- **Embeddings**: OpenAI and Cohere providers
- **Document Processing**: PDF and text file parsing, chunking, embedding
- **Benchmarking**: Run RAG evaluations on datasets
- **Storage**: File-based JSON storage (easy to swap for Postgres/Redis)

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first. `uv` manages the Python version, virtual environment, and dependencies for this project.

```bash
cd ragger-backend
uv sync --extra dev
cp .env.example .env  # Edit with your API keys
uv run uvicorn app.main:app --reload --port 8000
```

On Windows PowerShell, use `Copy-Item .env.example .env` instead of `cp`.

### Knowledge base for the developer-docs evaluation

The Week 6 cases in `evals/developer_docs_cases.json` expect `current-version.md`, `old-version.md`
and the Apple Combine PDF in the knowledge base. Load them from `uploads/` (idempotent; add `--dry-run` to preview):

```bash
uv run python scripts/seed_github_kb.py
```

### Check which retrieval features work

Vector search and the local reranker need the optional `sentence-transformers` (torch) stack. If it is not
installable on your Python version, retrieval falls back to BM25 and logs a warning at startup. Check with:

```bash
curl localhost:8000/api/v1/health/capabilities
```

### Week 10 benchmark (single agent vs squad)

```bash
DEFAULT_STRATEGY=bm25 uv run python scripts/run_week10_experiment.py   # writes artifacts/week10/
```

### Traces, search and cost (Week 11)

Every agent and squad request is stored as one trace (spans for model calls, tools, retrieval and handoffs; provider-reported
tokens; prompt version; redacted inputs) in `app/data/traces.jsonl`.

```bash
uv run python scripts/trace_search.py --query-contains assignee --answer-contains "could not find"   # find a bad answer
uv run python scripts/trace_export.py <trace_id> out.json                                           # complete trace
uv run python scripts/cost_by_stage.py --trace-id <trace_id>                                        # retrieval / generation / tools cost
DEFAULT_STRATEGY=bm25 uv run python scripts/week11_live_regression.py 5                             # live prompt v1 vs v2 (slow, uses the LLM)
```

Run the test suite and quality checks with:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy app
```

## API Endpoints

All endpoints are under `/api/v1`:

| Method | Endpoint            | Description                       |
| ------ | ------------------- | --------------------------------- |
| GET    | `/health`           | Health check                      |
| POST   | `/chat/stream`      | SSE stream of RAG pipeline events |
| POST   | `/chat/send`        | Non-streaming chat response       |
| GET    | `/documents`        | List documents                    |
| POST   | `/documents/upload` | Upload and process document       |
| DELETE | `/documents/{id}`   | Delete document                   |
| GET    | `/knowledge-bases`  | List knowledge bases              |
| POST   | `/knowledge-bases`  | Create knowledge base             |
| GET    | `/datasets`         | List datasets                     |
| POST   | `/datasets`         | Create dataset                    |
| POST   | `/benchmark/start`  | Start benchmark run               |
| GET    | `/benchmark/runs`   | List benchmark runs               |

## Configuration

Environment variables (see `.env.example`):

- `GROQ_API_KEY` / `GEMINI_API_KEY` — LLM provider keys
- `EMBEDDING_PROVIDER` — `openai`, `cohere`, or `none`
- `EMBEDDING_API_KEY` — Embedding API key
- `LLM_PROVIDER` — `groq` or `gemini`
- `DEFAULT_STRATEGY` — RAG strategy
- `DEFAULT_TOP_K` — Default retrieval count
- `CORS_ORIGINS` — Allowed CORS origins (comma-separated)

## Docker

```bash
docker build -t ragger-backend .
docker run -p 8000:8000 --env-file .env ragger-backend
```

Or with docker-compose (from repo root):

```bash
docker-compose up
```

The Docker image uses `uv sync --locked` and does not require a host Python installation.

## API Compatibility

The API responses match the frontend's expected format exactly:

- SSE events use `data: {...}\n\n` format
- Chat stream emits `llm.token` events with `content` field
- Document/chat/knowledge-base responses use `{ success: true, data: [...] }` format
- Error responses use `{ success: false, error: {...} }` format

No frontend changes required.

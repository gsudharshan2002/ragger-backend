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

```bash
cd ragger-backend
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows
pip install -e ".[dev]"
cp .env.example .env  # Edit with your API keys
uvicorn app.main:app --reload --port 8000
```

## API Endpoints

All endpoints are under `/api/v1`:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/chat/stream` | SSE stream of RAG pipeline events |
| POST | `/chat/send` | Non-streaming chat response |
| GET | `/documents` | List documents |
| POST | `/documents/upload` | Upload and process document |
| DELETE | `/documents/{id}` | Delete document |
| GET | `/knowledge-bases` | List knowledge bases |
| POST | `/knowledge-bases` | Create knowledge base |
| GET | `/datasets` | List datasets |
| POST | `/datasets` | Create dataset |
| POST | `/benchmark/start` | Start benchmark run |
| GET | `/benchmark/runs` | List benchmark runs |

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

## API Compatibility

The API responses match the frontend's expected format exactly:

- SSE events use `data: {...}\n\n` format
- Chat stream emits `llm.token` events with `content` field
- Document/chat/knowledge-base responses use `{ success: true, data: [...] }` format
- Error responses use `{ success: false, error: {...} }` format

No frontend changes required.

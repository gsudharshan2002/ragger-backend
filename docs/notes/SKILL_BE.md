# SKILL_BE — Ragger Backend Engineering

## Purpose

Build a fast, reliable, production-ready backend for **Ragger** using FastAPI and Python.

The backend should prioritize:

- Fast development
- Minimal code
- Strong typing
- Clear architecture
- Async I/O
- Reliable API contracts
- Good error handling
- Easy testing
- Easy maintenance
- RAG-friendly architecture

---

# 1. Backend Stack

The Ragger backend uses:

- Python `>=3.11`
- FastAPI `>=0.110.0`
- Uvicorn
- Pydantic `>=2.7.0`
- Pydantic Settings
- HTTPX
- OpenAI
- Cohere
- NumPy
- scikit-learn
- rank-bm25
- pypdf
- python-multipart
- aiofiles
- python-dotenv
- SQLAlchemy `>=2.0.30`
- Alembic
- asyncpg
- Redis
- structlog
- sentence-transformers
- sse-starlette

Development:

- pytest
- pytest-asyncio
- pytest-cov
- Ruff
- mypy

Project configuration:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

---

# 2. Core Rules

Before writing backend code:

1. Inspect the existing project.
2. Understand the current architecture.
3. Reuse existing utilities.
4. Reuse existing models and schemas.
5. Do not introduce unnecessary dependencies.
6. Do not create unnecessary abstraction.
7. Keep endpoints thin.
8. Keep business logic outside route handlers.
9. Use async I/O for I/O-bound operations.
10. Validate data at API boundaries.

Prefer the smallest clean implementation.

---

# 3. Recommended Architecture

Prefer a simple layered structure:

```text
app/
├── main.py
├── api/
│   ├── routes/
│   └── dependencies.py
├── core/
│   ├── config.py
│   └── logging.py
├── schemas/
├── models/
├── services/
├── repositories/
├── rag/
├── db/
├── utils/
└── tests/
```

Do not create every folder automatically.

Create a layer only when the project actually needs it.

---

# 4. Responsibilities

## Routes

Routes should handle:

- HTTP request
- Validation
- Authentication/dependencies
- Calling services
- HTTP response

Routes should NOT contain large business logic.

Bad:

```python
@router.post("/search")
async def search(request: SearchRequest):
    # 100 lines of RAG logic
    ...
```

Better:

```python
@router.post("/search")
async def search(request: SearchRequest):
    return await search_service.search(request)
```

---

# 5. Services

Services contain business logic.

Examples:

```text
DocumentService
SearchService
RagService
EmbeddingService
ChatService
```

A service should represent a meaningful business operation.

Avoid creating a service for every tiny function.

---

# 6. Repositories

Use repositories when database access becomes sufficiently complex.

Example:

```python
class DocumentRepository:
    async def get_by_id(self, document_id: UUID):
        ...
```

Keep SQL/database-specific logic out of API routes.

For simple CRUD operations, avoid unnecessary repository abstractions if they add no value.

---

# 7. Pydantic Schemas

Use Pydantic for request and response validation.

Example:

```python
from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
```

Use separate request and response schemas when their responsibilities differ.

Avoid returning raw database models directly from public APIs.

---

# 8. Type Safety

Use Python type hints everywhere.

Prefer:

```python
async def search(query: str, top_k: int = 5) -> list[SearchResult]:
    ...
```

Avoid:

```python
async def search(query, top_k=5):
    ...
```

Avoid `Any` unless genuinely necessary.

The project uses:

```toml
disallow_untyped_defs = true
```

so functions should have explicit types.

---

# 9. FastAPI Routes

Use `APIRouter`.

Example:

```python
from fastapi import APIRouter

router = APIRouter(prefix="/documents", tags=["documents"])
```

Keep route modules focused by domain.

Example:

```text
routes/
├── documents.py
├── search.py
├── chat.py
└── health.py
```

---

# 10. HTTP Status Codes

Use correct HTTP status codes.

Common cases:

```text
200 → Successful request
201 → Resource created
204 → Successful request with no body
400 → Invalid request
401 → Authentication required
403 → Forbidden
404 → Resource not found
409 → Conflict
422 → Validation error
500 → Unexpected server error
```

Do not return `200` for every failure.

---

# 11. Error Handling

Use FastAPI's exception handling.

Example:

```python
from fastapi import HTTPException

raise HTTPException(
    status_code=404,
    detail="Document not found",
)
```

User-facing errors should be clear.

Do not expose:

- Stack traces
- API keys
- Database credentials
- Internal filesystem paths
- Provider secrets
- Sensitive implementation details

---

# 12. Error Strategy

Separate expected errors from unexpected errors.

Expected:

```text
Document not found
Invalid file type
Invalid query
External provider unavailable
Duplicate resource
```

Unexpected errors should be logged with context and return a safe generic response.

Do not use:

```python
except Exception:
    pass
```

Never silently swallow errors.

---

# 13. Async Programming

Use async for I/O-bound operations.

Good candidates:

- Database operations
- HTTP requests
- Redis
- File operations
- LLM APIs
- Streaming responses

Prefer:

```python
async def get_document(...):
    ...
```

when the underlying operation is asynchronous.

Do not use blocking operations inside async endpoints when an async alternative exists.

---

# 14. HTTP Client

Use:

```text
httpx
```

for external HTTP APIs.

Reuse clients where appropriate rather than creating unnecessary clients for every request.

Always configure:

- Timeout
- Error handling
- Connection lifecycle

Do not allow external requests to hang indefinitely.

---

# 15. OpenAI / Cohere

Keep LLM and embedding provider logic isolated from API routes.

Prefer:

```text
Route
  ↓
Service
  ↓
Provider
```

Example:

```text
rag_service.py
    ↓
embedding_service.py
    ↓
OpenAI / Cohere / Sentence Transformers
```

Do not scatter provider calls throughout the application.

---

# 16. RAG Architecture

Ragger is a RAG backend.

Keep the RAG pipeline logically separated.

Typical flow:

```text
User Query
    ↓
Query Validation
    ↓
Query Processing
    ↓
Embedding
    ↓
Vector Retrieval
    ↓
BM25 Retrieval
    ↓
Hybrid Ranking
    ↓
Optional Reranking
    ↓
Context Selection
    ↓
LLM
    ↓
Response
```

Each stage should have a clear responsibility.

Do not put the entire pipeline into one giant function.

---

# 17. Hybrid Search

The project includes:

- NumPy
- scikit-learn
- rank-bm25
- sentence-transformers

These can be used for hybrid retrieval.

Typical approach:

```text
Semantic Search
+
BM25 Search
↓
Score Normalization
↓
Score Fusion
↓
Ranking
↓
Top K
```

Keep retrieval logic independent from HTTP concerns.

---

# 18. Embeddings

Embedding generation should be isolated behind a clear interface.

Example:

```python
class EmbeddingProvider:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...
```

The application should not need to know whether embeddings come from:

- OpenAI
- Cohere
- Sentence Transformers

unless provider-specific behavior is explicitly required.

---

# 19. Document Processing

The backend supports PDF/document processing.

Use:

```text
pypdf
python-multipart
aiofiles
```

for appropriate operations.

Typical flow:

```text
Upload
  ↓
Validate
  ↓
Store
  ↓
Extract Text
  ↓
Chunk
  ↓
Generate Embeddings
  ↓
Index
```

Validate:

- File type
- File size
- Filename
- Content availability

Do not trust client-provided filenames or MIME types alone.

---

# 20. File Uploads

Use FastAPI's `UploadFile`.

Example:

```python
from fastapi import UploadFile


async def upload_document(file: UploadFile):
    ...
```

Avoid loading unnecessarily large files entirely into memory.

Prefer streaming/chunked processing where appropriate.

Use safe temporary/storage paths.

Never execute uploaded files.

---

# 21. Database

Use:

```text
SQLAlchemy 2.x
asyncpg
Alembic
```

for PostgreSQL access.

Prefer SQLAlchemy's modern async patterns.

Use:

```python
AsyncSession
```

for async database operations.

Do not create a database connection for every request manually.

Use proper session lifecycle management.

---

# 22. Database Models

Keep ORM models separate from API schemas.

Example:

```text
models/
    document.py

schemas/
    document.py
```

Database model:

```python
class Document(Base):
    ...
```

API schema:

```python
class DocumentResponse(BaseModel):
    ...
```

Do not expose internal database fields unnecessarily.

---

# 23. Migrations

Use Alembic for schema changes.

Never manually modify production database schemas as the normal workflow.

Migration flow:

```text
Model Change
    ↓
Create Migration
    ↓
Review Migration
    ↓
Run Migration
```

Keep migrations small and understandable.

---

# 24. Redis

Use Redis for appropriate use cases such as:

- Caching
- Temporary state
- Rate limiting
- Job coordination
- Frequently requested data

Do not use Redis as a replacement for PostgreSQL unless the data is intentionally ephemeral.

Set sensible expiration times for temporary cache data.

---

# 25. Streaming / SSE

The project includes:

```text
sse-starlette
```

Use Server-Sent Events when the client needs incremental streaming.

Typical RAG chat flow:

```text
Client
  ↓
POST /chat
  ↓
RAG Processing
  ↓
LLM Stream
  ↓
SSE Events
  ↓
Frontend
```

Keep stream event formats consistent.

Example:

```text
event: token
data: {"text":"Hello"}

event: done
data: {}
```

Handle client disconnects gracefully.

---

# 26. Configuration

Use:

```text
pydantic-settings
python-dotenv
```

for configuration.

Example:

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    openai_api_key: str
```

Do not hard-code secrets.

Use environment variables.

---

# 27. Secrets

Never commit:

```text
OPENAI_API_KEY
COHERE_API_KEY
DATABASE_URL
REDIS_URL
SECRET_KEY
```

to source control.

Never return secrets through API responses.

Never log API keys.

---

# 28. Logging

Use:

```text
structlog
```

for structured logging.

Logs should provide useful context:

```text
request_id
endpoint
operation
document_id
duration
status
error
```

Do not log:

- API keys
- Passwords
- Authorization tokens
- Sensitive user content unnecessarily

Use appropriate log levels.

---

# 29. Performance

Prioritize:

- Async I/O
- Database connection pooling
- Redis caching where useful
- Efficient document processing
- Batch embedding where supported
- Avoiding duplicate LLM calls
- Avoiding unnecessary database queries
- Streaming long responses

Do not optimize prematurely.

Measure before adding complex optimizations.

---

# 30. Dependency Injection

Use FastAPI dependencies for shared infrastructure.

Good candidates:

```text
Database session
Current user
Settings
Redis
Service dependencies
```

Example:

```python
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    ...
```

Do not use dependency injection for every ordinary function.

---

# 31. API Naming

Prefer predictable REST-style endpoints.

Examples:

```text
GET    /health
GET    /documents
POST   /documents
GET    /documents/{id}
DELETE /documents/{id}

POST   /search
POST   /chat
```

Use nouns for resources.

Avoid inconsistent endpoint naming.

---

# 32. API Response Design

Keep response structures predictable.

Example:

```python
class SearchResponse(BaseModel):
    results: list[SearchResult]
    total: int
```

For errors, return a consistent structure where practical.

Do not return different shapes for the same endpoint depending on the error.

---

# 33. Pagination

Use pagination for potentially large collections.

Prefer:

```text
limit
offset
```

for simple internal APIs.

For high-volume datasets, consider cursor-based pagination.

Never return thousands of database records by default.

---

# 34. Validation

Validate at the API boundary.

Check:

- Required values
- String length
- Numeric ranges
- Enum values
- File constraints
- IDs
- Query limits

Example:

```python
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
```

Prefer declarative validation with Pydantic.

---

# 35. Testing

Use:

```text
pytest
pytest-asyncio
pytest-cov
```

Test important behavior.

Prioritize:

- API endpoints
- Validation
- RAG retrieval
- Ranking
- Document processing
- Database behavior
- Error handling
- Streaming behavior

Avoid tests that only verify implementation details.

---

# 36. External Services in Tests

Do not make real LLM/provider calls in normal unit tests.

Mock:

```text
OpenAI
Cohere
HTTP APIs
Redis
External services
```

Use integration tests for real infrastructure when required.

Tests should be deterministic.

---

# 37. Ruff

Use Ruff for linting and formatting-related quality checks.

Project configuration:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"
```

Fix the underlying issue rather than disabling Ruff rules.

Avoid unnecessary `# noqa`.

---

# 38. Mypy

The project uses strict function typing requirements:

```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

Functions should have:

- Parameter types
- Return types

Avoid `Any` where a concrete type is possible.

---

# 39. Code Generation Rules

When generating backend code:

1. Inspect existing code first.
2. Follow the existing architecture.
3. Reuse existing services.
4. Reuse existing schemas.
5. Reuse existing dependencies.
6. Keep route handlers thin.
7. Keep business logic in services.
8. Add types.
9. Handle errors.
10. Avoid unnecessary files.
11. Avoid unnecessary abstractions.
12. Avoid unnecessary dependencies.

Prefer:

```text
1 good service
```

over:

```text
5 unnecessary abstraction layers
```

---

# 40. Fast Implementation Rule

When implementing a small feature, prefer this sequence:

```text
Understand
   ↓
Reuse
   ↓
Implement
   ↓
Validate
   ↓
Test
```

Do not spend time designing infrastructure that the feature does not need.

---

# 41. Debugging Rule

When fixing an error:

1. Read the complete traceback.
2. Identify the actual failing layer.
3. Fix the root cause.
4. Avoid unrelated refactoring.
5. Run the relevant test.
6. Run lint/type checks when appropriate.

Do not hide errors with:

```python
try:
    ...
except Exception:
    pass
```

or broad suppression.

---

# 42. Quality Checklist

Before completing a backend task:

- [ ] API contract is clear
- [ ] Request validation exists
- [ ] Response schema is typed
- [ ] Route is thin
- [ ] Business logic is separated when needed
- [ ] Async I/O is used appropriately
- [ ] Errors are handled
- [ ] Secrets are protected
- [ ] Database sessions are managed correctly
- [ ] External calls have timeouts
- [ ] Tests cover important behavior
- [ ] Ruff passes
- [ ] Mypy passes where applicable
- [ ] No unnecessary dependencies
- [ ] No unnecessary abstraction

---

# 43. Ragger RAG Quality Checklist

For RAG-related changes, additionally verify:

- [ ] Documents are validated
- [ ] Text extraction works
- [ ] Chunking is deterministic
- [ ] Embeddings are generated correctly
- [ ] Retrieval returns relevant documents
- [ ] BM25 retrieval works when enabled
- [ ] Semantic retrieval works when enabled
- [ ] Ranking/fusion is deterministic
- [ ] Top-K limits are enforced
- [ ] Context size is controlled
- [ ] LLM failures are handled
- [ ] Streaming disconnects are handled
- [ ] Source metadata is preserved where required

---

# 44. Golden Rule

Build the **simplest backend that is correct, typed, testable, and production-ready**.

Optimize for:

```text
Less code
+
Clear APIs
+
Strong types
+
Async I/O
+
Reliable errors
+
Reusable services
+
Easy testing
```

Avoid:

```text
Unnecessary abstraction
+
Duplicate logic
+
Blocking I/O
+
Untyped code
+
Hidden errors
+
Hard-coded secrets
```

The best Ragger backend is not the backend with the most code.

It is the backend with the **clearest flow and fewest unnecessary moving parts**.

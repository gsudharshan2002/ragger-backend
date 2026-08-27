FROM python:3.11-slim

# Install system dependencies for PDF processing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml ./
COPY app ./app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn[standard] pydantic pydantic-settings httpx \
    openai cohere numpy scikit-learn rank-bm25 pypdf python-multipart aiofiles \
    python-dotenv sqlalchemy alembic asyncpg redis structlog sse-starlette

# Create data directories
RUN mkdir -p /app/data /app/uploads

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

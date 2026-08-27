from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict
from pydantic.alias_generators import to_camel


class BaseSchema(BaseModel):
    """Base model that accepts and emits camelCase field names
    (matching the original Next.js API contract)."""

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


class RagStrategy(str, Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    HYBRID_RRF = "hybrid-rrf"
    HYBRID_RERANK = "hybrid-rerank"
    HYBRID_RERANK_MMR = "hybrid-rerank-mmr"


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class MetricKey(str, Enum):
    HIT_RATE = "hitRate"
    RECALL = "recall"
    PRECISION = "precision"
    MRR = "mrr"
    NDCG = "ndcg"
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answerRelevance"
    CONTEXT_PRECISION = "contextPrecision"
    CONTEXT_RECALL = "contextRecall"
    LATENCY_MS = "latencyMs"
    INPUT_TOKENS = "inputTokens"
    OUTPUT_TOKENS = "outputTokens"
    TOTAL_TOKENS = "totalTokens"
    COST = "cost"


class Chunk(BaseSchema):
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    document_name: str
    content: str
    page: int
    section: Optional[str] = None
    token_count: int = 0
    embedding: Optional[list[float]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class StoredChunk(Chunk):
    pass


class VectorSearchConfig(BaseSchema):
    embedding_model: str = "text-embedding-3-large"
    top_k: int = 5
    similarity: Literal["cosine", "dot_product", "l2"] = "cosine"
    similarity_threshold: float = 0.0


class BM25Config(BaseSchema):
    top_k: int = 5
    language: str = "english"
    tokenizer: Literal["standard", "whitespace", "porter"] = "standard"


class RRFConfig(BaseSchema):
    k: int = 60
    vector_weight: float = 1.0
    bm25_weight: float = 1.0


class RerankerConfig(BaseSchema):
    enabled: bool = False
    model: Optional[str] = None
    candidate_count: int = 20
    top_n: int = 10


class MMRConfig(BaseSchema):
    enabled: bool = False
    lambda_: float = 0.7
    candidate_count: int = 15
    final_count: int = 8


class LLMConfig(BaseSchema):
    model: str = "openai/gpt-oss-20b"
    temperature: float = 0.7
    max_tokens: int = 2048


class RagEngineConfig(BaseSchema):
    strategy: RagStrategy = RagStrategy.HYBRID_RERANK_MMR
    vector: VectorSearchConfig = Field(default_factory=VectorSearchConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    rrf: RRFConfig = Field(default_factory=RRFConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    mmr: MMRConfig = Field(default_factory=MMRConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)


class ChatMessage(BaseSchema):
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    strategy: Optional[RagStrategy] = None
    trace_id: Optional[str] = None
    sources: list[dict[str, Any]] = Field(default_factory=list)


class Session(BaseSchema):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = "New Session"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    messages: list[ChatMessage] = Field(default_factory=list)
    documents: list[str] = Field(default_factory=list)


class UploadedDocument(BaseSchema):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    size: int
    type: str
    pages: int = 0
    chunks: int = 0
    token_count: int = 0
    path: Optional[str] = None
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    knowledge_base_id: Optional[str] = None


class KnowledgeBase(BaseSchema):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""
    document_count: int = 0
    chunk_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class BenchmarkCase(BaseSchema):
    id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    expectedAnswer: Optional[str] = None
    context: Optional[str] = None
    difficulty: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class DocumentVersion(BaseSchema):
    id: str = Field(default_factory=lambda: str(uuid4()))
    version: str
    cases_count: int
    cases: list[BenchmarkCase] = Field(default_factory=list)
    change_note: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Dataset(BaseSchema):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""
    current_version: str = "v1"
    versions: list[DocumentVersion] = Field(default_factory=list)
    knowledge_base_id: Optional[str] = None


class EvaluationMetrics(BaseSchema):
    hit_rate: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0


class BenchmarkConfig(BaseSchema):
    strategy: RagStrategy = RagStrategy.HYBRID_RERANK_MMR
    dataset_id: str
    dataset_version: str
    vector: VectorSearchConfig = Field(default_factory=VectorSearchConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    rrf: RRFConfig = Field(default_factory=RRFConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    mmr: MMRConfig = Field(default_factory=MMRConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    metrics: list[MetricKey] = Field(default_factory=lambda: [
        MetricKey.HIT_RATE, MetricKey.RECALL, MetricKey.PRECISION,
        MetricKey.MRR, MetricKey.NDCG, MetricKey.FAITHFULNESS,
        MetricKey.ANSWER_RELEVANCE, MetricKey.CONTEXT_PRECISION,
        MetricKey.CONTEXT_RECALL
    ])


class BenchmarkRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BenchmarkRun(BaseSchema):
    id: str = Field(default_factory=lambda: str(uuid4()))
    config: BenchmarkConfig
    status: BenchmarkRunStatus = BenchmarkRunStatus.PENDING
    aggregate_metrics: EvaluationMetrics = Field(default_factory=EvaluationMetrics)
    total_tests: int = 0
    passed_tests: int = 0
    partial_tests: int = 0
    failed_tests: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class TraceEvent(BaseSchema):
    type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: dict[str, Any]


class FullTrace(BaseSchema):
    id: str
    run_id: str
    session_id: str
    request_id: str
    timestamp: datetime
    query: str
    strategy: RagStrategy
    config: RagEngineConfig
    query_processing: dict[str, Any]
    context: dict[str, Any]
    prompt: dict[str, Any]
    llm: dict[str, Any]
    sources: list[dict[str, Any]]
    total_latency_ms: int
    status: Literal["completed", "failed", "partial"]
    error: Optional[str] = None
    vector_search: Optional[dict[str, Any]] = None
    bm25: Optional[dict[str, Any]] = None
    rrf: Optional[dict[str, Any]] = None
    reranker: Optional[dict[str, Any]] = None
    mmr: Optional[dict[str, Any]] = None
    difficulty_breakdown: dict[str, Any] = Field(default_factory=dict)
    tag_breakdown: dict[str, Any] = Field(default_factory=dict)
    failure_categories: list[dict[str, Any]] = Field(default_factory=list)


# API Request/Response Models
class ChatRequest(BaseSchema):
    query: str
    strategy: Optional[RagStrategy] = None
    knowledge_base_id: Optional[str] = None


class ChatResponse(BaseSchema):
    answer: str
    trace: FullTrace
    sources: list[dict[str, Any]]


class EmbedRequest(BaseSchema):
    texts: list[str]


class EmbedResponse(BaseSchema):
    embeddings: list[list[float]]


class RerankRequest(BaseSchema):
    query: str
    documents: list[str]
    top_n: int = 10
    model: Optional[str] = None


class RerankResponse(BaseSchema):
    results: list[dict[str, Any]]


class HealthResponse(BaseSchema):
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
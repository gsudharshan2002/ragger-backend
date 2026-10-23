"""Week 11: Per-request observability and span instrumentation.

Provides:
- Span model (trace_id, span_id, parent_span_id, stage, timing, tokens, cost)
- TraceContext aggregating spans for a single request
- Helpers to start/finish spans and estimate cost
- A contextvar holding the active trace context so nested async code can
  record spans without threading an explicit argument through every call.

All fields are privacy-safe: no API keys, authorization headers, or
personal information are recorded. Request inputs are stored only as a
sanitized reference (the query string itself, which is the user-facing
search text, not credentials).
"""

import re
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from app.core.config import settings

_ACTIVE_TRACE: ContextVar[Optional["TraceContext"]] = ContextVar(
    "ragger_active_trace", default=None
)


def new_id() -> str:
    return str(uuid.uuid4())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost from token counts using the configured rate.

    Labelled as an estimate: settings.COST_PER_TOKEN is a flat
    approximation, not provider-reported usage.
    """
    return (input_tokens + output_tokens) * settings.COST_PER_TOKEN


@dataclass
class Span:
    """A single timed, instrumented operation within a request trace."""

    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    stage: str
    start_time: str
    end_time: Optional[str] = None
    duration_ms: Optional[int] = None
    status: str = "in_progress"
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    context_ids: list = field(default_factory=list)
    tool_name: Optional[str] = None
    tool_outcome: Optional[str] = None
    retry_count: int = 0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def finish(
        self,
        status: str = "success",
        error: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> "Span":
        end = datetime.now(timezone.utc)
        start = datetime.fromisoformat(self.start_time)
        self.end_time = end.isoformat()
        self.duration_ms = int((end - start).total_seconds() * 1000)
        self.status = status
        self.error = error
        if input_tokens is not None:
            self.input_tokens = input_tokens
        if output_tokens is not None:
            self.output_tokens = output_tokens
        if self.input_tokens or self.output_tokens:
            self.cost_usd = estimate_cost_usd(self.input_tokens, self.output_tokens)
        # Accumulate into the active trace context totals.
        ctx = get_active_trace()
        if ctx is not None:
            ctx.input_tokens += self.input_tokens
            ctx.output_tokens += self.output_tokens
            ctx.total_cost_usd += self.cost_usd
        return self

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TraceContext:
    """Aggregates all spans and request-level telemetry for one request."""

    trace_id: str
    request_id: str
    prompt_version: str
    model_provider: str
    model_id: str
    endpoint: str = ""
    mode: str = ""
    sanitized_input: str = ""
    spans: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0
    cache_hit: Optional[bool] = None
    started_at: str = field(default_factory=now_iso)
    # Span that top-level spans of this request hang under (e.g. the agent run).
    root_span_id: Optional[str] = None

    def add_span(self, span: Span) -> None:
        self.spans.append(span)

    def get_span(self, span_id: str) -> Optional[Span]:
        for s in self.spans:
            if s.span_id == span_id:
                return s
        return None

    def totals(self) -> dict:
        input_tokens = sum(s.input_tokens for s in self.spans)
        output_tokens = sum(s.output_tokens for s in self.spans)
        cost = sum(s.cost_usd for s in self.spans)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": round(cost, 6),
        }

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "prompt_version": self.prompt_version,
            "model_provider": self.model_provider,
            "model_id": self.model_id,
            "endpoint": self.endpoint,
            "mode": self.mode,
            "sanitized_input": self.sanitized_input,
            "started_at": self.started_at,
            "spans": [s.to_dict() for s in self.spans],
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "cache_hit": self.cache_hit,
        }


def get_active_trace() -> Optional[TraceContext]:
    return _ACTIVE_TRACE.get()


def set_active_trace(ctx: Optional[TraceContext]) -> None:
    _ACTIVE_TRACE.set(ctx)


def start_span(
    trace_id: str,
    stage: str,
    parent_span_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Span:
    ctx = get_active_trace()
    if ctx is None:
        raise RuntimeError("No active trace context; call begin_trace() first")
    span = Span(
        span_id=new_id(),
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        stage=stage,
        start_time=now_iso(),
        tool_name=tool_name,
        metadata=metadata or {},
    )
    ctx.add_span(span)
    return span


def begin_trace(
    request_id: str,
    prompt_version: str,
    model_provider: str,
    model_id: str,
    endpoint: str = "",
    mode: str = "",
    sanitized_input: str = "",
) -> TraceContext:
    ctx = TraceContext(
        trace_id=new_id(),
        request_id=request_id,
        prompt_version=prompt_version,
        model_provider=model_provider,
        model_id=model_id,
        endpoint=endpoint,
        mode=mode,
        sanitized_input=sanitized_input,
    )
    set_active_trace(ctx)
    return ctx


@contextmanager
def span_ctx(
    stage: str,
    parent_span_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Optional[Span]:
    """Context manager that starts and finishes a span on the active trace.

    Safe to use when no trace is active (yields None).
    """
    ctx = get_active_trace()
    if ctx is None:
        yield None
        return
    span = Span(
        span_id=new_id(),
        trace_id=ctx.trace_id,
        parent_span_id=parent_span_id,
        stage=stage,
        start_time=now_iso(),
        tool_name=tool_name,
        metadata=metadata or {},
    )
    ctx.add_span(span)
    try:
        yield span
    except Exception as e:
        span.finish(status="error", error=str(e))
        raise
    else:
        span.finish(status="success")



# --- Redaction ---------------------------------------------------------------
# Credential-looking keys only. Plain "token" is deliberately not matched on its own:
# it would also hide token-COUNT fields such as max_tokens / tokens_estimated.
_SENSITIVE_KEY = re.compile(
    r"(api[_-]?key|authorization|(access|refresh|id|auth|session|bearer|csrf)[_-]?token|^token$|secret|password|passwd|credential|cookie)",
    re.I,
)
_SENSITIVE_VALUE = re.compile(r"(Bearer\s+[A-Za-z0-9._\-]+|sk-[A-Za-z0-9_\-]{8,}|nvapi-[A-Za-z0-9_\-]{8,}|gsk_[A-Za-z0-9_\-]{8,})")


def redact(value: Any, _depth: int = 0) -> Any:
    """Return a copy of `value` safe to write to logs/traces: values under
    credential-looking keys are replaced, and bearer tokens / API-key shaped
    strings inside any string are masked."""
    if _depth > 6:
        return "[truncated]"
    if isinstance(value, dict):
        return {
            k: ("[REDACTED]" if _SENSITIVE_KEY.search(str(k)) else redact(v, _depth + 1))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v, _depth + 1) for v in value]
    if isinstance(value, str):
        return _SENSITIVE_VALUE.sub("[REDACTED]", value)
    return value


def record_span(
    stage: str,
    duration_ms: int,
    status: str = "success",
    *,
    parent_span_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    tool_outcome: Optional[str] = None,
    context_ids: Optional[list] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    error: Optional[str] = None,
    retry_count: int = 0,
    metadata: Optional[dict] = None,
) -> Optional[Span]:
    """Record an already-finished operation as a span on the active trace.

    For code paths where wrapping the operation in a context manager is
    awkward (generators, existing timing variables). The span is back-dated
    by `duration_ms`. No-op (returns None) when no trace is active, so
    instrumentation can never break a request.
    """
    ctx = get_active_trace()
    if ctx is None:
        return None
    try:
        end = datetime.now(timezone.utc)
        span = Span(
            span_id=new_id(),
            trace_id=ctx.trace_id,
            parent_span_id=parent_span_id if parent_span_id is not None else ctx.root_span_id,
            stage=stage,
            start_time=(end - timedelta(milliseconds=duration_ms)).isoformat(),
            end_time=end.isoformat(),
            duration_ms=int(duration_ms),
            status=status,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=estimate_cost_usd(input_tokens, output_tokens) if (input_tokens or output_tokens) else 0.0,
            context_ids=[str(i) for i in (context_ids or [])],
            tool_name=tool_name,
            tool_outcome=tool_outcome,
            retry_count=retry_count,
            error=(redact(error)[:300] if error else None),
            metadata=redact(metadata or {}),
        )
        ctx.add_span(span)
        ctx.input_tokens += span.input_tokens
        ctx.output_tokens += span.output_tokens
        ctx.total_cost_usd += span.cost_usd
        return span
    except Exception:  # observability must never break the request
        return None

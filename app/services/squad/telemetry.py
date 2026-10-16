"""Week 10: Telemetry collection for squad vs single-agent comparison.

Tracks per-case results, latency, token usage, cost, and every inter-agent
delegation hop (handoffs) in structured JSON Lines format.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HANDOFFS_LOG_PATH = Path(__file__).resolve().parents[3] / "artifacts" / "week10" / "handoffs.log"


@dataclass
class HandoffRecord:
    """A single inter-agent delegation hop."""

    timestamp: str
    case_id: str
    source_agent: str
    destination_agent: str
    delegation_purpose: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: int
    status: str  # success | failure | timeout
    retry_number: int = 0
    error: str | None = None
    tokens_estimated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class TelemetryCollector:
    """Collects telemetry for a single benchmark run (one arm, one case)."""

    def __init__(self, arm: str, case_id: str):
        self.arm = arm
        self.case_id = case_id
        self.run_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.end_time: float | None = None

        self.latency_ms = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.cost = 0.0

        self.passed = False
        self.answer = ""
        self.error: str | None = None
        self.status: str = "pending"  # pending | completed | failed

        self.handoffs: list[HandoffRecord] = []

    def record_handoff(
        self,
        source_agent: str,
        destination_agent: str,
        delegation_purpose: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        status: str,
        retry_number: int = 0,
        error: str | None = None,
        tokens_estimated: bool = False,
    ) -> None:
        hop = HandoffRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            case_id=self.case_id,
            source_agent=source_agent,
            destination_agent=destination_agent,
            delegation_purpose=delegation_purpose,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=latency_ms,
            status=status,
            retry_number=retry_number,
            error=error,
            tokens_estimated=tokens_estimated,
        )
        self.handoffs.append(hop)

    def finish(self, passed: bool, answer: str, error: str | None = None, status: str = "completed") -> None:
        self.end_time = time.time()
        self.latency_ms = int((self.end_time - self.start_time) * 1000)
        self.passed = passed
        self.answer = answer
        self.error = error
        self.status = status

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "arm": self.arm,
            "case_id": self.case_id,
            "passed": self.passed,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "answer": self.answer,
            "error": self.error,
            "status": self.status,
            "handoffs": [h.to_dict() for h in self.handoffs],
        }


class BenchmarkTelemetry:
    """Aggregate telemetry across all cases for one arm."""

    def __init__(self, arm: str):
        self.arm = arm
        self.cases: list[TelemetryCollector] = []

    def add(self, collector: TelemetryCollector) -> None:
        self.cases.append(collector)

    def summary(self) -> dict:
        n = len(self.cases)
        passed = sum(1 for c in self.cases if c.passed)
        latencies = [c.latency_ms for c in self.cases if c.latency_ms > 0]
        latencies.sort()

        def percentile(values: list[float], pct: float) -> float:
            if not values:
                return 0.0
            k = max(0, min(len(values) - 1, int(__import__("math").ceil(pct / 100 * len(values))) - 1))
            return values[k]

        total_input = sum(c.input_tokens for c in self.cases)
        total_output = sum(c.output_tokens for c in self.cases)
        total_tokens = total_input + total_output
        total_cost = sum(c.cost for c in self.cases)

        all_handoffs = [h for c in self.cases for h in c.handoffs]

        return {
            "arm": self.arm,
            "case_count": n,
            "passed": passed,
            "pass_rate": round(passed / n, 4) if n else 0.0,
            "p50_latency_ms": int(percentile(latencies, 50)) if latencies else 0,
            "p99_latency_ms": int(percentile(latencies, 99)) if latencies else 0,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 6),
            "cost_per_question": round(total_cost / n, 6) if n else 0.0,
            "cases": [c.to_dict() for c in self.cases],
            "handoffs": [h.to_dict() for h in all_handoffs],
        }


def write_handoffs_jsonl(collectors: list[TelemetryCollector], path: Path = HANDOFFS_LOG_PATH) -> None:
    """Write all handoff records to JSON Lines file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in collectors:
            for h in c.handoffs:
                f.write(h.to_jsonl() + "\n")


def estimate_tokens(text: str) -> int:
    """Word-count-based token estimate (same heuristic as rag_engine.py)."""
    if not text:
        return 0
    return max(1, int(len(text.split()) / 0.75))

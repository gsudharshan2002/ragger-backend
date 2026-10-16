"""Week 11: Seed a labeled synthetic incident fixture.

Creates a deterministic incident trace representing a "deprecated method" bad
answer: the model recommended the deprecated `assignee` field when creating a
GitHub issue, instead of the current `assignees` replacement.

The trace is stored in app/data/traces.jsonl (so the implemented search
mechanism can find it) and exported to artifacts/week11/trace.json.

This is explicitly a SYNTHETIC fixture, not an actual customer incident. It is
grounded in real documentation content (the chunks record that `assignee` is
deprecated in favor of `assignees`).
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.storage import add_trace  # noqa: E402

TRACE_ID = "week11-incident-deprecated-assignee-001"
CASE_ID = "gh-deprecated-001"
NOW = datetime.now(timezone.utc).isoformat()


def build_incident() -> dict:
    spans = [
        {
            "span_id": "span-retrieval-1",
            "trace_id": TRACE_ID,
            "parent_span_id": None,
            "stage": "retrieval.vector",
            "start_time": NOW,
            "end_time": NOW,
            "duration_ms": 14,
            "status": "success",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "context_ids": ["4704", "5262"],
            "tool_name": "vector_search",
            "tool_outcome": "found",
            "retry_count": 0,
            "error": None,
            "metadata": {"top_k": 5, "query": "assign users to github issue"},
        },
        {
            "span_id": "span-rerank-1",
            "trace_id": TRACE_ID,
            "parent_span_id": None,
            "stage": "retrieval.reranker",
            "start_time": NOW,
            "end_time": NOW,
            "duration_ms": 312,
            "status": "success",
            "input_tokens": 120,
            "output_tokens": 40,
            "cost_usd": 0.00008,
            "context_ids": ["4704"],
            "tool_name": "reranker",
            "tool_outcome": "reranked",
            "retry_count": 0,
            "error": None,
            "metadata": {"model": "rerank-english-v3", "candidates": 10, "top_n": 5},
        },
        {
            "span_id": "span-llm-1",
            "trace_id": TRACE_ID,
            "parent_span_id": None,
            "stage": "model.generation",
            "start_time": NOW,
            "end_time": NOW,
            "duration_ms": 8420,
            "status": "completed",
            "input_tokens": 940,
            "output_tokens": 210,
            "cost_usd": 0.000575,
            "context_ids": ["4704", "5262"],
            "tool_name": "llm",
            "tool_outcome": "generated",
            "retry_count": 0,
            "error": None,
            "metadata": {"model": "openai/gpt-oss-20b", "temperature": 0.2},
        },
        {
            "span_id": "span-response-1",
            "trace_id": TRACE_ID,
            "parent_span_id": None,
            "stage": "response.assembly",
            "start_time": NOW,
            "end_time": NOW,
            "duration_ms": 8,
            "status": "success",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "context_ids": ["4704", "5262"],
            "tool_name": None,
            "tool_outcome": "assembled",
            "retry_count": 0,
            "error": None,
            "metadata": {},
        },
    ]

    input_tokens = sum(s["input_tokens"] for s in spans)
    output_tokens = sum(s["output_tokens"] for s in spans)
    cost_usd = round(sum(s["cost_usd"] for s in spans), 6)
    total_latency_ms = sum(s["duration_ms"] for s in spans)

    trace = {
        "id": TRACE_ID,
        "run_id": "week11-incident-run-001",
        "session_id": "week11-incident-session-001",
        "request_id": "week11-incident-request-001",
        "timestamp": NOW,
        "query": "When creating a GitHub issue via POST /repos/{owner}/{repo}/issues, "
        "which field do I use to assign users?",
        "strategy": "hybrid-rerank-mmr",
        "config": {},
        "events": [],
        "query_processing": {
            "original_query": "When creating a GitHub issue via POST /repos/{owner}/{repo}/issues, "
            "which field do I use to assign users?",
            "token_count": 21,
            "processing_duration_ms": 326,
        },
        "context": {
            "chunks": [
                {
                    "chunk_id": "4704",
                    "document_id": "09d46ce8-c141-4465-b10a-6fc485caa016",
                    "document_name": "github_combined_reference.pdf",
                    "page": 5,
                    "section": "Replaces the",
                    "content": "Replaces the deprecated assignee field. labels array of strings No [] "
                    "Labels to associate with this issue. type string No null ... assignees array of strings",
                    "token_count": 45,
                    "score": 0.92,
                    "method": "final",
                },
                {
                    "chunk_id": "5262",
                    "document_id": "09d46ce8-c141-4465-b10a-6fc485caa016",
                    "document_name": "github_combined_reference.pdf",
                    "page": 9,
                    "section": "Previous version (2022-11-28)",
                    "content": "The prior version of this endpoint only accepted a single assignee "
                    "(string) rather than an array.",
                    "token_count": 26,
                    "score": 0.78,
                    "method": "final",
                },
            ],
            "total_tokens": 71,
            "chunk_count": 2,
            "document_count": 1,
        },
        "prompt": {
            "system": ("You are a RAG assistant... - Be concise..."),
            "user": "When creating a GitHub issue ... assign users?",
            "context": "Replaces the deprecated assignee field ... assignees ...",
            "system_tokens": 120,
            "context_tokens": 71,
            "user_tokens": 21,
            "total_tokens": 212,
        },
        "llm": {
            "provider": "groq",
            "model": "openai/gpt-oss-20b",
            "latency_ms": 8420,
            "input_tokens": 940,
            "output_tokens": 210,
            "total_tokens": 1150,
            "status": "completed",
        },
        "sources": [
            {
                "document_id": "09d46ce8-c141-4465-b10a-6fc485caa016",
                "document_name": "github_combined_reference.pdf",
                "page": 5,
                "section": "Replaces the",
                "chunk_id": "4704",
                "retrieval_method": "final",
                "score": 0.92,
            },
            {
                "document_id": "09d46ce8-c141-4465-b10a-6fc485caa016",
                "document_name": "github_combined_reference.pdf",
                "page": 9,
                "section": "Previous version (2022-11-28)",
                "chunk_id": "5262",
                "retrieval_method": "final",
                "score": 0.78,
            },
        ],
        "total_latency_ms": total_latency_ms,
        "status": "completed",
        "error": None,
        "spans": spans,
        "prompt_version": "1",
        "cost_usd": cost_usd,
        "cache_hit": False,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "kind": "rag",
        "knowledge_base_id": None,
        "answer": "Use the `assignee` field (string) to assign a user when creating "
        "an issue via POST /repos/{owner}/{repo}/issues.",
        "synthetic": True,
        "incident": True,
        "incident_type": "deprecated_method",
        "deprecated_method": "assignee",
        "correct_replacement": "assignees",
        "root_cause": "Retrieved outdated 2022-11-28 documentation that still describes "
        "`assignee` as valid; prompt v1 had no deprecation-guard instruction, so the "
        "model recommended the deprecated field as the current implementation.",
    }
    return trace


async def main() -> int:
    trace = build_incident()
    await add_trace(trace)
    out_dir = Path("artifacts") / "week11"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trace.json").write_text(
        json.dumps(trace, indent=2, default=str), encoding="utf-8"
    )
    print(f"seeded incident trace {TRACE_ID} into traces.jsonl")
    print(f"exported complete trace to artifacts/week11/trace.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

"""Week 11: Regression test for the deprecated-method incident.

Covers case gh-deprecated-001: the application must not recommend the
deprecated `assignee` field when creating a GitHub issue; it must direct
users to the current `assignees` replacement.

This is a deterministic regression test (no live LLM / API key required). It
verifies the targeted fix at the instruction level: the system prompt must
contain a deprecation-guard rule, and the retrieved context must surface the
current-version replacement.

RED (pre-fix): prompt version 1 has no deprecation guard -> this test fails.
GREEN (post-fix): prompt version 2 adds the guard -> this test passes.
"""

import asyncio
import json
from types import SimpleNamespace

from app.core.config import settings
from app.models.schemas import RagEngineConfig, RagStrategy
from app.services.rag_engine import build_prompt
from app.services.storage import add_trace, search_traces


CASE_ID = "gh-deprecated-001"
DEPRECATED_METHOD = "assignee"
CURRENT_REPLACEMENT = "assignees"
INCIDENT_TRACE_ID = "week11-incident-deprecated-assignee-001"


def _chunk(chunk_id, document, page, section, content):
    return SimpleNamespace(
        id=chunk_id,
        document_id="doc-deprecated-001",
        document_name=document,
        page=page,
        section=section,
        content=content,
        token_count=len(content.split()),
    )


def _build_case_context():
    return [
        _chunk(
            "chunk-dep-old",
            "old-version.md",
            1,
            "Previous version (2022-11-28)",
            "The prior version of this endpoint only accepted a single assignee "
            "(string).",
        ),
        _chunk(
            "chunk-dep-current",
            "current-version.md",
            1,
            "Create an issue",
            "Replaces the deprecated assignee field. Use assignees (array of "
            "strings) to assign users when creating an issue.",
        ),
    ]


def _seed_incident_trace():
    trace = {
        "id": INCIDENT_TRACE_ID,
        "run_id": "week11-incident-run-001",
        "session_id": "week11-incident-session-001",
        "request_id": "week11-incident-request-001",
        "timestamp": "2026-10-16T00:00:00+00:00",
        "query": "When creating a GitHub issue via POST /repos/{owner}/{repo}/issues, "
        "which field assigns users?",
        "strategy": "hybrid-rerank-mmr",
        "config": {},
        "events": [],
        "query_processing": {
            "original_query": "assign users github issue",
            "token_count": 5,
            "processing_duration_ms": 100,
        },
        "context": {
            "chunks": [],
            "total_tokens": 0,
            "chunk_count": 0,
            "document_count": 0,
        },
        "prompt": {
            "system": "system",
            "user": "assign users?",
            "context": "assignees",
            "system_tokens": 10,
            "context_tokens": 10,
            "user_tokens": 3,
            "total_tokens": 23,
        },
        "llm": {
            "provider": "groq",
            "model": "openai/gpt-oss-20b",
            "latency_ms": 100,
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "status": "completed",
        },
        "sources": [],
        "total_latency_ms": 100,
        "status": "completed",
        "error": None,
        "spans": [],
        "prompt_version": "1",
        "cost_usd": 0.0,
        "cache_hit": False,
        "input_tokens": 10,
        "output_tokens": 5,
        "kind": "rag",
        "knowledge_base_id": None,
        "answer": "Use the `assignee` field.",
        "synthetic": True,
        "incident": True,
        "incident_type": "deprecated_method",
        "deprecated_method": DEPRECATED_METHOD,
        "correct_replacement": CURRENT_REPLACEMENT,
        "root_cause": "Outdated retrieval; no deprecation guard in prompt v1.",
    }
    asyncio.run(add_trace(trace))


def test_dataset_case_exists():
    """The regression case is part of the permanent core evaluation suite."""
    with open("evals/developer_docs_cases.json") as f:
        cases = json.load(f)
    ids = {c["id"] for c in cases}
    assert CASE_ID in ids
    case = next(c for c in cases if c["id"] == CASE_ID)
    assert CURRENT_REPLACEMENT in case["expected_answer_keywords"]
    assert DEPRECATED_METHOD in case.get("prohibited_keywords", [])


def test_system_prompt_prohibits_deprecated_methods():
    """Targeted fix: the system prompt must instruct against deprecated methods.

    RED: prompt version 1 lacks this rule -> assertion fails.
    GREEN: prompt version 2 adds the rule -> assertion passes.
    """
    system = settings.SYSTEM_PROMPT
    assert "deprecated" in system.lower(), (
        "system prompt must warn against deprecated methods (RED: missing)"
    )
    assert CURRENT_REPLACEMENT in system, (
        "system prompt must name the current replacement (RED: missing)"
    )


def test_retrieved_context_surfaces_current_replacement():
    """Retrieval must surface the current-version `assignees` guidance so the
    model can answer correctly instead of repeating the deprecated `assignee`."""
    config = RagEngineConfig(
        strategy=RagStrategy.HYBRID_RERANK_MMR,
        top_k=5,
    )
    prompt = build_prompt(
        query=(
            "When creating a GitHub issue via POST /repos/{owner}/{repo}/issues, "
            "which field assigns users instead of the deprecated assignee field?"
        ),
        context_chunks=_build_case_context(),
        config=config,
    )
    context = prompt["context"]
    assert CURRENT_REPLACEMENT in context, (
        "retrieved context must surface the current replacement"
    )


def test_incident_trace_is_searchable():
    """The synthetic incident trace must be findable via the search mechanism."""
    _seed_incident_trace()
    results = asyncio.run(
        search_traces({"trace_id": INCIDENT_TRACE_ID}, limit=5)
    )
    assert len(results) == 1
    tr = results[0]
    assert tr.get("synthetic") is True
    assert tr.get("incident_type") == "deprecated_method"
    assert tr.get("deprecated_method") == DEPRECATED_METHOD
    assert tr.get("correct_replacement") == CURRENT_REPLACEMENT

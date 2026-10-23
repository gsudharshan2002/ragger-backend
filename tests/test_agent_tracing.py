"""One agent run = one trace: spans for the model calls and tools, searchable by
trace id and by retrieved chunk id, token totals from the model-call spans, and
credentials redacted. Uses a scripted LLM so it needs no network or API key."""

import asyncio
import json

from app.models.schemas import AgentRunRequest, RagStrategy, StoredChunk
from app.services import llm
from app.services.agent_engine import AgentEngine
from app.services.observability import get_active_trace, redact
from app.services.storage import add_chunks, init_storage, search_traces


def _scripted_llm(monkeypatch):
    calls = {"n": 0}

    async def fake(system_prompt, user_prompt, model=None, temperature=0.7, max_tokens=1024, top_p=1.0):
        calls["n"] += 1
        if "ReAct" in system_prompt:
            tool = "retrieve" if calls["n"] == 1 else "answer"
            text = json.dumps({"thought": "t", "tool": tool, "inputs": {"query": "assignees array"} if tool == "retrieve" else {}})
        else:
            text = "Use assignees (array)."
        yield {"type": "token", "content": text}
        yield {"type": "done", "tokens": {"input": 100, "output": 10, "total": 110, "estimated": False}}

    monkeypatch.setattr(llm, "_stream_completion", fake)

    async def configured():
        return True

    monkeypatch.setattr("app.services.agent_engine.is_llm_configured", configured)
    return calls


def _run(query: str) -> dict:
    async def go():
        await init_storage()
        await add_chunks([
            StoredChunk(id="c-assignees", document_id="d1", document_name="current-version.md",
                        content="Use the assignees array. The assignee field was removed.", page=1),
            StoredChunk(id="c-rate", document_id="d1", document_name="current-version.md",
                        content="Read rate limits from resources.core on GET /rate_limit.", page=1),
            StoredChunk(id="c-repo", document_id="d1", document_name="current-version.md",
                        content="The has_downloads property was removed from the repository object.", page=1),
            StoredChunk(id="c-pr", document_id="d1", document_name="current-version.md",
                        content="The merge_commit_sha field was removed from pull request responses.", page=1),
        ])
        data = {}
        async for ev in AgentEngine().run_stream(AgentRunRequest(query=query, max_steps=4, strategy=RagStrategy.BM25)):
            if ev.get("type") == "trace.completed":
                data = ev["data"]
        found = await search_traces({"trace_id": data["traceId"]})
        by_chunk = await search_traces({"chunk_id": "c-assignees"})
        return {"response": data, "trace": found[0] if found else None, "by_chunk": by_chunk}

    return asyncio.run(go())


def test_agent_run_is_one_searchable_trace(monkeypatch):
    calls = _scripted_llm(monkeypatch)
    out = _run("which field assigns users, assignees? my key is Bearer abc123secret")
    trace, resp = out["trace"], out["response"]

    assert trace is not None, "trace must be findable by its trace id"
    assert trace["id"] == resp["traceId"]
    stages = {s["stage"] for s in trace["spans"]}
    assert {"agent.run", "llm.generate", "tool.retrieve", "tool.answer"} <= stages
    assert all(s["trace_id"] == trace["id"] for s in trace["spans"])

    llm_spans = [s for s in trace["spans"] if s["stage"] == "llm.generate"]
    assert len(llm_spans) == calls["n"]
    # response tokens are the sum of model-call spans, each counted once
    assert resp["inputTokens"] == sum(s["input_tokens"] for s in llm_spans) == 100 * calls["n"]
    assert trace["cost_usd"] > 0 and trace["prompt_version"] and trace["llm"]["model"]

    retrieve = next(s for s in trace["spans"] if s["stage"] == "tool.retrieve")
    assert "c-assignees" in retrieve["context_ids"]
    assert any(t["id"] == trace["id"] for t in out["by_chunk"]), "searchable by retrieved chunk id"

    assert "abc123secret" not in json.dumps(trace)
    assert get_active_trace() is None, "trace must not leak into the next request"


def test_redact_masks_credentials():
    cleaned = redact({"api_key": "k", "headers": {"Authorization": "Bearer zzz"}, "q": "x nvapi-abcdefgh1234"})
    assert cleaned["api_key"] == "[REDACTED]"
    assert cleaned["headers"]["Authorization"] == "[REDACTED]"
    assert "nvapi" not in cleaned["q"]


def test_pure_vector_strategy_falls_back_to_bm25_when_vector_search_is_unavailable(monkeypatch):
    """Regression for a live trace: the agent picked `vector`, which returned no
    chunks because no embedding provider works, wasting a step."""
    from app.core import capabilities
    from app.services.rag_engine import RagEngine

    monkeypatch.setattr(capabilities, "detect_capabilities", lambda: {"vector_search_available": False, "degraded": []})

    async def go():
        await init_storage()
        await add_chunks([
            StoredChunk(id="v-1", document_id="d9", document_name="current-version.md", content="Use the assignees array.", page=1),
            StoredChunk(id="v-2", document_id="d9", document_name="current-version.md", content="Read rate limits from resources.core.", page=1),
            StoredChunk(id="v-3", document_id="d9", document_name="current-version.md", content="has_downloads was removed.", page=1),
            StoredChunk(id="v-4", document_id="d9", document_name="current-version.md", content="merge_commit_sha was removed.", page=1),
        ])
        return await RagEngine().get_context("assignees array", RagStrategy.VECTOR)

    chunks = asyncio.run(go())
    assert [c.id for c in chunks][:1] == ["v-1"]


def test_traces_are_findable_by_question_and_answer_text(monkeypatch):
    _scripted_llm(monkeypatch)
    out = _run("which field assigns users in the issue payload?")
    tid = out["trace"]["id"]

    async def find(**f):
        return [t["id"] for t in await search_traces(f)]

    assert tid in asyncio.run(find(query_contains="ASSIGNS USERS"))
    assert tid in asyncio.run(find(answer_contains="assignees (array)", query_contains="issue payload"))
    assert tid not in asyncio.run(find(answer_contains="no such text in any answer"))


def test_redact_keeps_token_count_fields_but_masks_credentials():
    cleaned = redact({"max_tokens": 512, "tokens_estimated": False, "input_tokens": 9,
                      "access_token": "abc", "token": "abc", "client_secret": "s", "Cookie": "c"})
    assert cleaned["max_tokens"] == 512 and cleaned["tokens_estimated"] is False and cleaned["input_tokens"] == 9
    assert cleaned["access_token"] == cleaned["token"] == cleaned["client_secret"] == cleaned["Cookie"] == "[REDACTED]"

import asyncio

import pytest

from app.services import llm
from app.services.squad.fault_injection import inject_fault_for, is_fault_enabled, should_inject_fault
from app.services.squad.orchestrator import SquadOrchestrator
from app.services.squad.telemetry import TelemetryCollector
from app.services.squad.version_worker import resolve_api_version


def _fake_llm(route: str = '["version_deprecation"]', answer: str = "It was removed [doc p.1]."):
    async def fake(system_prompt, user_prompt, model=None, temperature=0.7, max_tokens=1024, top_p=1.0):
        text = route if "multi-agent RAG squad" in system_prompt else answer
        yield {"type": "token", "content": text}
        yield {"type": "done", "tokens": {"input": 100, "output": 10, "total": 110, "estimated": False}}
    return fake


def _worker(results):
    """A worker whose successive run() calls return `results` in order."""
    calls = {"n": 0}

    class W:
        timeout_seconds = 30.0

        async def run(self, query, case_id, *a, **k):
            i = min(calls["n"], len(results) - 1)
            calls["n"] += 1
            return dict(results[i])

    w = W()
    w.calls = calls
    return w


OK = {"status": "success", "evidence": "[d p.1] x removed", "api_version": "2026-03-10",
      "input_tokens": 5, "output_tokens": 50, "latency_ms": 3, "error": None}
HTTP500 = {"status": "failure", "evidence": "", "error": "HTTP 500: boom", "fault_injected": True,
           "input_tokens": 5, "output_tokens": 3, "latency_ms": 1}
NO_EVIDENCE = {"status": "error", "evidence": "", "error": "no_deprecation_info_found",
               "input_tokens": 5, "output_tokens": 0, "latency_ms": 1}


def _run(monkeypatch, version_results, **llm_kwargs):
    monkeypatch.setattr(llm, "generate_completion_stream", _fake_llm(**llm_kwargs))
    orch = SquadOrchestrator(max_retries=2, retry_backoff_s=0)
    orch.version_worker = _worker(version_results)
    orch.code_sample_worker = _worker([{"status": "success", "evidence": "doc", "code_samples": [],
                                        "input_tokens": 1, "output_tokens": 1, "latency_ms": 1}])
    tc = TelemetryCollector(arm="squad", case_id="t-1")
    result = asyncio.run(orch.run("Is has_downloads removed?", "t-1", tc))
    return orch, tc, result


def test_http_500_is_retried_within_limit_and_disclosed(monkeypatch):
    orch, tc, result = _run(monkeypatch, [HTTP500])
    hops = [h for h in tc.handoffs if h.destination_agent == "version_deprecation"]
    assert [h.retry_number for h in hops] == [0, 1, 2]  # 1 attempt + max_retries
    assert all(h.status == "failure" and h.error for h in hops)
    assert orch.version_worker.calls["n"] == 3
    assert result["version_verification"] == "unavailable"
    assert "version/deprecation evidence was unavailable" in result["answer"]


def test_transient_failure_recovers_without_disclosure(monkeypatch):
    orch, tc, result = _run(monkeypatch, [HTTP500, OK])
    hops = [h for h in tc.handoffs if h.destination_agent == "version_deprecation"]
    assert [h.status for h in hops] == ["failure", "success"]
    assert result["version_verification"] == "verified"
    assert "unavailable" not in result["answer"]


def test_no_evidence_is_not_retried_but_is_disclosed(monkeypatch):
    orch, tc, result = _run(monkeypatch, [NO_EVIDENCE])
    assert orch.version_worker.calls["n"] == 1
    hop = next(h for h in tc.handoffs if h.destination_agent == "version_deprecation")
    assert hop.status == "failure" and hop.error == "no_deprecation_info_found"
    assert result["version_verification"] == "no_evidence"
    assert "unverified" in result["answer"]


def test_timeout_hop_status(monkeypatch):
    timeout = dict(HTTP500, error="version/deprecation check timed out after 30.0s", fault_injected=False)
    _, tc, _ = _run(monkeypatch, [timeout])
    hops = [h for h in tc.handoffs if h.destination_agent == "version_deprecation"]
    assert {h.status for h in hops} == {"timeout"}


def test_totals_count_only_model_calls_once(monkeypatch):
    _, tc, result = _run(monkeypatch, [OK])
    model_hops = [h for h in tc.handoffs if h.model_call]
    assert len(model_hops) == 2  # routing + final answer
    assert result["input_tokens"] == 200 and result["output_tokens"] == 20
    assert result["total_tokens"] == 220
    assert result["tokens_estimated"] is False
    assert not any(h.model_call for h in tc.handoffs if h.destination_agent in ("version_deprecation", "code_sample"))


def test_unparseable_route_runs_both_specialists(monkeypatch):
    _, tc, result = _run(monkeypatch, [OK], route="not json")
    dests = {h.destination_agent for h in tc.handoffs}
    assert {"version_deprecation", "code_sample"} <= dests
    assert result["steps"][0]["routed_by"] == "default_both"


def test_fault_injection_is_off_by_default_and_scoped(monkeypatch):
    monkeypatch.delenv("RAGGER_FAULT_INJECTION", raising=False)
    monkeypatch.delenv("RAGGER_FAULT_CASE_ID", raising=False)
    assert not is_fault_enabled() and not should_inject_fault("gh-006")
    with inject_fault_for("gh-006"):
        assert should_inject_fault("gh-006")
        assert not should_inject_fault("gh-007")  # only the designated case
        assert not should_inject_fault("gh-006", "code_sample")  # only this worker
    assert not should_inject_fault("gh-006")  # restored


@pytest.mark.parametrize("query,expected", [
    ("Using API Version 2022-11-28, what status code ...", "2022-11-28"),
    ("On API Version 2026-03-10 does X exist", "2026-03-10"),
    ("What is the current version behaviour?", "2026-03-10"),
    ("Using API version 2019-01-01 ...", "2026-03-10"),  # unindexed date is not guessed
])
def test_resolve_api_version(query, expected):
    assert resolve_api_version(query) == expected

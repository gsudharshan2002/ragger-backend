"""Week 10: Benchmark harness for single-agent vs multi-agent squad.

Runs the same 10 evaluation cases through both arms:
  Arm A: existing single-agent (AgentEngine at /agent/run)
  Arm B: new orchestrator-plus-two-specialists squad

Collects pass rate, p50/p99 latency, token usage, cost, and handoffs.
Generates artifacts/week10/race_table.md, handoffs.log, failure_case.md, verdict.md.

Run:  uv run python -m app.services.squad.benchmark
"""

import asyncio
import csv
import json
import math
import statistics
import time
from pathlib import Path

from app.core.config import settings
from app.services.evaluation.race_harness import _percentile, _passed
from app.services.squad.eval_cases import load_selected_cases
from app.services.squad.telemetry import (
    TelemetryCollector,
    BenchmarkTelemetry,
    write_handoffs_jsonl,
    estimate_tokens,
)
from app.services.squad.orchestrator import SquadOrchestrator
from app.services.squad.fault_experiment import generate_verdict

RESULTS_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "week10"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


async def run_single_agent(query: str) -> dict:
    """Run one query through AgentEngine (single-agent baseline)."""
    from app.models.schemas import AgentRunRequest
    from app.services.agent_engine import AgentEngine

    engine = AgentEngine()
    request = AgentRunRequest(query=query, max_steps=5)

    answer = ""
    total_latency_ms = 0
    input_tokens = 0
    output_tokens = 0

    async for event in engine.run_stream(request):
        if event.get("type") == "trace.completed":
            data = event.get("data", {})
            answer = data.get("answer", "")
            total_latency_ms = data.get("totalLatencyMs", 0)
            input_tokens = data.get("inputTokens", 0)
            output_tokens = data.get("outputTokens", 0)

    return {
        "answer": answer,
        "total_latency_ms": total_latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


async def run_squad(query: str, case_id: str, telemetry: TelemetryCollector) -> dict:
    """Run one query through the squad orchestrator."""
    orchestrator = SquadOrchestrator(max_steps=5, temperature=0.2)
    return await orchestrator.run(query, case_id, telemetry)


def _cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens + output_tokens) * settings.COST_PER_TOKEN


async def run_benchmark() -> dict:
    cases = load_selected_cases()
    single = BenchmarkTelemetry(arm="single_agent")
    squad = BenchmarkTelemetry(arm="squad")

    for case in cases:
        case_id = case["id"]
        question = case["question"]
        keywords = case.get("expected_answer_keywords", case.get("expected_keywords", []))
        print(f"Case {case_id}: {question[:70]}...")

        # --- Arm A: single agent ---
        tc_single = TelemetryCollector(arm="single_agent", case_id=case_id)
        result_a = await run_single_agent(question)
        passed_a = _passed(result_a["answer"], keywords)
        tc_single.input_tokens = result_a["input_tokens"]
        tc_single.output_tokens = result_a["output_tokens"]
        tc_single.total_tokens = tc_single.input_tokens + tc_single.output_tokens
        tc_single.cost = _cost(tc_single.input_tokens, tc_single.output_tokens)
        tc_single.finish(passed=passed_a, answer=result_a["answer"])
        single.add(tc_single)
        print(f"  single: pass={passed_a} latency={result_a['total_latency_ms']}ms "
              f"tokens={tc_single.total_tokens}")

        # --- Arm B: squad ---
        tc_squad = TelemetryCollector(arm="squad", case_id=case_id)
        result_b = await run_squad(question, case_id, tc_squad)
        passed_b = _passed(result_b["answer"], keywords)
        tc_squad.finish(passed=passed_b, answer=result_b["answer"], error=result_b.get("error"))
        squad.add(tc_squad)
        print(f"  squad:  pass={passed_b} latency={result_b['total_latency_ms']}ms "
              f"tokens={tc_squad.total_tokens} hops={len(tc_squad.handoffs)}")

    # Write artifacts
    single_summary = single.summary()
    squad_summary = squad.summary()

    # handoffs.log (squad only)
    write_handoffs_jsonl(squad.cases)

    # race_table.md
    _write_race_table(single_summary, squad_summary, cases)

    # verdict.md (from actual results)
    generate_verdict(single_summary, squad_summary)

    # bonus: agentcard.json
    _write_agentcard()

    # CSV details
    _write_csv(single.cases, squad.cases)

    return {
        "single_agent": single_summary,
        "squad": squad_summary,
    }


def _write_race_table(single: dict, squad: dict, cases: list[dict]) -> None:
    lines = []
    lines.append("# Week 10: Single-Agent vs Multi-Agent Squad Comparison")
    lines.append("")
    lines.append(f"Dataset: `evals/developer_docs_cases.json` (10 GitHub API version-specific cases)")
    lines.append(f"Case IDs: {', '.join(c['id'] for c in cases)}")
    lines.append(f"Evaluator: keyword-presence (all expected keywords must appear in answer)")
    lines.append(f"Model: `{settings.LLM_PROVIDER}` / `{settings.GROQ_MODEL if settings.LLM_PROVIDER=='groq' else settings.GEMINI_MODEL if settings.LLM_PROVIDER=='gemini' else settings.OPENROUTER_MODEL}`")
    lines.append(f"Cost per token: `{settings.COST_PER_TOKEN}` (estimated, not provider-reported)")
    lines.append(f"Token method: estimated (word-count heuristic); not provider-reported usage")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Single Agent | Squad |")
    lines.append("|--------|-------------|-----|")
    lines.append(f"| Pass rate | {single['pass_rate']*100:.1f}% ({single['passed']}/{single['case_count']}) | {squad['pass_rate']*100:.1f}% ({squad['passed']}/{squad['case_count']}) |")
    lines.append(f"| p50 latency (ms) | {single['p50_latency_ms']} | {squad['p50_latency_ms']} |")
    lines.append(f"| p99 latency (ms) | {single['p99_latency_ms']} | {squad['p99_latency_ms']} |")
    lines.append(f"| Total input tokens | {single['input_tokens']} | {squad['input_tokens']} |")
    lines.append(f"| Total output tokens | {single['output_tokens']} | {squad['output_tokens']} |")
    lines.append(f"| Total tokens | {single['total_tokens']} | {squad['total_tokens']} |")
    lines.append(f"| Total cost | ${single['total_cost']:.6f} | ${squad['total_cost']:.6f} |")
    lines.append(f"| Avg cost/question | ${single['cost_per_question']:.6f} | ${squad['cost_per_question']:.6f} |")

    # Context re-send multiplier
    if single["total_tokens"] > 0:
        multiplier = round(squad["total_tokens"] / single["total_tokens"], 1)
    else:
        multiplier = None
        mult_str = "unavailable"
    lines.append("")
    lines.append(f"| Context Re-Send Multiplier | {multiplier if multiplier is not None else 'unavailable'} |")
    if multiplier is not None:
        lines.append(f"| Notes | Multi-Agent Total Tokens / Single-Agent Total Tokens = {squad['total_tokens']} / {single['total_tokens']} |")

    lines.append("")
    lines.append("## Per-Case Results")
    lines.append("")
    lines.append("| Case | Single Pass | Single Latency (ms) | Single Tokens | Squad Pass | Squad Latency (ms) | Squad Tokens | Squad Hops |")
    lines.append("|------|-------------|---------------------|---------------|------------|--------------------|--------------|------------|")
    for c in cases:
        cid = c["id"]
        s_row = next((x for x in single["cases"] if x["case_id"] == cid), None)
        b_row = next((x for x in squad["cases"] if x["case_id"] == cid), None)
        s_pass = "Y" if s_row and s_row["passed"] else "N"
        b_pass = "Y" if b_row and b_row["passed"] else "N"
        s_lat = s_row["latency_ms"] if s_row else 0
        b_lat = b_row["latency_ms"] if b_row else 0
        s_tok = s_row["total_tokens"] if s_row else 0
        b_tok = b_row["total_tokens"] if b_row else 0
        b_hops = len(b_row["handoffs"]) if b_row else 0
        lines.append(f"| {cid} | {s_pass} | {s_lat} | {s_tok} | {b_pass} | {b_lat} | {b_tok} | {b_hops} |")

    lines.append("")
    lines.append("## Methodology Notes")
    lines.append("- Token counts are estimated via word-count heuristic (same as `rag_engine.estimate_tokens`), not provider-reported.")
    lines.append("- Costs are estimates based on `settings.COST_PER_TOKEN`; actual provider pricing may differ.")
    lines.append("- p50/p99 use nearest-rank percentile; p99 of 10 samples is the max value.")
    lines.append("- Squad token counts include orchestrator + both specialist workers + final answer generation.")
    lines.append("- All handoffs are logged in `artifacts/week10/handoffs.log` (JSON Lines).")

    # Largest handoff bottleneck
    all_handoffs = [h for c in squad["cases"] for h in c["handoffs"]]
    if all_handoffs:
        bottleneck = max(all_handoffs, key=lambda h: h["total_tokens"])
        squad_total = squad["total_tokens"]
        pct = round(bottleneck["total_tokens"] / squad_total * 100, 1) if squad_total > 0 else 0
        lines.append("")
        lines.append("## Largest Handoff Bottleneck")
        lines.append(f"- Source: `{bottleneck['source_agent']}`")
        lines.append(f"- Destination: `{bottleneck['destination_agent']}`")
        lines.append(f"- Purpose: {bottleneck['delegation_purpose']}")
        lines.append(f"- Tokens: {bottleneck['total_tokens']} ({'estimated' if bottleneck['tokens_estimated'] else 'measured'})")
        lines.append(f"- % of squad total tokens: {pct}%")
        lines.append(f"- Case: {bottleneck['case_id']}")
        lines.append(f"- Status: {bottleneck['status']}")

    (RESULTS_DIR / "race_table.md").write_text("\n".join(lines), encoding="utf-8")


def _write_csv(single_cases, squad_cases) -> None:
    path = RESULTS_DIR / "benchmark_details.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "case_id", "passed", "latency_ms", "input_tokens", "output_tokens", "total_tokens", "cost", "handoffs"])
        for c in single_cases:
            w.writerow(["single_agent", c.case_id, c.passed, c.latency_ms, c.input_tokens, c.output_tokens, c.total_tokens, c.cost, len(c.handoffs)])
        for c in squad_cases:
            w.writerow(["squad", c.case_id, c.passed, c.latency_ms, c.input_tokens, c.output_tokens, c.total_tokens, c.cost, len(c.handoffs)])


def _write_agentcard() -> None:
    """Bonus: Publish a machine-readable AgentCard for the orchestrator.

    Describes identity, capabilities, interfaces, delegation, auth, and
    endpoint/version info. No authentication protections are claimed (the
    deployment has no auth).
    """
    card = {
        "schema": "https://agentcard.org/v1",
        "agent": {
            "name": "ragger-squad-orchestrator",
            "version": "1.0.0",
            "description": "Manager-plus-two-specialists multi-agent RAG orchestrator for developer-documentation questions. Delegates to version/deprecation and code-sample workers, combines evidence, and returns a final answer with source references.",
            "purpose": "answer_developer_documentation_questions",
        },
        "capabilities": [
            "question_analysis",
            "version_deprecation_check",
            "code_sample_retrieval",
            "evidence_combining",
            "final_answer_generation",
            "graceful_degradation_on_worker_failure",
        ],
        "workers": [
            {
                "name": "version_deprecation",
                "role": "checks API version/deprecation status via RAG tools",
                "tools": ["check_deprecation", "search_knowledge_base"],
            },
            {
                "name": "code_sample",
                "role": "retrieves documentation and code examples",
                "tools": ["search_knowledge_base"],
            },
        ],
        "input": {
            "endpoint": "/api/v1/squad/run",
            "method": "POST",
            "body": {
                "query": "string (developer-documentation question)",
                "strategy": "optional RAG strategy",
                "knowledge_base_id": "optional knowledge base id",
                "max_steps": "optional int (1-10)",
                "temperature": "optional float",
            },
        },
        "output": {
            "format": "Server-Sent Events (text/event-stream)",
            "events": ["squad.progress", "squad.completed", "[DONE]"],
            "fields": ["answer", "steps", "specialist_results", "handoffs", "input_tokens", "output_tokens", "total_tokens", "cost"],
        },
        "delegation": {
            "strategy": "manager_plus_two_specialists",
            "hops_logged": "artifacts/week10/handoffs.log (JSON Lines)",
            "fault_injection": "RAGGER_FAULT_INJECTION=1 + RAGGER_FAULT_CASE_ID=<case> (experiment only)",
        },
        "authentication": {
            "required": False,
            "note": "No authentication or authorization is enforced in this deployment.",
        },
        "mcp": {
            "servers": ["ragger-tools", "package-registry"],
            "transport": "HTTP Streamable (stateless)",
            "mount": "/api/v1/mcp",
        },
    }
    (RESULTS_DIR / "agentcard.json").write_text(json.dumps(card, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(run_benchmark())

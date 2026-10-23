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
from app.services.evaluation.scoring import passed_strict
from app.services.llm import track_usage
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
    """Run one query through AgentEngine (single-agent baseline).

    Tokens come from the LLM-layer usage tracker, i.e. every model call the
    agent made (reasoning steps and the final answer, with retrieved context),
    measured the same way as the squad. AgentEngine's own input/output counts
    only cover the query and the final answer, so they are not used here.
    """
    from app.models.schemas import AgentRunRequest
    from app.services.agent_engine import AgentEngine

    engine = AgentEngine()
    request = AgentRunRequest(query=query, max_steps=5)

    answer = ""
    with track_usage() as usage:
        async for event in engine.run_stream(request):
            if event.get("type") == "trace.completed":
                answer = event.get("data", {}).get("answer", "")

    return {
        "answer": answer,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "tokens_estimated": usage.any_estimated or not usage.calls,
        "llm_calls": len(usage.calls),
    }


async def run_squad(query: str, case_id: str, telemetry: TelemetryCollector) -> dict:
    """Run one query through the squad orchestrator."""
    orchestrator = SquadOrchestrator(max_steps=5, temperature=0.2)
    return await orchestrator.run(query, case_id, telemetry)


def _cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens + output_tokens) * settings.COST_PER_TOKEN


def _errored(answer: str, error: str | None) -> bool:
    """True when the arm failed to produce an answer at all (provider/LLM
    error), as opposed to producing a wrong answer."""
    return bool(error) or (answer or "").startswith("No answer was generated")


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
        passed_a = passed_strict(result_a["answer"], keywords)
        tc_single.input_tokens = result_a["input_tokens"]
        tc_single.output_tokens = result_a["output_tokens"]
        tc_single.total_tokens = tc_single.input_tokens + tc_single.output_tokens
        tc_single.cost = _cost(tc_single.input_tokens, tc_single.output_tokens)
        tc_single.tokens_estimated = result_a["tokens_estimated"]
        err_a = "llm_error" if _errored(result_a["answer"], None) else None
        tc_single.finish(passed=passed_a, answer=result_a["answer"], error=err_a)
        single.add(tc_single)
        print(f"  single: pass={passed_a} latency={tc_single.latency_ms}ms "
              f"tokens={tc_single.total_tokens} llm_calls={result_a['llm_calls']}")

        # --- Arm B: squad ---
        tc_squad = TelemetryCollector(arm="squad", case_id=case_id)
        result_b = await run_squad(question, case_id, tc_squad)
        passed_b = passed_strict(result_b["answer"], keywords)
        err_b = result_b.get("error") or ("llm_error" if _errored(result_b["answer"], None) else None)
        tc_squad.finish(passed=passed_b, answer=result_b["answer"], error=err_b)
        squad.add(tc_squad)
        print(f"  squad:  pass={passed_b} latency={tc_squad.latency_ms}ms "
              f"tokens={tc_squad.total_tokens} hops={len(tc_squad.handoffs)}")

    # Write artifacts
    single_summary = single.summary()
    squad_summary = squad.summary()

    # handoffs.log (squad only)
    write_handoffs_jsonl(squad.cases)

    # race_table.md
    provider, model = await _provider_and_model()
    _write_race_table(single_summary, squad_summary, cases, provider, model)

    # verdict.md (from actual results)
    generate_verdict(single_summary, squad_summary)

    # bonus: agentcard.json
    _write_agentcard()

    # CSV details
    _write_csv(single.cases, squad.cases)

    # Full per-case results incl. answers, so borderline keyword scores can be reviewed.
    (RESULTS_DIR / "benchmark_results.json").write_text(
        json.dumps({"single_agent": single_summary, "squad": squad_summary,
                    "provider": provider, "model": model}, indent=2, default=str),
        encoding="utf-8",
    )

    return {
        "single_agent": single_summary,
        "squad": squad_summary,
    }


async def _provider_and_model() -> tuple[str, str]:
    from app.services.llm import resolve_provider_and_model

    return await resolve_provider_and_model()


def _write_race_table(single: dict, squad: dict, cases: list[dict], provider: str, model: str) -> None:
    lines = []
    lines.append("# Week 10: Single-Agent vs Multi-Agent Squad Comparison")
    lines.append("")
    lines.append(f"Dataset: `evals/developer_docs_cases.json` (10 GitHub API version-specific cases)")
    lines.append(f"Case IDs: {', '.join(c['id'] for c in cases)}")
    lines.append("Evaluator: strict keyword scorer (`app/services/evaluation/scoring.py`): whole-word match, all expected keywords required, refusals never pass")
    lines.append(f"Model: `{provider}` / `{model}` (same model and evaluator for both arms)")
    lines.append(f"Retrieval strategy: `{settings.DEFAULT_STRATEGY}` (same for both arms)")
    lines.append(f"Cost per token: `{settings.COST_PER_TOKEN}` (flat configured rate, not provider billing)")
    lines.append(
        "Token method: single agent = "
        + ("estimated" if single["tokens_estimated"] else "provider-reported")
        + ", squad = "
        + ("estimated" if squad["tokens_estimated"] else "provider-reported")
        + " (both measured at the LLM layer, every model call counted once)"
    )
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
    lines.append("")
    lines.append("## Context Re-Send Multiplier")
    lines.append("")
    if single["total_tokens"] > 0:
        multiplier = round(squad["total_tokens"] / single["total_tokens"], 1)
        lines.append(
            f"**{multiplier}** = squad total tokens / single-agent total tokens "
            f"= {squad['total_tokens']} / {single['total_tokens']} (same 10 cases)"
        )
    else:
        lines.append("unavailable (single-agent total tokens is zero)")

    def _status(row: dict | None) -> str:
        if not row:
            return "-"
        if row.get("error"):
            return "ERR"  # arm produced no answer (provider/LLM error), not a wrong answer
        return "Y" if row["passed"] else "N"

    lines.append("")
    lines.append("## Per-Case Results")
    lines.append("")
    lines.append("Result: Y = pass, N = wrong/refused answer, ERR = no answer produced (provider/LLM error).")
    lines.append("")
    lines.append("| Case | Single | Single ms | Single tokens | Squad | Squad ms | Squad tokens | Squad hops |")
    lines.append("|------|--------|-----------|---------------|-------|----------|--------------|------------|")
    for c in cases:
        cid = c["id"]
        s_row = next((x for x in single["cases"] if x["case_id"] == cid), None)
        b_row = next((x for x in squad["cases"] if x["case_id"] == cid), None)
        lines.append(
            f"| {cid} | {_status(s_row)} | {s_row['latency_ms'] if s_row else 0} | {s_row['total_tokens'] if s_row else 0} "
            f"| {_status(b_row)} | {b_row['latency_ms'] if b_row else 0} | {b_row['total_tokens'] if b_row else 0} "
            f"| {len(b_row['handoffs']) if b_row else 0} |"
        )
    n_err_s = sum(1 for x in single["cases"] if x.get("error"))
    n_err_b = sum(1 for x in squad["cases"] if x.get("error"))
    lines.append("")
    lines.append(f"Errored cases (no answer produced): single {n_err_s}, squad {n_err_b}. These are provider/LLM failures, not wrong answers.")

    known = Path(__file__).resolve().parents[3] / "evals" / "known_scoring_issues.json"
    if known.exists():
        issues = {k: v for k, v in json.loads(known.read_text()).items() if not k.startswith("_")}
        hit = [cid for cid in issues if any(not r["passed"] for arm in (single, squad) for r in arm["cases"] if r["case_id"] == cid)]
        if hit:
            lines.append("")
            lines.append("## Known scorer weaknesses (scores above are NOT adjusted)")
            for cid in hit:
                lines.append(f"- `{cid}`: keyword `{issues[cid]['keyword']}` is a {issues[cid]['issue']}. {issues[cid]['evidence']}")

    lines.append("")
    lines.append("## Methodology Notes")
    lines.append("- Tokens for both arms are measured at the LLM layer (`app/services/llm.py` usage tracker): every model call is counted once, with its full prompt (retrieved context included).")
    lines.append("- A call's tokens are provider-reported when the stream carries a usage chunk; otherwise estimated by word count and flagged above.")
    lines.append("- Squad totals include only the two orchestrator model calls (routing, final answer). Worker hops move retrieved text and make no model call, so their payload size is logged in `handoffs.log` (`model_call: false`) but not added to totals (it is already inside the final-answer prompt).")
    lines.append("- Cost = total tokens x `settings.COST_PER_TOKEN` (one flat configured rate for input and output; not provider billing).")
    lines.append("- Latency is wall-clock per question, measured identically for both arms. p50/p99 use nearest-rank percentile; with 10 samples p99 is the maximum, so it is a weak estimate.")
    lines.append("- Scoring is a keyword check (see Evaluator). It cannot judge correctness beyond keyword presence; read the answers for borderline cases.")
    lines.append("- All handoffs, including every retry attempt, are in `artifacts/week10/handoffs.log` (JSON Lines).")

    # Largest handoff bottleneck: among billable model-call hops (these are what
    # the totals are made of), plus the largest worker payload for reference.
    all_handoffs = [h for c in squad["cases"] for h in c["handoffs"]]
    model_hops = [h for h in all_handoffs if h.get("model_call")]
    if model_hops:
        bottleneck = max(model_hops, key=lambda h: h["total_tokens"])
        squad_total = squad["total_tokens"]
        pct = round(bottleneck["total_tokens"] / squad_total * 100, 1) if squad_total > 0 else 0
        lines.append("")
        lines.append("## Largest Handoff Bottleneck")
        lines.append(f"- Source: `{bottleneck['source_agent']}`")
        lines.append(f"- Destination: `{bottleneck['destination_agent']}`")
        lines.append(f"- Purpose: {bottleneck['delegation_purpose']}")
        lines.append(f"- Tokens: {bottleneck['total_tokens']} ({'estimated' if bottleneck['tokens_estimated'] else 'provider-reported'})")
        lines.append(f"- % of squad total tokens: {pct}%")
        lines.append(f"- Case: {bottleneck['case_id']}")
        lines.append(f"- Status: {bottleneck['status']}")
        payload_hops = [h for h in all_handoffs if not h.get("model_call")]
        if payload_hops:
            big = max(payload_hops, key=lambda h: h["total_tokens"])
            lines.append(
                f"- Largest worker payload (not a model call, not in totals): `{big['destination_agent']}` "
                f"{big['total_tokens']} tokens (estimated) in {big['case_id']}"
            )

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

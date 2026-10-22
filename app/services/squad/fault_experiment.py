"""Week 10: Controlled HTTP 500 fault injection experiment.

Forces the version/deprecation worker to return an HTTP 500 for exactly one
designated evaluation case, then observes the orchestrator's actual behavior.

Enabled via env vars:
  RAGGER_FAULT_INJECTION=1
  RAGGER_FAULT_CASE_ID=gh-006

Disabled by default (never affects normal production requests).
"""

import asyncio
import json
import os
from pathlib import Path

from app.services.squad.eval_cases import load_selected_cases
from app.services.squad.telemetry import TelemetryCollector
from app.services.squad.orchestrator import SquadOrchestrator
from app.services.squad.fault_injection import (
    is_fault_enabled,
    should_inject_fault,
    InjectedHTTP500Error,
    FAULT_INJECTION_CASE_ID,
    FAULT_INJECTION_ENABLED,
)

RESULTS_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "week10"


async def run_failure_experiment(case_id: str | None = None) -> dict:
    """Run the squad with HTTP 500 injected into the version worker for one case.

    Returns the observed execution trace and orchestrator response.
    """
    target = case_id or FAULT_INJECTION_CASE_ID or "gh-006"

    cases = load_selected_cases()
    target_case = next((c for c in cases if c["id"] == target), None)
    if not target_case:
        raise ValueError(f"Case {target} not found in dataset")

    telemetry = TelemetryCollector(arm="squad", case_id=target)
    orchestrator = SquadOrchestrator(max_steps=5, temperature=0.2)

    print(f"[FAULT EXPERIMENT] Injecting HTTP 500 into version_deprecation worker for case {target}")
    print(f"  Fault enabled: {is_fault_enabled()}")
    print(f"  should_inject_fault({target}, version_deprecation): {should_inject_fault(target, 'version_deprecation')}")

    result = await orchestrator.run(target_case["question"], target, telemetry)

    trace = {
        "case_id": target,
        "question": target_case["question"],
        "fault_injected": result.get("specialist_results", {}).get("version_deprecation", {}).get("fault_injected", False),
        "fault_enabled": is_fault_enabled(),
        "orchestrator_answer": result.get("answer", ""),
        "total_latency_ms": result.get("total_latency_ms", 0),
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
        "total_tokens": result.get("total_tokens", 0),
        "steps": result.get("steps", []),
        "specialist_results": result.get("specialist_results", {}),
        "handoffs": [h for h in telemetry.handoffs],
    }

    # Write failure_case.md
    _write_failure_case(trace)

    return trace


def _write_failure_case(trace: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    version_res = trace["specialist_results"].get("version_deprecation", {})
    code_res = trace["specialist_results"].get("code_sample", {})

    answer = trace["orchestrator_answer"]
    has_unsupported_version_claim = any(
        k in answer.lower() for k in ["deprecated", "removed", "migrated", "version"]
    ) and not version_res.get("evidence")

    lines = []
    lines.append("# Week 10: Controlled HTTP 500 Failure Injection")
    lines.append("")
    lines.append(f"Selected case: `{trace['case_id']}`")
    lines.append(f"Question: {trace['question']}")
    lines.append("")
    lines.append("## Fault-Injection Configuration")
    lines.append(f"- Fault enabled: `{trace['fault_enabled']}`")
    lines.append(f"- Worker: `version_deprecation`")
    lines.append(f"- Fault injected: `{trace['fault_injected']}`")
    lines.append(f"- Response: HTTP 500 (simulated via `InjectedHTTP500Error`)")
    lines.append("")
    lines.append("## Retry Behavior")
    lines.append("- Retries: not applicable (fault injection is deterministic, not transient)")
    lines.append("- Retry limits respected: N/A (no retry attempted for injected faults)")
    lines.append("- Timeouts respected: N/A")
    lines.append("")
    lines.append("## Actual HTTP 500 Response")
    lines.append("```json")
    lines.append(json.dumps({
        "error": version_res.get("error", "HTTP 500: version/deprecation check failed"),
        "status": version_res.get("status"),
        "fault_injected": version_res.get("fault_injected"),
    }, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Actual Orchestrator Response")
    lines.append("```")
    lines.append(answer)
    lines.append("```")
    lines.append("")
    lines.append("## Graceful Degradation")
    lines.append(f"- Complete answer: `{'yes' if answer and 'could not find' not in answer.lower() else 'partial'}`")
    lines.append(f"- Missing version verification disclosed: `{'yes' if 'version/deprecation verification was unavailable' in answer else 'no'}`")
    lines.append(f"- Unsupported version claims/hallucinated code: `{'yes' if has_unsupported_version_claim else 'no'}`")
    lines.append("")
    lines.append("## Execution Trace References")
    lines.append(f"- Total latency: {trace['total_latency_ms']}ms")
    lines.append(f"- Input tokens: {trace['input_tokens']} (estimated)")
    lines.append(f"- Output tokens: {trace['output_tokens']} (estimated)")
    lines.append(f"- Total tokens: {trace['total_tokens']} (estimated)")
    lines.append(f"- Handoffs: {len(trace['handoffs'])}")
    lines.append("")
    lines.append("## Version/Deprecation Worker Result")
    lines.append(f"- Status: `{version_res.get('status')}`")
    lines.append(f"- Evidence: `{bool(version_res.get('evidence'))}`")
    lines.append(f"- Error: `{version_res.get('error')}`")
    lines.append("")
    lines.append("## Code-Sample Worker Result")
    lines.append(f"- Status: `{code_res.get('status')}`")
    lines.append(f"- Evidence: `{bool(code_res.get('evidence'))}`")
    lines.append("")
    lines.append("## Final Conclusion")
    graceful = "version/deprecation verification was unavailable" in answer or "could not find" in answer.lower()
    no_hallucination = not has_unsupported_version_claim
    lines.append(f"- Graceful degradation: `{'yes' if graceful else 'no'}`")
    lines.append(f"- No unsupported claims: `{'yes' if no_hallucination else 'no'}`")
    lines.append("")
    lines.append(f"The orchestrator {'gracefully degraded and disclosed the missing verification' if graceful else 'did not adequately disclose'} the version/deprecation failure.")
    lines.append(f"{'No hallucinated version-specific claims were detected.' if no_hallucination else 'WARNING: unsupported version claims may be present.'}")

    (RESULTS_DIR / "failure_case.md").write_text("\n".join(lines), encoding="utf-8")


def generate_verdict(single_summary: dict, squad_summary: dict) -> None:
    """Write verdict.md based on actual benchmark results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    single_pass = single_summary["pass_rate"]
    squad_pass = squad_summary["pass_rate"]
    single_lat = single_summary["p50_latency_ms"]
    squad_lat = squad_summary["p50_latency_ms"]
    single_tok = single_summary["total_tokens"]
    squad_tok = squad_summary["total_tokens"]
    single_cost = single_summary["cost_per_question"]
    squad_cost = squad_summary["cost_per_question"]

    if squad_pass > single_pass + 0.05:
        decision = "KEEP"
        reason = "squad meaningfully improves pass rate"
    elif squad_pass < single_pass - 0.05:
        decision = "KILL"
        reason = "squad lowers pass rate while increasing cost and latency"
    else:
        decision = "KILL"
        reason = ("squad provides no meaningful pass-rate benefit while adding "
                  "latency, token consumption, and cost")

    multiplier = None
    if single_tok > 0:
        multiplier = round(squad_tok / single_tok, 1)

    lines = []
    lines.append("# Week 10: KEEP or KILL Verdict")
    lines.append("")
    lines.append(f"Decision: **{decision}**")
    lines.append("")
    lines.append(f"- Single-agent pass rate: {single_pass*100:.1f}% | Squad pass rate: {squad_pass*100:.1f}%")
    lines.append(f"- Single-agent p50 latency: {single_lat}ms | Squad p50 latency: {squad_lat}ms")
    lines.append(f"- Single-agent total tokens: {single_tok} | Squad total tokens: {squad_tok}")
    lines.append(f"- Single-agent avg cost/question: ${single_cost:.6f} | Squad avg cost/question: ${squad_cost:.6f}")
    lines.append(f"- Context Re-Send Multiplier: {multiplier if multiplier is not None else 'unavailable'}")
    lines.append("")
    lines.append(f"Trade-off: {reason}. The squad adds {len(squad_summary.get('cases', []))} specialist hops per question,")
    lines.append("increasing latency and token consumption. Unless it delivers a measurable correctness gain,")
    lines.append("the added cost and complexity are not justified.")
    lines.append("")
    lines.append("Sunk-cost bias: the decision is based solely on measured benchmark results, not on the")
    lines.append("implementation effort already invested in the squad. If the data does not show a clear")
    lines.append("advantage, the squad is KILL regardless of build time.")
    lines.append("")
    lines.append(f"Data quality: token counts are estimated (word-count heuristic), not provider-reported.")
    lines.append(f"If provider-reported usage or real pricing becomes available, revisit this verdict.")

    (RESULTS_DIR / "verdict.md").write_text("\n".join(lines), encoding="utf-8")

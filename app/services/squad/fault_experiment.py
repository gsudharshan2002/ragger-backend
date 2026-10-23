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
    fault_case_id,
    inject_fault_for,
    is_fault_enabled,
    should_inject_fault,
)

RESULTS_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "week10"


async def run_failure_experiment(case_id: str | None = None) -> dict:
    """Run the squad with HTTP 500 injected into the version worker for one case.

    Returns the observed execution trace and orchestrator response.
    """
    target = case_id or fault_case_id() or "gh-006"

    cases = load_selected_cases()
    target_case = next((c for c in cases if c["id"] == target), None)
    if not target_case:
        raise ValueError(f"Case {target} not found in dataset")

    telemetry = TelemetryCollector(arm="squad", case_id=target)
    orchestrator = SquadOrchestrator(max_steps=5, temperature=0.2)

    print(f"[FAULT EXPERIMENT] Injecting HTTP 500 into version_deprecation worker for case {target}")
    # Injection is enabled only inside this block, so nothing else in the
    # process (e.g. the benchmark) is ever affected.
    with inject_fault_for(target):
        fault_active = is_fault_enabled()
        print(f"  Fault enabled: {fault_active}")
        print(f"  should_inject_fault({target}, version_deprecation): {should_inject_fault(target, 'version_deprecation')}")
        result = await orchestrator.run(target_case["question"], target, telemetry)

    trace = {
        "case_id": target,
        "question": target_case["question"],
        "fault_injected": result.get("specialist_results", {}).get("version_deprecation", {}).get("fault_injected", False),
        "fault_enabled": fault_active,
        "orchestrator_answer": result.get("answer", ""),
        "total_latency_ms": result.get("total_latency_ms", 0),
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
        "total_tokens": result.get("total_tokens", 0),
        "steps": result.get("steps", []),
        "specialist_results": result.get("specialist_results", {}),
        "handoffs": [h.to_dict() for h in telemetry.handoffs],
        "version_verification": result.get("version_verification"),
        "max_retries": orchestrator.max_retries,
        "worker_timeout_s": orchestrator.version_worker.timeout_seconds,
        "error": result.get("error"),
    }

    # Write failure_case.md
    _write_failure_case(trace)

    return trace


_CLAIM_PHRASES = ["is deprecated", "is removed", "was removed", "has been removed", "has been deprecated",
                  "no longer supported", "replaced by", "still includes", "still available"]


def _write_failure_case(trace: dict) -> None:
    """Write failure_case.md. Every statement is derived from the recorded
    trace (handoffs, specialist results, final answer) - nothing is assumed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    version_res = trace["specialist_results"].get("version_deprecation", {})
    code_res = trace["specialist_results"].get("code_sample", {})
    answer = trace["orchestrator_answer"]
    body, _, note = answer.partition("\n\nNote: ")
    disclosed = bool(note) and "unavailable" in note.lower()

    hops = [h for h in trace["handoffs"] if h["destination_agent"] == "version_deprecation"]
    attempts = len(hops)
    retries = max(0, attempts - 1)
    max_attempts = trace["max_retries"] + 1
    retry_limit_respected = attempts <= max_attempts
    timeout_s = trace["worker_timeout_s"]
    timeouts_respected = all(h["latency_ms"] <= timeout_s * 1000 for h in hops)
    state = trace.get("version_verification")

    body_lower = body.lower()
    claim_hits = [p for p in _CLAIM_PHRASES if p in body_lower]
    claims_unverified = bool(claim_hits) and state != "verified"

    refused = "could not find" in body_lower
    if trace.get("error"):
        task_state = "failed (answer generation error)"
    elif state != "verified" and body.strip() and not refused:
        task_state = "completed with partial answer (version verification missing)"
    elif refused:
        task_state = "completed: could not answer from evidence"
    else:
        task_state = "completed"
    graceful = (not trace.get("error")) and (state == "verified" or disclosed)

    lines = ["# Week 10: Controlled HTTP 500 Failure Injection", ""]
    lines.append(f"Selected case: `{trace['case_id']}`")
    lines.append(f"Question: {trace['question']}")
    lines.append("")
    lines.append("## Fault-Injection Configuration")
    lines.append(f"- Fault enabled (env `RAGGER_FAULT_INJECTION`/`RAGGER_FAULT_CASE_ID`): `{trace['fault_enabled']}`")
    lines.append("- Target worker: `version_deprecation` (only this case, only this worker)")
    lines.append(f"- Fault injected on this run: `{trace['fault_injected']}`")
    lines.append("- Mechanism: the worker returns an HTTP 500 error result in-process (there is no network boundary between manager and workers in this build)")
    lines.append("")
    lines.append("## Retry Behavior (from the recorded handoffs)")
    lines.append(f"- Attempts: {attempts} (1 initial + {retries} retr{'y' if retries == 1 else 'ies'}); configured limit: {max_attempts} attempts")
    lines.append(f"- Retry limit respected: `{retry_limit_respected}`")
    lines.append(f"- Per-attempt timeout {timeout_s}s respected: `{timeouts_respected}`")
    for h in hops:
        lines.append(f"  - attempt {h['retry_number']}: status=`{h['status']}` latency={h['latency_ms']}ms error={h['error']!r}")
    lines.append("")
    lines.append("## Actual HTTP 500 Response")
    lines.append("```json")
    lines.append(json.dumps({
        "error": version_res.get("error"),
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
    lines.append("## Outcome")
    lines.append(f"- Final task state: **{task_state}**")
    lines.append(f"- Version verification state: `{state}`")
    lines.append(f"- Missing verification disclosed to the user: `{disclosed}`")
    lines.append(f"- Affirmative version/deprecation claims in the answer body: `{claim_hits or 'none'}`"
                 + (" (unverified, but disclosed)" if claims_unverified and disclosed else "")
                 + (" (UNVERIFIED AND UNDISCLOSED)" if claims_unverified and not disclosed else ""))
    lines.append(f"- Code sample presented as version-verified: `{bool(code_res.get('code_samples')) and any(c.get('version_verified') for c in code_res.get('code_samples', []))}`")
    lines.append(f"- Graceful degradation: `{graceful}`")
    lines.append(f"- Error details: `{trace.get('error')}`")
    lines.append("")
    lines.append("## Execution Trace References")
    lines.append(f"- Total latency: {trace['total_latency_ms']}ms")
    lines.append(f"- Model-call tokens: {trace['input_tokens']} in / {trace['output_tokens']} out / {trace['total_tokens']} total")
    lines.append(f"- Handoffs recorded: {len(trace['handoffs'])} (see `handoffs.log`)")
    lines.append("")
    lines.append("## Worker Results")
    lines.append(f"- version_deprecation: status=`{version_res.get('status')}` evidence=`{bool(version_res.get('evidence'))}` attempts=`{version_res.get('attempts')}`")
    lines.append(f"- code_sample: status=`{code_res.get('status')}` evidence=`{bool(code_res.get('evidence'))}` attempts=`{code_res.get('attempts')}`")
    lines.append("")
    lines.append("## Conclusion")
    if graceful and not (claims_unverified and not disclosed):
        lines.append("The orchestrator retried within its limits, did not present unverified version claims as verified, and told the user verification was unavailable.")
    else:
        lines.append("WARNING: the orchestrator did not degrade gracefully (see Outcome above).")

    (RESULTS_DIR / "failure_case.md").write_text("\n".join(lines), encoding="utf-8")


def generate_verdict(single_summary: dict, squad_summary: dict) -> None:
    """Write verdict.md (10 lines max) from the measured results.

    The decision is a function of the numbers: keep the squad only if it is
    meaningfully more correct; otherwise kill it, since it costs more tokens
    and an extra routing call. Ties go to the simpler single agent."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sp, qp = single_summary["pass_rate"], squad_summary["pass_rate"]
    st, qt = single_summary["total_tokens"], squad_summary["total_tokens"]
    n = squad_summary["case_count"]
    mult = f"{qt / st:.1f}x" if st > 0 else "unavailable"

    # With n=10 one case is 10 points, so a one-case gap is noise. Correctness
    # decides only when the gap is more than one case; inside that band the
    # cheaper/faster arm wins (squad must save >=20% tokens without being slower).
    saves = st > 0 and qt <= 0.8 * st
    faster = squad_summary["p50_latency_ms"] <= single_summary["p50_latency_ms"]
    if qp > sp + 0.10:
        decision, why = "KEEP", "the squad is meaningfully more correct"
    elif qp < sp - 0.10:
        decision, why = "KILL", "the squad is less correct and costs more"
    elif saves and faster:
        decision, why = "KEEP", (f"correctness is within one case of the single agent (noise at n={n}) "
                                 f"while the squad uses {100 - round(100 * qt / st)}% fewer tokens and is not slower")
    else:
        decision, why = "KILL", "no meaningful correctness, token or latency advantage to justify the extra calls"

    est = single_summary["tokens_estimated"] or squad_summary["tokens_estimated"]
    lines = [
        "# Week 10: KEEP or KILL Verdict",
        f"Decision: **{decision}**, because {why}.",
        f"- Pass rate: single {sp*100:.0f}% ({single_summary['passed']}/{n}) vs squad {qp*100:.0f}% ({squad_summary['passed']}/{n}).",
        f"- p50 latency: single {single_summary['p50_latency_ms']}ms vs squad {squad_summary['p50_latency_ms']}ms.",
        f"- Tokens: single {st} vs squad {qt} (squad/single = {mult}); avg cost/question ${single_summary['cost_per_question']:.6f} vs ${squad_summary['cost_per_question']:.6f}.",
        f"- Trade-off: {'the squad trades at most one case of correctness for fewer tokens and lower latency' if decision == 'KEEP' and qp <= sp else 'the squad adds a routing call and extra hop and must earn that with correctness, tokens or latency'}.",
        "- Sunk cost: the build effort already spent is not a reason to keep it; the decision follows the measured numbers only.",
        f"- Data caveats: n={n} cases (p99 = max); keyword scoring is coarse; token counts {'include estimates' if est else 'are provider-reported'}; cost uses one flat rate.",
        "- Re-run `python scripts/run_week10_experiment.py` after any change to re-derive this verdict.",
    ]
    (RESULTS_DIR / "verdict.md").write_text("\n".join(lines), encoding="utf-8")

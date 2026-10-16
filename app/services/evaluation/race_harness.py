"""Task Set E, Step 4 (revised to match the refined spec): the race harness.

Runs the SAME 10 questions (race_questions.py) through both systems -
run_agent_via_engine (agent) and run_fixed_workflow (workflow) - and
reports: pass rate, p50 AND p99 latency, input/output tokens reported
SEPARATELY (not just a combined total), and cost both per-question (all
10, failures included) and per-SUCCESSFUL-execution (only the passes) -
the latter surfaces that an unreliable system pays more per correct
answer, which cost-per-question alone hides.

Pass/fail ground truth is keyword presence in the final answer - the same
approach used throughout this codebase, since no real LLM-judge exists
anywhere here.

The agent side runs through AgentEngine (app/services/agent_engine.py) -
the same class the real chat (/agent/run) and the Week 8 trajectory eval
use - instead of agent_loop.py's separate "race-harness skeleton". That
skeleton was its own independent implementation with its own system
prompt/budgets/dispatch logic, kept in sync with AgentEngine by hand for
each fix (dedup guard, tightened tool descriptions, etc.) - a real drift
risk if a future fix ever landed in one and not the other. Racing the
actual product agent removes that risk; agent_loop.py itself is untouched
and still used by scripts/demo_budget_termination.py.
"""
import asyncio
import csv
import statistics
from math import ceil
from pathlib import Path

from app.core.config import settings
from app.services.evaluation.race_questions import RACE_QUESTIONS
from app.services.workflow_docs import run_fixed_workflow

RESULTS_DIR = Path(__file__).resolve().parents[3]  # ragger-backend/


async def run_agent_via_engine(query: str) -> dict:
    """Run one query through AgentEngine and return the same
    {answer, total_latency_ms, input_tokens, output_tokens, steps} shape
    run_fixed_workflow returns, so _run_one can treat both systems
    identically."""
    from app.models.schemas import AgentRunRequest
    from app.services.agent_engine import AgentEngine

    engine = AgentEngine()
    request = AgentRunRequest(query=query, max_steps=5)

    answer = ""
    total_latency_ms = 0
    input_tokens = 0
    output_tokens = 0
    steps: list = []

    async for event in engine.run_stream(request):
        if event.get("type") == "trace.completed":
            data = event.get("data", {})
            answer = data.get("answer", "")
            total_latency_ms = data.get("totalLatencyMs", 0)
            input_tokens = data.get("inputTokens", 0)
            output_tokens = data.get("outputTokens", 0)
            steps = data.get("steps", [])

    return {
        "answer": answer,
        "total_latency_ms": total_latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "steps": steps,
    }


def _passed(answer: str, expected_keywords: list[str]) -> bool:
    lowered = (answer or "").lower()
    return all(kw.lower() in lowered for kw in expected_keywords)


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile - simple and honest for a 10-item sample
    (no interpolation guesswork). p99 of 10 values is, by construction,
    the max - that's an accurate statement about a sample this small, not
    a bug."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = max(0, min(len(sorted_vals) - 1, ceil(pct / 100 * len(sorted_vals)) - 1))
    return sorted_vals[k]


async def _run_one(system_name: str, run_fn, case: dict) -> dict:
    result = await run_fn(case["question"])
    input_tokens = result["input_tokens"]
    output_tokens = result["output_tokens"]
    total_tokens = input_tokens + output_tokens
    cost = total_tokens * settings.COST_PER_TOKEN
    passed = _passed(result["answer"], case["expected_keywords"])

    return {
        "system": system_name,
        "case_id": case["id"],
        "passed": passed,
        "latency_ms": result["total_latency_ms"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost": cost,
        "answer": result["answer"],
    }


def _summarize(rows: list[dict]) -> dict:
    n = len(rows)
    latencies = [r["latency_ms"] for r in rows]
    passed_rows = [r for r in rows if r["passed"]]
    total_cost = sum(r["cost"] for r in rows)

    return {
        "pass_rate": round(len(passed_rows) / n, 3),
        "p50_latency_ms": int(statistics.median(latencies)),
        "p99_latency_ms": int(_percentile(latencies, 99)),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "cost_per_question": round(total_cost / n, 6),
        "cost_per_success": round(total_cost / len(passed_rows), 6) if passed_rows else None,
    }


async def run_race() -> dict:
    agent_rows = []
    workflow_rows = []

    for case in RACE_QUESTIONS:
        print(f"Racing case {case['id']}: {case['question'][:70]}...")

        agent_row = await _run_one("agent", run_agent_via_engine, case)
        agent_rows.append(agent_row)
        print(
            f"  agent:    pass={agent_row['passed']} latency={agent_row['latency_ms']}ms "
            f"in={agent_row['input_tokens']} out={agent_row['output_tokens']}"
        )

        workflow_row = await _run_one("workflow", run_fixed_workflow, case)
        workflow_rows.append(workflow_row)
        print(
            f"  workflow: pass={workflow_row['passed']} latency={workflow_row['latency_ms']}ms "
            f"in={workflow_row['input_tokens']} out={workflow_row['output_tokens']}"
        )

    summary = {"agent": _summarize(agent_rows), "workflow": _summarize(workflow_rows)}

    with open(RESULTS_DIR / "race.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "system", "pass_rate", "p50_latency_ms", "p99_latency_ms",
            "input_tokens", "output_tokens", "cost_per_question", "cost_per_success",
        ])
        for system, metrics in summary.items():
            writer.writerow([
                system, metrics["pass_rate"], metrics["p50_latency_ms"], metrics["p99_latency_ms"],
                metrics["input_tokens"], metrics["output_tokens"],
                metrics["cost_per_question"], metrics["cost_per_success"],
            ])

    with open(RESULTS_DIR / "race_details.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "system", "case_id", "passed", "latency_ms",
            "input_tokens", "output_tokens", "cost", "answer",
        ])
        for row in agent_rows + workflow_rows:
            writer.writerow([
                row["system"], row["case_id"], row["passed"], row["latency_ms"],
                row["input_tokens"], row["output_tokens"], row["cost"], row["answer"],
            ])

    print("\n=== RACE SUMMARY ===")
    for system, metrics in summary.items():
        print(f"{system}: {metrics}")

    return summary


if __name__ == "__main__":
    asyncio.run(run_race())

"""Week 10: Master experiment runner.

Phase 3: Execute the full experiment.
  1. Benchmark both arms (single-agent + squad) over the 10 cases.
  2. Run the controlled HTTP 500 failure experiment.
  3. Generate all artifacts (race_table.md, handoffs.log, failure_case.md, verdict.md).

Usage:
  # Normal benchmark (no fault injection)
  uv run python scripts/run_week10_experiment.py

  # Choose which case gets the HTTP 500 in the failure experiment (default gh-006).
  # The benchmark itself always runs with faults disabled.
  RAGGER_FAULT_CASE_ID=gh-007 uv run python scripts/run_week10_experiment.py
"""

import asyncio
import os
import sys

from app.services.squad.benchmark import run_benchmark
from app.services.squad.fault_experiment import run_failure_experiment


async def main():
    # The benchmark must never run with fault injection on, even if the caller's
    # shell has it set; the failure experiment enables it itself, scoped.
    os.environ.pop("RAGGER_FAULT_INJECTION", None)
    print("=== Week 10 Multi-Agent Squad Experiment ===")
    print()

    # Phase 3: Run benchmark both arms
    print("--- Running benchmark: single-agent vs squad ---")
    results = await run_benchmark()
    single = results["single_agent"]
    squad = results["squad"]
    print()
    print(f"Single-agent: pass={single['pass_rate']*100:.1f}% p50={single['p50_latency_ms']}ms tokens={single['total_tokens']}")
    print(f"Squad:        pass={squad['pass_rate']*100:.1f}% p50={squad['p50_latency_ms']}ms tokens={squad['total_tokens']}")
    print()

    # Phase 3: Run the isolated HTTP 500 experiment. Injection is enabled only
    # for this one scoped run (the benchmark above ran with faults disabled).
    case_id = os.environ.get("RAGGER_FAULT_CASE_ID", "gh-006")
    print(f"--- Running HTTP 500 fault experiment (case={case_id}) ---")
    trace = await run_failure_experiment(case_id)
    print(f"Fault injected: {trace['fault_injected']}")
    print(f"Version verification state: {trace['version_verification']}")
    print()

    print("=== Artifacts written to artifacts/week10/ ===")
    print("  race_table.md")
    print("  handoffs.log")
    print("  failure_case.md")
    print("  verdict.md")
    print("  benchmark_details.csv")


if __name__ == "__main__":
    asyncio.run(main())

"""Week 10: Master experiment runner.

Phase 3: Execute the full experiment.
  1. Benchmark both arms (single-agent + squad) over the 10 cases.
  2. Run the controlled HTTP 500 failure experiment.
  3. Generate all artifacts (race_table.md, handoffs.log, failure_case.md, verdict.md).

Usage:
  # Normal benchmark (no fault injection)
  uv run python scripts/run_week10_experiment.py

  # With fault injection enabled (HTTP 500 into version worker for gh-006)
  RAGGER_FAULT_INJECTION=1 RAGGER_FAULT_CASE_ID=gh-006 uv run python scripts/run_week10_experiment.py
"""

import asyncio
import os
import sys

from app.services.squad.benchmark import run_benchmark
from app.services.squad.fault_experiment import run_failure_experiment


async def main():
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

    # Phase 3: Run fault injection experiment
    fault_enabled = os.environ.get("RAGGER_FAULT_INJECTION", "0") == "1"
    case_id = os.environ.get("RAGGER_FAULT_CASE_ID", "gh-006")
    print(f"--- Running HTTP 500 fault experiment (enabled={fault_enabled}, case={case_id}) ---")
    trace = await run_failure_experiment(case_id)
    print(f"Fault injected: {trace['fault_injected']}")
    print(f"Orchestrator disclosed missing verification: {'version/deprecation verification was unavailable' in trace['orchestrator_answer']}")
    print()

    print("=== Artifacts written to artifacts/week10/ ===")
    print("  race_table.md")
    print("  handoffs.log")
    print("  failure_case.md")
    print("  verdict.md")
    print("  benchmark_details.csv")


if __name__ == "__main__":
    asyncio.run(main())

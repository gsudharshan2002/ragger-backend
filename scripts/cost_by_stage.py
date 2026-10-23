"""Week 11/12: cost and latency by stage, computed from a stored trace.

Attribution rule (each billable operation is counted exactly once):
  generation  every `llm.generate` span (agent reasoning, final answer, squad routing/answer).
              A model call made inside a tool is counted here, never under tools.
  retrieval   `tool.retrieve`, `tool.check_deprecation` and `retrieval.*` spans. Local BM25 /
              local embeddings / local reranker have no per-request API charge; a billed
              embedding or rerank call would appear as its own span with tokens.
  tools       any other tool or handoff span (`handoff.*`, `tool.finish`, MCP calls).
  (excluded)  `tool.answer` and the root span: containers whose time is already covered by
              the `llm.generate` span they contain, so they are not summed again.

Cost = provider-reported tokens x settings.COST_PER_TOKEN (one flat rate; not provider billing).

Usage:  python scripts/cost_by_stage.py --trace-id <id> [--out artifacts/week11/cost_by_stage.md]
"""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from app.services.storage import export_trace, init_storage  # noqa: E402

EXCLUDED = {"tool.answer", "agent.run", "squad.run"}


def stage_of(span: dict) -> str | None:
    s = span.get("stage", "")
    if s in EXCLUDED:
        return None
    if s == "llm.generate":
        return "generation"
    if s in ("tool.retrieve", "tool.check_deprecation") or s.startswith("retrieval"):
        return "retrieval"
    return "tools"


def breakdown(trace: dict) -> dict:
    rows = {k: {"calls": 0, "input": 0, "output": 0, "latency_ms": 0, "cost": 0.0, "estimated_calls": 0}
            for k in ("retrieval", "generation", "tools")}
    for sp in trace.get("spans") or []:
        stage = stage_of(sp)
        if stage is None:
            continue
        r = rows[stage]
        r["calls"] += 1
        r["input"] += sp.get("input_tokens") or 0
        r["output"] += sp.get("output_tokens") or 0
        r["latency_ms"] += sp.get("duration_ms") or 0
        r["cost"] += sp.get("cost_usd") or 0.0
        if (sp.get("metadata") or {}).get("tokens_estimated"):
            r["estimated_calls"] += 1
    return rows


def render(trace: dict, rows: dict) -> str:
    total_in = sum(r["input"] for r in rows.values())
    total_out = sum(r["output"] for r in rows.values())
    total_cost = sum(r["cost"] for r in rows.values())
    est = sum(r["estimated_calls"] for r in rows.values())
    llm_calls = rows["generation"]["calls"]
    lines = [
        "# Cost by stage",
        "",
        f"Trace: `{trace.get('id')}` (kind `{trace.get('kind')}`, mode `{trace.get('mode')}`, "
        f"prompt v{trace.get('prompt_version')}, model `{(trace.get('llm') or {}).get('model')}`)",
        f"Question: {trace.get('query')}",
        "",
        "| Stage | Calls | Input tokens | Output tokens | Latency (ms) | Cost (USD) |",
        "|-------|------:|-------------:|--------------:|-------------:|-----------:|",
    ]
    for name in ("retrieval", "generation", "tools"):
        r = rows[name]
        lines.append(f"| {name} | {r['calls']} | {r['input']} | {r['output']} | {r['latency_ms']} | ${r['cost']:.6f} |")
    lines += [
        f"| **total** | | **{total_in}** | **{total_out}** | | **${total_cost:.6f}** |",
        "",
        f"- Token source: {'provider-reported' if est == 0 else f'{est} of {llm_calls} model calls estimated'} "
        "(read from each `llm.generate` span).",
        f"- Price: flat `settings.COST_PER_TOKEN` = {settings.COST_PER_TOKEN} USD/token for input and output; "
        "this is a configured estimate, not provider billing.",
        "- Retrieval cost is $0: BM25, local embeddings and the local reranker have no per-request API charge here.",
        "- Latency per stage is the sum of its spans; stages run sequentially in the agent, so they add up to about the request latency.",
        f"- Request latency (from trace): {trace.get('total_latency_ms')} ms.",
    ]
    return "\n".join(lines) + "\n"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-id", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    await init_storage()
    trace = await export_trace(args.trace_id)
    if not trace:
        print(f"trace {args.trace_id} not found", file=sys.stderr)
        return 1
    text = render(trace, breakdown(trace))
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

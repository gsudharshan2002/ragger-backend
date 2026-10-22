"""Week 11: Search traces from the command line.

Usage:
  python scripts/trace_search.py --trace-id <id>
  python scripts/trace_search.py --status failed --limit 10
  python scripts/trace_search.py --chunk-id <chunk_id> --min-latency 1000
  python scripts/trace_search.py --prompt-version 1 --error

Filters map to the /api/v1/traces/search endpoint.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.storage import search_traces  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Search stored RAG traces")
    parser.add_argument("--trace-id")
    parser.add_argument("--request-id")
    parser.add_argument("--status")
    parser.add_argument("--prompt-version")
    parser.add_argument("--model")
    parser.add_argument("--mode")
    parser.add_argument("--chunk-id")
    parser.add_argument("--error", action="store_true", help="traces with an error")
    parser.add_argument("--no-error", action="store_true", help="traces without an error")
    parser.add_argument("--min-latency-ms", type=int)
    parser.add_argument("--min-cost-usd", type=float)
    parser.add_argument("--started-after")
    parser.add_argument("--started-before")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    error = None
    if args.error:
        error = True
    elif args.no_error:
        error = False

    filters = {
        k: v
        for k, v in {
            "trace_id": args.trace_id,
            "request_id": args.request_id,
            "status": args.status,
            "prompt_version": args.prompt_version,
            "model": args.model,
            "mode": args.mode,
            "chunk_id": args.chunk_id,
            "error": error,
            "min_latency_ms": args.min_latency_ms,
            "min_cost_usd": args.min_cost_usd,
            "started_after": args.started_after,
            "started_before": args.started_before,
        }.items()
        if v is not None
    }

    results = asyncio.run(search_traces(filters, limit=args.limit))
    for t in results:
        print(
            f"{t.get('id')} | {t.get('status')} | pv={t.get('prompt_version')} "
            f"| lat={t.get('total_latency_ms')}ms | cost={t.get('cost_usd')} | "
            f"model={(t.get('llm') or {}).get('model')} | q={t.get('query','')[:60]}"
        )
    print(f"\n{len(results)} trace(s) matched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

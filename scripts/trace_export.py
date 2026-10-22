"""Week 11: Export a complete trace to a JSON file.

Usage:
  python scripts/trace_export.py <trace_id> [output.json]

If no output path is given, writes artifacts/week11/<trace_id>.json
and prints the path.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.storage import export_trace  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/trace_export.py <trace_id> [output.json]")
        return 2
    trace_id = sys.argv[1]
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    trace = asyncio.run(export_trace(trace_id))
    if not trace:
        print(f"trace not found: {trace_id}")
        return 1

    if output is None:
        out_dir = Path("artifacts") / "week11"
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / f"{trace_id}.json"

    output.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

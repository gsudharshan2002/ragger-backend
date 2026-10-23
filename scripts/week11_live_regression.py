"""Week 11: live RED -> GREEN measurement for the deprecated-method case.

Runs the real agent (live LLM, real retrieval, real traces) on regression case
gh-deprecated-001 under prompt v1 (no deprecation-guard rule) and prompt v2
(the fix), N times each, and scores every answer deterministically:

  correct      = names the current replacement `assignees` (whole word) and does not
                 tell the user to use the singular `assignee`
  wrong        = anything else

It reports what actually happened. If the live model does not reproduce the
failure under v1, the report says so; the deterministic prompt-presence test in
tests/test_week11_regression.py is a separate, weaker check.

Usage:  DEFAULT_STRATEGY=bm25 python scripts/week11_live_regression.py [N]
        DEFAULT_STRATEGY=bm25 python scripts/week11_live_regression.py once   # one v2 run, prints its trace id
Output: artifacts/week11/live_regression.json  (answers, trace ids, counts)
"""

import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from app.models.schemas import AgentRunRequest  # noqa: E402
from app.services.agent_engine import AgentEngine  # noqa: E402
from app.services.storage import init_storage  # noqa: E402

CASE_ID = "gh-deprecated-001"
GUARD_MARKER = "- Do not recommend deprecated methods"


def prompt_v1() -> str:
    """Prompt version 1 = the current prompt without the deprecation-guard rule."""
    return settings.SYSTEM_PROMPT.split(GUARD_MARKER)[0]


def score(answer: str) -> dict:
    has_plural = re.search(r"\bassignees\b", answer, re.I) is not None
    # "use assignee", "set the assignee field", "assignee (string)" as the recommendation
    recommends_singular = re.search(
        r"\b(use|set|pass|send|provide)\s+(the\s+)?`?assignee`?(?!s)\b(?!\s+field\s+(is|was|has been)\s+(deprecated|removed))",
        answer,
        re.I,
    ) is not None
    return {"has_assignees": has_plural, "recommends_singular": recommends_singular,
            "correct": has_plural and not recommends_singular}


async def run_once(question: str) -> dict:
    engine = AgentEngine()
    data = {}
    async for ev in engine.run_stream(AgentRunRequest(query=question, max_steps=5)):
        if ev.get("type") == "trace.completed":
            data = ev["data"]
    answer = data.get("answer", "")
    return {"trace_id": data.get("traceId") or data.get("trace_id"), "answer": answer,
            "input_tokens": data.get("inputTokens"), "output_tokens": data.get("outputTokens"),
            "latency_ms": data.get("totalLatencyMs"), **score(answer)}


async def main(n: int) -> None:
    await init_storage()
    cases = json.loads((ROOT / "evals" / "developer_docs_cases.json").read_text())
    question = next(c["question"] for c in cases if c["id"] == CASE_ID)
    fixed_prompt, fixed_version = settings.SYSTEM_PROMPT, settings.PROMPT_VERSION
    out = {"case_id": CASE_ID, "question": question, "n_per_arm": n,
           "strategy": settings.DEFAULT_STRATEGY, "arms": {}}
    try:
        for label, prompt, version in (("v1_baseline", prompt_v1(), "1"), ("v2_fixed", fixed_prompt, fixed_version)):
            settings.SYSTEM_PROMPT, settings.PROMPT_VERSION = prompt, version
            runs = []
            for i in range(n):
                r = await run_once(question)
                runs.append(r)
                print(f"{label} #{i + 1}: correct={r['correct']} trace={r['trace_id']}", flush=True)
            out["arms"][label] = {
                "prompt_version": version,
                "correct": sum(r["correct"] for r in runs),
                "recommends_singular": sum(r["recommends_singular"] for r in runs),
                "n": len(runs),
                "runs": runs,
            }
    finally:
        settings.SYSTEM_PROMPT, settings.PROMPT_VERSION = fixed_prompt, fixed_version
    dest = ROOT / "artifacts" / "week11" / "live_regression.json"
    dest.write_text(json.dumps(out, indent=2))
    for label, arm in out["arms"].items():
        print(f"{label}: {arm['correct']}/{arm['n']} correct, {arm['recommends_singular']} recommended singular `assignee`")
    print(f"wrote {dest}")


async def once() -> None:
    await init_storage()
    cases = json.loads((ROOT / "evals" / "developer_docs_cases.json").read_text())
    question = next(c["question"] for c in cases if c["id"] == CASE_ID)
    r = await run_once(question)
    print("ONCE", json.dumps({k: r[k] for k in ("trace_id", "correct", "latency_ms", "input_tokens", "output_tokens")}))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "once":
        asyncio.run(once())
    else:
        asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 5))

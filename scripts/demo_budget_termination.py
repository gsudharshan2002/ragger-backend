"""Task Set E, Step 5 deliverable: demonstrate one budget firing cleanly.

Temporarily lowers MAX_TOKEN_BUDGET (imported, not edited in the source
file) so it trips after exactly one real step against the real LLM/KB -
enough to show a genuine step happen, then a clean BUDGET HIT termination
instead of a crash or a silent hang. The production constant in
agent_loop.py is untouched; this only patches the module attribute for
this one demonstration process.
"""
import asyncio

from app.services import agent_loop

agent_loop.MAX_TOKEN_BUDGET = 50  # deliberately far below a single step's real cost


async def main():
    result = await agent_loop.run_agent_loop_skeleton(
        "Does the current version of GET /repos/{owner}/{repo} still include a has_downloads field?"
    )
    print("\n=== FINAL RESULT ===")
    print(f"answer: {result['answer']!r}")
    print(f"steps completed: {len(result['steps'])}")
    print(f"total_latency_ms: {result['total_latency_ms']}")
    print(f"input_tokens: {result['input_tokens']}, output_tokens: {result['output_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())

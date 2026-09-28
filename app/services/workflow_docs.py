"""Task Set E, Step 2: the fixed workflow.

Reuses the EXACT SAME tools, same model, same output contract as the
agent (agent_loop.py) - the only difference is that this file hard-codes
the step order in Python instead of letting an LLM decide it. That's the
whole point of the race in Step 4: any speed/cost/reliability difference
we measure comes from "loop vs. no loop", not from one system having
better tools, a different prompt, or a different model.

This fixed order (retrieve -> check_deprecation(current) -> answer) is
only valid because of THIS task's domain: every question in this task
set is about migrating to the current API version (2026-03-10), so
"check deprecation against the current version" is always a relevant
step, regardless of which specific endpoint is asked about. A workflow
can only hard-code an argument like api_version=ApiVersion.CURRENT when
the task domain guarantees that argument is constant - if the target
version varied per question, this shortcut would need a reasoning step
first, and the workflow could not exist in this form.
"""
import time

from app.services.agent_tools import ApiVersion, check_deprecation, generate_answer, search_knowledge_base


async def run_fixed_workflow(query: str) -> dict:
    start = time.time()
    steps = []

    t0 = time.time()
    retrieve_result = await search_knowledge_base(query)
    steps.append({"tool": "retrieve", "latency_ms": int((time.time() - t0) * 1000)})

    t0 = time.time()
    deprecation_result = await check_deprecation(query, ApiVersion.CURRENT)
    steps.append({"tool": "check_deprecation", "latency_ms": int((time.time() - t0) * 1000)})

    retrieved_chunks = retrieve_result["raw_chunks"] + deprecation_result["raw_chunks"]

    t0 = time.time()
    answer_result = await generate_answer(query, retrieved_chunks)
    steps.append({"tool": "answer", "latency_ms": int((time.time() - t0) * 1000)})

    return {
        "answer": answer_result.get("answer", ""),
        "error": answer_result.get("error"),
        "total_latency_ms": int((time.time() - start) * 1000),
        "input_tokens": answer_result.get("input_tokens", 0),
        "output_tokens": answer_result.get("output_tokens", 0),
        "steps": steps,
    }


if __name__ == "__main__":
    import asyncio

    result = asyncio.run(
        run_fixed_workflow("Does the current version of GET /repos/{owner}/{repo} still include a has_downloads field?")
    )
    print(result)
"""Task Set E, Steps 4-5: accurate per-lap token accounting, shared output
contract, and all four required budgets enforced in the loop.

Step 4 fix: `reason()` reports the REAL input+output tokens for its own
LLM call (system prompt + user prompt sent, and the raw completion
received), and run_agent_loop_skeleton sums these over EVERY lap - not
just the final answer call. This fixes the flagged mistake: the loop
re-sends the whole prompt every lap, so per-lap tokens must be summed or
cost is understated by multiples. run_agent_loop_skeleton also now returns
{answer, total_latency_ms, input_tokens, output_tokens, steps} - the same
shape run_fixed_workflow returns, so the race harness can treat both
systems identically.

Step 5: all four budgets are enforced, not just two.
  1. Max iterations   -> MAX_STEPS (the for-loop bound)
  2. Max tokens        -> MAX_TOKEN_BUDGET
  3. Max cost          -> MAX_COST_USD (tokens * settings.COST_PER_TOKEN)
  4. Wall-clock        -> MAX_WALL_CLOCK_SECONDS (whole run, not per-step)
Every termination triggered by one of these four is logged with a
"BUDGET HIT:" prefix so it's easy to find in a log and tell apart from the
separate per-step TIMEOUT guard (Concept 5's original addition, which
catches one hung call rather than an exhausted budget).

BUG FIX (found while cross-checking against a later spec revision):
check_deprecation was added to agent_tools.py's TOOLS registry in Task Set
E Step 1, but was never actually wired into this file - the system prompt
below still only listed retrieve/answer/finish, and act() never dispatched
to it. The agent could never call its own third tool. Fixed here by
generating the system prompt's tool list FROM the TOOLS registry instead
of hand-duplicating it as a second string that can silently drift out of
sync - that duplication is exactly how this bug happened. act() also now
validates every tool's inputs against agent_tools.py's Pydantic schemas
before dispatch (strict, type-safe parameters - malformed inputs become a
clean recoverable observation instead of a crash deep inside a tool).
"""
import asyncio
import json
import re
import time

from app.core.config import settings
from app.services.agent_memory import recall, remember
from app.services.agent_tools import (
    TOOLS,
    check_deprecation,
    generate_answer,
    search_knowledge_base,
    validate_tool_inputs,
)
from app.services.llm import generate_completion_stream

MAX_STEPS = 5
STEP_TIMEOUT_SECONDS = 30
MAX_TOKEN_BUDGET = 2000
MAX_COST_USD = 0.01
MAX_WALL_CLOCK_SECONDS = 120
SUMMARIZE_AFTER_STEPS = 2
KEEP_RECENT_STEPS = 1


def _build_tools_section() -> str:
    """Render the Tools section of the system prompt FROM the TOOLS
    registry (agent_tools.py), instead of a hand-maintained duplicate
    string - the registry is the single source of truth `act()` also
    dispatches against, so the prompt can't silently drift out of sync
    with what's actually callable (the bug this fix corrects)."""
    lines = []
    for name, spec in TOOLS.items():
        inputs_desc = ", ".join(f"{k}: {v}" for k, v in spec["inputs"].items()) or "no inputs"
        lines.append(f'- "{name}": {spec["description"]} inputs: {{{inputs_desc}}}')
    return "\n".join(lines)


REACT_SYSTEM_PROMPT = f"""You are a ReAct agent. At each step, respond with ONE JSON object and nothing else:

{{"thought": "<your reasoning about what to do next>", "tool": "<{'|'.join(TOOLS.keys())}>", "inputs": {{<tool inputs>}}}}

Tools:
{_build_tools_section()}

Always retrieve before answering, and check deprecation status when the question concerns whether something still works on a specific API version. Respond with ONLY the JSON object described above."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _estimate_tokens(text: str) -> int:
    return max(1, int(len((text or "").split()) / 0.75))


def _choose_action(raw_output: str, query: str) -> dict:
    match = _JSON_RE.search(raw_output or "")
    if not match:
        return {
            "thought": "Could not find JSON in model output; defaulting to retrieve.",
            "tool": "retrieve",
            "inputs": {"query": query},
        }

    try:
        parsed = json.loads(match.group(0))
        return {
            "thought": parsed.get("thought", "(no thought)"),
            "tool": parsed.get("tool", "retrieve"),
            "inputs": parsed.get("inputs") or {},
        }
    except json.JSONDecodeError:
        return {
            "thought": "Model output was not valid JSON; defaulting to retrieve.",
            "tool": "retrieve",
            "inputs": {"query": query},
        }


async def summarize_history(older_steps: list[dict]) -> str:
    if not older_steps:
        return ""

    steps_text = "\n".join(
        f"- thought=\"{h['thought']}\" tool={h['tool']} observation={h['observation']}"
        for h in older_steps
    )
    prompt = (
        "Summarize the following agent steps into 2-3 short sentences, "
        "focused only on: what was searched for, what was found (or not "
        "found), and what remains unresolved. Do not restate the question.\n\n"
        + steps_text
    )

    summary = ""
    async for chunk in generate_completion_stream(
        "You compress agent reasoning history into brief factual summaries.",
        prompt,
        None,
        0.2,
        150,
        1.0,
    ):
        if chunk["type"] == "token" and chunk.get("content"):
            summary += chunk["content"]
        elif chunk["type"] == "error":
            return "(summary unavailable)"

    return summary.strip()


async def reason(query: str, history: list[dict], memories: list[dict]) -> tuple[dict, int, int]:
    """Ask the LLM what to do next. Returns (decision, input_tokens,
    output_tokens) for THIS specific call, so the caller can sum real
    per-lap cost instead of only counting the final answer."""
    prompt_lines = [f"Question: {query}", ""]

    if memories:
        prompt_lines.append("Relevant past interactions (from earlier, unrelated sessions):")
        for m in memories:
            prompt_lines.append(f"- Q: \"{m['query']}\" -> A: \"{m['answer']}\" (similarity={m['similarity']})")
        prompt_lines.append("")

    if len(history) > SUMMARIZE_AFTER_STEPS:
        older_steps = history[:-KEEP_RECENT_STEPS]
        recent_steps = history[-KEEP_RECENT_STEPS:]
        print(f"  [memory] summarizing {len(older_steps)} older step(s) before reasoning...")
        summary = await summarize_history(older_steps)
        prompt_lines.append(f"Summary of earlier steps: {summary}")
        prompt_lines.append("")
    else:
        recent_steps = history

    if recent_steps:
        prompt_lines.append("Recent steps:")
        for h in recent_steps:
            prompt_lines.append(
                f"- thought=\"{h['thought']}\" tool={h['tool']} observation={h['observation']}"
            )
        prompt_lines.append("")

    prompt_lines.append("What is your next step? Respond with ONLY the JSON object.")
    user_prompt = "\n".join(prompt_lines)

    raw_output = ""
    async for chunk in generate_completion_stream(REACT_SYSTEM_PROMPT, user_prompt, None, 0.2, 512, 1.0):
        if chunk["type"] == "token" and chunk.get("content"):
            raw_output += chunk["content"]
        elif chunk["type"] == "error":
            raise RuntimeError(chunk.get("error", "LLM call failed"))

    input_tokens = _estimate_tokens(REACT_SYSTEM_PROMPT + user_prompt)
    output_tokens = _estimate_tokens(raw_output)
    return _choose_action(raw_output, query), input_tokens, output_tokens


async def act(decision: dict, query: str, retrieved_chunks: list) -> dict:
    """Dispatch the LLM's chosen tool name to the real function that
    implements it. Inputs are validated against agent_tools.py's Pydantic
    schemas before dispatch - a missing/malformed field becomes a clean
    error observation the next reasoning step can see and recover from,
    instead of a raw exception from deep inside the tool."""
    tool = decision["tool"]
    raw_inputs = decision.get("inputs", {}) or {}

    validated, validation_error = validate_tool_inputs(tool, raw_inputs)
    if validation_error:
        return {"tool": tool, "output": {"error": f"Invalid inputs for '{tool}': {validation_error}"}}

    if tool == "retrieve":
        search_query = validated.query or query
        result = await search_knowledge_base(search_query)
        retrieved_chunks.extend(result["raw_chunks"])
        return {"tool": tool, "output": result["summary"]}

    if tool == "check_deprecation":
        endpoint_or_feature = validated.endpoint_or_feature or query
        result = await check_deprecation(endpoint_or_feature, validated.api_version)
        retrieved_chunks.extend(result["raw_chunks"])
        return {"tool": tool, "output": result["summary"]}

    if tool == "answer":
        if not retrieved_chunks:
            result = await search_knowledge_base(query)
            retrieved_chunks.extend(result["raw_chunks"])
        result = await generate_answer(query, retrieved_chunks)
        return {"tool": tool, "output": result}

    # finish
    return {"tool": tool, "output": {"message": validated.message}}


async def run_agent_loop_skeleton(query: str) -> dict:
    run_start = time.time()
    history: list[dict] = []
    retrieved_chunks: list = []
    used_queries: set[str] = set()
    total_input_tokens = 0
    total_output_tokens = 0
    final_answer = ""

    memories = await recall(query)
    if memories:
        print(f"  [memory] recalled {len(memories)} relevant past interaction(s).")

    for step_number in range(1, MAX_STEPS + 1):
        elapsed = time.time() - run_start
        if elapsed >= MAX_WALL_CLOCK_SECONDS:
            print(
                f"BUDGET HIT: wall-clock budget ({MAX_WALL_CLOCK_SECONDS}s) exceeded "
                f"({elapsed:.1f}s elapsed) before step {step_number}. Terminating cleanly."
            )
            break

        current_cost = (total_input_tokens + total_output_tokens) * settings.COST_PER_TOKEN
        if current_cost >= MAX_COST_USD:
            print(
                f"BUDGET HIT: cost budget (${MAX_COST_USD:.4f}) exceeded "
                f"(${current_cost:.6f} spent) before step {step_number}. Terminating cleanly."
            )
            break

        if total_input_tokens + total_output_tokens >= MAX_TOKEN_BUDGET:
            print(
                f"BUDGET HIT: token budget ({MAX_TOKEN_BUDGET}) exhausted "
                f"({total_input_tokens + total_output_tokens} used) before step {step_number}. "
                f"Terminating cleanly."
            )
            break

        try:
            decision, in_tok, out_tok = await asyncio.wait_for(
                reason(query, history, memories), timeout=STEP_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            print(f"TIMEOUT: step {step_number} timed out during reasoning; stopping.")
            break

        total_input_tokens += in_tok
        total_output_tokens += out_tok

        if decision["tool"] == "retrieve":
            search_query = decision["inputs"].get("query", query)
            if search_query in used_queries:
                print(f"Step {step_number}: repeated retrieve query detected - forcing 'answer' instead.")
                decision = {
                    "thought": "Already searched for this; answering with what's available instead of repeating.",
                    "tool": "answer",
                    "inputs": {},
                }
            else:
                used_queries.add(search_query)

        try:
            observation = await asyncio.wait_for(
                act(decision, query, retrieved_chunks), timeout=STEP_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            print(f"TIMEOUT: step {step_number} timed out during tool execution; stopping.")
            break

        if decision["tool"] == "answer":
            total_input_tokens += observation["output"].get("input_tokens", 0)
            total_output_tokens += observation["output"].get("output_tokens", 0)
            final_answer = observation["output"].get("answer", "")

        step = {
            "step_number": step_number,
            "thought": decision["thought"],
            "tool": decision["tool"],
            "observation": observation,
        }
        history.append(step)

        print(f"Step {step_number}: thought=\"{step['thought']}\" tool={step['tool']} -> {observation['output']}")

        if decision["tool"] in ("answer", "finish"):
            print(f"Stopping after step {step_number}: agent chose '{decision['tool']}'.")
            break
    else:
        print(f"BUDGET HIT: max iterations ({MAX_STEPS}) reached without answering. Terminating cleanly.")

    if not final_answer:
        if history and history[-1]["tool"] == "finish":
            final_answer = history[-1]["observation"]["output"].get("message", "") or "Finished without producing an answer."
        else:
            final_answer = "I could not find a sufficient answer within the allotted budget."
    elif history and history[-1]["tool"] == "answer":
        await remember(query, final_answer)
        print("  [memory] stored this interaction for future recall.")

    return {
        "answer": final_answer,
        "total_latency_ms": int((time.time() - run_start) * 1000),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "steps": history,
    }


if __name__ == "__main__":
    result = asyncio.run(run_agent_loop_skeleton("What is in the docs?"))
    print(result)

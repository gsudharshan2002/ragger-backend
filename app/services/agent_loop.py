"""Concept 10: Summarisation.

`reason()` no longer pastes the FULL history into every prompt. Once
history passes SUMMARIZE_AFTER_STEPS, everything older than the most
recent KEEP_RECENT_STEPS gets compressed into a short LLM-generated
summary instead of being sent verbatim - this is the fix for the token
growth problem flagged in Concept 8 (every step's full history gets
re-sent on every subsequent call).

Known simplification: this re-summarizes the same older steps from
scratch every time it's triggered, rather than caching/incrementally
extending one running summary. Simple to follow, wasteful in practice -
worth knowing the difference.
"""
import asyncio
import json
import re

from app.services.agent_memory import recall, remember
from app.services.agent_tools import generate_answer, search_knowledge_base
from app.services.llm import generate_completion_stream

MAX_STEPS = 5
STEP_TIMEOUT_SECONDS = 30
MAX_TOKEN_BUDGET = 4000
SUMMARIZE_AFTER_STEPS = 2   # once history has more than this many steps...
KEEP_RECENT_STEPS = 1       # ...compress everything except the most recent N

REACT_SYSTEM_PROMPT = """You are a ReAct agent. At each step, respond with ONE JSON object and nothing else:

{"thought": "<your reasoning about what to do next>", "tool": "<retrieve|answer|finish>", "inputs": {<tool inputs>}}

Tools:
- "retrieve": search the knowledge base. inputs: {"query": "<search text>"}
- "answer": produce the final answer using context retrieved so far. inputs: {}
- "finish": stop without answering. inputs: {"message": "<optional note>"}

Always retrieve before answering. Respond with ONLY the JSON object described above."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _estimate_tokens(text: str) -> int:
    """Same rough word-count heuristic used elsewhere in this codebase
    (rag_engine.py's estimate_tokens) - good enough for a budget guard,
    not a claim of real tokenizer accuracy."""
    return max(1, int(len((text or "").split()) / 0.75))


def _choose_action(raw_output: str, query: str) -> dict:
    """Parse the LLM's freeform text into a structured decision. If the
    model doesn't return valid JSON, fall back to a safe default instead
    of crashing - a malformed model response should degrade gracefully."""
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
    """Compress older steps into a short paragraph instead of feeding them
    to the model verbatim forever."""
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


async def reason(query: str, history: list[dict], memories: list[dict]) -> dict:
    """Ask the LLM what to do next. History beyond SUMMARIZE_AFTER_STEPS
    gets compressed (Concept 10); the rest of this IS still the ReAct step
    from Concept 2 - Thought and Action produced together, as one parsed
    JSON object."""
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

    return _choose_action(raw_output, query)


async def act(decision: dict, query: str, retrieved_chunks: list) -> dict:
    """Dispatch the LLM's chosen tool name to the real function that
    implements it - this IS "tool calling." Native function-calling APIs
    do this same dispatch for you; here it's a plain if-chain keyed on
    tool name, which works with any LLM provider."""
    tool = decision["tool"]
    inputs = decision.get("inputs", {})

    if tool == "retrieve":
        search_query = inputs.get("query") or query
        result = await search_knowledge_base(search_query)
        retrieved_chunks.extend(result["raw_chunks"])
        return {"tool": tool, "output": result["summary"]}

    if tool == "answer":
        if not retrieved_chunks:
            result = await search_knowledge_base(query)
            retrieved_chunks.extend(result["raw_chunks"])
        result = await generate_answer(query, retrieved_chunks)
        return {"tool": tool, "output": result}

    # finish
    return {"tool": tool, "output": {"message": inputs.get("message", "")}}


async def run_agent_loop_skeleton(query: str) -> list[dict]:
    history: list[dict] = []
    retrieved_chunks: list = []
    used_queries: set[str] = set()
    tokens_used = 0

    memories = await recall(query)
    if memories:
        print(f"  [memory] recalled {len(memories)} relevant past interaction(s).")

    for step_number in range(1, MAX_STEPS + 1):
        if tokens_used >= MAX_TOKEN_BUDGET:
            print(f"Stopping: token budget ({MAX_TOKEN_BUDGET}) exhausted before step {step_number}.")
            break

        try:
            decision = await asyncio.wait_for(reason(query, history, memories), timeout=STEP_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            print(f"Step {step_number} timed out during reasoning; stopping.")
            break

        tokens_used += _estimate_tokens(decision.get("thought", ""))

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
            print(f"Step {step_number} timed out during tool execution; stopping.")
            break

        if decision["tool"] == "answer":
            tokens_used += _estimate_tokens(observation["output"].get("answer", ""))

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
        print(f"Stopping: reached MAX_STEPS={MAX_STEPS} without answering.")

    if not history or history[-1]["tool"] not in ("answer", "finish"):
        print("No definitive answer was reached within budget - a real caller should show a clear fallback message here.")
    elif history[-1]["tool"] == "answer":
        final_answer = history[-1]["observation"]["output"].get("answer", "")
        if final_answer:
            await remember(query, final_answer)
            print("  [memory] stored this interaction for future recall.")

    return history


if __name__ == "__main__":
    asyncio.run(run_agent_loop_skeleton("What is in the docs?"))
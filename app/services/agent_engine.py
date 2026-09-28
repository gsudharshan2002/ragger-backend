import json
import re
import time
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional
from uuid import uuid4

from app.core.logging_config import get_logger
from app.models.schemas import (
    AgentAction,
    AgentObservation,
    AgentRunRequest,
    AgentRunResponse,
    AgentStep,
    AgentTool,
    RagEngineConfig,
    RagStrategy,
)
from app.services.agent_tools import ApiVersion, check_deprecation
from app.services.llm import generate_completion_stream, is_llm_configured
from app.services.rag_engine import RagEngine, build_prompt, estimate_tokens

logger = get_logger(__name__)

REACT_SYSTEM_PROMPT = """You are a ReAct-style research agent that answers questions using a document knowledge base.

You reason step by step in a Thought -> Action -> Observation loop. At each step you must respond with a single JSON object and nothing else - no markdown fences, no commentary before or after it:

{"thought": "<your reasoning about what to do next>", "tool": "<retrieve|check_deprecation|answer|finish>", "inputs": {<tool inputs>}}

Available tools:
- "retrieve": search the knowledge base for context relevant to the question. inputs: {"query": "<search query>", "strategy": "<optional: vector | bm25 | hybrid | hybrid-rrf | hybrid-rerank | hybrid-rerank-mmr>"}. Leave strategy unset for most questions - it uses the configured default. Only set "bm25" when the query contains an exact technical term, field/parameter name, or identifier likely to appear verbatim in the docs (e.g. "sortParamsByRequiredFlag", "has_downloads"). Only set "vector" for a conceptual/paraphrased question with no exact term to match. Do not guess between the other strategies - if it's not clearly one of these two cases, leave strategy unset.
- "check_deprecation": determine if a specific API endpoint or feature is deprecated or removed in a given API version, and what replaced it. ONLY use when the question explicitly mentions a specific API version date (e.g., '2022-11-28', '2026-03-10'). Do NOT use for general questions, version-agnostic questions, or questions that don't mention a version date — use retrieve for those. Using this tool on a general question wastes a step. inputs: {"endpoint_or_feature": "<string>", "api_version": "2022-11-28 | 2026-03-10"}
- "answer": produce the final answer using everything retrieved so far. inputs: {}
- "finish": stop without producing a new answer (e.g. you already answered in a prior step). inputs: {"message": "<optional note>"}

Rules:
- Always retrieve at least once before answering, unless earlier steps already retrieved relevant context.
- Never repeat a "retrieve" call with the same query you already used.
- Only check deprecation status when the question explicitly names a specific API version date.
- Once you have enough context to answer, call "answer" - do not keep retrieving indefinitely.
- Respond with ONLY the JSON object described in the system prompt.
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_ANSWER_MAX_TOKENS = 1024
_REASON_MAX_TOKENS = 512


def _truncate_json(value: Any, limit: int = 500) -> str:
    text = json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


def _serialize_chunk(chunk: Any) -> dict[str, Any]:
    return {
        "chunk_id": chunk.id,
        "document": chunk.document_name,
        "document_id": chunk.document_id,
        "page": chunk.page,
        "section": chunk.section,
        "excerpt": chunk.content[:280],
    }


def _chunk_to_source(chunk: Any) -> dict[str, Any]:
    return {
        "document": chunk.document_name,
        "document_id": chunk.document_id,
        "page": chunk.page,
        "section": chunk.section,
        "chunk_id": chunk.id,
    }


class AgentEngine:
    def __init__(self, enable_dedup_guard: bool = True):
        self.rag_engine = RagEngine()
        # Exists so evals/trajectory_eval.py's run_mitigation_experiment can
        # measure a real before/after: False reproduces the pre-mitigation
        # behavior (an LLM can repeat the same retrieve query indefinitely,
        # up to max_steps) for the "before" run, True (the default, and the
        # only mode used in production) is the fix.
        self.enable_dedup_guard = enable_dedup_guard

    def _build_react_prompt(self, query: str, history: list[dict[str, Any]], max_steps: int) -> str:
        lines = [f"Question: {query}", f"You have at most {max_steps} reasoning steps total.", ""]

        if history:
            lines.append("Steps so far:")
            for h in history:
                observation = h.get("observation") or {}
                obs_summary = _truncate_json(observation.get("output", {}))
                action = h["action"]
                lines.append(
                    f'- Step {h["step_number"]}: thought="{h["reason"]}" '
                    f'action={action["tool"]}({json.dumps(action["inputs"])}) '
                    f"observation={obs_summary}"
                )
            lines.append("")

        lines.append("Decide the next step. Respond with ONLY the JSON object described in the system prompt.")
        return "\n".join(lines)

    async def _reason(
        self,
        query: str,
        history: list[dict[str, Any]],
        max_steps: int,
        temperature: float,
    ) -> str:
        prompt = self._build_react_prompt(query, history, max_steps)
        text = ""

        async for chunk in generate_completion_stream(
            REACT_SYSTEM_PROMPT, prompt, None, temperature, _REASON_MAX_TOKENS, 1.0
        ):
            if chunk["type"] == "token" and chunk.get("content"):
                text += chunk["content"]
            elif chunk["type"] == "error":
                raise RuntimeError(chunk.get("error", "LLM reasoning call failed"))

        return text

    def _choose_action(self, raw_output: str, query: str) -> tuple[str, AgentAction]:
        """Parse the LLM's JSON action. Any parsing failure - missing JSON,
        invalid JSON, or an unknown tool name - falls back to a retrieve
        action so a malformed model response degrades gracefully instead of
        crashing the run."""
        match = _JSON_BLOCK_RE.search(raw_output or "")
        if not match:
            logger.warning("Agent: no JSON object found in LLM output; falling back to retrieve")
            return (
                "Could not find a JSON action in the model output; defaulting to retrieval.",
                AgentAction(tool=AgentTool.RETRIEVE, inputs={"query": query}),
            )

        try:
            parsed = json.loads(match.group(0))
            tool = AgentTool(parsed.get("tool", "retrieve"))
            thought = str(parsed.get("thought", "")).strip() or "(no thought provided)"
            inputs = parsed.get("inputs")
            if not isinstance(inputs, dict):
                inputs = {}
            return thought, AgentAction(tool=tool, inputs=inputs)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Agent: malformed LLM action output ({e}); falling back to retrieve")
            return (
                "Model output could not be parsed as a valid action; defaulting to retrieval.",
                AgentAction(tool=AgentTool.RETRIEVE, inputs={"query": query}),
            )

    async def run_stream(self, request: AgentRunRequest) -> AsyncGenerator[dict, None]:
        """Run the ReAct loop, yielding SSE-shaped progress events for each
        step and a final trace.completed event carrying the AgentRunResponse."""
        trace_id = str(uuid4())
        start_time = time.time()
        query = request.query
        strategy = request.strategy
        knowledge_base_id = request.knowledge_base_id
        max_steps = request.max_steps
        temperature = request.temperature

        def emit(event_type: str, data: dict) -> dict:
            return {
                "type": event_type,
                "stage": "agent",
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        if not await is_llm_configured():
            error_msg = "LLM API key not configured. Cannot run the agent."
            yield emit("trace.failed", {"error": error_msg})
            yield {"type": "error", "error": error_msg}
            return

        history: list[dict[str, Any]] = []
        steps: list[AgentStep] = []
        retrieved_chunks: list = []
        used_queries: set[str] = set()
        final_answer = ""
        sources: list[dict[str, Any]] = []

        for step_number in range(1, max_steps + 1):
            yield emit("agent.reason", {"step_number": step_number, "status": "thinking"})

            try:
                raw_output = await self._reason(query, history, max_steps, temperature)
            except Exception as e:
                logger.error(f"Agent reasoning call failed at step {step_number}: {e}")
                raw_output = ""

            reason, action = self._choose_action(raw_output, query)

            if action.tool == AgentTool.RETRIEVE and self.enable_dedup_guard:
                search_query = action.inputs.get("query") or query
                if search_query in used_queries:
                    logger.info(
                        f"Agent: repeated retrieve query at step {step_number} ('{search_query}'); forcing answer instead"
                    )
                    reason = "Already searched for this; answering with what's available instead of repeating."
                    action = AgentAction(tool=AgentTool.ANSWER, inputs={})
                else:
                    used_queries.add(search_query)

            yield emit(
                "agent.reason",
                {"step_number": step_number, "status": "done", "reason": reason, "tool": action.tool.value},
            )
            yield emit(
                "agent.action.start",
                {"step_number": step_number, "tool": action.tool.value, "inputs": action.inputs},
            )

            step_start = time.time()

            if action.tool == AgentTool.RETRIEVE:
                search_query = action.inputs.get("query") or query
                call_strategy = strategy
                raw_strategy = action.inputs.get("strategy")
                if raw_strategy:
                    try:
                        call_strategy = RagStrategy(raw_strategy)
                    except ValueError:
                        logger.warning(f"Agent: invalid retrieve strategy '{raw_strategy}' at step {step_number}; using default")
                yield emit(
                    "agent.action.progress",
                    {"step_number": step_number, "status": "searching", "query": search_query},
                )
                try:
                    chunks = await self.rag_engine.get_context(search_query, call_strategy, knowledge_base_id)
                    retrieved_chunks.extend(chunks)
                    observation = AgentObservation(
                        tool=action.tool,
                        output={"chunk_count": len(chunks), "chunks": [_serialize_chunk(c) for c in chunks]},
                        latency_ms=int((time.time() - step_start) * 1000),
                    )
                except Exception as e:
                    logger.error(f"Agent retrieve failed at step {step_number}: {e}")
                    observation = AgentObservation(
                        tool=action.tool,
                        output={},
                        latency_ms=int((time.time() - step_start) * 1000),
                        error=str(e),
                    )

            elif action.tool == AgentTool.CHECK_DEPRECATION:
                endpoint_or_feature = action.inputs.get("endpoint_or_feature") or query
                api_version_str = action.inputs.get("api_version", "2026-03-10")
                try:
                    api_version = ApiVersion(api_version_str)
                except ValueError:
                    api_version = ApiVersion.CURRENT
                yield emit(
                    "agent.action.progress",
                    {"step_number": step_number, "status": "checking_deprecation", "endpoint": endpoint_or_feature, "api_version": api_version.value},
                )
                try:
                    result = await check_deprecation(endpoint_or_feature, api_version)
                    retrieved_chunks.extend(result["raw_chunks"])
                    # Sanitize tool output against prompt injection
                    from evals.trajectory_eval import sanitize_tool_output
                    result["summary"] = sanitize_tool_output(result["summary"])
                    observation = AgentObservation(
                        tool=action.tool,
                        output=result["summary"],
                        latency_ms=int((time.time() - step_start) * 1000),
                    )
                except Exception as e:
                    logger.error(f"Agent check_deprecation failed at step {step_number}: {e}")
                    observation = AgentObservation(
                        tool=action.tool,
                        output={},
                        latency_ms=int((time.time() - step_start) * 1000),
                        error=str(e),
                    )

            elif action.tool == AgentTool.ANSWER:
                if not retrieved_chunks:
                    try:
                        retrieved_chunks.extend(await self.rag_engine.get_context(query, strategy, knowledge_base_id))
                    except Exception as e:
                        logger.error(f"Agent implicit retrieve before answer failed: {e}")

                prompt = build_prompt(query, retrieved_chunks, RagEngineConfig())
                answer_text = ""
                answer_error: Optional[str] = None

                try:
                    async for chunk in generate_completion_stream(
                        prompt["system"] + "\n\n--- Context ---\n\n" + prompt["context"],
                        prompt["user"],
                        None,
                        temperature,
                        _ANSWER_MAX_TOKENS,
                        1.0,
                    ):
                        if chunk["type"] == "token" and chunk.get("content"):
                            answer_text += chunk["content"]
                            yield emit(
                                "agent.action.progress",
                                {"step_number": step_number, "content": chunk["content"]},
                            )
                        elif chunk["type"] == "error":
                            answer_error = chunk.get("error")
                except Exception as e:
                    answer_error = str(e)

                # Run output guardrails on generated answer
                from evals.trajectory_eval import check_output_guardrails
                guardrail = check_output_guardrails(answer_text)
                if not guardrail["clean"]:
                    answer_text = "[ANSWER BLOCKED: potentially unsafe content detected]"

                final_answer = answer_text or final_answer
                if not final_answer:
                    # The LLM call itself errored or streamed nothing - never
                    # leave this blank, since a blank answer looks like the
                    # agent silently did nothing rather than surfacing that
                    # the call failed.
                    final_answer = (
                        f"No answer was generated for this question ({answer_error})."
                        if answer_error
                        else "No answer was generated for this question."
                    )
                observation = AgentObservation(
                    tool=action.tool,
                    output={"answer": answer_text},
                    latency_ms=int((time.time() - step_start) * 1000),
                    error=answer_error,
                )
                sources = [_chunk_to_source(c) for c in retrieved_chunks]

            else:  # FINISH
                observation = AgentObservation(
                    tool=action.tool,
                    output={"message": action.inputs.get("message", "")},
                    latency_ms=int((time.time() - step_start) * 1000),
                )

            yield emit(
                "agent.action.done",
                {
                    "step_number": step_number,
                    "tool": action.tool.value,
                    "latency_ms": observation.latency_ms,
                    "error": observation.error,
                },
            )
            yield emit(
                "agent.observation",
                {
                    "step_number": step_number,
                    "tool": observation.tool.value,
                    "output": observation.output,
                    "latency_ms": observation.latency_ms,
                },
            )

            step = AgentStep(
                step_number=step_number,
                reason=reason,
                action=action,
                observation=observation,
                latency_ms=observation.latency_ms,
            )
            steps.append(step)
            history.append(
                {
                    "step_number": step_number,
                    "reason": reason,
                    "action": {"tool": action.tool.value, "inputs": action.inputs},
                    "observation": {"tool": observation.tool.value, "output": observation.output},
                }
            )

            yield emit("agent.step.completed", step.model_dump(by_alias=True))

            if action.tool in (AgentTool.ANSWER, AgentTool.FINISH):
                break

        BUDGET_EXHAUSTED_MSG = "I could not find a sufficient answer within the allotted reasoning steps."

        if not final_answer:
            if steps and steps[-1].action.tool == AgentTool.FINISH:
                final_answer = steps[-1].observation.output.get("message") or "Finished without producing an answer."
            elif retrieved_chunks:
                # Budget exhausted, but the agent DID retrieve something along
                # the way - make one best-effort answer from it instead of
                # discarding everything gathered, while still appending the
                # budget notice so it's clear this wasn't a clean "answer"
                # tool call (matters for trajectory_pass in the Week 8 eval).
                prompt = build_prompt(query, retrieved_chunks, RagEngineConfig())
                best_effort_answer = ""
                try:
                    async for chunk in generate_completion_stream(
                        prompt["system"] + "\n\n--- Context ---\n\n" + prompt["context"],
                        prompt["user"],
                        None,
                        temperature,
                        _ANSWER_MAX_TOKENS,
                        1.0,
                    ):
                        if chunk["type"] == "token" and chunk.get("content"):
                            best_effort_answer += chunk["content"]
                except Exception as e:
                    logger.error(f"Agent best-effort answer on budget exhaustion failed: {e}")

                if best_effort_answer:
                    from evals.trajectory_eval import check_output_guardrails
                    guardrail = check_output_guardrails(best_effort_answer)
                    if not guardrail["clean"]:
                        best_effort_answer = "[ANSWER BLOCKED: potentially unsafe content detected]"
                    final_answer = f"{best_effort_answer}\n\n{BUDGET_EXHAUSTED_MSG}"
                    sources = [_chunk_to_source(c) for c in retrieved_chunks]
                else:
                    final_answer = BUDGET_EXHAUSTED_MSG
            else:
                final_answer = BUDGET_EXHAUSTED_MSG

        total_latency_ms = int((time.time() - start_time) * 1000)

        response = AgentRunResponse(
            answer=final_answer,
            steps=steps,
            total_latency_ms=total_latency_ms,
            input_tokens=estimate_tokens(query),
            output_tokens=estimate_tokens(final_answer),
            trace_id=trace_id,
            sources=sources,
        )

        try:
            from app.services.storage import add_trace

            await add_trace(
                {
                    **response.model_dump(),
                    "kind": "agent",
                    "query": query,
                    "strategy": strategy.value if strategy else None,
                    "knowledge_base_id": knowledge_base_id,
                }
            )
        except Exception as e:
            logger.error(f"Failed to persist agent trace: {e}")

        yield emit("trace.completed", response.model_dump(by_alias=True))

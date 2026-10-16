"""LangGraph-based ReAct Agent implementation.

Provides an agent implementation using LangGraph's StateGraph while maintaining
compatibility with the existing MCP client, RAG engine, and tool schemas.
"""

import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict
from uuid import uuid4

from langgraph.graph import StateGraph, END

from app.core.logging_config import get_logger
from app.models.schemas import (
    AgentAction,
    AgentRunRequest,
    AgentRunResponse,
    AgentTool,
    RagEngineConfig,
    RagStrategy,
)
from app.services.agent_tools import TOOL_SCHEMAS, validate_tool_inputs
from app.services.llm import generate_completion_stream, is_llm_configured
from app.services.mcp_client import get_mcp_manager, MCPManager
from app.services.rag_engine import RagEngine, build_prompt, estimate_tokens
from app.services.storage import add_trace

logger = get_logger(__name__)

REACT_SYSTEM_PROMPT_FALLBACK = """You are a ReAct-style research agent that answers questions using a document knowledge base.

You reason step by step in a Thought -> Action -> Observation loop. At each step you must respond with a single JSON object and nothing else - no markdown fences, no commentary before or after it:

{"thought": "<your reasoning about what to do next>", "tool": "<retrieve|check_deprecation|answer|finish>", "inputs": {<tool inputs>}}

Available tools:
- "retrieve": Open-ended search of the knowledge base for context relevant to a query. inputs: {query: string, strategy: optional enum vector|bm25|hybrid|hybrid-rrf|hybrid-rerank|hybrid-rerank-mmr}
- "check_deprecation": Check whether a specific API endpoint or feature is deprecated or removed in a given API version, and what replaced it. ONLY use when the question explicitly mentions a specific API version date. inputs: {endpoint_or_feature: string, api_version: enum 2022-11-28|2026-03-10}
- "answer": Produce the final answer using everything retrieved so far. inputs: {}
- "finish": Stop without producing a new answer. inputs: {message: string - optional note}

Rules:
- Always retrieve at least once before answering, unless earlier steps already retrieved relevant context.
- Never repeat a retrieve call with the same query you already used.
- Only check deprecation status when the question explicitly names a specific API version date.
- Once you have enough context to answer, call answer - do not keep retrieving indefinitely.
- Respond with ONLY the JSON object described in the system prompt.
"""

_ANSWER_MAX_TOKENS = 1024
_REASON_MAX_TOKENS = 512
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class AgentState(TypedDict):
    """State for the LangGraph agent."""
    messages: List[Dict[str, Any]]
    query: str
    strategy: Optional[RagStrategy]
    knowledge_base_id: Optional[str]
    max_steps: int
    temperature: float
    step_count: int
    final_answer: str
    retrieved_chunks: List[Any]
    used_queries: set
    sources: List[Dict[str, Any]]
    mcp: bool
    last_action: Optional[Dict[str, Any]]
    step_outputs: List[Dict[str, Any]]
    input_tokens: int
    output_tokens: int


def _serialize_chunk(chunk: Any) -> Dict[str, Any]:
    return {
        "chunk_id": chunk.id,
        "document": chunk.document_name,
        "document_id": chunk.document_id,
        "page": chunk.page,
        "section": chunk.section,
        "excerpt": chunk.content[:280],
    }


def _chunk_to_source(chunk: Any) -> Dict[str, Any]:
    return {
        "document": chunk.document_name,
        "document_id": chunk.document_id,
        "page": chunk.page,
        "section": chunk.section,
        "chunk_id": chunk.id,
    }


class LangGraphAgent:
    """LangGraph-based ReAct agent with MCP support."""

    def __init__(self, enable_dedup_guard: bool = True):
        self.rag_engine = RagEngine()
        self.enable_dedup_guard = enable_dedup_guard
        self.graph = self._build_graph()
        logger.info("LangGraphAgent initialized")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph with ReAct pattern."""
        workflow = StateGraph(AgentState)

        workflow.add_node("reason", self._reason_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("check_deprecation", self._check_deprecation_node)
        workflow.add_node("answer", self._answer_node)
        workflow.add_node("finish", self._finish_node)

        workflow.add_conditional_edges(
            "reason",
            self._route_action,
            {
                "retrieve": "retrieve",
                "check_deprecation": "check_deprecation",
                "answer": "answer",
                "finish": "finish",
            },
        )

        workflow.add_edge("retrieve", "reason")
        workflow.add_edge("check_deprecation", "reason")

        workflow.set_entry_point("reason")

        return workflow.compile()

    def _parse_action(self, raw_output: str, query: str) -> tuple[str, AgentAction]:
        """Parse the LLM's JSON action output."""
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
            inputs = parsed.get("inputs", {})
            if not isinstance(inputs, dict):
                inputs = {}
            return thought, AgentAction(tool=tool, inputs=inputs)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Agent: malformed LLM action output ({e}); falling back to retrieve")
            return (
                "Model output could not be parsed as a valid action; defaulting to retrieval.",
                AgentAction(tool=AgentTool.RETRIEVE, inputs={"query": query}),
            )

    async def _reason_node(self, state: AgentState) -> Dict[str, Any]:
        """Reason about what action to take next."""
        if state["step_count"] >= state["max_steps"]:
            return self._build_final_state(state)

        system_prompt = await self._build_system_prompt(state)
        user_prompt = self._build_user_prompt(state)

        raw_output = ""
        async for chunk in generate_completion_stream(
            system_prompt, user_prompt, None,
            state["temperature"], _REASON_MAX_TOKENS, 1.0
        ):
            if chunk.get("type") == "token" and chunk.get("content"):
                raw_output += chunk["content"]
            elif chunk.get("type") == "error":
                logger.error(f"Agent reasoning call failed at step {state['step_count'] + 1}: {chunk.get('error')}")
                raw_output = ""

        thought, action = self._parse_action(raw_output, state["query"])

        used_queries = state["used_queries"]
        if (
            action.tool == AgentTool.RETRIEVE
            and self.enable_dedup_guard
        ):
            search_query = action.inputs.get("query") or state["query"]
            if search_query in used_queries:
                logger.info(f"Agent: repeated retrieve query at step {state['step_count'] + 1} ('{search_query}'); forcing answer")
                thought = "Already searched for this; answering with what's available instead of repeating."
                action = AgentAction(tool=AgentTool.ANSWER, inputs={})
            else:
                used_queries = used_queries | {search_query}

        new_message = {
            "role": "assistant",
            "content": json.dumps({
                "thought": thought,
                "tool": action.tool.value,
                "inputs": action.inputs,
            }),
        }

        return {
            "step_count": state["step_count"] + 1,
            "messages": state["messages"] + [new_message],
            "last_action": {
                "tool": action.tool.value,
                "inputs": action.inputs,
            },
            "reason": thought,
            "used_queries": used_queries,
        }

    def _route_action(self, state: AgentState) -> str:
        """Route to the appropriate node based on the last action."""
        last_action = state.get("last_action")
        if not last_action:
            return "retrieve"

        tool = last_action.get("tool", "retrieve")
        return tool if tool in {"retrieve", "check_deprecation", "answer", "finish"} else "retrieve"

    async def _retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute retrieve tool action."""
        last_action = state.get("last_action")
        if last_action:
            try:
                action = AgentAction(
                    tool=AgentTool(last_action.get("tool", "retrieve")),
                    inputs=last_action.get("inputs", {}),
                )
            except (ValueError, TypeError):
                action = AgentAction(tool=AgentTool.RETRIEVE, inputs={"query": state["query"]})
        else:
            action = AgentAction(tool=AgentTool.RETRIEVE, inputs={"query": state["query"]})

        search_query = action.inputs.get("query") or state["query"]
        call_strategy = state["strategy"]
        raw_strategy = action.inputs.get("strategy")
        if raw_strategy:
            try:
                call_strategy = RagStrategy(raw_strategy)
            except ValueError:
                logger.warning(f"Agent: invalid retrieve strategy '{raw_strategy}'; using default")

        try:
            chunks = await self.rag_engine.get_context(
                search_query, call_strategy, state["knowledge_base_id"]
            )

            observation = {
                "tool": AgentTool.RETRIEVE.value,
                "output": {
                    "chunk_count": len(chunks),
                    "chunks": [_serialize_chunk(c) for c in chunks],
                },
                "latency_ms": 0,
            }

            return {
                "retrieved_chunks": state["retrieved_chunks"] + chunks,
                "messages": state["messages"] + [
                    {
                        "role": "user",
                        "content": json.dumps({
                            "tool": AgentTool.RETRIEVE.value,
                            "inputs": action.inputs,
                            "observation": observation,
                        }),
                    }
                ],
                "step_outputs": state["step_outputs"]
                + [
                    {
                        "step_number": state["step_count"],
                        "reason": state.get("reason", ""),
                        "action": action,
                        "observation": observation,
                    }
                ],
            }
        except Exception as e:
            logger.error(f"Agent retrieve failed at step {state['step_count']}: {e}")
            observation = {
                "tool": AgentTool.RETRIEVE.value,
                "output": {},
                "latency_ms": 0,
                "error": str(e),
            }
            return {
                "messages": state["messages"] + [
                    {
                        "role": "user",
                        "content": json.dumps({
                            "tool": AgentTool.RETRIEVE.value,
                            "inputs": action.inputs,
                            "observation": observation,
                        }),
                    }
                ],
                "step_outputs": state["step_outputs"]
                + [
                    {
                        "step_number": state["step_count"],
                        "reason": state.get("reason", ""),
                        "action": action,
                        "observation": observation,
                    }
                ],
            }

    async def _check_deprecation_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute check_deprecation tool action."""
        last_action = state.get("last_action")
        if last_action:
            action_inputs = last_action.get("inputs", {})
        else:
            action_inputs = {}

        endpoint_or_feature = action_inputs.get("endpoint_or_feature") or state["query"]
        api_version_str = action_inputs.get("api_version", "2026-03-10")

        try:
            from app.services.agent_tools import ApiVersion
            api_version = ApiVersion(api_version_str)
        except ValueError:
            api_version = ApiVersion.CURRENT

        try:
            mcp_manager = await get_mcp_manager()
            result = await mcp_manager.call_tool(
                "check_deprecation_status",
                {
                    "endpoint_or_feature": endpoint_or_feature,
                    "api_version": api_version.value,
                },
            )

            if result:
                content = ""
                if hasattr(result, "content") and result.content:
                    if isinstance(result.content, list) and len(result.content) > 0:
                        content = getattr(result.content[0], "text", str(result.content[0]))
                    else:
                        content = str(result.content)
                else:
                    content = str(result)

                try:
                    parsed_content = json.loads(content)
                    output_data = parsed_content
                except (json.JSONDecodeError, TypeError):
                    output_data = {"summary": content}

                # Also try to get raw chunks from the result
                raw_chunks = []
                if isinstance(output_data, dict) and output_data.get("chunks"):
                    raw_chunks = output_data["chunks"]

                observation = {
                    "tool": AgentTool.CHECK_DEPRECATION.value,
                    "output": output_data,
                    "latency_ms": 0,
                }

                return {
                    "retrieved_chunks": state["retrieved_chunks"],
                    "messages": state["messages"] + [
                        {
                            "role": "user",
                            "content": json.dumps({
                                "tool": AgentTool.CHECK_DEPRECATION.value,
                                "inputs": action_inputs,
                                "observation": observation,
                            }),
                        }
                    ],
                    "step_outputs": state["step_outputs"]
                    + [
                        {
                            "step_number": state["step_count"],
                            "reason": state.get("reason", ""),
                            "action": AgentAction(
                                tool=AgentTool.CHECK_DEPRECATION,
                                inputs=action_inputs,
                            ),
                            "observation": observation,
                        }
                    ],
                }
            else:
                observation = {
                    "tool": AgentTool.CHECK_DEPRECATION.value,
                    "output": {},
                    "latency_ms": 0,
                    "error": "Tool 'check_deprecation_status' not found on any MCP server",
                }
                return self._emit_observation(state, AgentTool.CHECK_DEPRECATION, observation, action_inputs)
        except Exception as e:
            logger.error(f"Agent check_deprecation failed at step {state['step_count']}: {e}")
            observation = {
                "tool": AgentTool.CHECK_DEPRECATION.value,
                "output": {},
                "latency_ms": 0,
                "error": str(e),
            }
            return self._emit_observation(state, AgentTool.CHECK_DEPRECATION, observation, action_inputs)

    def _emit_observation(self, state: AgentState, tool: AgentTool, observation: dict, inputs: dict) -> Dict[str, Any]:
        """Helper to emit observation state for non-retrieval tools."""
        return {
            "messages": state["messages"] + [
                {
                    "role": "user",
                    "content": json.dumps({
                        "tool": tool.value,
                        "inputs": inputs,
                        "observation": observation,
                    }),
                }
            ],
            "step_outputs": state["step_outputs"]
            + [
                {
                    "step_number": state["step_count"],
                    "reason": state.get("reason", ""),
                    "action": AgentAction(tool=tool, inputs=inputs),
                    "observation": observation,
                }
            ],
        }

    async def _answer_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate final answer from retrieved chunks."""
        retrieved_chunks = state["retrieved_chunks"]

        if not retrieved_chunks:
            try:
                retrieved_chunks = await self.rag_engine.get_context(
                    state["query"], state["strategy"], state["knowledge_base_id"]
                )
            except Exception as e:
                logger.error(f"Agent implicit retrieve before answer failed: {e}")

        try:
            prompt = build_prompt(state["query"], retrieved_chunks, RagEngineConfig())
            answer_text = ""
            input_tokens = estimate_tokens(prompt["system"] + prompt["context"] + prompt["user"])
            output_tokens = 0

            async for chunk in generate_completion_stream(
                prompt["system"] + "\n\n--- Context ---\n\n" + prompt["context"],
                prompt["user"],
                None,
                state["temperature"],
                _ANSWER_MAX_TOKENS,
                1.0,
            ):
                if chunk.get("type") == "token" and chunk.get("content"):
                    answer_text += chunk["content"]
                elif chunk.get("type") == "error":
                    logger.error(f"Answer error: {chunk.get('error')}")

            # Run guardrails
            from evals.trajectory_eval import check_output_guardrails
            guardrail = check_output_guardrails(answer_text)
            if not guardrail["clean"]:
                answer_text = "[ANSWER BLOCKED: potentially unsafe content detected]"

            if not answer_text.strip():
                answer_text = "No answer was generated for this question."

            sources = [_chunk_to_source(c) for c in retrieved_chunks]
            output_tokens = estimate_tokens(answer_text)

            return {
                "final_answer": answer_text,
                "sources": sources,
                "input_tokens": state["input_tokens"] + input_tokens,
                "output_tokens": state["output_tokens"] + output_tokens,
                "messages": state["messages"] + [
                    {
                        "role": "assistant",
                        "content": json.dumps({
                            "tool": AgentTool.ANSWER.value,
                            "answer": answer_text,
                        }),
                    }
                ],
                "step_outputs": state["step_outputs"]
                + [
                    {
                        "step_number": state["step_count"],
                        "reason": state.get("reason", ""),
                        "action": AgentAction(tool=AgentTool.ANSWER, inputs={}),
                        "observation": {
                            "tool": AgentTool.ANSWER.value,
                            "output": {"answer": answer_text},
                            "latency_ms": 0,
                        },
                    }
                ],
            }
        except Exception as e:
            logger.error(f"Agent answer generation failed: {e}")
            return {
                "final_answer": "An error occurred while generating the answer.",
                "sources": [],
            }

    async def _finish_node(self, state: AgentState) -> Dict[str, Any]:
        """Finish the agent run."""
        last_action = state.get("last_action")
        message = ""
        if last_action and isinstance(last_action.get("inputs"), dict):
            message = last_action["inputs"].get("message", "")

        return {
            "final_answer": message or "Finished without producing an answer.",
            "messages": state["messages"] + [
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "tool": AgentTool.FINISH.value,
                        "inputs": last_action or {},
                    }),
                }
            ],
        }

    def _build_final_state(self, state: AgentState) -> Dict[str, Any]:
        """Build final state when max steps reached."""
        final_answer = state.get("final_answer", "")
        if not final_answer:
            if state["retrieved_chunks"]:
                import asyncio

                async def _best_effort():
                    prompt = build_prompt(
                        state["query"], state["retrieved_chunks"], RagEngineConfig()
                    )
                    text = ""
                    async for chunk in generate_completion_stream(
                        prompt["system"] + "\n\n--- Context ---\n\n" + prompt["context"],
                        prompt["user"], None, state["temperature"], _ANSWER_MAX_TOKENS, 1.0,
                    ):
                        if chunk.get("type") == "token" and chunk.get("content"):
                            text += chunk["content"]
                    if text.strip():
                        from evals.trajectory_eval import check_output_guardrails
                        guardrail = check_output_guardrails(text)
                        if not guardrail["clean"]:
                            text = "[ANSWER BLOCKED: potentially unsafe content detected]"
                        return f"{text}\n\nI could not find a sufficient answer within the allotted reasoning steps."
                    return "I could not find a sufficient answer within the allotted reasoning steps."

                final_answer = asyncio.get_event_loop().run_until_complete(_best_effort())
            else:
                final_answer = "I could not find a sufficient answer within the allotted reasoning steps."

        sources = [_chunk_to_source(c) for c in state["retrieved_chunks"]]

        return {
            "final_answer": final_answer,
            "sources": sources,
            "messages": state["messages"],
            "step_outputs": state["step_outputs"],
        }

    async def _build_system_prompt(self, state: AgentState) -> str:
        """Build system prompt dynamically from available tools."""
        if state["mcp"]:
            try:
                mcp_manager = await get_mcp_manager()
                mcp_tools = await mcp_manager.discover_tools()
                if mcp_tools:
                    tool_descriptions = []
                    for name, tool in mcp_tools.items():
                        desc = tool.description or "No description available"
                        schema = tool.inputSchema or {}
                        properties = schema.get("properties", {})
                        if properties:
                            parts = [f"{k}: {v.get('type', 'any')}" for k, v in properties.items()]
                            inputs_str = f" inputs: {{{', '.join(parts)}}}"
                        else:
                            inputs_str = " inputs: {}"
                        tool_descriptions.append(f'- "{name}": {desc}{inputs_str}')

                    tools_section = "\n".join(tool_descriptions)
                    return (
                        "You are a ReAct-style research agent that answers questions "
                        "using a document knowledge base.\n\n"
                        "You reason step by step in a Thought -> Action -> Observation "
                        "loop. At each step you must respond with a single JSON object "
                        "and nothing else:\n\n"
                        '{"thought": "<reasoning>", "tool": "<TOOL_NAME>", "inputs": {<inputs>}}\n\n'
                        f"Available tools:\n{tools_section}\n\n"
                        "Rules:\n"
                        "- Always retrieve at least once before answering.\n"
                        "- Never repeat a retrieve call with the same query.\n"
                        "- Once you have enough context to answer, call answer.\n"
                        "- Respond with ONLY the JSON object."
                    )
            except Exception as e:
                logger.warning(f"Failed to get MCP tools for system prompt: {e}")

        return REACT_SYSTEM_PROMPT_FALLBACK

    def _build_user_prompt(self, state: AgentState) -> str:
        """Build the user prompt with conversation history."""
        lines = [f"Question: {state['query']}", f"You have at most {state['max_steps']} reasoning steps total.", ""]

        if state["messages"]:
            lines.append("Steps so far:")
            for h in state["messages"]:
                if h.get("role") == "assistant":
                    try:
                        content = h.get("content", "")
                        if content.startswith("{") and content.endswith("}"):
                            parsed = json.loads(content)
                            thought = parsed.get("thought", "")
                            tool = parsed.get("tool", "")
                            inputs = parsed.get("inputs", {})
                            lines.append(
                                f'- Step {state["step_count"]}: thought="{thought}" '
                                f'action={tool}({json.dumps(inputs)})'
                            )
                    except (json.JSONDecodeError, KeyError):
                        continue
            lines.append("")

        lines.append(
            "Decide the next step. Respond with ONLY the JSON object described in the system prompt."
        )
        return "\n".join(lines)

    async def run_stream(self, request: AgentRunRequest):
        """Run the agent and yield SSE events, compatible with AgentEngine.run_stream."""
        from datetime import datetime, timezone

        trace_id = str(uuid4())
        start_time = time.time()

        def emit(event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
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

        initial_state: AgentState = {
            "messages": [],
            "query": request.query,
            "strategy": request.strategy,
            "knowledge_base_id": request.knowledge_base_id,
            "max_steps": request.max_steps,
            "temperature": request.temperature,
            "step_count": 0,
            "final_answer": "",
            "retrieved_chunks": [],
            "used_queries": set(),
            "sources": [],
            "mcp": request.mcp or getattr(request, "langgraph", False),
            "last_action": None,
            "step_outputs": [],
            "input_tokens": 0,
            "output_tokens": 0,
        }

        # Stream events from the graph
        try:
            async for event in self.graph.astream(initial_state):
                for node_name, node_state in event.items():
                    if node_name == "__end__":
                        final_state = node_state
                        if final_state.get("final_answer"):
                            response = AgentRunResponse(
                                answer=final_state["final_answer"],
                                steps=[
                                    AgentStep(
                                        step_number=s.get("step_number", 0),
                                        reason=s.get("reason", ""),
                                        action=s.get("action"),
                                        observation=s.get("observation"),
                                    )
                                    for s in final_state.get("step_outputs", [])
                                ],
                                total_latency_ms=int((time.time() - start_time) * 1000),
                                input_tokens=final_state.get("input_tokens", estimate_tokens(request.query)),
                                output_tokens=final_state.get("output_tokens", estimate_tokens(final_state.get("final_answer", ""))),
                                trace_id=trace_id,
                                sources=final_state.get("sources", []),
                            )
                            try:
                                await add_trace({
                                    **response.model_dump(),
                                    "kind": "agent",
                                    "query": request.query,
                                    "strategy": request.strategy.value if request.strategy else None,
                                    "knowledge_base_id": request.knowledge_base_id,
                                })
                            except Exception as e:
                                logger.error(f"Failed to persist agent trace: {e}")

                            yield emit("trace.completed", response.model_dump(by_alias=True))
                        continue

                    # Emit step events for each node
                    step_outputs = node_state.get("step_outputs", [])
                    for s in step_outputs:
                        if s.get("action") is not None:
                            action = s["action"]
                            if hasattr(action, "model_dump"):
                                action = action.model_dump()
                            elif hasattr(action, "dict"):
                                action = action.dict()
                            else:
                                action = action if isinstance(action, dict) else {}

                            yield emit(
                                "agent.reason",
                                {
                                    "step_number": s["step_number"],
                                    "status": "done",
                                    "reason": s.get("reason", ""),
                                    "tool": action.get("tool", ""),
                                },
                            )
                            yield emit(
                                "agent.action.start",
                                {
                                    "step_number": s["step_number"],
                                    "tool": action.get("tool", ""),
                                    "inputs": action.get("inputs", {}),
                                },
                            )

                            if s.get("observation"):
                                obs = s["observation"]
                                yield emit(
                                    "agent.action.done",
                                    {
                                        "step_number": s["step_number"],
                                        "tool": obs.get("tool", ""),
                                        "latency_ms": obs.get("latency_ms", 0),
                                        "error": obs.get("error"),
                                    },
                                )
                                yield emit(
                                    "agent.observation",
                                    {
                                        "step_number": s["step_number"],
                                        "tool": obs.get("tool", ""),
                                        "output": obs.get("output", {}),
                                        "latency_ms": obs.get("latency_ms", 0),
                                    },
                                )

                            yield emit("agent.step.completed", AgentStep(
                                step_number=s["step_number"],
                                reason=s.get("reason", ""),
                                action=AgentAction(
                                    tool=AgentTool(action.get("tool", "retrieve")),
                                    inputs=action.get("inputs", {}),
                                ),
                                observation=None if not s.get("observation") else AgentObservation(
                                    tool=AgentTool(s["observation"].get("tool", "retrieve")),
                                    output=s["observation"].get("output", {}),
                                    latency_ms=s["observation"].get("latency_ms", 0),
                                    error=s["observation"].get("error"),
                                ),
                            ).model_dump(by_alias=True))
        except Exception as e:
            logger.error(f"LangGraph agent execution failed: {e}")
            yield emit("trace.failed", {"error": str(e)})
            yield {"type": "error", "error": str(e)}
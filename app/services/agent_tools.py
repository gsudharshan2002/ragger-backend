"""Concept 3: Tool design.

A tool is a narrow, single-purpose capability the agent can call - not a
general escape hatch. Each tool here has one job, returns a plain dict
observation, and never lets an exception escape past its own boundary.

Notice the two different "shapes" of output below: `search_knowledge_base`
returns a SUMMARY (short, LLM/UI-friendly) *and* the RAW chunks (full
content, needed internally so `generate_answer` can build a real prompt
later). Tool output almost always needs to serve two different audiences
like this - keep them as separate keys instead of guessing which fields
are "safe" to expose.
"""
from typing import Any, Optional
from enum import Enum

from pydantic import BaseModel, ValidationError

from app.models.schemas import RagEngineConfig, RagStrategy
from app.services.llm import generate_completion_stream
from app.services.rag_engine import RagEngine, build_prompt, estimate_tokens

_rag_engine = RagEngine()


async def search_knowledge_base(
    query: str,
    strategy: Optional[RagStrategy] = None,
    knowledge_base_id: Optional[str] = None,
) -> dict[str, Any]:
    """Tool: search the knowledge base for context relevant to `query`."""
    try:
        chunks = await _rag_engine.get_context(query, strategy, knowledge_base_id)
    except Exception as e:
        return {"summary": {"chunk_count": 0, "chunks": [], "error": str(e)}, "raw_chunks": []}

    summary = {
        "chunk_count": len(chunks),
        "chunks": [
            {"document": c.document_name, "page": c.page, "section": c.section, "excerpt": c.content[:280]}
            for c in chunks
        ],
    }
    return {"summary": summary, "raw_chunks": chunks}


async def generate_answer(query: str, retrieved_chunks: list) -> dict[str, Any]:
    """Tool: produce the final answer using whatever context has been
    retrieved so far. Reports its own token cost so both the agent and the
    fixed workflow can account for it the same way, rather than each
    guessing independently."""
    prompt = build_prompt(query, retrieved_chunks, RagEngineConfig())
    input_tokens = estimate_tokens(prompt["system"] + prompt["context"] + prompt["user"])
    answer = ""

    try:
        async for chunk in generate_completion_stream(
            prompt["system"] + "\n\n--- Context ---\n\n" + prompt["context"],
            prompt["user"],
            None,
            0.2,
            1024,
            1.0,
        ):
            if chunk["type"] == "token" and chunk.get("content"):
                answer += chunk["content"]
            elif chunk["type"] == "error":
                return {"answer": answer, "input_tokens": input_tokens, "output_tokens": estimate_tokens(answer), "error": chunk.get("error")}
    except Exception as e:
        return {"answer": answer, "input_tokens": input_tokens, "output_tokens": estimate_tokens(answer), "error": str(e)}

    return {"answer": answer, "input_tokens": input_tokens, "output_tokens": estimate_tokens(answer)}

class ApiVersion(str, Enum):
    OLD = "2022-11-28"
    CURRENT = "2026-03-10"


# --- Strict, type-safe input schemas -----------------------------------
# One Pydantic model per tool. Validating the LLM's JSON `inputs` against
# these BEFORE dispatch (see validate_tool_inputs) turns a malformed tool
# call - a missing required field, a version string that isn't a real
# enum value - into a clean, recoverable observation instead of a raw
# TypeError/ValueError happening deep inside the tool function itself.
class RetrieveInputs(BaseModel):
    query: Optional[str] = None


class CheckDeprecationInputs(BaseModel):
    endpoint_or_feature: Optional[str] = None
    api_version: ApiVersion


class AnswerInputs(BaseModel):
    pass


class FinishInputs(BaseModel):
    message: str = ""


TOOL_SCHEMAS: dict[str, type[BaseModel]] = {
    "retrieve": RetrieveInputs,
    "check_deprecation": CheckDeprecationInputs,
    "answer": AnswerInputs,
    "finish": FinishInputs,
}


def validate_tool_inputs(tool: str, raw_inputs: dict) -> tuple[Optional[BaseModel], Optional[str]]:
    """Validate the LLM-supplied `inputs` for `tool` against its strict
    schema. Returns (validated_model, None) on success, or (None,
    error_message) on failure - callers should treat a validation failure
    the same way a malformed-JSON action is treated (Concept 2): fail
    gracefully with a clear observation, never let it crash the loop."""
    schema = TOOL_SCHEMAS.get(tool)
    if schema is None:
        return None, f"Unknown tool: {tool}"
    try:
        return schema(**raw_inputs), None
    except ValidationError as e:
        return None, str(e)


async def check_deprecation(endpoint_or_feature: str, api_version: ApiVersion) -> dict[str, Any]:
    """Tool: Determine if a specific API endpoint or feature has been
    deprecated or removed in a given API version, and what replaced it.

    This is NOT a general search — it is a targeted deprecation lookup.
    Use it when the question asks about version-specific changes like:
      - "Does X still exist in version Y?"
      - "Was Z removed in 2026-03-10?"
      - "What replaced [endpoint] between versions?"

    Returns a summary of matching deprecation changelog entries and the
    raw chunks so callers can build a full prompt from them.

    The search is scoped to the specific api_version (e.g. "2022-11-28"
    or "2026-03-10"), which are the two versions indexed in the knowledge base.
    """
    search_query = f"{endpoint_or_feature} deprecated changelog {api_version.value}"
    result = await search_knowledge_base(search_query)

    return {
        "summary": {
            "endpoint_or_feature": endpoint_or_feature,
            "api_version": api_version.value,
            "chunk_count": result["summary"]["chunk_count"],
            "chunks": result["summary"]["chunks"],
        },
        "raw_chunks": result["raw_chunks"],
    }


# The TOOL REGISTRY - the single source of truth for "what can the agent
# do." Each `description` becomes part of the system prompt, so the LLM
# knows *when* to reach for that tool. Concept 4 (tool calling) will use
# this dict's keys to dispatch the LLM's chosen tool name to the actual
# function above - so adding a new tool later is a one-line addition
# here, and nothing in agent_loop.py has to change.
TOOLS: dict[str, dict[str, Any]] = {
    "retrieve": {
        "description": "Open-ended search of the knowledge base for context relevant to a query.",
        "inputs": {"query": "string - the search text"},
    },
    "check_deprecation": {
        "description": "Check whether a specific API endpoint or feature is deprecated or removed in a given API version, and what replaced it. Use when the question asks about version-specific changes like 'Does X still exist in version Y?' or 'Was Z removed in 2026-03-10?'. Do NOT use for general knowledge searches — use retrieve for those.",
        "inputs": {"endpoint_or_feature": "string - the endpoint or feature name", "api_version": "enum: 2022-11-28 | 2026-03-10 - the API version to check against"},
    },
    "answer": {
        "description": "Produce the final answer using everything retrieved so far.",
        "inputs": {},
    },
    "finish": {
        "description": "Stop without producing a new answer.",
        "inputs": {"message": "string - optional note"},
    },
}



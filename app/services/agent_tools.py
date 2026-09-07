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

from app.models.schemas import RagEngineConfig, RagStrategy
from app.services.llm import generate_completion_stream
from app.services.rag_engine import RagEngine, build_prompt

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
    retrieved so far."""
    prompt = build_prompt(query, retrieved_chunks, RagEngineConfig())
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
                return {"answer": answer, "error": chunk.get("error")}
    except Exception as e:
        return {"answer": answer, "error": str(e)}

    return {"answer": answer}


# The TOOL REGISTRY - the single source of truth for "what can the agent
# do." Each `description` becomes part of the system prompt, so the LLM
# knows *when* to reach for that tool. Concept 4 (tool calling) will use
# this dict's keys to dispatch the LLM's chosen tool name to the actual
# function above - so adding a new tool later is a one-line addition
# here, and nothing in agent_loop.py has to change.
TOOLS: dict[str, dict[str, Any]] = {
    "retrieve": {
        "description": "Search the knowledge base for context relevant to a query.",
        "inputs": {"query": "string - the search text"},
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
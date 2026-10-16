"""Standalone MCP server exposing every tool from the in-process agent.

Wraps the same functions the agent loop dispatches to
(app/services/agent_tools.py, app/services/agent_loop.py) so any MCP
client can drive this project's RAG pipeline directly - search,
deprecation lookups, and answer generation - without going through
the agent loop or the HTTP API.

Each call is stateless: `answer` doesn't share memory with a prior
`retrieve` call in the same session, so pass the exact `chunks` list a
prior `retrieve`/`check_deprecation` call returned back into `answer`.

Run:
    uv run python -m app.mcp_server

Or, for anyone else, point an HTTP MCP client at the copy mounted inside
the main FastAPI app (see app/main.py) - no separate process needed:
    http://<host>:<port>/api/v1/mcp
"""
from typing import Any, Optional

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from app.models.schemas import Chunk, RagStrategy
from app.services.agent_tools import ApiVersion, check_deprecation, generate_answer, search_knowledge_base

mcp = MCPServer("ragger-tools")


def _serialize_chunks(chunks: list) -> list[dict]:
    return [
        {
            "document_id": c.document_id,
            "document": c.document_name,
            "page": c.page,
            "section": c.section,
            "content": c.content,
        }
        for c in chunks
    ]


@mcp.tool()
async def retrieve(
    query: str,
    strategy: Optional[str] = None,
    knowledge_base_id: Optional[str] = None,
) -> dict:
    """Open-ended search of the knowledge base for context relevant to `query`.

    strategy: optional - one of vector | bm25 | hybrid | hybrid-rrf |
    hybrid-rerank | hybrid-rerank-mmr. Leave unset to use the server's
    configured default. Use 'bm25' when the query contains an exact
    technical term, field/parameter name, or identifier likely to appear
    verbatim in the docs. Use 'vector' for a conceptual/paraphrased
    question with no exact term to match. Leave unset otherwise.

    knowledge_base_id: optional - restrict the search to a single
    knowledge base instead of searching across all of them.

    Returns the matching chunks. Pass them to `answer` to generate a
    final answer once you've gathered enough context.
    """
    parsed_strategy = RagStrategy(strategy) if strategy else None
    result = await search_knowledge_base(query, parsed_strategy, knowledge_base_id)

    if result["summary"].get("error"):
        return {"error": result["summary"]["error"], "chunk_count": 0, "chunks": []}

    return {
        "chunk_count": result["summary"]["chunk_count"],
        "chunks": _serialize_chunks(result["raw_chunks"]),
    }


@mcp.tool()
async def check_deprecation_status(
    endpoint_or_feature: str,
    api_version: str,
) -> dict:
    """Check whether a specific API endpoint or feature is deprecated or
    removed in a given API version, and what replaced it.

    ONLY use when the question explicitly mentions a specific API version
    date. Do NOT use for general or version-agnostic questions - use
    `retrieve` for those; using this tool on a general question wastes a
    step.

    api_version: one of 2022-11-28 | 2026-03-10 - the two versions indexed
    in the knowledge base.

    Returns matching deprecation changelog chunks. Pass them to `answer`
    to generate a final answer.
    """
    try:
        parsed_version = ApiVersion(api_version)
    except ValueError:
        return {
            "error": f"Unknown api_version {api_version!r}; expected one of {[v.value for v in ApiVersion]}",
            "chunk_count": 0,
            "chunks": [],
        }

    result = await check_deprecation(endpoint_or_feature, parsed_version)
    return {
        "endpoint_or_feature": endpoint_or_feature,
        "api_version": parsed_version.value,
        "chunk_count": result["summary"]["chunk_count"],
        "chunks": _serialize_chunks(result["raw_chunks"]),
    }


@mcp.tool()
async def answer(query: str, chunks: list[dict[str, Any]]) -> dict:
    """Produce the final answer to `query` using previously retrieved
    context.

    chunks: the exact `chunks` list returned by a prior `retrieve` or
    `check_deprecation_status` call (call one of those first - this tool
    does not search the knowledge base itself). Pass an empty list to get
    a "no context" answer.
    """
    reconstructed = [
        Chunk(
            document_id=c.get("document_id", ""),
            document_name=c["document"],
            content=c["content"],
            page=c["page"],
            section=c.get("section"),
        )
        for c in chunks
    ]
    result = await generate_answer(query, reconstructed)

    output = {
        "answer": result["answer"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
    }
    if result.get("error"):
        output["error"] = result["error"]
    return output


@mcp.tool()
def finish(message: str = "") -> dict:
    """Stop without producing a new answer. Doesn't touch the knowledge
    base or the LLM - use it purely to end a multi-step research sequence
    cleanly and leave an optional closing note."""
    return {"message": message}


def build_mcp_http_app():
    """Build the ASGI app used to mount this server into the main FastAPI
    app (see app/main.py) at API_PREFIX + "/mcp" - e.g. /api/v1/mcp.

    stateless_http + json_response: every call is a plain HTTP
    request/response with no session affinity or SSE stream to keep
    alive, so any HTTP-capable MCP client (not just ones that speak SSE)
    can reach it, and it works the same as the rest of this API behind a
    normal load balancer.
    """
    return mcp.streamable_http_app(
        streamable_http_path="/",
        stateless_http=True,
        json_response=True,
    )


if __name__ == "__main__":
    # Local stdio transport - see the module docstring above.
    mcp.run()

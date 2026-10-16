import json
import os
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from app.core.rate_limit import check_chat_rate_limit
from app.models.schemas import AgentRunRequest
from app.services.agent_engine import AgentEngine
from app.services.agent_memory import _MEMORY_PATH

agent_router = APIRouter()


@agent_router.post("/run")
async def agent_run(request: AgentRunRequest, req: Request):
    """Stream a ReAct agent run as Server-Sent Events."""
    check_chat_rate_limit(req)

    engine = AgentEngine()

    async def event_generator() -> AsyncGenerator[str, None]:
        async for event in engine.run_stream(request):
            yield f"data: {json.dumps(jsonable_encoder(event))}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@agent_router.get("/memory")
async def list_agent_memory(page: int = 1, page_size: int = 10) -> dict:
    """Paginated view of the agent's stored (query, answer) memories
    (agent_memory.jsonl), newest first. Excludes each record's embedding
    vector - it's never useful in a UI and can be large."""
    page = max(1, page)
    page_size = max(1, min(page_size, 100))

    records: list[dict] = []
    if os.path.exists(_MEMORY_PATH):
        with open(_MEMORY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                records.append({
                    "id": record.get("id"),
                    "query": record.get("query"),
                    "answer": record.get("answer"),
                    "timestamp": record.get("timestamp"),
                })

    records.reverse()  # newest first
    total = len(records)
    start = (page - 1) * page_size
    page_items = records[start:start + page_size]

    return {
        "success": True,
        "data": {
            "items": page_items,
            "total": total,
            "page": page,
            "pageSize": page_size,
            "totalPages": max(1, (total + page_size - 1) // page_size),
        },
    }

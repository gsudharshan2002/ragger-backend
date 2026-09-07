import json
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from app.core.rate_limit import check_chat_rate_limit
from app.models.schemas import AgentRunRequest
from app.services.agent_engine import AgentEngine

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

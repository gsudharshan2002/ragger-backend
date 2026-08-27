import json
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag_engine import execute_rag

router = APIRouter()


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """Stream RAG pipeline events as Server-Sent Events."""

    async def event_generator() -> AsyncGenerator[str, None]:
        async for event in execute_rag(
            request.query,
            request.strategy,
            None,
            request.knowledge_base_id,
        ):
            if event.get("type") == "llm.token" and "content" in event:
                payload = {
                    "type": "llm.token",
                    "content": event["content"],
                }
            else:
                payload = event

            yield f"data: {json.dumps(jsonable_encoder(payload))}\n\n"

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


@router.post("/send")
async def chat_send(request: ChatRequest):
    """Non-streaming chat."""

    trace = None
    answer = ""
    sources = []

    async for event in execute_rag(
        request.query,
        request.strategy,
        None,
        request.knowledge_base_id,
        start_time=datetime.now(timezone.utc),
    ):
        if event.get("type") == "llm.token" and "content" in event:
            answer += event["content"]

        elif (
            event.get("stage") == "trace"
            and event.get("event") == "trace.completed"
        ):
            trace = event["data"]
            answer = trace.get("llm", {}).get("answer", answer)
            sources = trace.get("sources", [])

        elif event.get("type") == "error":
            return {
                "success": False,
                "error": event.get("error", "Unknown error"),
            }

    return ChatResponse(
        answer=answer,
        trace=trace,
        sources=sources,
    )
import json
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from app.core.rate_limit import check_chat_rate_limit
from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag_engine import execute_rag

router = APIRouter()


@router.post("/stream")
async def chat_stream(request: ChatRequest, req: Request):
    """Stream RAG pipeline events as Server-Sent Events."""
    check_chat_rate_limit(req)

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
async def chat_send(request: ChatRequest, req: Request):
    """Non-streaming chat."""
    check_chat_rate_limit(req)

    trace = None
    answer = ""
    sources = []

    async for event in execute_rag(
        request.query,
        request.strategy,
        None,
        request.knowledge_base_id,
    ):
        if event.get("type") == "llm.token" and "content" in event:
            answer += event["content"]

        elif (
            event.get("stage") == "trace"
            and (
                event.get("type") == "trace.completed"
                or event.get("event") == "trace.completed"
            )
        ):
            trace = event["data"]
            answer = trace.get("llm", {}).get("answer", answer)
            sources = trace.get("sources", [])

        elif event.get("type") == "trace.failed" or event.get("event") == "trace.failed":
            return {
                "success": False,
                "error": event.get("data", {}).get("error", "RAG pipeline failed"),
            }

        elif event.get("type") == "error":
            return {
                "success": False,
                "error": event.get("error", "Unknown error"),
            }

    if trace is None:
        return {
            "success": False,
            "error": "RAG pipeline failed before producing a trace",
        }

    return ChatResponse(
        answer=answer,
        trace=trace,
        sources=sources,
    )
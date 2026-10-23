"""Week 10: Multi-Agent Squad API endpoint.

Additive endpoint: POST /api/v1/squad/run

Runs the orchestrator-plus-two-specialists squad for a developer-documentation
question. Preserves the existing single-agent /api/v1/agent/run endpoint
unchanged.
"""

import json
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from app.core.logging_config import get_logger
from app.core.rate_limit import check_chat_rate_limit
from app.models.schemas import AgentRunRequest
from app.services.squad.orchestrator import SquadOrchestrator
from app.services.squad.telemetry import TelemetryCollector

logger = get_logger(__name__)
router = APIRouter()


@router.post("/run")
async def squad_run(request: AgentRunRequest, req: Request):
    """Stream a multi-agent squad run as Server-Sent Events."""
    check_chat_rate_limit(req)

    telemetry = TelemetryCollector(arm="squad", case_id="unknown")
    orchestrator = SquadOrchestrator(
        max_steps=request.max_steps,
        temperature=request.temperature,
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        yield f"data: {json.dumps(jsonable_encoder({'type': 'squad.progress', 'data': {'status': 'running'}}))}\n\n"

        try:
            result = await orchestrator.run(request.query, "squad-case", telemetry)
            event = {"type": "squad.completed", "data": result}
        except Exception as e:
            logger.error(f"Squad run failed: {e}")
            event = {"type": "squad.error", "data": {"error": f"{type(e).__name__}: {e}"}}

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

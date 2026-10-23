from fastapi import APIRouter
from datetime import datetime

from app.core.capabilities import detect_capabilities
from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        timestamp=datetime.utcnow(),
    )


@router.get("/health/capabilities")
async def health_capabilities():
    """Additive: which retrieval features work here (vector search, reranker)."""
    return {"success": True, "data": detect_capabilities()}

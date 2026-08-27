from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Optional

from app.services.storage import get_settings, update_settings, get_all_documents, get_all_chunks

router = APIRouter()


class SettingsUpdate(BaseModel):
    llmProvider: Optional[str] = None
    groqModel: Optional[str] = None
    groqApiKey: Optional[str] = None
    geminiApiKey: Optional[str] = None
    embeddingProvider: Optional[str] = None
    embeddingModel: Optional[str] = None
    embeddingApiKey: Optional[str] = None
    chunkSize: Optional[int] = None
    chunkOverlap: Optional[int] = None
    defaultTopK: Optional[int] = None
    defaultStrategy: Optional[str] = None
    systemPrompt: Optional[str] = None
    rerankerModel: Optional[str] = None
    mmrLambda: Optional[float] = None


@router.get("/config")
async def get_rag_config() -> dict:
    settings = await get_settings()
    documents = await get_all_documents()
    chunks = await get_all_chunks()

    embedding_config_status = (
        settings.get("embeddingProvider") not in (None, "none")
        and bool(settings.get("embeddingModel"))
        and bool(settings.get("embeddingApiKey"))
    )
    if not embedding_config_status:
        embedding_config_status = settings.get("embeddingProvider") not in (None, "none")

    return {
        "success": True,
        "data": {
            "settings": settings,
            "embeddingConfigStatus": "configured" if embedding_config_status else "not_configured",
            "documentCount": len(documents),
            "totalChunks": len(chunks),
        },
    }


@router.put("/config")
async def update_rag_config(payload: SettingsUpdate) -> dict:
    updates = {k: v for k, v in payload.model_dump(by_alias=True).items() if v is not None}
    updated = await update_settings(updates)
    return {"success": True, "data": updated}

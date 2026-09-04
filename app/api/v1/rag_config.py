from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Optional

from app.services.storage import (
    get_settings,
    update_settings,
    get_all_documents,
    get_all_chunks,
    update_chunk_embeddings,
)

router = APIRouter()


class SettingsUpdate(BaseModel):
    llmProvider: Optional[str] = None
    groqModel: Optional[str] = None
    geminiModel: Optional[str] = None
    openrouterModel: Optional[str] = None
    groqApiKey: Optional[str] = None
    geminiApiKey: Optional[str] = None
    openrouterApiKey: Optional[str] = None
    embeddingProvider: Optional[str] = None
    embeddingModel: Optional[str] = None
    cohereEmbedModel: Optional[str] = None
    vectorSimilarity: Optional[str] = None
    embeddingApiKey: Optional[str] = None
    chunkSize: Optional[int] = None
    chunkOverlap: Optional[int] = None
    defaultTopK: Optional[int] = None
    defaultStrategy: Optional[str] = None
    systemPrompt: Optional[str] = None
    rerankerProvider: Optional[str] = None
    rerankerModel: Optional[str] = None
    cohereRerankModel: Optional[str] = None
    mmrLambda: Optional[float] = None
    costPerToken: Optional[float] = None


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


@router.post("/reindex-embeddings")
async def reindex_embeddings() -> dict:
    """Re-embed every existing chunk's stored text under whichever embedding
    provider is currently active. Documents and chunk text are untouched -
    only the vectors are recomputed, so switching provider never requires
    the user to re-upload anything."""
    from app.services.embeddings import get_embeddings_for_texts

    settings = await get_settings()
    provider = settings.get("embeddingProvider") or "local"

    chunks = await get_all_chunks()
    if not chunks:
        return {"success": True, "data": {"reindexed": 0, "provider": provider}}

    texts = [c.content for c in chunks]
    embeddings = await get_embeddings_for_texts(texts, input_type="search_document")
    if not embeddings or len(embeddings) != len(chunks):
        return {
            "success": False,
            "error": f"Failed to generate embeddings for the '{provider}' provider.",
        }

    embeddings_by_id = {chunk.id: emb for chunk, emb in zip(chunks, embeddings)}
    reindexed = await update_chunk_embeddings(embeddings_by_id)

    return {"success": True, "data": {"reindexed": reindexed, "provider": provider}}

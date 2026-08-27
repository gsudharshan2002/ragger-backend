from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging_config import get_logger
from app.services.storage import (
    get_all_knowledge_bases,
    get_knowledge_base,
    add_knowledge_base,
    update_knowledge_base,
    delete_knowledge_base,
    get_all_documents,
    get_all_chunks,
)

logger = get_logger(__name__)
router = APIRouter()


class KnowledgeBaseCreate(BaseModel):
    name: str
    description: str = ""
    tags: list[str] = []


class KnowledgeBaseUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[list[str]] = None


def _kb_response(kb) -> dict:
    created = kb.created_at.isoformat() if hasattr(kb.created_at, "isoformat") else str(kb.created_at)
    return {
        "id": kb.id,
        "name": kb.name,
        "description": kb.description,
        "tags": kb.tags,
        "createdAt": created,
        "updatedAt": created,
        "documentCount": kb.document_count,
        "chunkCount": kb.chunk_count,
        "settings": {
            "defaultChunkSize": settings.CHUNK_SIZE,
            "defaultChunkOverlap": settings.CHUNK_OVERLAP,
            "embeddingProvider": settings.EMBEDDING_PROVIDER,
            "embeddingModel": settings.EMBEDDING_MODEL,
        },
    }


@router.get("")
async def list_knowledge_bases() -> dict:
    kbs = await get_all_knowledge_bases()
    return {"success": True, "data": [_kb_response(kb) for kb in kbs]}


@router.get("/{kb_id}")
async def get_kb(kb_id: str) -> dict:
    kb = await get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    documents = [d for d in await get_all_documents() if d.knowledge_base_id == kb_id]
    document_ids = {d.id for d in documents}
    chunks = [c for c in await get_all_chunks() if c.document_id in document_ids]
    response = _kb_response(kb)
    response["stats"] = {
        "documentCount": len(documents),
        "readyDocuments": len(documents),
        "processingDocuments": 0,
        "failedDocuments": 0,
        "chunkCount": len(chunks),
        "totalTokens": sum(c.token_count for c in chunks),
        "indexedChunks": sum(1 for c in chunks if c.embedding),
    }
    return {"success": True, "data": response}


@router.post("", status_code=201)
async def create_knowledge_base(payload: KnowledgeBaseCreate) -> dict:
    from app.models.schemas import KnowledgeBase
    kb = KnowledgeBase(id=str(uuid4()), name=payload.name, description=payload.description, tags=payload.tags)
    kb = await add_knowledge_base(kb)
    return {"success": True, "data": _kb_response(kb)}


@router.put("/{kb_id}")
async def update_kb(kb_id: str, payload: KnowledgeBaseUpdate) -> dict:
    kb = await update_knowledge_base(kb_id, payload.model_dump(exclude_none=True))
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return {"success": True, "data": _kb_response(kb)}


@router.delete("/{kb_id}")
async def delete_kb(kb_id: str) -> dict:
    success = await delete_knowledge_base(kb_id)
    if not success:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return {"success": True}

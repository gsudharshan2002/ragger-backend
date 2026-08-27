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
    delete_knowledge_base,
)

logger = get_logger(__name__)
router = APIRouter()


class KnowledgeBaseCreate(BaseModel):
    name: str
    description: str = ""


def _kb_response(kb) -> dict:
    created = kb.created_at.isoformat() if hasattr(kb.created_at, "isoformat") else str(kb.created_at)
    return {
        "id": kb.id,
        "name": kb.name,
        "description": kb.description,
        "tags": [],
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
    return {"success": True, "data": _kb_response(kb)}


@router.post("", status_code=201)
async def create_knowledge_base(payload: KnowledgeBaseCreate) -> dict:
    from app.models.schemas import KnowledgeBase
    kb = KnowledgeBase(id=str(uuid4()), name=payload.name, description=payload.description)
    kb = await add_knowledge_base(kb)
    return {"success": True, "data": _kb_response(kb)}


@router.delete("/{kb_id}")
async def delete_kb(kb_id: str) -> dict:
    success = await delete_knowledge_base(kb_id)
    if not success:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return {"success": True}

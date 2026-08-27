import os
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from pydantic import BaseModel
from uuid import uuid4

from app.services.storage import (
    get_all_documents,
    get_knowledge_base,
    add_document_version,
    add_processing_history,
)
from app.api.v1.documents import _document_response

router = APIRouter()


@router.get("/{kb_id}/documents")
async def list_kb_documents(kb_id: str, folder_id: str = Query(None)) -> dict:
    docs = await get_all_documents()
    kb_docs = [d for d in docs if getattr(d, "knowledge_base_id", None) == kb_id]
    return {"success": True, "data": [d.model_dump(by_alias=True) for d in kb_docs]}


@router.post("/{kb_id}/documents", status_code=201)
async def upload_kb_document(kb_id: str, file: UploadFile = File(...), folder_id: str = Query(None)) -> dict:
    from app.core.config import settings
    from app.services.document_processor import process_pdf, process_text_file
    from app.services.storage import add_document

    kb = await get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    upload_dir = settings.UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)
    unique_filename = f"{uuid4().hex}_{file.filename}"
    file_path = os.path.join(upload_dir, unique_filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    doc = None
    try:
        if file.filename and file.filename.lower().endswith(".pdf"):
            doc = await process_pdf(file_path, file.filename, kb_id)
        else:
            doc = await process_text_file(file_path, file.filename or "document.txt", kb_id)
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process: {e}")

    # Keep original file for preview
    from app.services.storage import (
        _documents_cache,
        _save_documents,
        add_document_version,
        make_document_version,
        add_processing_history,
    )
    if doc and doc.id in _documents_cache:
        _documents_cache[doc.id].path = file_path
        await _save_documents()

    await add_document_version(doc.id, make_document_version(doc.id, doc, 1))
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    await add_processing_history(doc.id, {
        "id": uuid4().hex,
        "documentId": doc.id,
        "knowledgeBaseId": kb_id,
        "action": "upload",
        "status": "completed",
        "startedAt": now,
        "completedAt": now,
        "durationMs": 0,
        "resultSummary": f"Processed into {doc.chunks} chunks",
    })

    return {"success": True, "data": _document_response(doc)}


class FolderCreate(BaseModel):
    name: str
    parentId: str = None


@router.get("/{kb_id}/folders")
async def list_folders(kb_id: str) -> dict:
    from app.services.storage import get_kb_folders
    folders = await get_kb_folders(kb_id)
    return {"success": True, "data": folders}


@router.post("/{kb_id}/folders", status_code=201)
async def create_folder(kb_id: str, payload: FolderCreate) -> dict:
    from app.services.storage import create_folder
    folder = await create_folder(kb_id, payload.name, payload.parentId)
    return {"success": True, "data": folder}

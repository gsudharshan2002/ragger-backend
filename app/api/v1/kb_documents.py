import os
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from pydantic import BaseModel
from uuid import uuid4

from app.services.storage import (
    get_all_documents,
    get_knowledge_base,
    get_document,
    update_document,
    add_document_version,
    add_processing_history,
)
from app.api.v1.documents import _document_response
from app.services.embeddings import generate_query_embedding
from app.services.search import vector_search

router = APIRouter()


class KnowledgeBaseSearch(BaseModel):
    query: str
    top_k: int = 20


@router.get("/{kb_id}/documents")
async def list_kb_documents(kb_id: str, folder_id: str = Query(None)) -> dict:
    docs = await get_all_documents()
    kb_docs = [d for d in docs if getattr(d, "knowledge_base_id", None) == kb_id]
    if folder_id:
        kb_docs = [d for d in kb_docs if getattr(d, "folder_id", None) == folder_id]
    return {"success": True, "data": [_document_response(d) for d in kb_docs]}


@router.post("/{kb_id}/documents/{doc_id}/attach")
async def attach_existing_document(
    kb_id: str,
    doc_id: str,
    folder_id: str = Query(None),
) -> dict:
    if not await get_knowledge_base(kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    document = await get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if document.knowledge_base_id and document.knowledge_base_id != kb_id:
        raise HTTPException(status_code=409, detail="Document already belongs to another knowledge base")

    document.knowledge_base_id = kb_id
    document.folder_id = folder_id
    await update_document(document)
    return {"success": True, "data": _document_response(document)}


@router.post("/{kb_id}/search")
async def search_kb(kb_id: str, payload: KnowledgeBaseSearch) -> dict:
    if not await get_knowledge_base(kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    if not payload.query.strip():
        raise HTTPException(status_code=422, detail="Search query cannot be empty")

    documents = [d for d in await get_all_documents() if d.knowledge_base_id == kb_id]
    document_ids = {d.id for d in documents}
    from app.services.storage import get_all_chunks
    chunks = [c for c in await get_all_chunks() if c.document_id in document_ids]
    embedding = await generate_query_embedding(payload.query)
    results = vector_search(embedding or [], chunks, max(1, min(payload.top_k, 100)))
    return {
        "success": True,
        "data": [
            {
                "chunkId": result.chunk_id,
                "content": result.chunk.content,
                "documentName": result.chunk.document_name,
                "documentId": result.chunk.document_id,
                "page": result.chunk.page,
                "score": result.score,
            }
            for result in results
        ],
    }


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
            doc = await process_pdf(file_path, file.filename, kb_id, folder_id)
        else:
            doc = await process_text_file(file_path, file.filename or "document.txt", kb_id, folder_id)
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
    from app.services.storage import create_folder, get_kb_folders
    if not await get_knowledge_base(kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    if payload.parentId:
        folders = await get_kb_folders(kb_id)
        if not any(folder["id"] == payload.parentId for folder in folders):
            raise HTTPException(status_code=400, detail="Parent folder not found in knowledge base")
    folder = await create_folder(kb_id, payload.name, payload.parentId)
    return {"success": True, "data": folder}

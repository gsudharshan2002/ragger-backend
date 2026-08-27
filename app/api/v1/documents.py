import os
import shutil
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Query

from app.core.config import settings
from app.core.logging_config import get_logger
from app.services.document_processor import process_pdf, process_text_file
from app.services.storage import (
    get_all_documents,
    get_document,
    delete_document,
    get_all_chunks,
)

logger = get_logger(__name__)
router = APIRouter()


def _document_response(doc: "object") -> dict:
    """Map an UploadedDocument to the frontend's expected DocumentMetadata shape."""
    uploaded = doc.uploaded_at.isoformat() if hasattr(doc.uploaded_at, "isoformat") else str(doc.uploaded_at)
    return {
        "id": doc.id,
        "name": doc.name,
        "size": doc.size,
        "type": doc.type,
        "mimeType": doc.type,
        "pages": doc.pages,
        "pageCount": doc.pages,
        "chunks": doc.chunks,
        "chunkCount": doc.chunks,
        "tokenCount": doc.token_count,
        "path": doc.path,
        "status": "ready",
        "createdAt": uploaded,
        "updatedAt": uploaded,
        "uploadedAt": uploaded,
        "knowledgeBaseId": doc.knowledge_base_id,
        "folderId": doc.folder_id,
    }


@router.get("")
async def list_documents() -> dict:
    docs = await get_all_documents()
    return {"success": True, "data": [_document_response(d) for d in docs]}


@router.get("/{doc_id}")
async def get_document_by_id(doc_id: str) -> dict:
    doc = await get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"success": True, "data": _document_response(doc)}


@router.delete("/{doc_id}")
async def delete_document_by_id(doc_id: str) -> dict:
    success = await delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"success": True}


@router.get("/{doc_id}/chunks")
async def get_document_chunks(doc_id: str) -> dict:
    chunks = await get_all_chunks()
    doc_chunks = [c for c in chunks if c.document_id == doc_id]
    return {
        "success": True,
        "data": [
            {
                "id": c.id,
                "document_id": c.document_id,
                "document_name": c.document_name,
                "content": c.content,
                "page": c.page,
                "section": c.section,
                "token_count": c.token_count,
            }
            for c in doc_chunks
        ],
    }


@router.post("/upload", status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    knowledge_base_id: Optional[str] = Query(None),
) -> dict:
    upload_dir = settings.UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, f"{os.urandom(8).hex()}_{file.filename}")

    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    doc = None
    try:
        content_type = file.content_type or ""
        if file.filename and file.filename.lower().endswith(".pdf"):
            doc = await process_pdf(file_path, file.filename, knowledge_base_id)
        elif "text" in content_type or (file.filename and file.filename.lower().endswith((".txt", ".md"))):
            doc = await process_text_file(file_path, file.filename, knowledge_base_id)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}")
    except HTTPException:
        os.remove(file_path)
        raise
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")
    finally:
        # Keep the original file on disk for preview
        pass

    # Record initial version
    from app.services.storage import add_document_version, make_document_version
    await add_document_version(doc.id, make_document_version(doc.id, doc, 1))

    return {"success": True, "data": _document_response(doc)}

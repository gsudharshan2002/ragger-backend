import os
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, FileResponse

from app.services.storage import (
    get_document,
    get_document_versions,
    add_document_version,
    get_next_version_number,
    make_document_version,
    get_document_history,
    add_processing_history,
)
from app.api.v1.documents import _document_response

router = APIRouter()


@router.get("/{doc_id}/versions")
async def get_versions(doc_id: str) -> dict:
    versions = await get_document_versions(doc_id)
    # Mark the latest version (highest versionNumber)
    if versions:
        sorted_versions = sorted(versions, key=lambda v: v.get("versionNumber", 0), reverse=True)
        for i, v in enumerate(sorted_versions):
            v = dict(v)
            v["isLatest"] = i == 0
            sorted_versions[i] = v
        return {"success": True, "data": sorted_versions}
    return {"success": True, "data": []}


@router.get("/{doc_id}/history")
async def get_history(doc_id: str) -> dict:
    history = await get_document_history(doc_id)
    return {"success": True, "data": history}


def _make_history_event(doc_id: str, action: str, kb_id: str | None, summary: str) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": __import__("uuid").uuid4().hex,
        "documentId": doc_id,
        "knowledgeBaseId": kb_id,
        "action": action,
        "status": "completed",
        "startedAt": now,
        "completedAt": now,
        "durationMs": 0,
        "resultSummary": summary,
    }


async def _rerun(doc_id: str, action: str) -> dict:
    """Re-extract and re-chunk an existing document, then record a new version."""
    doc = await get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if not doc.path or not os.path.exists(doc.path):
        raise HTTPException(status_code=400, detail="Source file is not available for reprocessing")

    from app.services.document_processor import reprocess_document

    await add_processing_history(doc_id, _make_history_event(doc_id, action, doc.knowledge_base_id, "Reprocessing started"))

    updated = await reprocess_document(doc_id, doc.path, doc.name, doc.knowledge_base_id)

    version_number = await get_next_version_number(doc_id)
    version = make_document_version(doc_id, updated, version_number)
    await add_document_version(doc_id, version)

    await add_processing_history(
        doc_id,
        _make_history_event(doc_id, action, doc.knowledge_base_id, f"Reprocessed into {updated.chunks} chunks"),
    )

    return {"success": True, "data": _document_response(updated)}


@router.post("/{doc_id}/reprocess")
async def reprocess_document_endpoint(doc_id: str) -> dict:
    return await _rerun(doc_id, "parse")


@router.post("/{doc_id}/rechunk")
async def rechunk_document_endpoint(doc_id: str) -> dict:
    return await _rerun(doc_id, "chunk")


@router.get("/{doc_id}/preview")
async def preview_document(doc_id: str, range: str = Query(None)) -> Response:
    doc = await get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if not doc.path or not os.path.exists(doc.path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    if range:
        # Serve range manually
        import mimetypes
        file_size = os.path.getsize(doc.path)
        mime_type = mimetypes.guess_type(doc.path)[0] or "application/octet-stream"
        parts = range.replace("bytes=", "").split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else min(start + 1024 * 1024 - 1, file_size - 1)
        if start >= file_size or end >= file_size or start > end:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})
        with open(doc.path, "rb") as f:
            f.seek(start)
            data = f.read(end - start + 1)
        return Response(
            content=data,
            status_code=206,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(end - start + 1),
                "Content-Type": mime_type,
            },
        )

    return FileResponse(doc.path)

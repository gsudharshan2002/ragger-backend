from fastapi import APIRouter, HTTPException

from app.services.storage import list_traces, get_trace, delete_trace

router = APIRouter()


@router.get("")
async def get_traces(limit: int = 50) -> dict:
    traces = await list_traces(limit)
    return {"success": True, "data": traces}


@router.get("/{trace_id}")
async def get_trace_by_id(trace_id: str) -> dict:
    trace = await get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {"success": True, "data": trace}


@router.delete("/{trace_id}")
async def delete_trace_by_id(trace_id: str) -> dict:
    deleted = await delete_trace(trace_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {"success": True, "data": {"id": trace_id}}

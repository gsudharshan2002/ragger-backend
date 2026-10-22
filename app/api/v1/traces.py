from fastapi import APIRouter, HTTPException, Query

from app.services.storage import list_traces, get_trace, delete_trace, search_traces, export_trace

router = APIRouter()


@router.get("")
async def get_traces(limit: int = 50) -> dict:
    traces = await list_traces(limit)
    return {"success": True, "data": traces}


@router.get("/search")
async def search_traces_endpoint(
    trace_id: str | None = Query(default=None),
    request_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    prompt_version: str | None = Query(default=None),
    model: str | None = Query(default=None),
    mode: str | None = Query(default=None),
    chunk_id: str | None = Query(default=None),
    error: bool | None = Query(default=None),
    min_latency_ms: int | None = Query(default=None),
    min_cost_usd: float | None = Query(default=None),
    started_after: str | None = Query(default=None),
    started_before: str | None = Query(default=None),
    limit: int = Query(default=100),
) -> dict:
    filters = {
        k: v
        for k, v in {
            "trace_id": trace_id,
            "request_id": request_id,
            "status": status,
            "prompt_version": prompt_version,
            "model": model,
            "mode": mode,
            "chunk_id": chunk_id,
            "error": error,
            "min_latency_ms": min_latency_ms,
            "min_cost_usd": min_cost_usd,
            "started_after": started_after,
            "started_before": started_before,
        }.items()
        if v is not None
    }
    results = await search_traces(filters, limit=limit)
    return {"success": True, "data": results, "count": len(results)}


@router.get("/{trace_id}")
async def get_trace_by_id(trace_id: str) -> dict:
    trace = await get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {"success": True, "data": trace}


@router.get("/{trace_id}/export")
async def export_trace_endpoint(trace_id: str) -> dict:
    trace = await export_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {"success": True, "data": trace}


@router.delete("/{trace_id}")
async def delete_trace_by_id(trace_id: str) -> dict:
    deleted = await delete_trace(trace_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {"success": True, "data": {"id": trace_id}}

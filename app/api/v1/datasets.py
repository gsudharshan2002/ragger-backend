from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.storage import (
    get_all_datasets,
    get_dataset,
    add_dataset,
    delete_dataset,
)

router = APIRouter()


class DatasetCreate(BaseModel):
    name: str
    description: str = ""


def _dataset_response(ds) -> dict:
    data = ds.model_dump(by_alias=True)
    data.setdefault("tags", [])
    return data


@router.get("")
async def list_datasets() -> dict:
    datasets = await get_all_datasets()
    return {"success": True, "data": [_dataset_response(d) for d in datasets]}


@router.get("/{dataset_id}")
async def get_dataset_by_id(dataset_id: str) -> dict:
    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"success": True, "data": _dataset_response(dataset)}


@router.post("", status_code=201)
async def create_dataset(payload: DatasetCreate) -> dict:
    from app.models.schemas import Dataset, DocumentVersion
    from datetime import datetime

    dataset = Dataset(
        id=str(uuid4()),
        name=payload.name,
        description=payload.description,
        current_version="v1",
        versions=[DocumentVersion(version="v1", cases_count=0)],
    )
    dataset = await add_dataset(dataset)
    return {"success": True, "data": _dataset_response(dataset)}


@router.delete("/{dataset_id}")
async def delete_dataset_by_id(dataset_id: str) -> dict:
    success = await delete_dataset(dataset_id)
    if not success:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"success": True}


@router.put("/{dataset_id}")
async def update_dataset(dataset_id: str, payload: dict) -> dict:
    from app.services.storage import get_dataset, add_dataset
    from datetime import datetime
    from app.models.schemas import Dataset

    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    updated = Dataset(**{
        **dataset.model_dump(by_alias=True),
        **payload,
        "id": dataset_id,
        "updated_at": datetime.utcnow(),
    })
    await add_dataset(updated)
    return {"success": True, "data": _dataset_response(updated)}


@router.post("/{dataset_id}")
async def create_dataset_version_endpoint(dataset_id: str, payload: dict) -> dict:
    from app.services.storage import get_dataset, add_dataset
    from app.models.schemas import DocumentVersion
    from datetime import datetime

    dataset = await get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    version = DocumentVersion(
        version=payload.get("version", f"v{len(dataset.versions) + 1}"),
        cases_count=payload.get("casesCount", 0),
        change_note=payload.get("changeNote"),
    )
    dataset.versions.append(version)
    dataset.current_version = version.version
    await add_dataset(dataset)
    return {"success": True, "data": _dataset_response(dataset)}

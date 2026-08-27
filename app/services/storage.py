import asyncio
import json
import os
import shutil
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from app.core.config import settings
from app.core.logging_config import get_logger
from app.models.schemas import (
    StoredChunk,
    UploadedDocument,
    KnowledgeBase,
    Dataset,
    DocumentVersion,
)

logger = get_logger(__name__)

DATA_DIR = os.environ.get("RAGGER_DATA_DIR", "./app/data")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
KB_DIR = os.path.join(DATA_DIR, "knowledge_bases")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")

# In-memory cache
_chunks_cache: list[StoredChunk] = []
_documents_cache: dict[str, UploadedDocument] = {}
_kb_cache: dict[str, KnowledgeBase] = {}
_dataset_cache: dict[str, Dataset] = {}

_initialized = False


async def _ensure_dirs():
    for d in [DATA_DIR, DOCUMENTS_DIR, CHUNKS_DIR, KB_DIR, DATASETS_DIR]:
        os.makedirs(d, exist_ok=True)


async def init_storage():
    global _initialized, _chunks_cache, _documents_cache, _kb_cache, _dataset_cache
    if _initialized:
        return

    await _ensure_dirs()
    await _load_all()
    _initialized = True
    logger.info(f"Storage initialized: {len(_chunks_cache)} chunks, {len(_documents_cache)} documents")


async def _load_all():
    global _chunks_cache, _documents_cache, _kb_cache, _dataset_cache
    # Load chunks
    chunks_file = os.path.join(CHUNKS_DIR, "chunks.json")
    if os.path.exists(chunks_file):
        try:
            with open(chunks_file) as f:
                data = json.load(f)
                _chunks_cache = [StoredChunk(**c) for c in data]
        except Exception as e:
            logger.error(f"Failed to load chunks: {e}")

    # Load documents
    docs_file = os.path.join(DOCUMENTS_DIR, "documents.json")
    if os.path.exists(docs_file):
        try:
            with open(docs_file) as f:
                data = json.load(f)
                _documents_cache = {d["id"]: UploadedDocument(**d) for d in data}
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")

    # Load KBs
    kb_file = os.path.join(KB_DIR, "knowledge_bases.json")
    if os.path.exists(kb_file):
        try:
            with open(kb_file) as f:
                data = json.load(f)
                _kb_cache = {k["id"]: KnowledgeBase(**k) for k in data}
        except Exception as e:
            logger.error(f"Failed to load KBs: {e}")

    # Load datasets
    ds_file = os.path.join(DATASETS_DIR, "datasets.json")
    if os.path.exists(ds_file):
        try:
            with open(ds_file) as f:
                data = json.load(f)
                _dataset_cache = {d["id"]: Dataset(**d) for d in data}
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")


async def _save_chunks():
    chunks_file = os.path.join(CHUNKS_DIR, "chunks.json")
    with open(chunks_file, "w") as f:
        json.dump([c.model_dump() for c in _chunks_cache], f, default=str)


async def _save_documents():
    docs_file = os.path.join(DOCUMENTS_DIR, "documents.json")
    with open(docs_file, "w") as f:
        json.dump([d.model_dump() for d in _documents_cache.values()], f, default=str)


async def _save_kbs():
    kb_file = os.path.join(KB_DIR, "knowledge_bases.json")
    with open(kb_file, "w") as f:
        json.dump([k.model_dump() for k in _kb_cache.values()], f, default=str)


async def _save_datasets():
    ds_file = os.path.join(DATASETS_DIR, "datasets.json")
    with open(ds_file, "w") as f:
        json.dump([d.model_dump() for d in _dataset_cache.values()], f, default=str)


async def get_all_chunks() -> list[StoredChunk]:
    await init_storage()
    return list(_chunks_cache)


async def get_all_documents() -> list[UploadedDocument]:
    await init_storage()
    return list(_documents_cache.values())


async def get_document(doc_id: str) -> Optional[UploadedDocument]:
    await init_storage()
    return _documents_cache.get(doc_id)


async def add_document(doc: UploadedDocument) -> UploadedDocument:
    await init_storage()
    _documents_cache[doc.id] = doc
    await _save_documents()
    return doc


async def update_document(doc: UploadedDocument) -> UploadedDocument:
    await init_storage()
    _documents_cache[doc.id] = doc
    await _save_documents()
    return doc


async def delete_chunks_for_document(doc_id: str) -> int:
    await init_storage()
    global _chunks_cache
    before = len(_chunks_cache)
    _chunks_cache = [c for c in _chunks_cache if c.document_id != doc_id]
    removed = before - len(_chunks_cache)
    if removed:
        await _save_chunks()
    return removed


async def delete_document(doc_id: str) -> bool:
    await init_storage()
    if doc_id in _documents_cache:
        del _documents_cache[doc_id]
        await _save_documents()
        # Also remove chunks
        global _chunks_cache
        _chunks_cache = [c for c in _chunks_cache if c.document_id != doc_id]
        await _save_chunks()
        return True
    return False


async def add_chunks(chunks: list[StoredChunk]) -> list[StoredChunk]:
    await init_storage()
    global _chunks_cache
    _chunks_cache.extend(chunks)
    await _save_chunks()
    return chunks


async def get_all_knowledge_bases() -> list[KnowledgeBase]:
    await init_storage()
    return list(_kb_cache.values())


async def get_knowledge_base(kb_id: str) -> Optional[KnowledgeBase]:
    await init_storage()
    return _kb_cache.get(kb_id)


async def add_knowledge_base(kb: KnowledgeBase) -> KnowledgeBase:
    await init_storage()
    _kb_cache[kb.id] = kb
    await _save_kbs()
    return kb


async def delete_knowledge_base(kb_id: str) -> bool:
    await init_storage()
    if kb_id in _kb_cache:
        del _kb_cache[kb_id]
        await _save_kbs()
        return True
    return False


async def get_all_datasets() -> list[Dataset]:
    await init_storage()
    return list(_dataset_cache.values())


async def get_dataset(dataset_id: str) -> Optional[Dataset]:
    await init_storage()
    return _dataset_cache.get(dataset_id)


async def add_dataset(dataset: Dataset) -> Dataset:
    await init_storage()
    _dataset_cache[dataset.id] = dataset
    await _save_datasets()
    return dataset


async def delete_dataset(dataset_id: str) -> bool:
    await init_storage()
    if dataset_id in _dataset_cache:
        del _dataset_cache[dataset_id]
        await _save_datasets()
        return True
    return False


def get_data_dir() -> str:
    return DATA_DIR


def get_upload_dir() -> str:
    upload_dir = settings.UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir


# ===================== Settings =====================
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
_settings_cache: Optional[dict] = None


async def get_settings() -> dict:
    global _settings_cache
    await _ensure_dirs()
    if _settings_cache is None:
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE) as f:
                    _settings_cache = json.load(f)
            except Exception:
                _settings_cache = {}
        else:
            _settings_cache = {}
    # Merge with defaults from config
    merged = {
        "llmProvider": settings.LLM_PROVIDER,
        "groqModel": settings.GROQ_MODEL,
        "groqApiKey": settings.GROQ_API_KEY,
        "geminiApiKey": settings.GEMINI_API_KEY,
        "embeddingProvider": settings.EMBEDDING_PROVIDER,
        "embeddingModel": settings.EMBEDDING_MODEL,
        "embeddingApiKey": settings.EMBEDDING_API_KEY,
        "chunkSize": settings.CHUNK_SIZE,
        "chunkOverlap": settings.CHUNK_OVERLAP,
        "defaultTopK": settings.DEFAULT_TOP_K,
        "defaultStrategy": settings.DEFAULT_STRATEGY,
        "systemPrompt": settings.SYSTEM_PROMPT,
        "rerankerModel": settings.RERANKER_MODEL,
        "mmrLambda": settings.MMR_LAMBDA,
    }
    merged.update(_settings_cache)
    return merged


async def update_settings(updates: dict) -> dict:
    global _settings_cache
    await _ensure_dirs()
    current = await get_settings()
    current.update(updates)
    _settings_cache = current
    with open(SETTINGS_FILE, "w") as f:
        json.dump(current, f, default=str)
    return current


# ===================== Traces =====================
TRACES_FILE = os.path.join(DATA_DIR, "traces.json")
_traces_cache: list[dict] = []
_traces_loaded = False


async def _load_traces():
    global _traces_cache, _traces_loaded
    if _traces_loaded:
        return
    if os.path.exists(TRACES_FILE):
        try:
            with open(TRACES_FILE) as f:
                _traces_cache = json.load(f)
        except Exception:
            _traces_cache = []
    _traces_loaded = True


async def _save_traces():
    with open(TRACES_FILE, "w") as f:
        json.dump(_traces_cache, f, default=str)


async def list_traces(limit: int = 50) -> list[dict]:
    await _load_traces()
    return _traces_cache[:limit]


async def get_trace(trace_id: str) -> Optional[dict]:
    await _load_traces()
    return next((t for t in _traces_cache if t.get("id") == trace_id), None)


async def add_trace(trace: dict) -> dict:
    await _load_traces()
    _traces_cache.insert(0, trace)
    await _save_traces()
    return trace


async def delete_trace(trace_id: str) -> bool:
    global _traces_cache
    await _load_traces()
    before = len(_traces_cache)
    _traces_cache = [t for t in _traces_cache if t.get("id") != trace_id]
    if len(_traces_cache) != before:
        await _save_traces()
        return True
    return False


# ===================== Document Versions =====================
async def get_document_versions(doc_id: str) -> list[dict]:
    await _ensure_dirs()
    versions_file = os.path.join(DOCUMENTS_DIR, f"{doc_id}_versions.json")
    if os.path.exists(versions_file):
        try:
            with open(versions_file) as f:
                return json.load(f)
        except Exception:
            return []
    return []


async def add_document_version(doc_id: str, version: dict) -> dict:
    await _ensure_dirs()
    versions = await get_document_versions(doc_id)
    versions.append(version)
    versions_file = os.path.join(DOCUMENTS_DIR, f"{doc_id}_versions.json")
    with open(versions_file, "w") as f:
        json.dump(versions, f, default=str)
    return version


async def get_next_version_number(doc_id: str) -> int:
    versions = await get_document_versions(doc_id)
    return max([v.get("versionNumber", 0) for v in versions], default=0) + 1


def make_document_version(doc_id: str, doc: UploadedDocument, version_number: int, status: str = "ready") -> dict:
    return {
        "id": str(uuid4()),
        "documentId": doc_id,
        "versionNumber": version_number,
        "filePath": doc.path or "",
        "fileSize": doc.size,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "pageCount": doc.pages,
        "chunkCount": doc.chunks,
        "tokenCount": doc.token_count,
        "error": None,
        "isLatest": True,
    }


# ===================== Processing History =====================
async def get_document_history(doc_id: str) -> list[dict]:
    await _ensure_dirs()
    history_file = os.path.join(DOCUMENTS_DIR, f"{doc_id}_history.json")
    if os.path.exists(history_file):
        try:
            with open(history_file) as f:
                return json.load(f)
        except Exception:
            return []
    return []


async def add_processing_history(doc_id: str, event: dict) -> dict:
    await _ensure_dirs()
    history = await get_document_history(doc_id)
    history.append(event)
    history_file = os.path.join(DOCUMENTS_DIR, f"{doc_id}_history.json")
    with open(history_file, "w") as f:
        json.dump(history, f, default=str)
    return event


# ===================== Folders =====================
FOLDERS_FILE = os.path.join(KB_DIR, "folders.json")
_folders_cache: list[dict] = []
_folders_loaded = False


async def _load_folders():
    global _folders_cache, _folders_loaded
    if _folders_loaded:
        return
    if os.path.exists(FOLDERS_FILE):
        try:
            with open(FOLDERS_FILE) as f:
                _folders_cache = json.load(f)
        except Exception:
            _folders_cache = []
    _folders_loaded = True


async def _save_folders():
    with open(FOLDERS_FILE, "w") as f:
        json.dump(_folders_cache, f, default=str)


async def get_kb_folders(kb_id: str) -> list[dict]:
    await _load_folders()
    return [f for f in _folders_cache if f.get("knowledgeBaseId") == kb_id]


async def create_folder(kb_id: str, name: str, parent_id: Optional[str] = None) -> dict:
    await _load_folders()
    folder = {
        "id": str(uuid4()),
        "knowledgeBaseId": kb_id,
        "name": name,
        "parentId": parent_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }
    _folders_cache.append(folder)
    await _save_folders()
    return folder


# ===================== Benchmark Results =====================
RESULTS_FILE = os.path.join(DATASETS_DIR, "benchmark-results.json")
_results_cache: list[dict] = []
_results_loaded = False


async def _load_results():
    global _results_cache, _results_loaded
    if _results_loaded:
        return
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE) as f:
                _results_cache = json.load(f)
        except Exception:
            _results_cache = []
    _results_loaded = True


async def _save_results():
    with open(RESULTS_FILE, "w") as f:
        json.dump(_results_cache, f, default=str)


async def list_benchmark_results() -> list[dict]:
    await _load_results()
    return _results_cache


async def add_benchmark_result(result: dict) -> dict:
    await _load_results()
    _results_cache.append(result)
    await _save_results()
    return result


async def get_benchmark_result(result_id: str) -> Optional[dict]:
    await _load_results()
    return next((r for r in _results_cache if r.get("id") == result_id), None)


async def delete_benchmark_result(result_id: str) -> bool:
    global _results_cache
    await _load_results()
    before = len(_results_cache)
    _results_cache = [r for r in _results_cache if r.get("id") != result_id]
    if len(_results_cache) != before:
        await _save_results()
        return True
    return False

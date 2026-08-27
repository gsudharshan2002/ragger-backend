import os
import re
import uuid
from typing import Optional

from app.core.config import settings
from app.core.logging_config import get_logger
from app.models.schemas import StoredChunk, UploadedDocument
from app.services.embeddings import get_embeddings_for_texts
from app.services.storage import (
    add_chunks,
    add_document,
    update_document,
    delete_chunks_for_document,
    get_data_dir,
    get_settings,
)

logger = get_logger(__name__)


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping chunks by character count."""
    if not text:
        return []

    # Split into sentences first for better chunk boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = ""
    current_size = 0

    for sentence in sentences:
        sentence_size = len(sentence)
        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep overlap
            overlap_text = current_chunk[-overlap:] if overlap > 0 else ""
            current_chunk = overlap_text
            current_size = len(overlap_text)

        current_chunk += " " + sentence if current_chunk else sentence
        current_size += sentence_size

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


async def process_pdf(file_path: str, file_name: str, knowledge_base_id: Optional[str] = None, folder_id: Optional[str] = None) -> UploadedDocument:
    """Process a PDF file: extract text, chunk, embed, store."""
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)

    total_text = "\n".join(pages_text)
    return await _process_document_text(
        total_text, file_name, len(reader.pages), knowledge_base_id, "application/pdf", file_path, folder_id=folder_id
    )


async def process_text_file(file_path: str, file_name: str, knowledge_base_id: Optional[str] = None, folder_id: Optional[str] = None) -> UploadedDocument:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return await _process_document_text(text, file_name, 1, knowledge_base_id, "text/plain", file_path, folder_id=folder_id)


async def _process_document_text(
    text: str,
    file_name: str,
    pages: int,
    knowledge_base_id: Optional[str] = None,
    file_type: str = "text/plain",
    original_path: Optional[str] = None,
    doc_id: Optional[str] = None,
    folder_id: Optional[str] = None,
) -> UploadedDocument:
    doc_id = doc_id or str(uuid.uuid4())
    persisted = await get_settings()
    chunk_size = int(persisted.get("chunkSize", settings.CHUNK_SIZE) or settings.CHUNK_SIZE)
    chunk_overlap = int(persisted.get("chunkOverlap", settings.CHUNK_OVERLAP) or settings.CHUNK_OVERLAP)
    chunk_texts = _chunk_text(text, chunk_size, chunk_overlap)

    # Create chunks
    chunks: list[StoredChunk] = []
    for i, chunk_text in enumerate(chunk_texts):
        chunks.append(
            StoredChunk(
                id=str(uuid.uuid4()),
                document_id=doc_id,
                document_name=file_name,
                content=chunk_text,
                page=i % max(pages, 1) + 1,
                section=None,
                token_count=max(1, int(len(chunk_text.split()) / 0.75)),
                knowledge_base_id=knowledge_base_id,
            )
        )

    # Generate embeddings
    embeddings = await get_embeddings_for_texts(chunk_texts)
    if embeddings:
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

    # Store
    await add_chunks(chunks)

    doc = UploadedDocument(
        id=doc_id,
        name=file_name,
        size=len(text.encode("utf-8")),
        type=file_type,
        pages=pages,
        chunks=len(chunks),
        token_count=sum(c.token_count for c in chunks),
        path=original_path,
        knowledge_base_id=knowledge_base_id,
        folder_id=folder_id,
    )
    await add_document(doc)

    logger.info(f"Processed document {file_name}: {len(chunks)} chunks")
    return doc


async def reprocess_document(
    doc_id: str,
    file_path: str,
    file_name: str,
    knowledge_base_id: Optional[str] = None,
    folder_id: Optional[str] = None,
) -> UploadedDocument:
    """Re-extract, re-chunk, re-embed and replace an existing document's content."""
    if file_path.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        pages_text = [page.extract_text() or "" for page in reader.pages]
        total_text = "\n".join(pages_text)
        file_type = "application/pdf"
        pages = len(reader.pages)
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            total_text = f.read()
        file_type = "text/plain"
        pages = 1

    # Remove previous chunks for this document
    await delete_chunks_for_document(doc_id)

    # Rebuild with the same document id
    doc = await _process_document_text(
        total_text, file_name, pages, knowledge_base_id, file_type, file_path, doc_id=doc_id, folder_id=folder_id
    )
    await update_document(doc)
    logger.info(f"Reprocessed document {file_name}: {doc.chunks} chunks")
    return doc

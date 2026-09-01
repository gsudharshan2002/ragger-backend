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

# Browser "print to PDF" artifacts: a timestamp+title header line and a
# "Page N of M<url>" footer line, both glued onto adjacent text with no
# separating space by the source renderer. pypdf extracts these as if they
# were body content, which otherwise lands mid-chunk.
_PRINT_TIMESTAMP_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*[AP]M")
_PRINT_PAGE_FOOTER_RE = re.compile(r"^Page\s+\d+\s+of\s+\d+")

_HEADING_SKIP_PREFIXES = (
    "protocol ", "class ", "struct ", "enum ", "func ",
    "extension ", "case ", "var ", "let ", "typealias ",
)


def _clean_page_text(text: str) -> str:
    """Strip browser-print header/footer artifacts before chunking."""
    lines = text.split("\n")
    kept = [
        line for line in lines
        if not _PRINT_TIMESTAMP_RE.match(line.strip())
        and not _PRINT_PAGE_FOOTER_RE.match(line.strip())
    ]
    return "\n".join(kept)


def _looks_like_heading(line: str) -> bool:
    """Best-effort heuristic: a short, capitalized, non-sentence line that
    isn't an API signature (protocol/class/struct/...) is treated as a
    section heading. Imperfect on arbitrary PDF layouts, but a reasonable
    default when no real heading markup survives text extraction."""
    line = line.strip()
    if not line or len(line) > 60:
        return False
    if not line[0].isupper():
        return False
    if line.endswith((".", ":", ",", ";")):
        return False
    lowered = line.lower()
    if any(lowered.startswith(p) for p in _HEADING_SKIP_PREFIXES):
        return False
    return True


def _extract_headings(pages_text: list[str]) -> list[tuple[int, str]]:
    """(offset, heading_text) pairs in document order, offsets into the
    text produced by joining pages_text with '\\n'."""
    headings: list[tuple[int, str]] = []
    cursor = 0
    for text in pages_text:
        search_from = 0
        for line in text.split("\n"):
            if _looks_like_heading(line):
                idx = text.find(line, search_from)
                if idx != -1:
                    headings.append((cursor + idx, line.strip()))
                    search_from = idx + len(line)
        cursor += len(text) + 1  # +1 for the '\n' join separator
    return headings


def _section_for_offset(offset: int, headings: list[tuple[int, str]]) -> Optional[str]:
    """The nearest heading at or before this offset, if any."""
    section = None
    for h_offset, text in headings:
        if h_offset <= offset:
            section = text
        else:
            break
    return section


def _page_bounds(pages_text: list[str]) -> list[tuple[int, int, int]]:
    """(start, end, page_number) for each page within the joined text."""
    bounds = []
    cursor = 0
    for i, text in enumerate(pages_text):
        start = cursor
        end = start + len(text)
        bounds.append((start, end, i + 1))
        cursor = end + 1
    return bounds


def _page_for_span(start: int, end: int, bounds: list[tuple[int, int, int]]) -> int:
    """The page containing the largest share of the [start, end) span -
    not just whichever page a trailing pointer happened to land on."""
    if not bounds:
        return 1
    best_page = bounds[0][2]
    best_overlap = -1
    for p_start, p_end, page_num in bounds:
        overlap = min(end, p_end) - max(start, p_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_page = page_num
    return best_page


def _split_sentences_with_offsets(text: str) -> list[tuple[str, int]]:
    """Same sentence split as before, but paired with each sentence's
    character offset in the original text so chunks can be traced back to
    a source page/heading."""
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    result = []
    cursor = 0
    for raw in raw_sentences:
        s = raw.strip()
        if not s:
            continue
        idx = text.find(s, cursor)
        if idx == -1:
            idx = cursor
        result.append((s, idx))
        cursor = idx + len(s)
    return result


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[tuple[str, int, int]]:
    """Split text into overlapping chunks by character count. Returns
    (chunk_text, start_offset, end_offset) so callers can map each chunk
    back to the page(s) and heading it actually came from."""
    if not text:
        return []

    sentences = _split_sentences_with_offsets(text)
    if not sentences:
        return []

    chunks: list[tuple[str, int, int]] = []
    current_chunk = ""
    current_size = 0
    current_start = sentences[0][1]
    current_end = current_start

    for sentence, offset in sentences:
        sentence_size = len(sentence)
        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append((current_chunk.strip(), current_start, current_end))
            # Keep overlap
            overlap_text = current_chunk[-overlap:] if overlap > 0 else ""
            current_start = current_end - len(overlap_text) if overlap_text else offset
            current_chunk = overlap_text
            current_size = len(overlap_text)
        elif not current_chunk:
            current_start = offset

        current_chunk += " " + sentence if current_chunk else sentence
        current_size += sentence_size
        current_end = offset + len(sentence)

    if current_chunk.strip():
        chunks.append((current_chunk.strip(), current_start, current_end))

    return chunks


async def process_pdf(file_path: str, file_name: str, knowledge_base_id: Optional[str] = None, folder_id: Optional[str] = None) -> UploadedDocument:
    """Process a PDF file: extract text, chunk, embed, store."""
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    pages_text = [_clean_page_text(page.extract_text() or "") for page in reader.pages]

    total_text = "\n".join(pages_text)
    return await _process_document_text(
        total_text, file_name, len(reader.pages), knowledge_base_id, "application/pdf", file_path,
        pages_text=pages_text, folder_id=folder_id,
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
    pages_text: Optional[list[str]] = None,
) -> UploadedDocument:
    doc_id = doc_id or str(uuid.uuid4())
    persisted = await get_settings()
    chunk_size = int(persisted.get("chunkSize", settings.CHUNK_SIZE) or settings.CHUNK_SIZE)
    chunk_overlap = int(persisted.get("chunkOverlap", settings.CHUNK_OVERLAP) or settings.CHUNK_OVERLAP)
    chunk_spans = _chunk_text(text, chunk_size, chunk_overlap)

    pages_text = pages_text or [text]
    page_bounds = _page_bounds(pages_text)
    headings = _extract_headings(pages_text)

    # Create chunks
    chunks: list[StoredChunk] = []
    for chunk_text_value, start, end in chunk_spans:
        chunks.append(
            StoredChunk(
                id=str(uuid.uuid4()),
                document_id=doc_id,
                document_name=file_name,
                content=chunk_text_value,
                page=_page_for_span(start, end, page_bounds),
                section=_section_for_offset(start, headings),
                token_count=max(1, int(len(chunk_text_value.split()) / 0.75)),
                knowledge_base_id=knowledge_base_id,
            )
        )

    # Generate embeddings
    chunk_texts = [c.content for c in chunks]
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
    pages_text: Optional[list[str]] = None
    if file_path.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        pages_text = [_clean_page_text(page.extract_text() or "") for page in reader.pages]
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
        total_text, file_name, pages, knowledge_base_id, file_type, file_path,
        doc_id=doc_id, folder_id=folder_id, pages_text=pages_text,
    )
    await update_document(doc)
    logger.info(f"Reprocessed document {file_name}: {doc.chunks} chunks")
    return doc

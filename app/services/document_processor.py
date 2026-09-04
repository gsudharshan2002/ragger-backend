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

_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$")

# Known single-word section titles common in resumes/reports - exempted
# from the 2+ word rule below. A blanket "single word = never a heading"
# rule would also reject these real, common section titles; a blanket
# "single capitalized word = heading" rule is what caused "It"/"Job" style
# false positives in the first place. This bounded whitelist is the safe
# middle ground: known vocabulary is allowed through, anything else still
# needs 2+ words to count.
_COMMON_SECTION_WORDS = {
    "summary", "experience", "education", "skills", "responsibilities",
    "objective", "projects", "achievements", "certifications", "references",
    "highlights", "overview", "introduction", "background", "qualifications",
    "profile", "employment", "history", "publications", "awards",
    "languages", "interests", "activities", "volunteering", "expertise",
    "competencies", "accomplishments", "portfolio", "contact",
}

# Lightweight, embedding-free "semantic grouping" signal: a sentence that
# opens with one of these almost always continues the previous sentence's
# thought rather than starting a new one, so the two shouldn't be split
# across a chunk boundary just because the character budget was hit.
_CONTINUATION_STARTERS = (
    "it ", "it's ", "its ", "this ", "these ", "those ", "that ", "such ",
    "he ", "she ", "they ", "however", "additionally", "furthermore",
    "also", "moreover", "thus", "therefore", "then ", "so ", "meanwhile",
)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
    "with", "is", "are", "was", "were", "be", "been", "by", "as", "at",
    "from", "this", "that", "these", "those", "it", "its", "you", "your",
}

_BULLET_MARKER_RE = re.compile(r"[•●▪‣∙]")


def _clean_page_text(text: str) -> str:
    """Strip browser-print header/footer artifacts before chunking."""
    lines = text.split("\n")
    kept = [
        line for line in lines
        if not _PRINT_TIMESTAMP_RE.match(line.strip())
        and not _PRINT_PAGE_FOOTER_RE.match(line.strip())
    ]
    return "\n".join(kept)


def _normalize_whitespace(text: str) -> str:
    """Collapse any run of whitespace (including embedded newlines) into a
    single space. Some PDFs (design-tool exports especially) extract with
    roughly one word per line and a lone space character on its own line
    between words - real content, but unreadable as stored and confusing
    for the LLM to parse. Applied to chunk CONTENT only, after heading
    detection has already run on the original line-structured text, so
    line-based heading detection is unaffected."""
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_heading(line: str) -> bool:
    """A real markdown heading (`#`..`######`) always counts - that's
    unambiguous markup, not a guess. Otherwise fall back to a best-effort
    heuristic: a short, capitalized, multi-word, non-sentence line that
    isn't an API signature (protocol/class/struct/...) is treated as a
    section heading. Imperfect on arbitrary PDF layouts, but a reasonable
    default when no real heading markup survives text extraction.

    Requiring 2+ words matters: PDFs that extract with roughly one word per
    line (common from design-tool exports) otherwise trip this heuristic on
    ordinary sentence-initial words - "It", "Job", "Receive" - none of which
    are real headings, just the first word of a normal sentence. A genuine
    section title is essentially always a multi-word phrase, except for a
    bounded, known vocabulary of real single-word section titles (see
    _COMMON_SECTION_WORDS) - "Summary", "Responsibilities", and the like."""
    line = line.strip()
    if _MARKDOWN_HEADING_RE.match(line):
        return True
    if not line or len(line) > 60:
        return False
    if not line[0].isupper():
        return False
    if line.endswith((".", ":", ",", ";")):
        return False
    lowered = line.lower()
    if len(line.split()) < 2 and lowered not in _COMMON_SECTION_WORDS:
        return False
    if any(lowered.startswith(p) for p in _HEADING_SKIP_PREFIXES):
        return False
    return True


def _heading_text(line: str) -> str:
    """Strip markdown '#' markup off a heading line, if present."""
    match = _MARKDOWN_HEADING_RE.match(line.strip())
    return match.group(1).strip() if match else line.strip()


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
                    headings.append((cursor + idx, _heading_text(line)))
                    search_from = idx + len(line)
        cursor += len(text) + 1  # +1 for the '\n' join separator
    return headings


_MIN_SECTION_BODY_CHARS = 40


def _split_into_sections(text: str, headings: list[tuple[int, str]]) -> list[tuple[Optional[str], int, str]]:
    """Split `text` into (section_label, segment_start_offset, segment_text)
    at each heading boundary. Chunking is then run independently within
    each segment (see _chunk_text call site below), so a chunk can never
    span two sections - a hard split, not just after-the-fact metadata.
    With no detected headings, the whole document is one segment (section
    None), identical to the old whole-document chunking behavior.

    A heading immediately followed by another heading (or by nothing
    substantial) produces a segment that's just the heading line itself -
    e.g. a short capitalized label like "Job" or "State" that the PDF
    heading heuristic mistook for a section title, since it has no real
    heading markup to go on. Hard-splitting on that would turn it into its
    own near-empty, useless chunk, so such a segment is folded forward into
    the next one instead of being emitted on its own."""
    if not headings:
        return [(None, 0, text)]

    raw_segments: list[tuple[Optional[str], int, str]] = []
    if headings[0][0] > 0:
        raw_segments.append((None, 0, text[: headings[0][0]]))

    for i, (start, label) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        raw_segments.append((label, start, text[start:end]))

    raw_segments = [(label, start, seg) for label, start, seg in raw_segments if seg.strip()]

    merged: list[tuple[Optional[str], int, str]] = []
    carry_text = ""
    carry_start: Optional[int] = None
    for label, start, seg in raw_segments:
        # Body text is the segment with its own heading line removed - that
        # heading line itself shouldn't count towards "is there real content
        # here", or every one-line section would trivially pass.
        body = seg.split("\n", 1)[1] if (label is not None and "\n" in seg) else ("" if label is not None else seg)

        if carry_text:
            seg = carry_text + seg
            start = carry_start
            carry_text = ""
            carry_start = None

        if label is not None and len(body.strip()) < _MIN_SECTION_BODY_CHARS:
            carry_text = seg
            carry_start = start
            continue

        merged.append((label, start, seg))

    if carry_text:
        # A near-empty section trailing at the very end of the document has
        # nothing after it to merge into - fold it onto the previous real
        # section instead of silently dropping it.
        if merged:
            prev_label, prev_start, prev_seg = merged[-1]
            merged[-1] = (prev_label, prev_start, prev_seg + carry_text)
        else:
            merged.append((None, carry_start, carry_text))

    return merged


def _split_by_bullets(text: str) -> list[tuple[int, str]]:
    """(start_offset, item_text) pairs, one per bullet-point item, for
    sections that use bullet markers (resume-style "responsibilities" and
    "features" lists are the common case). Each bullet is a hard split -
    two short bullets are never packed into one chunk just because they'd
    both fit under the character budget, since each is its own atomic idea.

    Order-agnostic to whether the marker leads or trails the item text -
    PDF text extraction sometimes reorders it so the bullet glyph ends up
    after the sentence it introduces rather than before. Falls back to the
    whole text as a single item when no bullet markers are present, leaving
    non-bulleted documents (API docs, prose) unaffected."""
    markers = list(_BULLET_MARKER_RE.finditer(text))
    if not markers:
        return [(0, text)]

    items: list[tuple[int, str]] = []
    cursor = 0
    for m in markers:
        piece = text[cursor:m.start()]
        if piece.strip():
            items.append((cursor, piece))
        cursor = m.end()
    tail = text[cursor:]
    if tail.strip():
        items.append((cursor, tail))
    return items


def _continues_topic(prev_sentence: str, next_sentence: str) -> bool:
    """Lightweight, embedding-free heuristic: does `next_sentence` read as a
    direct continuation of `prev_sentence`'s thought rather than a new one?
    Used to avoid splitting a chunk in the middle of one continuous idea
    purely because the character budget was hit exactly on that boundary."""
    lowered_next = next_sentence.strip().lower()
    if lowered_next.startswith(_CONTINUATION_STARTERS):
        return True

    def _content_words(s: str) -> set[str]:
        return {w for w in re.findall(r"[a-z']+", s.lower()) if w not in _STOPWORDS and len(w) > 2}

    prev_words = _content_words(prev_sentence)
    next_words = _content_words(next_sentence)
    if not prev_words or not next_words:
        return False
    overlap = prev_words & next_words
    return len(overlap) >= 2 or (len(overlap) / min(len(prev_words), len(next_words))) >= 0.4


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
    """Split text into overlapping chunks by character count, packing whole
    sentences and preferring not to split two topically-continuous
    sentences apart (see _continues_topic). Returns (chunk_text,
    start_offset, end_offset) so callers can map each chunk back to the
    page(s) and heading it actually came from."""
    if not text:
        return []

    sentences = _split_sentences_with_offsets(text)
    if not sentences:
        return []

    # A chunk may run past chunk_size, but only far enough to keep two
    # tightly-continuing sentences together - never split what reads as one
    # continuous thought just because the budget was hit on that boundary,
    # but also never let that grace grow unbounded.
    soft_max = int(chunk_size * 1.3)

    chunks: list[tuple[str, int, int]] = []
    current_chunk = ""
    current_size = 0
    current_start = sentences[0][1]
    current_end = current_start
    prev_sentence = ""

    for sentence, offset in sentences:
        sentence_size = len(sentence)
        over_budget = current_size + sentence_size > chunk_size
        must_split = current_size + sentence_size > soft_max
        keep_together = (
            over_budget and not must_split and current_chunk
            and _continues_topic(prev_sentence, sentence)
        )

        if over_budget and current_chunk and not keep_together:
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
        prev_sentence = sentence

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

    pages_text = pages_text or [text]
    page_bounds = _page_bounds(pages_text)
    headings = _extract_headings(pages_text)
    sections = _split_into_sections(text, headings)

    # Create chunks - chunked independently per section, and within a
    # section independently per bullet item, so a chunk never spans two
    # sections or two bullets; see _split_into_sections / _split_by_bullets.
    chunks: list[StoredChunk] = []
    for section_label, segment_start, segment_text in sections:
        for item_start, item_text in _split_by_bullets(segment_text):
            for chunk_text_value, rel_start, rel_end in _chunk_text(item_text, chunk_size, chunk_overlap):
                start = segment_start + item_start + rel_start
                end = segment_start + item_start + rel_end
                clean_content = _normalize_whitespace(chunk_text_value)
                chunks.append(
                    StoredChunk(
                        id=str(uuid.uuid4()),
                        document_id=doc_id,
                        document_name=file_name,
                        content=clean_content,
                        page=_page_for_span(start, end, page_bounds),
                        section=section_label,
                        token_count=max(1, int(len(clean_content.split()) / 0.75)),
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

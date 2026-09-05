import os
import random
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
    get_all_chunks,
    get_data_dir,
    get_settings,
)

logger = get_logger(__name__)

# Chunk ids are integers in [0, 2**13] (0..8192). uuid ids remain valid for
# pre-existing chunks, but any chunk created from now on gets a numeric id.
CHUNK_ID_MAX = 2 ** 13


def _next_chunk_id(taken: set[int]) -> str:
    """Return a fresh numeric chunk id (0..CHUNK_ID_MAX) not already in
    `taken`, registering it as used. Falls back to a duplicate-allowed random
    id if the (unrealistically small) id space is exhausted."""
    if len(taken) <= CHUNK_ID_MAX:
        for _ in range((CHUNK_ID_MAX + 1) * 10):
            candidate = random.randint(0, CHUNK_ID_MAX)
            if candidate not in taken:
                taken.add(candidate)
                return str(candidate)
    return str(random.randint(0, CHUNK_ID_MAX))

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

# Recursive character splitting tries each separator in priority order -
# paragraph break, then line break, then word break - only falling through
# to individual characters ("") if a piece still doesn't fit chunk_size
# after all of those. See _atomic_pieces.
_RECURSIVE_SEPARATORS = ["\n\n", "\n", " ", ""]

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


def _atomic_pieces(
    text: str, start: int, chunk_size: int, separators: list[str]
) -> list[tuple[str, int, int]]:
    """Break `text` (which begins at absolute offset `start` in the
    original document) into pieces small enough to pack into chunks of
    chunk_size, by trying each separator in `separators` in order -
    paragraph break, then line break, then word break - and recursing into
    any piece still larger than chunk_size using the next separator down
    the list. The final separator, "" (no separator left), falls back to
    splitting at every character so any input can always be divided small
    enough. A piece within chunk_size is never split further, even if an
    earlier separator would have cut it up more finely - the goal is the
    fewest pieces that still all fit, not the smallest possible pieces.

    Returns (piece_text, start_offset, end_offset) triples with real
    offsets into the original document, so the caller can trace a chunk
    back to its source page/heading."""
    if not text:
        return []
    if len(text) <= chunk_size or not separators:
        return [(text, start, start + len(text))]

    sep, rest = separators[0], separators[1:]
    if sep == "":
        return [(ch, start + i, start + i + 1) for i, ch in enumerate(text)]
    if sep not in text:
        return _atomic_pieces(text, start, chunk_size, rest)

    pieces: list[tuple[str, int, int]] = []
    cursor = 0
    for part in text.split(sep):
        part_start = start + cursor
        if part:
            pieces.extend(_atomic_pieces(part, part_start, chunk_size, rest))
        cursor += len(part) + len(sep)
    return pieces


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[tuple[str, int, int]]:
    """Recursive character splitting: break text into small atomic pieces
    (see _atomic_pieces), then greedily pack consecutive pieces into chunks
    up to chunk_size, carrying the trailing `overlap` characters of each
    chunk into the start of the next so no boundary loses context. Returns
    (chunk_text, start_offset, end_offset) so callers can map each chunk
    back to the page(s) and heading it actually came from."""
    if not text:
        return []

    pieces = _atomic_pieces(text, 0, chunk_size, _RECURSIVE_SEPARATORS)
    if not pieces:
        return []

    def _joined(parts: list[tuple[str, int, int]]) -> str:
        return " ".join(p[0] for p in parts).strip()

    chunks: list[tuple[str, int, int]] = []
    current: list[tuple[str, int, int]] = []
    current_size = 0

    for piece in pieces:
        piece_text = piece[0]
        piece_len = len(piece_text)

        if current and current_size + piece_len > chunk_size:
            chunks.append((_joined(current), current[0][1], current[-1][2]))

            if overlap > 0:
                # Carry trailing pieces totaling up to `overlap` characters
                # into the next chunk, so it opens with context from the
                # end of this one instead of a hard cut. Always keeps at
                # least the last piece, even if it alone exceeds overlap.
                carried: list[tuple[str, int, int]] = []
                carried_size = 0
                for p in reversed(current):
                    if carried_size + len(p[0]) > overlap and carried:
                        break
                    carried.insert(0, p)
                    carried_size += len(p[0])
                current, current_size = carried, carried_size
            else:
                current, current_size = [], 0

        current.append(piece)
        current_size += piece_len

    if current:
        chunks.append((_joined(current), current[0][1], current[-1][2]))

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
    taken_chunk_ids = {
        int(c.id) for c in await get_all_chunks() if c.id.isdigit()
    }
    for section_label, segment_start, segment_text in sections:
        for item_start, item_text in _split_by_bullets(segment_text):
            for chunk_text_value, rel_start, rel_end in _chunk_text(item_text, chunk_size, chunk_overlap):
                start = segment_start + item_start + rel_start
                end = segment_start + item_start + rel_end
                clean_content = _normalize_whitespace(chunk_text_value)
                chunks.append(
                    StoredChunk(
                        id=_next_chunk_id(taken_chunk_ids),
                        document_id=doc_id,
                        document_name=file_name,
                        content=clean_content,
                        page=_page_for_span(start, end, page_bounds),
                        section=section_label,
                        token_count=max(1, int(len(clean_content.split()) / 0.75)),
                        knowledge_base_id=knowledge_base_id,
                    )
                )

    # Generate embeddings. get_embeddings_for_texts swallows its own errors
    # and returns None on total failure - checked explicitly here (rather
    # than silently storing embedding-less chunks) so a provider hiccup at
    # ingest doesn't quietly degrade every future MMR run touching this
    # document with no visible sign anything went wrong.
    chunk_texts = [c.content for c in chunks]
    embeddings = await get_embeddings_for_texts(chunk_texts)
    embedding_error: Optional[str] = None
    if not embeddings:
        embedding_error = (
            f"Embedding generation failed for all {len(chunks)} chunk(s) - stored without "
            "vectors. Vector search and MMR diversity won't work for this document until "
            "it's reprocessed or embeddings are reindexed."
        )
        logger.error(f"{file_name}: {embedding_error}")
    else:
        if len(embeddings) < len(chunks):
            missing = len(chunks) - len(embeddings)
            embedding_error = (
                f"Embedding generation returned {len(embeddings)}/{len(chunks)} vectors - "
                f"{missing} chunk(s) stored without an embedding. Vector search and MMR "
                "diversity won't work for those chunks until reindexed."
            )
            logger.error(f"{file_name}: {embedding_error}")
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

    # Extract retrieval keywords per chunk (stored under metadata["keywords"]).
    # Best-effort: prefer the configured LLM, fall back to local TF-IDF, and
    # never fail the document on keyword trouble - see extract_keywords().
    from app.services.keywords import extract_keywords

    keywords = await extract_keywords(chunk_texts)
    for chunk, kw in zip(chunks, keywords):
        if kw:
            chunk.metadata["keywords"] = kw

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
        embedding_error=embedding_error,
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

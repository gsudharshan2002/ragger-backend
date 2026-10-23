"""Seed the knowledge base with the documents the Week 6 cases expect.

The cases in evals/developer_docs_cases.json name `current-version.md` and
`old-version.md` (gh-*) and the Apple Combine PDF (combine-*) as their expected
sources. Those files live in uploads/ but were not in the knowledge base, so
those cases were unanswerable. This script makes the KB match the dataset,
idempotently:

  - ingests any missing expected source from uploads/ (matched by file name),
  - removes documents listed in REMOVE (github_combined_reference.pdf documents an
    API version "2025-06-01" that does not exist and contradicts the real
    2026-03-10 / 2022-11-28 versions).

Usage:  python scripts/seed_github_kb.py [--dry-run]
"""

import asyncio
import sys
from pathlib import Path

from app.core.config import settings
from app.services.document_processor import process_pdf, process_text_file
from app.services.storage import (
    delete_document,
    get_all_documents,
    init_storage,
)

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["current-version.md", "old-version.md", "Combine | Apple Developer Documentation.pdf"]
REMOVE = ["github_combined_reference.pdf"]


def _find_upload(name: str) -> Path | None:
    matches = sorted(Path(ROOT / settings.UPLOAD_DIR).glob(f"*_{name}"))
    return matches[0] if matches else None


async def main(dry_run: bool) -> int:
    await init_storage()
    docs = await get_all_documents()
    by_name = {d.name: d for d in docs}

    for name in REMOVE:
        doc = by_name.get(name)
        if doc:
            print(f"remove  {name} ({doc.chunks} chunks)")
            if not dry_run:
                await delete_document(doc.id)
        else:
            print(f"absent  {name}")

    missing = 0
    for name in REQUIRED:
        if name in by_name:
            print(f"present {name}")
            continue
        src = _find_upload(name)
        if not src:
            print(f"MISSING {name}: not found in {settings.UPLOAD_DIR}/", file=sys.stderr)
            missing += 1
            continue
        print(f"ingest  {name} <- {src.name}")
        if not dry_run:
            ingest = process_pdf if name.lower().endswith(".pdf") else process_text_file
            doc = await ingest(str(src), name)
            print(f"        {doc.chunks} chunks, embedding_error={doc.embedding_error}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main("--dry-run" in sys.argv)))

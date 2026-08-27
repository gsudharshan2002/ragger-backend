import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure the backend package is importable from the repo root
BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

# --- Isolate file storage for tests BEFORE importing the app ---
_tmp = os.environ.setdefault("RAGGER_DATA_DIR", tempfile.mkdtemp(prefix="ragger-test-data-"))
os.environ["UPLOAD_DIR"] = tempfile.mkdtemp(prefix="ragger-test-uploads-")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
import app.services.storage as storage  # noqa: E402


@pytest.fixture()
def client():
    """Yield an isolated TestClient with a fresh in-memory storage state."""
    # Reset every module-level cache so tests do not leak into each other.
    storage._initialized = False
    storage._chunks_cache = []
    storage._documents_cache = {}
    storage._kb_cache = {}
    storage._dataset_cache = {}
    storage._settings_cache = None
    storage._traces_cache = []
    storage._traces_loaded = False
    storage._folders_cache = []
    storage._folders_loaded = False
    storage._results_cache = []
    storage._results_loaded = False

    # Also clear the benchmark in-memory run store
    import app.api.v1.benchmark as benchmark
    benchmark._benchmark_runs = {}

    with TestClient(app) as c:
        yield c

"""End-to-end API tests for the FastAPI Ragger backend.

Each test uses an isolated storage directory (see conftest.py) so tests
are independent and deterministic.
"""

SAMPLE_TEXT = (
    "FastAPI is a modern Python web framework. It is built on Starlette and Pydantic. "
    "It provides automatic OpenAPI documentation. "
    "Ragger is a Retrieval-Augmented Generation backend. It ingests documents, "
    "chunks them, embeds them, and answers questions. "
    "The chunking strategy splits text into overlapping windows. "
    "The default top K for retrieval is five. MMR lambda is set to zero point seven. "
) * 20


def upload_text(client, filename="sample.txt", kb_id=None):
    url = "/api/v1/documents/upload"
    params = {"knowledge_base_id": kb_id} if kb_id else None
    return client.post(
        url,
        files={"file": (filename, SAMPLE_TEXT.encode("utf-8"), "text/plain")},
        params=params,
    )


# ============================== Health ==============================


def test_root_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_api_health_route(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == "1.0.0"


# ============================== Documents ==============================


def test_upload_text_document(client):
    r = upload_text(client)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["success"] is True
    doc = body["data"]
    # camelCase fields expected by the frontend
    for field in ("id", "name", "size", "type", "mimeType", "pages", "pageCount",
                  "chunks", "chunkCount", "tokenCount", "status", "createdAt",
                  "updatedAt", "uploadedAt", "knowledgeBaseId"):
        assert field in doc, f"missing {field}"
    assert doc["mimeType"] == "text/plain"
    assert doc["status"] == "ready"
    assert doc["chunkCount"] > 0


def test_upload_unsupported_type_returns_400(client):
    r = client.post(
        "/api/v1/documents/upload",
        files={"file": ("evil.exe", b"MZ....", "application/octet-stream")},
    )
    assert r.status_code == 400
    assert "Unsupported" in r.json()["detail"]


def test_list_documents_and_get_one(client):
    doc = upload_text(client).json()["data"]
    listing = client.get("/api/v1/documents").json()
    assert listing["success"] is True
    assert any(d["id"] == doc["id"] for d in listing["data"])

    one = client.get(f"/api/v1/documents/{doc['id']}").json()["data"]
    assert one["id"] == doc["id"]


def test_get_missing_document_404(client):
    r = client.get("/api/v1/documents/nonexistent")
    assert r.status_code == 404


def test_document_chunks_endpoint(client):
    doc = upload_text(client).json()["data"]
    r = client.get(f"/api/v1/documents/{doc['id']}/chunks")
    assert r.status_code == 200
    chunks = r.json()["data"]
    assert len(chunks) == doc["chunkCount"]
    assert all(c["document_id"] == doc["id"] for c in chunks)
    assert all("content" in c for c in chunks)


def test_chunk_size_and_overlap_respected(client):
    """Recursive character splitting (see document_processor._chunk_text)
    should keep chunks close to the configured chunkSize and produce
    overlapping content between consecutive chunks - not just "some
    chunks exist", but that the size/overlap settings actually take
    effect."""
    small_size, overlap = 100, 20
    r = client.put("/api/v1/rag/config", json={"chunkSize": small_size, "chunkOverlap": overlap})
    assert r.status_code == 200, r.text

    doc = upload_text(client, filename="chunking-sample.txt").json()["data"]
    chunks = client.get(f"/api/v1/documents/{doc['id']}/chunks").json()["data"]
    assert len(chunks) > 1, "long sample text should split into multiple chunks"

    # Whitespace normalization can add a little slack, but a chunk should
    # never balloon well past the configured size.
    for c in chunks:
        assert len(c["content"]) <= small_size * 2, c["content"]

    # Consecutive chunks should share trailing/leading content (the
    # overlap), not start exactly where the previous one ended.
    overlapping_pairs = sum(
        1 for prev, nxt in zip(chunks, chunks[1:])
        if set(prev["content"][-overlap:].split()) & set(nxt["content"][:overlap].split())
    )
    assert overlapping_pairs > 0, "expected at least one overlapping boundary between chunks"


def test_reprocess_and_rechunk_versions(client):
    doc = upload_text(client).json()["data"]

    reproc = client.post(f"/api/v1/documents/{doc['id']}/reprocess")
    assert reproc.status_code == 200, reproc.text

    rechunk = client.post(f"/api/v1/documents/{doc['id']}/rechunk")
    assert rechunk.status_code == 200, rechunk.text

    versions = client.get(f"/api/v1/documents/{doc['id']}/versions").json()["data"]
    nums = sorted(v["versionNumber"] for v in versions)
    assert nums == [1, 2, 3], nums
    # Only the newest (v3) should be flagged isLatest
    assert [v["isLatest"] for v in sorted(versions, key=lambda x: x["versionNumber"])] == [False, False, True]

    history = client.get(f"/api/v1/documents/{doc['id']}/history").json()["data"]
    assert len(history) >= 2  # reprocess + rechunk events
    for ev in history:
        for field in ("id", "documentId", "action", "status", "startedAt",
                      "completedAt", "durationMs", "resultSummary"):
            assert field in ev


def test_document_preview_range(client):
    doc = upload_text(client).json()["data"]
    r = client.get(f"/api/v1/documents/{doc['id']}/preview", headers={"Range": "bytes=0-9"})
    assert r.status_code == 206
    assert r.headers.get("Content-Range", "").startswith("bytes 0-9/")
    assert len(r.content) == 10


def test_delete_document(client):
    doc = upload_text(client).json()["data"]
    assert client.delete(f"/api/v1/documents/{doc['id']}").status_code == 200
    assert client.get(f"/api/v1/documents/{doc['id']}").status_code == 404
    chunks = client.get(f"/api/v1/documents/{doc['id']}/chunks").json()["data"]
    assert chunks == []


# ============================== Knowledge Bases ==============================


def test_knowledge_base_crud(client):
    create = client.post("/api/v1/knowledge-bases", json={"name": "Docs KB", "description": "test"})
    assert create.status_code == 201
    kb = create.json()["data"]
    assert kb["name"] == "Docs KB"
    assert "documentCount" in kb and "chunkCount" in kb

    listing = client.get("/api/v1/knowledge-bases").json()
    assert any(k["id"] == kb["id"] for k in listing["data"])

    one = client.get(f"/api/v1/knowledge-bases/{kb['id']}").json()["data"]
    assert one["id"] == kb["id"]

    assert client.delete(f"/api/v1/knowledge-bases/{kb['id']}").status_code == 200
    assert client.get(f"/api/v1/knowledge-bases/{kb['id']}").status_code == 404


def test_kb_get_missing_404(client):
    assert client.get("/api/v1/knowledge-bases/nope").status_code == 404


def test_kb_document_upload_and_list(client):
    kb = client.post("/api/v1/knowledge-bases", json={"name": "KB2"}).json()["data"]
    kb_id = kb["id"]

    up = client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        files={"file": ("kb.txt", SAMPLE_TEXT.encode("utf-8"), "text/plain")},
    )
    assert up.status_code == 201, up.text
    doc = up.json()["data"]
    assert doc["knowledgeBaseId"] == kb_id

    listing = client.get(f"/api/v1/knowledge-bases/{kb_id}/documents").json()["data"]
    assert any(d["id"] == doc["id"] for d in listing)


def test_kb_upload_missing_kb_404(client):
    r = client.post(
        "/api/v1/knowledge-bases/nope/documents",
        files={"file": ("kb.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 404


def test_kb_folders(client):
    kb = client.post("/api/v1/knowledge-bases", json={"name": "folders"}).json()["data"]
    kb_id = kb["id"]

    created = client.post(f"/api/v1/knowledge-bases/{kb_id}/folders", json={"name": "Folder A"})
    assert created.status_code == 201
    folder = created.json()["data"]
    assert folder["name"] == "Folder A"
    assert folder["knowledgeBaseId"] == kb_id

    listing = client.get(f"/api/v1/knowledge-bases/{kb_id}/folders").json()["data"]
    assert any(f["id"] == folder["id"] for f in listing)


# ============================== Datasets ==============================


def test_dataset_crud(client):
    create = client.post("/api/v1/datasets", json={"name": "Dataset", "description": "d"})
    assert create.status_code == 201
    ds = create.json()["data"]
    assert ds["name"] == "Dataset"
    assert ds["currentVersion"] == "v1"

    assert client.get(f"/api/v1/datasets/{ds['id']}").json()["data"]["id"] == ds["id"]
    listing = client.get("/api/v1/datasets").json()
    assert any(d["id"] == ds["id"] for d in listing["data"])

    # PUT update
    updated = client.put(f"/api/v1/datasets/{ds['id']}", json={"name": "Dataset v2", "description": "updated"})
    assert updated.status_code == 200
    assert updated.json()["data"]["name"] == "Dataset v2"

    # POST new version with cases
    ver = client.post(
        f"/api/v1/datasets/{ds['id']}",
        json={"version": "v2", "casesCount": 2, "changeNote": "added cases"},
    )
    assert ver.status_code == 200
    assert ver.json()["data"]["currentVersion"] == "v2"
    assert ver.json()["data"]["versions"][-1]["version"] == "v2"

    assert client.delete(f"/api/v1/datasets/{ds['id']}").status_code == 200
    assert client.get(f"/api/v1/datasets/{ds['id']}").status_code == 404


# ============================== Benchmark ==============================


def test_benchmark_run_flow(client):
    ds = client.post("/api/v1/datasets", json={"name": "Bench DS"}).json()["data"]

    # `/benchmark/run` is what the frontend calls; it returns a stored result.
    run = client.post("/api/v1/benchmark/run", json={"datasetId": ds["id"], "strategy": "bm25"})
    assert run.status_code == 200, run.text
    result = run.json()["data"]
    assert result["id"]
    assert result["datasetId"] == ds["id"]
    assert result["strategy"] == "bm25"

    # `/benchmark/runs` is a separate in-memory store driven by `/benchmark/start`.
    started = client.post(
        "/api/v1/benchmark/start",
        json={"config": {"strategy": "bm25", "datasetId": ds["id"], "datasetVersion": "v1"}},
    )
    assert started.status_code == 200, started.text
    run_id = started.json()["data"]["id"]

    runs = client.get("/api/v1/benchmark/runs").json()["data"]
    assert any(r["id"] == run_id for r in runs)

    one = client.get(f"/api/v1/benchmark/runs/{run_id}").json()["data"]
    assert one["id"] == run_id

    assert client.delete(f"/api/v1/benchmark/runs/{run_id}").status_code == 200
    assert client.get(f"/api/v1/benchmark/runs/{run_id}").status_code == 404


# ============================== Rag Config ==============================


def test_rag_config_get(client):
    r = client.get("/api/v1/rag/config")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    settings = body["data"]["settings"]
    for field in ("llmProvider", "groqModel", "groqApiKey", "geminiApiKey",
                  "embeddingProvider", "chunkSize", "chunkOverlap",
                  "defaultTopK", "defaultStrategy", "systemPrompt", "mmrLambda"):
        assert field in settings, f"missing {field} in settings"


def test_rag_config_save_groq_api_key_roundtrip(client):
    """Regression test for the dropped groqApiKey on save."""
    put = client.put(
        "/api/v1/rag/config",
        json={"groqApiKey": "gsk_TEST", "llmProvider": "groq", "groqModel": "model-x"},
    )
    assert put.status_code == 200, put.text
    saved = put.json()["data"]
    assert saved["groqApiKey"] == "gsk_TEST", "groqApiKey was dropped on save!"

    # Verify it persists on a fresh GET
    settings = client.get("/api/v1/rag/config").json()["data"]["settings"]
    assert settings["groqApiKey"] == "gsk_TEST"


def test_rag_config_saving_camel_and_snake(client):
    # Frontend sends camelCase; ensure it is accepted and returned camelCase.
    r = client.put("/api/v1/rag/config", json={"chunkSize": 256, "chunkOverlap": 32, "defaultTopK": 10})
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["chunkSize"] == 256
    assert data["chunkOverlap"] == 32
    assert data["defaultTopK"] == 10


# ============================== Traces / Chat ==============================


def test_chat_send_generates_trace_and_chat_nogrok(client):
    """Without an API key, chat should not 500; it records a trace."""
    upload_text(client)
    r = client.post("/api/v1/chat/send", json={"query": "What is FastAPI?", "strategy": "bm25"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "answer" in body or "error" in body

    traces = client.get("/api/v1/traces").json()
    assert traces["success"] is True


def test_traces_crud_with_seeded_trace(client):
    import app.services.storage as storage

    # Seed a trace directly (avoids needing an LLM provider).
    storage._traces_loaded = True
    from asyncio import run
    run(storage.add_trace({
        "id": "trace-1",
        "run_id": "run-1",
        "query": "q",
        "strategy": "bm25",
        "status": "completed",
    }))

    one = client.get("/api/v1/traces/trace-1").json()["data"]
    assert one["id"] == "trace-1"

    listing = client.get("/api/v1/traces").json()["data"]
    assert any(t["id"] == "trace-1" for t in listing)

    assert client.delete("/api/v1/traces/trace-1").status_code == 200
    assert client.delete("/api/v1/traces/trace-1").status_code == 404


def test_chat_stream_no_config(client):
    """Streaming chat should not raise and emits SSE frames ending with [DONE]."""
    r = client.post("/api/v1/chat/stream", json={"query": "hello", "strategy": "vector"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    text = r.text
    assert "[DONE]" in text

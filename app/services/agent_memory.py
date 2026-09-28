"""Concept 11: Vector memory.

Applies the SAME technique the RAG pipeline uses for documents (embed it,
store it, retrieve by similarity) to the agent's own past interactions
instead of uploaded files. This is what gives the agent recall ACROSS
separate runs - something short-term `history` (Concept 8) structurally
cannot do, since that list dies the moment the function returns.

Known simplification: memories are never pruned or expired here - this
file grows forever, same as traces.jsonl elsewhere in this app. A real
system needs a retention/decay policy; that's out of scope for this demo.
"""
import json
import math
import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.services.embeddings import generate_query_embedding

_MEMORY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "agent_memory.jsonl")


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def remember(query: str, answer: str) -> None:
    """Store a completed (query, answer) pair as a memory - called once a
    run finishes. This is what makes the interaction available to FUTURE,
    unrelated runs."""
    embedding = await generate_query_embedding(query)
    if not embedding:
        return

    record = {
        "id": str(uuid4()),
        "query": query,
        "answer": answer,
        "embedding": embedding,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(_MEMORY_PATH), exist_ok=True)
    with open(_MEMORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


async def recall(query: str, top_k: int = 3, min_similarity: float = 0.5) -> list[dict[str, Any]]:
    """Retrieve past memories relevant to `query` - the exact same shape
    as search_knowledge_base() from Concept 3 (embed -> compare -> return
    top-k), just over a different corpus."""
    if not os.path.exists(_MEMORY_PATH):
        return []

    query_embedding = await generate_query_embedding(query)
    if not query_embedding:
        return []

    scored = []
    with open(_MEMORY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            similarity = _cosine_similarity(query_embedding, record["embedding"])
            if similarity >= min_similarity:
                scored.append((similarity, record))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {"query": r["query"], "answer": r["answer"], "similarity": round(s, 3)}
        for s, r in scored[:top_k]
    ]

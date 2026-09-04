"""Lightweight RAGAS-style faithfulness and context precision metrics.

Not the `ragas` pip package (which pulls in the full LangChain stack for a
two-number bonus check) - these reimplement RAGAS's own published metric
definitions directly against whichever LLM provider is currently active,
using the exact retrieved context the pipeline actually fed the model
(trace["prompt"]["context"]), not a re-fetched approximation of it.

Faithfulness: break the answer into atomic factual claims, then check each
claim against the retrieved context - score = supported claims / total
claims. This says nothing about whether the context itself was the RIGHT
context - an answer can be perfectly faithful to confidently-retrieved
wrong-version documentation. That gap is exactly what the bonus challenge
is about.

Context precision: for each retrieved chunk, in the rank order retrieval
actually returned them, judge relevance to the question, then compute
average precision over that ranked list (RAGAS's own formula). A pipeline
that ranks the one right-document chunk below several wrong-document
chunks scores LOW here even when the final answer still sounds confident.
"""

import json
import re
from typing import Any

import httpx

from app.services.llm import get_api_url, get_persisted_provider_and_keys

_FAITHFULNESS_PROMPT = (
    "You will be given an answer and a context. Break the answer down into "
    "a list of atomic factual claims (one discrete fact per claim). For "
    "each claim, decide whether it can be directly verified or inferred "
    "from the context alone.\n\n"
    "Respond with strict JSON only, no other text: "
    '{"claims": [{"claim": "...", "supported": true or false}, ...]}'
)

_CONTEXT_PRECISION_PROMPT = (
    "You will be given a question and a numbered list of retrieved context "
    "chunks, in the order the retrieval system ranked them. For each "
    "chunk, decide whether it is actually relevant and useful for "
    "answering the question.\n\n"
    "Respond with strict JSON only, no other text: "
    '{"relevance": [true or false, ...]} - exactly one boolean per chunk, '
    "in the same order given."
)


_SOURCE_BLOCK_RE = re.compile(r'<source id="\d+">.*?</source>', re.DOTALL)


def split_context_chunks(context: str) -> list[str]:
    """Split the pipeline's built <retrieved_context> block back into one
    string per retrieved chunk, in ranked order - reversing the
    <source id="N">...</source> join done when the prompt was built (see
    rag_engine.py's build_prompt)."""
    return [m.group(0).strip() for m in _SOURCE_BLOCK_RE.finditer(context)]


async def _call_metrics_llm(system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
    """Same active-provider call pattern as evals/judge.py - returns None on
    any failure (missing key, provider error, malformed JSON) so callers can
    tell "couldn't score" apart from a real 0.0."""
    from app.services.storage import get_settings

    persisted = await get_settings()
    provider, api_key = get_persisted_provider_and_keys(persisted)
    if not api_key:
        return None
    model = (persisted.get("geminiModel") if provider == "gemini" else persisted.get("groqModel")) or ""

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 800,
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                get_api_url(provider),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=body,
            )
        if not response.is_success:
            return None
        content = response.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.strip("`").removeprefix("json").strip()
        return json.loads(content)
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError):
        return None


async def compute_faithfulness(answer: str, context: str) -> dict[str, Any]:
    """RAGAS faithfulness: fraction of the answer's atomic claims that are
    directly supported by the retrieved context. score is None (not 0.0)
    when it couldn't be computed at all, so callers can tell that apart
    from a genuinely unfaithful answer."""
    if not answer.strip() or not context.strip():
        return {"score": None, "claims": []}

    result = await _call_metrics_llm(
        _FAITHFULNESS_PROMPT,
        f"Context:\n{context}\n\nAnswer:\n{answer}",
    )
    claims = result.get("claims") if result else None
    if not isinstance(claims, list) or not claims:
        return {"score": None, "claims": []}

    supported = sum(1 for c in claims if c.get("supported") is True)
    return {"score": round(supported / len(claims), 4), "claims": claims}


async def compute_context_precision(question: str, context: str) -> dict[str, Any]:
    """RAGAS context precision (reference-free): average precision over the
    ranked retrieved chunks, judging each chunk's relevance to the question."""
    chunks = split_context_chunks(context)
    if not chunks:
        return {"score": None, "relevance": []}

    numbered = "\n\n".join(f"{i + 1}. {c}" for i, c in enumerate(chunks))
    result = await _call_metrics_llm(
        _CONTEXT_PRECISION_PROMPT,
        f"Question: {question}\n\nRetrieved chunks:\n{numbered}",
    )
    relevance = result.get("relevance") if result else None
    if not isinstance(relevance, list) or len(relevance) != len(chunks):
        return {"score": None, "relevance": []}

    relevance = [bool(r) for r in relevance]
    relevant_count = sum(relevance)
    if relevant_count == 0:
        return {"score": 0.0, "relevance": relevance}

    hits = 0
    precisions = []
    for k, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            hits += 1
            precisions.append(hits / k)
    return {"score": round(sum(precisions) / relevant_count, 4), "relevance": relevance}

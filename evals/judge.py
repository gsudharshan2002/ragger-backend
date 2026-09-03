"""LLM-as-judge for the developer-documentation eval's answer quality.

Grades each answer pass/fail using whichever LLM provider is currently
active (see app/services/llm.py's provider selection - Settings UI choice,
falling back to .env), as a replacement for pure keyword matching. Must be
validated against human grading (see validate_judge.py) before its number
is trusted.
"""

import json
from typing import Any

import httpx

from app.services.llm import get_api_url, get_persisted_provider_and_keys

_JUDGE_SYSTEM_PROMPT = (
    "You are grading whether an AI assistant's answer to a developer-documentation "
    "question is correct and helpful. You will be given the question, the key facts "
    "a good answer should cover, and the assistant's answer.\n\n"
    "Judge the answer holistically - correctness and helpfulness, not just whether "
    "it repeats exact words. An answer that conveys the same facts in different "
    "wording should pass. An answer that is vague, wrong, or refuses despite the "
    "facts being gradeable should fail.\n\n"
    "Do not check two specific things yourself - deterministic code verifies "
    "them separately, and re-litigating them here would be redundant: (1) "
    "whether an endpoint path the answer mentions actually exists in the API "
    "spec, and (2) whether a deprecated symbol the answer mentions is missing "
    "its required migration note. Grade everything else about correctness "
    "and helpfulness as usual - an answer can still fail for being wrong or "
    "unhelpful in ways unrelated to those two checks.\n\n"
    "Respond with strict JSON only, no other text: "
    '{"verdict": "pass" or "fail", "reasoning": "one short sentence"}'
)


def _judge_user_prompt(question: str, answer: str, expected_keywords: list[str]) -> str:
    facts = ", ".join(expected_keywords) if expected_keywords else "(none listed)"
    return (
        f"Question: {question}\n\n"
        f"Key facts a good answer should cover: {facts}\n\n"
        f"Assistant's answer: {answer or '(empty)'}"
    )


async def judge_answer(question: str, answer: str, expected_keywords: list[str]) -> dict[str, Any]:
    """Grade one answer pass/fail. Returns {"verdict", "reasoning"} - verdict is
    None (not "fail") when the judge call itself couldn't be made, so callers
    can tell "graded fail" apart from "couldn't grade"."""
    from app.services.storage import get_settings

    persisted = await get_settings()
    provider, api_key = get_persisted_provider_and_keys(persisted)
    if not api_key:
        env_var = "GEMINI_API_KEY" if provider == "gemini" else "GROQ_API_KEY"
        return {"verdict": None, "reasoning": f"{env_var} not set - judge skipped"}

    model = (
        persisted.get("geminiModel") if provider == "gemini" else persisted.get("groqModel")
    ) or ""

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": _judge_user_prompt(question, answer, expected_keywords)},
        ],
        "temperature": 0,
        "max_tokens": 300,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                get_api_url(provider),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=body,
            )
        if not response.is_success:
            return {"verdict": None, "reasoning": f"judge call failed ({response.status_code})"}

        content = response.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.strip("`").removeprefix("json").strip()
        parsed = json.loads(content)
        verdict = str(parsed.get("verdict", "")).strip().lower()
        if verdict not in ("pass", "fail"):
            return {"verdict": None, "reasoning": f"judge returned unrecognized verdict: {content!r}"}
        return {"verdict": verdict, "reasoning": str(parsed.get("reasoning", ""))}
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError) as e:
        return {"verdict": None, "reasoning": f"judge call errored: {e}"}

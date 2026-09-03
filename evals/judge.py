"""LLM-as-judge for the developer-documentation eval's answer quality.

Grades each answer pass/fail using Groq, as a replacement for pure keyword
matching. Must be validated against human grading (see validate_judge.py)
before its number is trusted.
"""

import json
from typing import Any

import httpx

from app.core.config import settings

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

_JUDGE_SYSTEM_PROMPT = (
    "You are grading whether an AI assistant's answer to a developer-documentation "
    "question is correct and helpful. You will be given the question, the key facts "
    "a good answer should cover, and the assistant's answer.\n\n"
    "Judge the answer holistically - correctness and helpfulness, not just whether "
    "it repeats exact words. An answer that conveys the same facts in different "
    "wording should pass. An answer that is vague, wrong, or refuses despite the "
    "facts being gradeable should fail.\n\n"
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
    api_key = settings.GROQ_API_KEY
    if not api_key:
        return {"verdict": None, "reasoning": "GROQ_API_KEY not set - judge skipped"}

    body = {
        "model": settings.GROQ_MODEL,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": _judge_user_prompt(question, answer, expected_keywords)},
        ],
        "temperature": 0,
        "max_tokens": 150,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                GROQ_API_URL,
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

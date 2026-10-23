"""Stricter keyword scoring for the developer-docs benchmark.

The plain substring check in race_harness._passed has two failure modes seen
in the Week 10 benchmark:
  - short keywords match inside unrelated words ("no" in "know"/"not"/"now"),
  - a refusal that merely restates the question ("no information about the
    submodule entry...") contains the keyword and is scored as a pass.

passed_strict fixes both without touching the dataset's expected keywords:
  - keywords match on word boundaries; keywords of <= 3 chars must match the
    whole word, longer ones may carry a suffix (so the stem "migrat" still
    matches "migrating"),
  - an answer that is a refusal / "not in the documents" never passes.

It is still a keyword check: it cannot tell "204 was replaced by 200" from a
correct answer. Use it for relative comparison, not as proof of correctness.
"""

import re

_REFUSAL_PATTERNS = [
    r"could not find (that|the|any|sufficient|relevant)",
    r"(evidence|documents?|documentation|context|sources?|excerpts?) (does|do|did) not "
    r"(contain|include|mention|provide|cover|have)",
    r"no (relevant |sufficient )?(information|evidence)\b.{0,60}\b(found|available|retrieved|provided)",
    r"cannot (be )?(answer|answered|determined|verified) (this |that )?from",
    r"unable to (find|answer|determine|verify)",
    r"not (present|available|covered|mentioned|found) in the (provided|retrieved)",
    r"insufficient (evidence|information)",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)
_WORD = r"A-Za-z0-9_"


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL_RE.search(answer or ""))


def keyword_present(answer: str, keyword: str) -> bool:
    kw = re.sub(r"\s+", " ", keyword.strip().lower())
    if not kw:
        return True
    text = re.sub(r"\s+", " ", (answer or "").lower())
    pattern = rf"(?<![{_WORD}]){re.escape(kw)}"
    if len(kw) <= 3:
        pattern += rf"(?![{_WORD}])"
    return re.search(pattern, text) is not None


def passed_strict(answer: str, expected_keywords: list[str]) -> bool:
    if is_refusal(answer):
        return False
    return all(keyword_present(answer, kw) for kw in expected_keywords)

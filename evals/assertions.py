"""Deterministic assertions for the GitHub API eval cases.

These replace two criteria the LLM judge used to implicitly fold into its
one "correct and helpful" verdict: whether an endpoint path mentioned in an
answer is real, and whether a deprecated symbol is mentioned without its
required migration note. Both are exact lookups against static fixtures
(github-api/openapi.yaml, github-api/deprecations.md) - a parser and a
lookup never have an off day, so there's no reason to pay an LLM to guess.
"""

import re
from pathlib import Path
from typing import Any

import yaml

GITHUB_API_DIR = Path(__file__).resolve().parent / "github-api"
OPENAPI_PATH = GITHUB_API_DIR / "openapi.yaml"
DEPRECATIONS_PATH = GITHUB_API_DIR / "deprecations.md"

_BACKTICK_RE = re.compile(r"`([^`]+)`")
_ENDPOINT_TOKEN_RE = re.compile(r"/[A-Za-z][A-Za-z0-9_\-{}]*(?:/[A-Za-z0-9_\-{}]+)*")
_STOPWORDS = {
    "read", "from", "instead", "with", "after", "that", "this", "have",
    "will", "your", "note", "code", "field", "endpoint", "property",
    "removed", "there", "replacement", "direct",
}


def load_openapi_paths(path: Path = OPENAPI_PATH) -> list[str]:
    """Every endpoint path template declared under `paths:` in the OpenAPI
    spec, e.g. "/repos/{owner}/{repo}/pulls/{pull_number}"."""
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    return list((spec or {}).get("paths", {}).keys())


def load_deprecations(path: Path = DEPRECATIONS_PATH) -> list[dict[str, Any]]:
    """Parse the `| symbol | endpoint | migration_note |` table in
    deprecations.md into [{symbols, endpoint, migration_note}, ...]."""
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        inner = line.strip("|").strip()
        if set(inner.replace("|", "").strip()) <= {"-", " "}:
            continue  # the "|---|---|---|" separator row
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 3 or cells[0].lower() == "symbol":
            continue  # header row or malformed row
        symbols = _BACKTICK_RE.findall(cells[0]) or [cells[0]]
        entries.append({"symbols": symbols, "endpoint": cells[1], "migration_note": cells[2]})
    return entries


def _path_to_regex(path_template: str) -> re.Pattern:
    """"/repos/{owner}/{repo}" -> a regex matching that shape with any
    concrete value standing in for each {param}."""
    escaped = re.escape(path_template)
    pattern = re.sub(r"\\\{[^}]+\\\}", r"[^/\\s]+", escaped)
    return re.compile(f"^{pattern}$")


def check_unknown_endpoints(answer: str, openapi_paths: list[str]) -> list[str]:
    """Endpoint-path-shaped tokens mentioned in `answer` that don't match any
    real path in the OpenAPI spec. Empty list = every endpoint checks out."""
    path_regexes = [_path_to_regex(p) for p in openapi_paths]
    unknown = []
    for token in _ENDPOINT_TOKEN_RE.findall(answer):
        if token in openapi_paths or any(rx.match(token) for rx in path_regexes):
            continue
        unknown.append(token)
    return unknown


def _significant_words(text: str) -> list[str]:
    cleaned = re.sub(r"[`*_]", "", text).lower()
    words = re.findall(r"[a-z][a-z0-9_.\-]{3,}", cleaned)
    return [w for w in words if w not in _STOPWORDS]


def check_deprecated_without_migration_note(
    answer: str, deprecations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Deprecation entries whose symbol is mentioned in `answer` with no
    trace of the required migration guidance. Empty list = clean - either
    the symbol isn't discussed at all, or it's discussed with its note."""
    lowered = answer.lower()
    violations = []
    for entry in deprecations:
        mentioned = any(
            re.search(r"\b" + re.escape(symbol.lower()) + r"\b", lowered)
            for symbol in entry["symbols"]
        )
        if not mentioned:
            continue
        signal_words = _significant_words(entry["migration_note"])
        if not any(word in lowered for word in signal_words):
            violations.append(entry)
    return violations

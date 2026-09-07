import re
from typing import Optional

from app.core.logging_config import get_logger

logger = get_logger(__name__)

MAX_KEYWORDS_PER_CHUNK = 8
NUMERIC_ONLY = re.compile(r"^[\d.,%/-]+$")


def _clean_keyword(keyword: str) -> Optional[str]:
    value = re.sub(r"[\"'\u2018\u2019\u201c\u201d]", "", keyword).strip()
    if not value:
        return None
    if len(value) < 2:
        return None
    return value


def _extract_keywords_local(texts: list[str]) -> list[list[str]]:
    """Offline keyword extraction: per-chunk top terms from a TF-IDF matrix
    fit over this document's chunks (scikit-learn, already a dependency).
    No provider calls, no network - always fully local."""
    results: list[list[str]] = [[] for _ in texts]
    if not texts:
        return results

    cleaned = [t.strip() or " " for t in texts]
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            stop_words="english",
            token_pattern=r"\b\w{3,}\b",
            lowercase=True,
            max_features=500,
        )
        matrix = vectorizer.fit_transform(cleaned)
        feature_names = vectorizer.get_feature_names_out()
        dense = matrix.toarray()

        for i, row in enumerate(dense):
            ranked = sorted(zip(feature_names, row), key=lambda p: p[1], reverse=True)
            keywords: list[str] = []
            for token, score in ranked:
                if len(keywords) >= MAX_KEYWORDS_PER_CHUNK:
                    break
                if score <= 0:
                    continue
                if NUMERIC_ONLY.match(token):
                    continue
                value = _clean_keyword(token)
                if value:
                    keywords.append(value)
            results[i] = keywords
    except Exception as e:
        logger.error(f"Local keyword extraction failed: {e}")
        return [[] for _ in texts]

    return results


async def extract_keywords(texts: list[str]) -> list[list[str]]:
    """Keyword extraction for a document's chunks. Purely local (TF-IDF), never
    raises and never calls an LLM - an empty keyword list just degrades BM25 to
    its pre-keyword behavior."""
    try:
        return _extract_keywords_local(texts)
    except Exception as e:
        logger.error(f"Keyword extraction failed entirely: {e}")
        return [[] for _ in texts]
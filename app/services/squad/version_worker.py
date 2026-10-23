"""Week 10: Version / Deprecation Worker.

A focused specialist responsible for:
- Resolving SDK/package versions where possible.
- Checking deprecation information through existing RAG tools and knowledge sources.
- Identifying relevant version constraints and migration information.
- Returning evidence, source references, and an explicit uncertainty/error state.
- Reporting token usage and execution metadata.

Reuses check_deprecation and retrieval services from agent_tools.py.
"""

import asyncio
import re
import time
from typing import Any

from app.services.agent_tools import ApiVersion, check_deprecation, search_knowledge_base
from app.services.squad.fault_injection import should_inject_fault, InjectedHTTP500Error
from app.services.squad.telemetry import estimate_tokens

_VERSION_RE = re.compile(r"\b(2022-11-28|2026-03-10)\b")


def resolve_api_version(query: str, default: str = ApiVersion.CURRENT.value) -> str:
    """Pick the API version to check from the question itself.

    An explicit supported version date wins; otherwise the question is about
    the "current" version. Dates the knowledge base does not index are ignored
    rather than guessed at.
    """
    match = _VERSION_RE.search(query or "")
    return match.group(1) if match else default


def format_chunks(chunks: list) -> str:
    """Render retrieved chunks as readable, citable evidence (full chunk text)."""
    lines = []
    for c in chunks:
        name = getattr(c, "document_name", None) or (c.get("document", "") if isinstance(c, dict) else "")
        page = getattr(c, "page", None) if not isinstance(c, dict) else c.get("page", 0)
        section = getattr(c, "section", None) if not isinstance(c, dict) else c.get("section")
        text = getattr(c, "content", None) if not isinstance(c, dict) else c.get("content", c.get("excerpt", ""))
        lines.append(f"[{name} p.{page}{' / ' + section if section else ''}] {text}")
    return "\n".join(lines)


class VersionDeprecationWorker:
    name = "version_deprecation"

    def __init__(self, timeout_seconds: float = 30.0, max_retries: int = 2):
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    async def run(
        self,
        query: str,
        case_id: str,
        endpoint_or_feature: str | None = None,
        api_version: str | None = None,
    ) -> dict[str, Any]:
        """Run the version/deprecation check.

        Returns a dict with:
          - evidence: summary of findings
          - sources: list of source references
          - uncertainty: explicit uncertainty/error state
          - input_tokens, output_tokens, total_tokens
          - latency_ms
          - status: success | failure | error
        """
        start = time.time()
        input_tokens = 0
        output_tokens = 0

        # Determine what to check
        feature = endpoint_or_feature or query
        api_version = api_version or resolve_api_version(query)

        # Controlled fault injection: HTTP 500 for the designated case
        if should_inject_fault(case_id, "version_deprecation"):
            input_tokens = estimate_tokens(feature + " " + api_version)
            output_tokens = estimate_tokens("Injected HTTP 500 error")
            latency_ms = int((time.time() - start) * 1000)
            return {
                "evidence": "",
                "sources": [],
                "uncertainty": "version_check_failed",
                "error": f"HTTP 500: version/deprecation check failed for {feature}@{api_version}",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "latency_ms": latency_ms,
                "status": "failure",
                "fault_injected": True,
            }

        result: dict[str, Any] = {}
        error: str | None = None

        try:
            result = await asyncio.wait_for(
                check_deprecation(feature, ApiVersion(api_version)),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            error = f"version/deprecation check timed out after {self.timeout_seconds}s"
        except InjectedHTTP500Error as e:
            error = str(e)
        except Exception as e:
            error = str(e)

        if error:
            input_tokens = estimate_tokens(feature + " " + api_version)
            output_tokens = estimate_tokens(error)
            latency_ms = int((time.time() - start) * 1000)
            return {
                "evidence": "",
                "sources": [],
                "uncertainty": "version_check_failed",
                "error": error,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "latency_ms": latency_ms,
                "status": "failure",
                "fault_injected": False,
            }

        chunks = result.get("raw_chunks", [])
        summary = result.get("summary", {})

        # search_knowledge_base reports retrieval failures in-band; surface
        # them as a failure instead of passing them off as "no evidence".
        retrieval_error = summary.get("error")
        if retrieval_error:
            return self._failure(feature, api_version, f"retrieval failed: {retrieval_error}", start)

        evidence_text = format_chunks(chunks)
        sources = [
            {
                "document": getattr(c, "document_name", ""),
                "page": getattr(c, "page", 0),
                "section": getattr(c, "section", None),
                "chunk_id": getattr(c, "id", None),
            }
            for c in chunks
        ]

        input_tokens = estimate_tokens(feature + " " + api_version)
        output_tokens = estimate_tokens(evidence_text)
        latency_ms = int((time.time() - start) * 1000)

        has_data = bool(chunks)
        return {
            "evidence": evidence_text,
            "sources": sources,
            "api_version": api_version,
            "uncertainty": "none" if has_data else "no_deprecation_info_found",
            "error": None if has_data else "no_deprecation_info_found",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency_ms": latency_ms,
            "status": "success" if has_data else "error",
            "fault_injected": False,
            "chunk_count": len(chunks),
        }

    def _failure(self, feature: str, api_version: str, error: str, start: float) -> dict[str, Any]:
        input_tokens = estimate_tokens(feature + " " + api_version)
        output_tokens = estimate_tokens(error)
        return {
            "evidence": "",
            "sources": [],
            "api_version": api_version,
            "uncertainty": "version_check_failed",
            "error": error,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency_ms": int((time.time() - start) * 1000),
            "status": "failure",
            "fault_injected": False,
        }

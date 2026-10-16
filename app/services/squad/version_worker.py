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
import time
from typing import Any

from app.services.agent_tools import ApiVersion, check_deprecation, search_knowledge_base
from app.services.squad.fault_injection import should_inject_fault, InjectedHTTP500Error
from app.services.squad.telemetry import estimate_tokens


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
        api_version: str = "2026-03-10",
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

        evidence = summary.get("chunks", [])
        if isinstance(evidence, list):
            evidence_text = " ".join(str(c) for c in evidence)
        else:
            evidence_text = str(evidence)

        sources = []
        for c in chunks:
            sources.append({
                "document": c.document_name if hasattr(c, "document_name") else c.get("document", ""),
                "page": c.page if hasattr(c, "page") else c.get("page", 0),
                "section": c.section if hasattr(c, "section") else c.get("section"),
            })

        input_tokens = estimate_tokens(feature + " " + api_version)
        output_tokens = estimate_tokens(evidence_text)
        latency_ms = int((time.time() - start) * 1000)

        has_data = bool(chunks) and summary.get("chunk_count", 0) > 0
        status = "success" if has_data else "error"

        return {
            "evidence": evidence_text,
            "sources": sources,
            "uncertainty": "none" if has_data else "no_deprecation_info_found",
            "error": None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency_ms": latency_ms,
            "status": status,
            "fault_injected": False,
            "chunk_count": summary.get("chunk_count", 0),
        }

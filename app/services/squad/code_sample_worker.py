"""Week 10: Code-Sample Worker.

A focused specialist responsible for:
- Retrieving relevant developer documentation.
- Finding code examples compatible with the requested SDK version.
- Returning code samples with their supporting source evidence.
- Reporting unsupported assumptions and missing version information.
- Returning execution metadata and token usage.

Does not fabricate version-specific code examples when compatibility cannot be verified.
"""

import asyncio
import time
from typing import Any

from app.services.agent_tools import search_knowledge_base
from app.services.squad.telemetry import estimate_tokens


class CodeSampleWorker:
    name = "code_sample"

    def __init__(self, timeout_seconds: float = 30.0):
        self.timeout_seconds = timeout_seconds

    async def run(
        self,
        query: str,
        case_id: str,
        sdk_version: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve documentation and code samples for the query.

        Returns a dict with:
          - evidence: retrieved documentation text
          - code_samples: list of code samples with source references
          - sources: list of source references
          - unsupported_assumptions: list of assumptions that cannot be verified
          - input_tokens, output_tokens, total_tokens
          - latency_ms
          - status: success | error
        """
        start = time.time()
        input_tokens = estimate_tokens(query + (sdk_version or ""))
        output_tokens = 0

        try:
            result = await asyncio.wait_for(
                search_knowledge_base(query),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            output_tokens = estimate_tokens("timeout")
            latency_ms = int((time.time() - start) * 1000)
            return {
                "evidence": "",
                "code_samples": [],
                "sources": [],
                "unsupported_assumptions": [f"code_sample_retrieval_timed_out_after_{self.timeout_seconds}s"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "latency_ms": latency_ms,
                "status": "failure",
                "error": "retrieval timed out",
            }
        except Exception as e:
            output_tokens = estimate_tokens(str(e))
            latency_ms = int((time.time() - start) * 1000)
            return {
                "evidence": "",
                "code_samples": [],
                "sources": [],
                "unsupported_assumptions": [f"retrieval_error: {e}"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "latency_ms": latency_ms,
                "status": "failure",
                "error": str(e),
            }

        chunks = result.get("raw_chunks", [])
        summary = result.get("summary", {})

        evidence_chunks = summary.get("chunks", [])
        evidence_text = " ".join(str(c) for c in evidence_chunks) if isinstance(evidence_chunks, list) else str(evidence_chunks)
        output_tokens = estimate_tokens(evidence_text)

        sources = []
        code_samples = []
        unsupported = []

        for c in chunks:
            doc_name = c.document_name if hasattr(c, "document_name") else c.get("document", "")
            page = c.page if hasattr(c, "page") else c.get("page", 0)
            section = c.section if hasattr(c, "section") else c.get("section")
            content = c.content if hasattr(c, "content") else c.get("content", "")

            sources.append({
                "document": doc_name,
                "page": page,
                "section": section,
            })

            # Heuristic: flag code-looking chunks as code samples, but only
            # when we have a source to back them. Never fabricate version
            # compatibility.
            if any(k in content.lower() for k in ["def ", "class ", "import ", "```", "curl", "sdk", "client"]):
                code_samples.append({
                    "snippet": content[:500],
                    "source_document": doc_name,
                    "source_page": page,
                    "version_verified": bool(sdk_version),
                    "compatibility_note": (
                        f"compatibility with SDK {sdk_version} not verified by this worker"
                        if sdk_version
                        else "no SDK version specified - compatibility unknown"
                    ),
                })

        if not chunks:
            unsupported.append("no_documentation_found_for_query")

        if sdk_version and not any(
            sdk_version in (s.get("document", "") or "") for s in sources
        ):
            unsupported.append(f"sdk_version_{sdk_version}_compatibility_unverified")

        latency_ms = int((time.time() - start) * 1000)
        status = "success" if chunks else "error"

        return {
            "evidence": evidence_text,
            "code_samples": code_samples,
            "sources": sources,
            "unsupported_assumptions": unsupported,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency_ms": latency_ms,
            "status": status,
            "chunk_count": summary.get("chunk_count", 0),
        }

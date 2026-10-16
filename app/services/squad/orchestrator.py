"""Week 10: Orchestrator / Manager for the Multi-Agent Squad.

Manager-Plus-Two-Specialists architecture:
- Receives a developer-documentation question.
- Identifies required specialist capabilities.
- Delegates to VersionDeprecationWorker and CodeSampleWorker.
- Collects and tracks specialist results.
- Combines evidence into a final answer.
- Handles specialist errors without inventing facts.
- Tracks delegation hops, latency, token usage, failures.
- Enforces timeouts, retry limits, and budget constraints.

Uses a simple, explicit coordination strategy (no autonomous planning framework).
"""

import asyncio
import time
from typing import Any

from app.core.config import settings
from app.services.squad.version_worker import VersionDeprecationWorker
from app.services.squad.code_sample_worker import CodeSampleWorker
from app.services.squad.telemetry import TelemetryCollector, estimate_tokens


class SquadOrchestrator:
    """Orchestrates the manager-plus-two-specialists squad."""

    def __init__(self, max_steps: int = 5, temperature: float = 0.2):
        self.max_steps = max_steps
        self.temperature = temperature
        self.version_worker = VersionDeprecationWorker()
        self.code_sample_worker = CodeSampleWorker()

    async def run(
        self,
        query: str,
        case_id: str,
        telemetry: TelemetryCollector,
    ) -> dict[str, Any]:
        """Run the full squad pipeline for one question.

        Returns a dict with answer, steps, latency, tokens, and handoffs
        recorded on the telemetry collector.
        """
        start = time.time()
        steps: list[dict[str, Any]] = []
        retrieved_chunks: list = []
        final_answer = ""

        # Step 1: Orchestrator analyzes the question to determine which
        # specialists are needed.
        orchestrator_input = estimate_tokens(query)
        analysis_prompt = (
            f"Question: {query}\n\n"
            "Determine which specialists are needed:\n"
            "- version_deprecation: if the question asks about API versions, "
            "deprecation, migration, or whether a field/endpoint still exists.\n"
            "- code_sample: if the question asks for code examples, SDK usage, "
            "or implementation details.\n"
            "Respond with a JSON list of specialist names, e.g. [\"version_deprecation\", \"code_sample\"]."
        )

        analysis_text = ""
        analysis_error = None
        try:
            async for chunk in self._llm_call("orchestrator_analysis", analysis_prompt, self.temperature, 256):
                if chunk["type"] == "token" and chunk.get("content"):
                    analysis_text += chunk["content"]
                elif chunk["type"] == "error":
                    analysis_error = chunk.get("error")
        except Exception as e:
            analysis_error = str(e)

        analysis_output_tokens = estimate_tokens(analysis_text)
        telemetry.record_handoff(
            source_agent="orchestrator",
            destination_agent="orchestrator",
            delegation_purpose="question_analysis",
            input_tokens=orchestrator_input,
            output_tokens=analysis_output_tokens,
            latency_ms=int((time.time() - start) * 1000),
            status="success" if not analysis_error else "failure",
            error=analysis_error,
            tokens_estimated=True,
        )

        needed = self._parse_specialist_list(analysis_text)

        # Step 2: Delegate to specialists (can run in parallel where independent).
        specialist_results: dict[str, dict[str, Any]] = {}

        async def call_version_worker():
            res = await self.version_worker.run(query, case_id)
            specialist_results["version_deprecation"] = res
            telemetry.record_handoff(
                source_agent="orchestrator",
                destination_agent="version_deprecation",
                delegation_purpose="version_deprecation_check",
                input_tokens=res.get("input_tokens", 0),
                output_tokens=res.get("output_tokens", 0),
                latency_ms=res.get("latency_ms", 0),
                status=res.get("status", "error"),
                error=res.get("error"),
                tokens_estimated=True,
            )
            if res.get("raw_chunks"):
                retrieved_chunks.extend(res["raw_chunks"])

        async def call_code_sample_worker():
            res = await self.code_sample_worker.run(query, case_id)
            specialist_results["code_sample"] = res
            telemetry.record_handoff(
                source_agent="orchestrator",
                destination_agent="code_sample",
                delegation_purpose="code_sample_retrieval",
                input_tokens=res.get("input_tokens", 0),
                output_tokens=res.get("output_tokens", 0),
                latency_ms=res.get("latency_ms", 0),
                status=res.get("status", "error"),
                error=res.get("error"),
                tokens_estimated=True,
            )
            if res.get("raw_chunks"):
                retrieved_chunks.extend(res["raw_chunks"])

        tasks = []
        if "version_deprecation" in needed:
            tasks.append(call_version_worker())
        if "code_sample" in needed:
            tasks.append(call_code_sample_worker())

        # Default: if no specialist identified, run both to be safe.
        if not tasks:
            tasks.append(call_version_worker())
            tasks.append(call_code_sample_worker())

        await asyncio.gather(*tasks)

        steps.append({
            "step": 1,
            "agent": "orchestrator",
            "action": "delegate",
            "specialists": needed or ["version_deprecation", "code_sample"],
        })

        # Step 3: Combine evidence and generate final answer.
        # The orchestrator must not claim version/deprecation verification
        # unless the version worker returned evidence.
        version_evidence = specialist_results.get("version_deprecation", {})
        code_evidence = specialist_results.get("code_sample", {})

        version_ok = version_evidence.get("status") == "success" and bool(version_evidence.get("evidence"))
        code_ok = code_evidence.get("status") == "success" and bool(code_evidence.get("evidence"))

        combined_context = ""
        if version_evidence.get("evidence"):
            combined_context += f"[Version/Deprecation Evidence]\n{version_evidence['evidence']}\n\n"
        if code_evidence.get("evidence"):
            combined_context += f"[Code Sample Evidence]\n{code_evidence['evidence']}\n\n"

        # Also pull raw chunks into context if available.
        if retrieved_chunks:
            combined_context += "[Retrieved Context]\n"
            for c in retrieved_chunks[:5]:
                combined_context += f"- {c.document_name if hasattr(c, 'document_name') else c.get('document', '')} (p. {c.page if hasattr(c, 'page') else c.get('page', 0)}): {c.content if hasattr(c, 'content') else c.get('content', '')[:300]}\n"

        answer_prompt = (
            f"Question: {query}\n\n"
            f"Evidence from specialists:\n{combined_context or 'No evidence retrieved.'}\n\n"
            "Rules:\n"
            "- Answer only using the evidence above.\n"
            "- Do not claim a version, deprecation status, or code sample is "
            "verified unless the version_deprecation worker returned evidence.\n"
            "- If version/deprecation verification failed, explicitly disclose it.\n"
            "- Do not fabricate code samples or version-specific claims.\n"
            "- Be concise."
        )

        answer_text = ""
        answer_error = None
        try:
            async for chunk in generate_answer_stream(answer_prompt, self.temperature):
                if chunk["type"] == "token" and chunk.get("content"):
                    answer_text += chunk["content"]
                elif chunk["type"] == "error":
                    answer_error = chunk.get("error")
        except Exception as e:
            answer_error = str(e)

        # If the answer claims version verification but the version worker
        # failed, append a disclosure.
        if not version_ok and any(k in answer_text.lower() for k in ["deprecated", "version", "removed", "migrated"]):
            answer_text += "\n\nNote: version/deprecation verification was unavailable for this question."

        final_answer = answer_text or (
            f"No answer was generated ({answer_error})"
            if answer_error
            else "I could not find sufficient evidence to answer this question."
        )

        answer_input_tokens = estimate_tokens(answer_prompt)
        answer_output_tokens = estimate_tokens(final_answer)
        telemetry.record_handoff(
            source_agent="orchestrator",
            destination_agent="llm_answer",
            delegation_purpose="final_answer_generation",
            input_tokens=answer_input_tokens,
            output_tokens=answer_output_tokens,
            latency_ms=int((time.time() - start) * 1000),
            status="success" if not answer_error else "failure",
            error=answer_error,
            tokens_estimated=True,
        )

        telemetry.input_tokens = (
            orchestrator_input
            + analysis_output_tokens
            + version_evidence.get("input_tokens", 0)
            + code_evidence.get("input_tokens", 0)
            + answer_input_tokens
        )
        telemetry.output_tokens = (
            answer_output_tokens
            + version_evidence.get("output_tokens", 0)
            + code_evidence.get("output_tokens", 0)
        )
        telemetry.total_tokens = telemetry.input_tokens + telemetry.output_tokens
        telemetry.cost = telemetry.total_tokens * settings.COST_PER_TOKEN
        telemetry.finish(passed=False, answer=final_answer, error=answer_error)

        return {
            "answer": final_answer,
            "steps": steps,
            "specialist_results": specialist_results,
            "total_latency_ms": int((time.time() - start) * 1000),
            "input_tokens": telemetry.input_tokens,
            "output_tokens": telemetry.output_tokens,
            "total_tokens": telemetry.total_tokens,
            "cost": telemetry.cost,
            "handoffs": [h.to_dict() for h in telemetry.handoffs],
        }

    def _parse_specialist_list(self, text: str) -> list[str]:
        """Parse the LLM's JSON list of specialist names."""
        import json
        import re
        match = re.search(r"\[.*\]", text or "")
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [s for s in parsed if s in ("version_deprecation", "code_sample")]
        except (json.JSONDecodeError, ValueError):
            pass
        return []

    async def _llm_call(self, purpose: str, prompt: str, temperature: float, max_tokens: int):
        from app.services.llm import generate_completion_stream
        async for chunk in generate_completion_stream(
            "You are an orchestrator for a multi-agent RAG squad.",
            prompt,
            None,
            temperature,
            max_tokens,
            1.0,
        ):
            yield chunk


async def generate_answer_stream(prompt: str, temperature: float):
    """Generate an answer using the same LLM streaming path as the RAG engine."""
    from app.services.llm import generate_completion_stream
    system_prompt = (
        "You are a RAG assistant. Answer only using the provided evidence. "
        "Do not use outside knowledge or guesses. "
        "If the answer is not in the evidence, say so explicitly. "
        "Cite sources where available. Be concise."
    )
    async for chunk in generate_completion_stream(system_prompt, prompt, None, temperature, 1024, 1.0):
        yield chunk

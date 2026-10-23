WEEK 12 CAPSTONE: SDK DOCS ANSWER ENGINE
1. Role and objective
Act as a senior Python backend engineer responsible for integrating, hardening, evaluating, and packaging the existing ragger-backend repository for the Week 12 capstone.

Your task is to implement the actual working application, not just provide a plan or generate documentation describing features that do not exist.

The final product must be an SDK Documentation Answer Engine that reliably answers developer questions for the correct SDK version, detects deprecated APIs, produces version-compatible code samples, cites authoritative documentation, and exposes enough telemetry to investigate failures and defend engineering decisions.

The completed application must be runnable by a new developer using a documented command, and it must support the four-part technical defense described in this prompt.

2. Existing project context
The project is a FastAPI/Python RAG backend replacing a Node.js/TypeScript backend while preserving the existing Next.js frontend contract.

Relevant existing components include:

app/main.py — FastAPI entry point, middleware, CORS, rate limiting, lifespan.
app/api/v1/router.py and API route modules — REST API.
app/services/rag_engine.py — vector search, BM25, RRF, reranking, MMR, execute_rag().
app/services/agent_engine.py — ReAct agent, SSE streaming, optional MCP mode.
app/services/agent_loop.py — alternative agent loop and budget enforcement.
app/services/agent_tools.py — retrieval, deprecation checking, answer generation, tool registry.
app/mcp_server.py — RAG tools MCP server.
app/mcp_package_registry.py — package registry MCP server.
app/services/mcp_client.py — dynamic MCP tool discovery and invocation.
app/core/config.py — configuration and provider settings.
app/models/schemas.py — Pydantic API schemas.
evals/developer_docs_cases.json — existing Week 6 developer documentation evaluation data.
app/data/traces.jsonl — existing trace records.
Existing Week 7, 8, 9, 10, and 11 artifacts, tests, and scripts where present.
The Week 12 implementation must inspect the actual working tree before deciding which earlier features exist. Do not assume previous weeks' implementations are complete simply because they were described in earlier requirements.

3. Mandatory coding workflow
Follow this workflow strictly.

Phase A — Inspect and implement
Inspect the repository structure, current branch, working-tree status, existing API contracts, configuration, evaluation datasets, trace schemas, MCP implementation, and prior-week artifacts relevant to integration.
Determine the smallest set of changes needed to integrate the existing components.
Identify integration seams and record their status as fixed, accepted, or ran-out-of-time.
Implement the required integration and application changes.
Complete the observability, evaluation, trace propagation, packaging, and artifact-generation code before running tests or benchmarks.
Do not run tests, benchmarks, evaluations, linting, type checks, or intermediate verification while implementation is in progress.

Do not repeatedly inspect files, perform unrelated refactoring, redesign working components, or introduce unnecessary dependencies.

Phase B — Verify after implementation
Once all implementation work is complete:

Run the relevant existing and new tests.
Fix integration failures caused by the implementation.
Rerun affected checks.
Verify the single-command execution path.
Run the integrated evaluations and required experiments.
The deliberate RED → GREEN experiment must preserve the correct order: reproduce and capture the baseline failure before applying the targeted fix, then demonstrate the passing result afterward.

Do not manufacture historical output, test failures, evaluation scores, timings, costs, or peer reviews.

If an external service, API key, SDK installation, or human reviewer is unavailable, complete the work possible and document the exact limitation.

4. Build one integrated application
Unify the existing RAG pipeline, agent loop, tools, MCP servers, logging, and evaluation suite into a coherent application.

The intended execution flow is:

Receive a developer question and identify the caller's SDK version.
Validate or resolve the target version using reliable evidence.
Retrieve documentation constrained to that version.
Check relevant API symbols and deprecation information.
Generate an answer using the retrieved evidence.
Validate the proposed code sample against the target version.
Return the answer with versioned citations, deprecation warnings, and any applicable migration guidance.
Record one unified trace spanning retrieval, model calls, tools, and MCP operations.
Prefer a minimal integration layer over a large rewrite of the existing architecture.

Single entry point
Provide one clearly documented entry point that supports the main application workflow and the required evaluation commands.

For example, use an existing project CLI or add a lightweight entry point with commands for:

Running the application.
Running the integrated evaluation suite.
Running the trajectory evaluation.
Exporting or inspecting traces.
Running the failure-to-test demonstration.
The exact command structure should follow the repository's existing conventions.

A stranger must be able to clone the repository, configure required credentials, install dependencies, initialize any required local data, and execute a documented command to obtain an answer.

Do not claim a single-command setup if mandatory configuration or initialization steps are missing from the README.

Preserve the existing FastAPI endpoints, request/response schemas, SSE streaming, MCP functionality, and frontend compatibility unless a change is essential and backward-compatible.

5. SDK version resolution and safe answers
The application serves SDK users across legacy v2.x and current v3.0 versions.

Breaking changes include the example transition from client.query() to client.search() and the introduction of required timeout arguments.

Use actual repository documentation and SDK behavior to verify the details. Treat these examples as requirements to investigate, not as permission to invent undocumented signatures.

Version resolution
Resolve the caller's SDK version before version-specific retrieval and answer generation.

Use the following priority where applicable:

Explicit version supplied in the request.
Trusted package or environment metadata available to the application.
A documented clarification flow when the version cannot be determined safely.
Do not silently infer the installed SDK version from the question's wording or use the newest version by default.

Record the version source and confidence or resolution status in the trace.

If the version remains unknown and a breaking API difference could change the answer, ask for clarification or provide a clearly labeled version-neutral explanation. Do not present a version-specific code sample as verified.

Version-scoped RAG
Enforce metadata filtering at the retrieval boundary.

Filter documents using the resolved target version.
Exclude incompatible version-specific snippets from the candidate context.
Preserve source identifiers, version metadata, and document provenance.
Keep the existing retrieval pipeline's ranking and reranking behavior where appropriate.
Ensure that later agent or MCP calls cannot bypass the version constraint unintentionally.
Version filtering must be a functional constraint, not merely a prompt instruction.

Versioned citations
Citations must identify the source and applicable SDK version.

Return stable source references or clickable links where actual URLs are available. Include relevant version information and document sections or chunk identifiers where possible.

Never fabricate a documentation URL, section, source identifier, or claim that a citation proves something it does not.

Deprecation and migration warnings
Surface relevant deprecations proactively, even if the user does not explicitly ask whether an API is deprecated.

When a breaking change affects the caller's version, explain:

The relevant version.
The deprecated or incompatible API.
The supported alternative, if verified.
The migration requirement.
Any required arguments or configuration.
The supporting documentation.
Do not recommend a v3-only signature to a v2 caller without clearly distinguishing it as a migration option rather than a compatible solution.

6. Integrate the agent loop and three or more tools
Integrate the existing agent implementation with the version-scoped retriever.

The agent must have at least these three functional capabilities:

Version-scoped documentation search — retrieves evidence only for the target version.
Changelog/deprecation lookup — checks breaking changes, deprecations, and migration information.
Version-specific symbol check — checks whether a method or symbol is supported in the target SDK version.
Reuse existing tools and extend their schemas as needed.

The symbol checker must use a trustworthy source, such as versioned SDK metadata, source code, installed SDK introspection where appropriate, or authoritative documentation. It must not simply accept a symbol because the language model says it exists.

The agent should select tools according to the question and evidence required. Record tool-choice decisions for trajectory evaluation.

Avoid duplicating the retrieval pipeline or building an unnecessarily complicated planning framework.

7. MCP capability: expose and consume
Both sides of MCP integration must work.

Expose
Expose version-scoped documentation search through an MCP server using the repository's existing MCP infrastructure.

The MCP tool should accept the required question or query and SDK version, apply the same version filtering as the main application, and return source identifiers, versions, and supporting evidence.

Consume
Demonstrate that the application or a separate MCP client can discover the tool and invoke it through the actual configured MCP transport.

Preserve the existing HTTP Streamable MCP implementation where applicable.

The MCP call must propagate the original trace ID and record its own span, latency, outcome, and measurable token/cost information.

A locally mocked function call is not proof of live MCP integration.

If the reviewer must run a separate process, provide exact commands for starting the server and invoking it from a separate client. Document any unavailable external-squad verification honestly.

8. Unified telemetry across agent and MCP hops
Every execution must have one trace ID that survives all relevant layers:

Incoming API request.
Retrieval.
Reranking.
Agent model calls.
Tool executions.
Orchestrator and specialist calls, if the multi-agent implementation exists.
Outgoing MCP requests.
MCP server tool execution.
Final response assembly.
Record a parent/child span relationship where appropriate.

Each relevant span should capture:

Trace ID and span ID.
Component or agent identity.
Stage name.
Start time and latency.
Outcome and error status.
Input/output token usage for model calls.
Cost in USD where measurable.
Retrieval query and context chunk IDs where applicable.
Tool name and sanitized arguments.
Prompt version and model identifier.
Retry or fallback behavior.
Redact credentials, access tokens, API keys, authorization headers, and other sensitive information before writing logs.

Avoid logging unnecessary personal information. Preserve enough privacy-safe information to reproduce and investigate a failure.

Use the existing Week 11 tracing system where possible. Do not build a competing logging architecture unless the existing implementation cannot support the required behavior.

9. Integrated evaluation suite: 25–40 cases
Build a single reproducible evaluation command covering the complete answer engine.

Target 25–40 benchmark cases. Reuse the existing Week 6 evaluation dataset and relevant regression cases from later weeks.

Include a representative mix of:

Explicit v2 questions.
Explicit v3 questions.
Breaking API changes.
Deprecated methods.
Required argument checks.
Version-specific code samples.
Incorrect-source and outdated-documentation cases.
Ambiguous or missing-version requests.
Retrieval failures or insufficient evidence.
MCP or tool execution failures.
Existing regression cases.
Preserve existing case IDs and expected-answer criteria where possible.

If the repository lacks enough suitable cases, create additional documented cases grounded in actual SDK documentation. Do not manufacture expected answers or change assertions simply to improve results.

Deterministic assertions first
Run deterministic checks before using any LLM judge.

Examples include:

The resolved version matches the expected version.
Retrieved chunks satisfy the target-version filter.
Required symbols and arguments are present.
Deprecated methods are not recommended as compatible APIs when a verified replacement exists.
Citations refer to real retrieved sources.
Required migration warnings are included.
Code samples meet the applicable validation criteria.
Use LLM-based evaluation for semantic correctness that deterministic assertions cannot reliably establish.

Record deterministic results and LLM judge results separately.

Judge agreement
Measure whether the LLM judge agrees with ground-truth human labels on at least 85% of the labeled comparison cases.

Requirements:

Maintain a set of independently human-labeled examples.
Preserve the ground-truth labels.
Use the same documented judge criteria across the evaluation.
Calculate agreement as the number of matching judgments divided by the number of comparable labeled examples.
Report the sample size and disagreements.
Do not use the LLM judge's own output as the ground truth.
Do not silently change labels or judge instructions after seeing results to force agreement.
If fewer human labels are available than required, prepare the labeling workflow and report the measured agreement on the available sample. Explicitly state that the 85% requirement remains unverified until the required human labels are obtained.

Required evaluation output
Report:

Dataset size and case IDs.
Deterministic pass rate.
Overall correctness pass rate.
LLM judge agreement and sample size.
Per-case failures.
Wrong-version code sample rate.
Relevant citation and deprecation failures.
Model and evaluator configuration.
Evaluation limitations.
The integrated suite must execute through the documented entry point.

10. Integrated trajectory evaluation: 10 or more cases
Evaluate at least 10 cases for agent trajectory quality.

For each case, capture:

Expected tool or capability selection.
Actual tool calls.
Tool-choice correctness.
Retrieval and symbol-check evidence.
Errors, retries, and unnecessary calls.
Final answer correctness.
Whether the agent followed the intended version-resolution and evidence-validation workflow.
Report tool-choice accuracy separately from final-answer accuracy.

Calculate and explain the outcome-versus-trajectory gap: cases where the final answer is correct despite an incorrect or wasteful trajectory, and cases where a reasonable trajectory still produced a wrong final answer.

Define the metrics clearly and use the same definitions across the evaluated cases.

Do not claim a good trajectory merely because the final answer happened to be correct.

11. Re-score the failure taxonomy
Create a taxonomy of four to seven failure modes observed during integrated evaluation.

For each failure mode, report:

Failure category.
Description.
Number of occurrences.
Frequency and denominator.
Severity.
Relevant trace IDs.
Root cause supported by evidence.
Fix or mitigation.
Whether it was introduced by integration or already existed in an earlier component.
Include at least one genuine integration-created failure mode.

Examples to investigate, not preselected findings:

A resolved version is lost between the orchestrator and MCP server.
A downstream tool returns results for a different SDK version.
A correct retrieval result is discarded during agent handoff.
A citation refers to a source that was not used to support the answer.
A model call succeeds but the trace ID changes across an HTTP boundary.
A fallback generates code without preserving required timeout arguments.
You must demonstrate the integration-created failure through a trace, reproducible test, or evaluation evidence. Do not label a hypothetical risk as an observed failure.

Include the top three weaknesses discovered during the evaluation, along with their supporting evidence.

If the suite produces no failures, investigate the known failure cases and run a controlled, clearly documented fault or regression scenario. Do not fabricate an observed failure or claim that a flawless suite proves the system is safe.

12. The failure-to-test loop: RED → GREEN
Demonstrate one complete integration failure cycle.

RED
Identify a failing integration query, preferably one where a v2 caller receives an incompatible v3 code sample.
Capture its request, trace ID, retrieved context, model/tool behavior, and incorrect result.
Create a permanent regression case in the core suite.
Run the relevant baseline suite before applying the fix.
Preserve the actual failing output and pass/fail counts.
Fix
Use the trace evidence to identify the root cause and apply the smallest effective correction.

Possible fixes include restoring the version constraint across an MCP boundary, correcting retrieval metadata filtering, improving deprecation evidence, or preventing unsupported code from passing validation.

Do not merely change the expected answer or weaken the assertion.

GREEN
Run the regression case after the fix.
Run the complete integrated evaluation suite.
Verify the original failure no longer occurs.
Confirm no regressions in the existing cases.
Preserve the actual output, including passing and failing counts.
Keep the regression case permanently in the test suite.

If the live LLM result is nondeterministic, use a deterministic regression fixture for the failure path while keeping live-model evaluation separate. Clearly document the distinction.

13. Validate that emitted code samples actually run
The application must distinguish between code that looks plausible and code that is compatible with the target SDK.

Implement a practical validation layer for generated code.

Use the strongest available verification mechanism in the repository:

Check the target version and required symbols against trusted SDK metadata or source.
Check required arguments and incompatible signatures.
Where a compatible SDK installation and safe test environment are available, execute a bounded compile/import or API validation test.
Use mocked clients or isolated fixtures when real credentials or network access are unnecessary.
Do not execute untrusted generated code directly in the production application process.
Report the level of validation accurately.

Python compilation alone does not prove that a particular SDK method exists or accepts the expected arguments. Likewise, a successful import does not prove that the sample will work against a live service.

If actual SDK installation or execution is unavailable, report that limitation and retain the static/version-aware checks that can be performed.

A sample that cannot be verified must not be labeled as fully tested.

14. Measure cost per query by stage
Produce a reproducible cost breakdown for the integrated answer engine.

Attribute billable operations to distinct stages:

Retrieval, including embedding and reranking model calls where applicable.
Generation, including orchestrator and specialist model calls.
Tool executions, including separately billable model-backed or external calls.
Record input/output tokens, latency, cost, provider/model identity, and pricing assumptions.

Count each billable operation exactly once. Do not assign the same model call to multiple stages.

Report:

Average cost per query.
Total cost for the evaluated workload.
Per-stage cost.
Token usage.
Latency.
Measurement methodology.
Whether cost values are provider-reported or calculated estimates.
If the project has a measured Week 11 baseline, compare the integrated cost against it using comparable requests and configuration.

Never invent exact cost values when provider pricing or usage information is unavailable.

15. Ten-times scaling plan for approximately 9,000 developers
Analyze the consequences of scaling from the documented current workload to a scenario where approximately 9,000 developers access the tool through their IDEs.

Do not assume that 9,000 active developers all submit requests simultaneously. Explicitly define a realistic request-rate and concurrency model, including a burst scenario.

Evaluate the likely first bottleneck among:

Cost.
p99 latency.
API rate limits.
Use measured request latency, request cost, concurrency, provider rate limits, and any relevant database constraints.

Estimate:

Baseline request rate.
10× request rate.
Expected model calls per request.
Expected cost at the stated workload.
Relevant API quota or throughput limits.
Likely saturation point.
Practical mitigations such as caching, rate limiting, concurrency control, model routing, and retrieval optimization.
Identify the leading bottleneck in a concise, evidence-backed statement.

If no measured evidence proves which component fails first, clearly state the leading risk and the assumptions needed to establish the conclusion. Do not describe a hypothetical calculation as an observed production failure.

16. Required submission artifacts
Create a dedicated capstone directory, preferably artifacts/week12/, unless the repository already has an established artifact convention.

Generate the following deliverables.

A. Runnable README
Document:

Prerequisites.
Installation.
Required environment variables.
Local data/index initialization.
One-command startup.
How to submit a question.
How to run the complete evaluation suite.
How to run trajectory evaluation.
How to start and invoke the MCP server.
How to inspect traces.
How to reproduce the failure-to-test demonstration.
Known limitations.
A new developer must be able to follow the guide without needing undocumented setup steps.

B. Seam list
Create or preserve the Tuesday integration seam list.

Mark each identified seam as:

fixed
accepted
ran-out-of-time
If the original Tuesday list exists, preserve its original content and status history. Do not rewrite it as if it had been recorded earlier.

If it does not exist, create a clearly labeled reconstructed list based on the actual integration work.

C. Architecture diagram
Create a one-page architecture diagram showing:

API entry point.
Version resolver.
Version-scoped retriever.
Agent and tools.
MCP client and server boundaries.
Model calls.
Telemetry and trace storage.
Evaluation pipeline.
Cost attribution boundaries.
Use Mermaid or another format supported by the repository. Show data flows and trace propagation clearly.

D. Integrated evaluation report
Include suite size, sample size, pass rates, deterministic checks, judge agreement, per-case results, and limitations.

E. Failure taxonomy
Include four to seven evidenced failure modes, their frequencies, severity, trace IDs, and the new integration-created failure mode.

F. Trajectory evaluation
Include at least 10 cases, tool-choice accuracy, outcome-versus-trajectory gap, and supporting evidence.

G. Unified trace evidence
Provide representative trace output showing one trace ID spanning the API, agent, tools, and MCP hops.

Redact sensitive information.

H. Failure-to-test evidence
Include the original failing integration case, RED output, targeted fix, GREEN output, and permanent regression test.

I. Cost and scaling report
Provide measured cost per query by stage and the 10× scaling analysis.

J. Top three weaknesses
Create a short document identifying the three most important known weaknesses before the final defense.

Base the weaknesses on actual inspection and evaluation evidence. Record the document's creation point honestly; do not claim it was written before the defense if it was created afterward.

K. Fresh-eyes audit
Provide a short checklist and review template for an external peer.

Record actual peer feedback only after another person has reviewed the application. If no external reviewer is available, mark the review as pending and provide exact instructions for completing it.

Do not fabricate reviewer identities, comments, or verification.

17. The four-part technical defense
Prepare a concise demo guide that can be executed in approximately 10 minutes.

Part 1 — Real run
Demonstrate a live v2.4 question.

Show:

Version resolution before retrieval.
The version-scoped retrieval evidence.
Versioned citations.
A relevant deprecation surfaced proactively.
An explicit migration warning.
A code sample validated for the target version.
Use actual application output rather than a prerecorded fabricated result.

Part 2 — Failures found
Show the trace where a v2.4 request received an incompatible v3 snippet, or the closest real reproduced integration failure.

Explain:

The root cause.
The re-scored taxonomy.
The new integration-created failure mode.
The top three weaknesses.
The evidence supporting each finding.
Part 3 — A number that moved
Present a measured before-and-after metric, preferably the rate of wrong-version code samples.

Include:

Metric definition.
Baseline and post-fix results.
Sample size 
n
.
Absolute and relative change where meaningful.
Judge agreement and sample size.
One explicit sentence stating what the evaluation suite does not measure.
Do not claim that a metric improved if the baseline and post-fix samples are not comparable.

Part 4 — What is next and scale
Demonstrate the permanent regression test, report cost per query, and identify the leading 10× bottleneck using numerical evidence.

Provide a realistic mitigation plan and acknowledge remaining limitations.

18. Final acceptance criteria
The capstone is complete only when:

The application has one documented entry point.
Existing functionality is integrated without unnecessary rewrites.
The caller's version is resolved before version-specific retrieval.
Retrieval enforces version metadata filters.
Deprecations and breaking changes are surfaced when relevant.
The agent uses at least three functional tools.
Version-scoped documentation search is exposed over MCP and consumed through a real MCP call.
A unified trace ID survives agent and MCP hops.
Sensitive credentials and tokens are redacted before logging.
The integrated evaluation suite runs 25–40 cases or documents the precise blocker.
Deterministic assertions run before LLM judging.
Judge agreement is measured against independently human-labeled ground truth.
At least 10 trajectory cases are evaluated.
The error taxonomy contains 4–7 evidenced failure modes, including an integration-created failure.
A permanent failure regression case is demonstrated through RED → GREEN.
Code validation accurately distinguishes static checks from actual SDK execution.
Per-query costs and the 10× scaling risk are supported by measurements or clearly labeled estimates.
Required artifacts and the runnable README are present.
The four-part technical defense can be followed from the repository.
Human peer verification and any unavailable external capabilities are reported honestly.
19. Final instruction
Implement the smallest complete and maintainable solution that satisfies the capstone.

Finish implementation before running verification. Then verify the integrated application, execute the required evaluation and failure experiments in the correct order, collect actual telemetry, generate the artifacts, and prepare the defense guide.

Do not fabricate data, hide failures, weaken tests to obtain passing results, or claim features that are not implemented.

At the end, provide a concise completion report listing changed files, commands executed, actual evaluation results, remaining limitations, and the location of each deliverable.
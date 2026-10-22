 WEEK 10 MULTI-AGENT RAG ENGINEERING CHALLENGE
Role
You are a senior Python backend engineer working directly in the existing ragger-backend repository. Your task is to implement and complete the Week 10 Multi-Agent Squad vs. Single Agent Engineering Challenge.

Primary objective: Complete all required implementation, benchmarking infrastructure, controlled failure injection, telemetry collection, and submission artifacts. Run tests and experiments only after all implementation work is complete.

Do not merely explain how to implement the solution. Make the actual code changes in the repository.

1. Existing Project Context
The project is a FastAPI backend that replaces a Node.js/TypeScript backend while preserving the API contract expected by an existing Next.js frontend.

Existing architecture:

app/main.py — FastAPI entry point, CORS, rate limiting, lifespan, MCP mounting.
app/api/v1/router.py and route modules — REST API endpoints.
app/services/rag_engine.py — vector retrieval, BM25, RRF, reranking, MMR, execute_rag().
app/services/agent_engine.py — existing ReAct agent, /api/v1/agent/run, SSE streaming, optional MCP mode.
app/services/agent_loop.py — alternative agent implementation with budget enforcement.
app/services/agent_tools.py — existing tools, including search_knowledge_base, generate_answer, and check_deprecation.
app/services/mcp_client.py — dynamic MCP tool discovery and invocation.
app/mcp_server.py — RAG tools MCP server.
app/mcp_package_registry.py — package-registry MCP server.
app/core/config.py — application settings and agent budgets.
app/models/schemas.py — Pydantic schemas and existing API contracts.
evals/developer_docs_cases.json — existing Week 6 developer-documentation evaluation dataset.
Current branch: week-10-task.

Previous work includes MCP integration and an existing single-agent baseline. Reuse the existing implementation wherever possible.

2. Mandatory Execution Rules
Follow these rules throughout the task.

Read the requirements and inspect only the relevant existing source files, configuration, evaluation data, and test fixtures needed to implement the challenge.
Produce a short implementation checklist internally, then begin coding immediately.
Do not run tests, benchmarks, evaluation cases, linting, type checking, or other intermediate verification while implementation is still in progress.
Do not repeatedly inspect the same files, conduct unnecessary repository-wide research, or spend time redesigning unrelated components.
Complete all required code changes, instrumentation, controlled failure-injection support, and artifact-generation code before starting verification.
After implementation is complete, run the relevant tests, fix implementation-related failures, and rerun the necessary checks.
Run the actual benchmark only after implementation and initial verification are complete.
Generate final reports from real collected results. Never fabricate telemetry, claim unexecuted experiments passed, or substitute illustrative values for actual measurements.
Do not ask for confirmation about implementation details that can reasonably be determined from the existing codebase. If a genuinely required external dependency or credential is missing, finish all work that can be completed and report the exact blocker.
Do not stop after producing a plan, partial implementation, or example report. Complete every mandatory deliverable.
3. Implement the Multi-Agent Squad
Build a Manager-Plus-Two-Specialists architecture that can be evaluated against the existing single agent.

A. Orchestrator / Manager
Implement an orchestrator responsible for:

Receiving a developer-documentation question.
Identifying the required specialist capabilities.
Delegating work to the appropriate specialists.
Collecting and tracking specialist results.
Combining evidence into a final answer.
Handling specialist errors without silently inventing facts.
Tracking delegation hops, latency, token usage, and failures.
Enforcing reasonable timeouts, retry limits, and existing agent budget constraints.
Use a simple, explicit coordination strategy. Do not build an unnecessary autonomous planning framework.

The orchestrator must not claim that a version, deprecation status, or code sample has been verified unless it has supporting evidence from the appropriate source.

B. Version / Deprecation Worker
Implement a focused specialist responsible for:

Resolving SDK or package versions where possible.
Checking deprecation information through existing RAG tools and knowledge sources.
Identifying relevant version constraints and migration information.
Returning evidence, source references where available, and an explicit uncertainty or error state.
Reporting its token usage and execution metadata.
Reuse check_deprecation, retrieval services, and existing tools when appropriate. Do not duplicate the existing retrieval pipeline without a clear reason.

C. Code-Sample Worker
Implement a focused specialist responsible for:

Retrieving relevant developer documentation.
Finding code examples compatible with the requested SDK version.
Returning code samples with their supporting source evidence.
Reporting unsupported assumptions and missing version information.
Returning execution metadata and token usage.
Do not fabricate a version-specific code example when its compatibility cannot be verified.

D. Integration
Integrate the squad into the existing backend using the smallest reasonable set of changes.

Preserve the existing single-agent implementation as the baseline.
Preserve current API routes, response schemas, SSE streaming, MCP behavior, and frontend compatibility.
Add a separate, clearly identifiable squad execution path or evaluation entry point.
If an API endpoint must be added, keep it additive and backward-compatible.
Follow existing project patterns for settings, schemas, error handling, logging, and dependency injection.
Avoid unnecessary dependencies and unrelated refactoring.
The single-agent baseline and squad must receive the same evaluation question and equivalent input context. Differences in retrieval access, tools, or system instructions must be documented if they cannot be made equivalent.

4. Evaluation Dataset and Fair Comparison
Use the existing Week 6 dataset at:

evals/developer_docs_cases.json

Identify the exact 10 evaluation cases required by the challenge. Inspect the dataset structure and existing evaluation conventions before selecting them. Preserve any existing case identifiers and expected-answer criteria.

Do not silently select arbitrary cases or create replacement cases merely to make the results better.

If the repository contains more than 10 cases and does not identify the required 10 unambiguously, use a deterministic, documented selection of 10 appropriate existing cases. Preserve the wrong_source cases when they are part of the intended evaluation set.

Run two arms:

Arm A: Existing single-agent implementation.
Arm B: New orchestrator-plus-two-specialists squad.
Requirements:

Use the identical 10 questions and expected-answer criteria for both arms.
Use the same evaluator model, evaluation prompt, scoring rules, and relevant model settings.
Keep evaluation order and benchmark conditions consistent.
Do not modify expected answers to improve measured pass rates.
Record model identifiers, configuration, dataset identifiers, and relevant environmental limitations.
Use the existing evaluator infrastructure if available. Extend it only where necessary.
If a separate evaluator model is configured, use that same evaluator for both arms.
Distinguish evaluator failures from genuine answer failures.
The benchmark must be reproducible through a documented command or script.

5. Required Telemetry
Collect the following metrics for both arms.

A. Pass Rate
Calculate:

Pass rate = passed cases / 10 × 100

Use the repository's existing scoring rules when available. If the scoring rules need to be introduced, define a consistent and documented correctness rubric and apply it identically to both arms.

Report individual case results as well as the aggregate score.

B. p50 and p99 Latency
Measure the elapsed wall-clock time for each complete question execution.

Report:

p50 latency.
p99 latency.
Units and percentile calculation method.
Per-case latency where practical.
Use a documented percentile convention consistently for both arms. Since the sample contains only 10 cases, explicitly acknowledge the limitations of the p99 estimate.

C. Total Token Consumption
Capture actual input and output token usage for each question and aggregate it for each arm.

For the squad, include token consumption across:

Orchestrator model calls.
Version/deprecation worker model calls.
Code-sample worker model calls.
Other model calls required to produce the final answer.
Avoid double-counting usage records.

Prefer actual provider-reported usage metadata. If the existing provider abstraction does not expose token usage, implement a compatible usage-tracking mechanism or a documented tokenizer-based estimate.

Clearly distinguish actual token counts from estimated token counts. Do not present estimates as exact provider usage.

D. Cost Per Question
Calculate the total cost for each arm using the configured model's applicable input and output token prices.

Report:

Total measured cost across 10 cases.
Average cost per question.
The pricing source or configuration used.
Whether the values are actual provider-reported costs or estimates calculated from token usage.
If pricing information is unavailable, do not invent a price. Report the missing pricing dependency and provide the available token metrics.

6. Delegation Instrumentation and handoffs.log
Record every inter-agent delegation hop made by the squad.

The log must contain sufficient information to identify:

Timestamp.
Evaluation case ID.
Source agent.
Destination agent.
Delegation purpose.
Input/context token count.
Output token count.
Total token count for the hop.
Latency.
Success, failure, or timeout status.
Retry number, where applicable.
Error information for failed calls.
Use a consistent structured format, such as JSON Lines, with a human-readable report if useful.

For each handoff, measure actual token usage when available. If the token count is estimated, mark it as estimated and document the method.

Do not count a character-length estimate as an exact token count. Avoid counting the same model call twice when aggregating usage.

Context Re-Send Multiplier
Calculate:

Context Re-Send Multiplier = Multi-Agent Total Tokens / Single-Agent Total Tokens

Round the reported multiplier to one decimal place.

Use the total token consumption from the same 10 evaluation cases for both arms.

If the denominator is zero or a valid comparison cannot be obtained, report the calculation as unavailable rather than manufacturing a result.

Largest Handoff Bottleneck
Identify the single handoff step responsible for the largest token consumption.

Report:

Source and destination agents.
Absolute tokens consumed.
Percentage of total squad token consumption.
Whether the count is measured or estimated.
Use the logged token data to identify the actual bottleneck. Do not hardcode an assumed bottleneck.

7. Controlled HTTP 500 Failure Injection
Implement a deterministic mechanism to force the version/deprecation worker to return an HTTP 500 error for exactly one designated evaluation case during the failure experiment.

Requirements:

The failure must be explicitly enabled for the experiment, not active during ordinary application operation.
The injection must affect the version/deprecation worker call, not merely simulate a final orchestrator error.
Use an actual HTTP worker boundary when required by the architecture, with the controlled response observable by the orchestrator.
Keep the fault isolated to the selected case and worker.
Record the injected failure in the logs and identify the affected case.
Ensure the fault-injection mechanism cannot accidentally affect normal production requests.
Observe and document the orchestrator's actual behavior.

Record:

Whether the request was retried.
The number of attempts.
Whether retry limits and timeouts were respected.
Whether the orchestrator returned a complete answer, a partial answer, or a clear failure.
Whether it disclosed that version/deprecation verification had failed.
Whether the final answer contained unsupported claims or a code sample presented as version-verified without evidence.
The final task state and error details.
Do not hardcode the expected result in the report. Derive it from the actual execution trace.

A graceful response may provide a partial answer if the remaining evidence is sufficient, but it must explicitly disclose missing verification. Otherwise, fail safely.

8. Required Submission Artifacts
Create or update the following files in a dedicated Week 10 artifact directory, preferably artifacts/week10/, unless the repository already has an established artifact convention.

race_table.md
Include:

A comparison of the single-agent and multi-agent arms.
Pass rate for each arm.
p50 latency for each arm.
p99 latency for each arm.
Total tokens for each arm.
Average cost per question for each arm.
The 10 individual case outcomes and useful per-case telemetry.
Evaluator and model configuration.
Dataset selection and measurement methodology.
Explicit notes on missing or estimated metrics.
Every reported number must be derived from actual benchmark output.

handoffs.log
Include every inter-agent delegation hop across the squad evaluation runs, with per-hop token counts and statuses.

Ensure the log is generated by the actual execution instrumentation.

failure_case.md
Include:

Selected evaluation case.
Fault-injection configuration.
Actual HTTP 500 response.
Retry behavior.
Actual orchestrator response.
Whether graceful degradation occurred.
Whether any unsupported version claims or hallucinated code samples occurred.
Relevant execution trace references.
Final conclusion about failure handling.
verdict.md
Write the final KEEP or KILL recommendation in 10 lines or fewer.

The verdict must:

Make one explicit decision.
Cite at least two real benchmark metrics.
Explain the practical trade-off between correctness, latency, token consumption, and cost.
Explicitly address sunk-cost bias: the team must not retain the squad merely because implementation effort has already been invested.
Acknowledge missing or unreliable data when it affects the decision.
Do not decide in advance that the multi-agent approach must be kept. Let the measured results determine the recommendation.

Reproducibility
Add or update the minimum required scripts and documentation so that another engineer can reproduce the benchmark, regenerate the telemetry, and reproduce the failure experiment.

Do not commit API keys, credentials, or sensitive runtime data.

9. Bonus Challenge: AgentCard and A2A Lifecycle
After all mandatory deliverables are implemented, address the bonus challenge without overengineering the system.

AgentCard
Publish a machine-readable AgentCard for the orchestrator, using a suitable project convention or a documented location.

Describe:

Agent identity and purpose.
Supported capabilities.
Input and output interfaces.
Worker delegation capabilities.
Authentication requirements, consistent with the actual deployment.
Relevant endpoint and version information.
Do not claim authentication protections that do not exist.

A2A Lifecycle Mapping
Document how the worker HTTP 500 failure maps to an A2A-style task lifecycle.

Identify whether the task should complete successfully, terminate as failed, or require additional input. A missing SDK version should only trigger an input-required state when user input is genuinely necessary.

Do not claim conformance to a formal A2A protocol unless the relevant protocol is actually implemented.

Protocol Value
Add a concise two-line comparison explaining what a formal A2A protocol provides over ordinary REST calls, distinguishing implemented features from proposed future benefits.

Prefer existing dependencies and infrastructure. Do not introduce a heavyweight framework just to satisfy the bonus.

10. Implementation and Verification Workflow
Follow this exact order.

Phase 1 — Inspect and implement

Read relevant files and the evaluation dataset.
Understand the existing agent execution path and provider usage tracking.
Implement the squad, integration, telemetry, fault injection, and artifact-generation mechanisms.
Finish all required code changes before running any verification command.
Phase 2 — Verify after implementation

Only after the implementation is complete:

Run the relevant existing and newly added tests.
Run the relevant linting and type checks when available.
Fix failures caused by the implementation.
Rerun the affected checks.
Do not change unrelated behavior simply to make unrelated checks pass.
Phase 3 — Execute the experiment

Run the same 10 evaluation cases through both arms.
Collect actual telemetry.
Run the isolated HTTP 500 failure experiment.
Calculate the context re-send multiplier and largest handoff bottleneck.
Generate all four required submission artifacts.
Execute the experiment again only when necessary to correct an implementation or measurement defect.
Phase 4 — Final report

Provide a concise completion summary containing:

Implemented architecture and integration points.
Files added or modified.
Test and check results.
Actual benchmark metrics for both arms.
Context re-send multiplier and largest handoff bottleneck.
Observed HTTP 500 recovery behavior.
The KEEP or KILL verdict.
Any blocked requirements and the precise reason.
11. Acceptance Criteria
The task is complete only when:

The existing single-agent baseline remains available.
The manager and both specialist workers are implemented and integrated.
Both arms can run the same 10 evaluation cases under the same evaluator conditions.
Pass rate, p50/p99 latency, token consumption, and cost are measured or explicitly marked unavailable with reasons.
Every inter-agent handoff is instrumented.
The context re-send multiplier is calculated from collected data when valid.
The largest token-consuming handoff is identified from the actual log.
A controlled HTTP 500 is injected into the version/deprecation worker and the resulting behavior is documented.
All four required artifacts are generated.
The final verdict is 10 lines or fewer and explicitly addresses sunk-cost bias.
The relevant final checks and benchmark execution results are reported honestly.
Existing API contracts and unrelated functionality are preserved.
Final instruction: Prioritize finishing the complete, minimal implementation over performing intermediate checks or producing lengthy explanations. Do not test while coding. Once all implementation work is finished, perform verification, run the actual experiment, generate the evidence-based reports, and provide the final completion summary. Do not stop at a partial implementation.
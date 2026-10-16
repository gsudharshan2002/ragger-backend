WEEK 11: PRODUCTION OBSERVABILITY, COST & FAILURE-TO-TEST LOOP
Role and objective
You are a senior Python backend engineer working directly in the existing ragger-backend repository.

Implement the Week 11 Production Engineering Challenge for the Developer Documentation track (Track E).

Your goal is to make the application observable, measurable, and resistant to repeated failures. Implement the required code changes, perform the incident investigation, convert a real or reproducibly injected failure into a permanent evaluation test, apply a targeted fix, and generate every required submission artifact.

Do not merely provide suggestions or code snippets. Make the actual changes in the repository.

1. Existing project context
The application is a FastAPI backend for a RAG application, preserving the existing Next.js frontend API contract.

Relevant existing files and components:

app/main.py — application entry point, middleware, CORS, rate limiting, lifespan.
app/api/v1/router.py and route modules — API endpoints.
app/services/rag_engine.py — vector search, BM25, RRF, reranking, MMR, execute_rag().
app/services/agent_engine.py — ReAct agent, agent execution, SSE streaming, optional MCP mode.
app/services/agent_loop.py — alternative agent loop and budget enforcement.
app/services/agent_tools.py — RAG and deprecation tools.
app/services/mcp_client.py — MCP tool discovery and invocation.
app/core/config.py — application configuration.
app/models/schemas.py — API request and response schemas.
evals/developer_docs_cases.json — Week 6 evaluation dataset, including cases tagged wrong_source.
app/data/traces.jsonl — existing trace data.
app/data/traces.json.bak — existing trace backup.
app/data/agent_memory.jsonl — existing agent memory.
The repository also contains previous evaluation and agent experiment artifacts. Inspect relevant files before deciding how to extend them.

Week 10 may have introduced a multi-agent orchestrator, version/deprecation worker, code-sample worker, handoff logging, and additional evaluation scripts. Inspect the actual current working tree instead of assuming those changes exist. Reuse the implemented Week 10 architecture if available.

Preserve existing API behavior, MCP functionality, streaming, and frontend compatibility.

2. Mandatory execution rules
Follow these rules strictly.

Inspect the relevant source files, trace formats, evaluation infrastructure, configuration, and existing tests.
Identify the smallest set of changes needed to satisfy the requirements.
Implement the complete required code changes before running tests or evaluations.
Do not run tests, linters, type checks, benchmarks, or intermediate verification while implementation is in progress.
Do not repeatedly inspect the same files, redesign unrelated components, or introduce unnecessary dependencies.
Do not replace working infrastructure when it can be extended safely.
Once implementation is complete, run the relevant tests and checks, fix implementation-related failures, and rerun them.
Capture a genuine failing evaluation before applying the targeted prompt or retrieval fix. The RED experiment is a required evaluation phase, not an intermediate development check.
Do not fabricate historical logs, timings, token usage, costs, test results, or human verification.
If the existing system lacks enough telemetry to reconstruct a real historical incident, create a clearly labeled, reproducible developer-documentation incident using a controlled fixture or test case. Do not represent a synthetic incident as an actual customer incident.
If an external service, API key, or required human reviewer is unavailable, finish all work possible and clearly document the limitation.
Do not stop after generating a plan or partially implementing the task.
3. Implement complete per-request observability
Extend the existing tracing and logging infrastructure to make individual requests searchable and replayable.

A. Request-level fields
Capture the following fields wherever applicable:

Unique request ID and trace ID.
Timestamp and request duration.
API endpoint and execution mode.
Sanitized request input or a privacy-safe replay reference.
Outcome status, including success, error, timeout, or partial answer.
Prompt version.
Model provider and model identifier.
Retrieved document and context chunk IDs.
Retrieval configuration and relevant source metadata.
Total input and output tokens.
Total request cost in USD.
Error type and sanitized error details.
Agent or worker identity for multi-agent execution.
Cache hit or miss status if caching is implemented.
Every request must have a trace ID that connects the request to its individual execution steps.

B. Span-level observability
Record distinct spans for the applicable stages:

Retrieval.
Reranking or other relevant retrieval substeps.
Tool executions.
Model generation.
Agent-to-agent handoffs, if applicable.
Final response assembly.
For every span, record:

Trace ID and span ID.
Parent span ID where applicable.
Stage name.
Start time and duration.
Success or failure status.
Input and output token usage for model calls.
Cost in USD where measurable.
Retrieved context IDs where relevant.
Tool name and outcome for tool calls.
Retry count and error information where applicable.
Use a consistent schema. Avoid counting the same model call or token usage twice when aggregating totals.

C. Storage and compatibility
Reuse app/data/traces.jsonl or the project's existing trace storage mechanism where practical.
Add fields compatibly so existing trace consumers do not break.
Keep log writes robust and prevent a logging failure from unnecessarily breaking a successful user request.
Use structured JSON logs where possible.
Avoid storing credentials, API keys, authorization headers, or unnecessary personal information.
Ensure request IDs remain consistent across nested operations and streaming requests.
Use existing project conventions. Do not add LangSmith, Phoenix, OpenTelemetry, or another observability service merely for its name. Add an external integration only if it provides necessary functionality and fits the current implementation.

4. Implement searchable incident investigation
Create a small, reliable mechanism for finding past requests using trace data.

Support useful filters such as:

Trace ID or request ID.
Time range.
Prompt version.
Error status.
Model or execution mode.
Retrieved context chunk ID.
Latency or cost outlier.
Relevant case or request metadata.
Prefer indexed or efficient filtering over scanning all logs repeatedly where the current storage volume justifies it. If the repository uses JSONL, an efficient filtering implementation is acceptable for this challenge.

The incident investigation workflow must be able to:

Locate the target bad response.
Identify the associated trace.
Retrieve the complete trace.
Inspect its spans and retrieved context.
Determine whether the failure resulted from outdated retrieval, missing evidence, tool failure, prompt behavior, or another supported cause.
Provide a reproducible command or script for searching traces and extracting a complete trace.

5. The support drill: Developer Documentation track
The assigned scenario is:

Find the code sample that used a deprecated method.

Inspect the existing developer-documentation evaluation dataset, especially wrong_source cases and any version/deprecation cases.

Prefer a real existing failure with sufficient trace evidence. If no suitable historical failure can be reconstructed, create a deterministic, explicitly labeled incident fixture that represents a deprecated-method answer.

The fixture must include enough information to exercise the actual trace lookup and debugging workflow.

Timed incident search
Perform the incident hunt using the implemented search mechanism.

Record:

Target incident identifier.
Search start and finish timestamps.
Elapsed time in mm:ss format.
Exact search filters or log slice used.
Fields that enabled the lookup.
The evidence that confirmed the bad answer.
Do not invent elapsed time or claim that a human completed the drill.

If a real five-minute timed drill is required, provide a documented command and procedure for a human reviewer to execute. Mark the human verification as pending until it has actually occurred.

If the search takes more than five minutes, document the exact missing field, inefficient lookup, or index limitation responsible.

Required drill.md
Record the incident, search procedure, measured time, log slice, root-cause evidence, and whether the result met the five-minute target.

Explicitly distinguish between actual historical incidents and synthetic fixtures.

6. Full trace extraction: trace.json
Extract the complete trace for the isolated incident.

The JSON must contain, where applicable:

Request and trace IDs.
Timestamp and execution status.
Prompt version.
Model identifiers.
Input and output token counts.
Per-span start times and latencies.
Per-span token counts and costs.
Total request latency and cost.
Retrieval queries and configuration.
Retrieved context chunk IDs.
Source identifiers and version metadata.
Tool executions and results.
Agent handoffs if applicable.
Errors, retries, and fallback behavior.
Final answer or a safe reference to the stored answer.
Include enough evidence to determine whether the deprecated code originated from outdated retrieved documentation, a tool failure, unsupported generation, or another demonstrated cause.

Never insert invented token counts, costs, source IDs, or span timings into a purported real trace. Mark unavailable fields explicitly. Synthetic fixtures must be labeled as synthetic.

Ensure the exported trace is valid JSON and contains the complete relevant execution path rather than only the final response.

7. Failure-to-test loop: RED → GREEN
Convert the isolated incident into a permanent regression case.

A. Create the evaluation case
Add a new case to the core evaluation suite, following the existing dataset and test conventions.

The case should test that the application does not recommend the deprecated method as the correct current implementation when reliable documentation establishes a replacement.

Include:

A stable case ID.
The developer-documentation question.
Relevant package and version information, if available.
Expected-answer criteria.
The deprecated method or prohibited behavior.
The correct replacement and supporting evidence, where known.
A regression assertion that catches the original failure.
Use the existing evaluation framework wherever possible. Avoid adding an unrelated test framework.

B. RED: reproduce the original failure
Before applying the fix:

Run the existing suite with the new regression case.
Capture the actual output and pass/fail counts.
Identify the failing case by its stable ID.
Preserve the RED output in a dedicated artifact.
Do not hardcode the expected RED result or alter the evaluator to force a failure.

If the incident cannot be reproduced by the live model because its output is nondeterministic, use a deterministic fixture or mock that reproduces the observed failure path. Clearly distinguish the deterministic regression test from a live-model evaluation.

The RED phase must occur before the targeted fix is applied.

C. Implement the targeted fix
Use the evidence from the trace to select the smallest appropriate fix.

Potential approaches include:

Correcting retrieval filtering or ranking.
Enforcing version-aware retrieval.
Excluding outdated documentation where appropriate.
Improving source selection and evidence requirements.
Updating the prompt to prohibit unsupported deprecated-method recommendations.
Adding a verification step for version-specific code samples.
Do not apply all these changes indiscriminately. Choose the fix justified by the actual incident.

Do not simply weaken the assertion or change the expected answer to make the test pass.

D. Prompt versioning
If the prompt changes, bump its explicit version using the existing convention or introduce a simple version constant if needed.

Record the previous and new prompt versions and their relationship to the incident.

If the fix is entirely retrieval-based, do not falsely claim a prompt change occurred. Record the prompt version used and document the retrieval configuration change instead.

Ensure the updated version is included in future request traces.

E. GREEN: rerun the complete suite
After the fix:

Run the full relevant evaluation suite, including the new regression case.
Capture the exact passing and failing counts.
Confirm the original incident no longer reproduces.
Check existing cases for regressions.
Preserve the actual GREEN output.
Do not claim the suite is green if unrelated or existing tests remain failing. Explain any pre-existing failures and separate them from new regressions.

F. Required evidence
Store RED and GREEN command output in separate files, such as:

artifacts/week11/red_output.log
artifacts/week11/green_output.log
Keep the new regression case in the permanent core test suite, not only in an isolated temporary file.

8. Cost attribution: cost_by_stage.md
Calculate the cost of the isolated request across three distinct categories.

Retrieval
Include measurable model or service costs attributable to retrieval, such as query-embedding calls or reranking model calls.

Vector database and local BM25 operations may have no direct per-request model API charge. If no defensible direct dollar amount is available, report it as zero direct API cost or not separately attributable, and explain the limitation.

Generation
Include the cost of all model calls that contribute to answer generation, including orchestrator and worker calls if the request uses a multi-agent execution path.

Tool executions
Include attributable costs of model-backed tool calls and external billed services used during tool execution.

Do not automatically assign all tool-related model calls to both the generation and tools categories.

Define a consistent attribution rule so that each billable operation is counted once.

Report requirements
Include:

Per-stage input and output tokens where applicable.
Per-stage latency.
Per-stage dollar cost.
Total request cost.
Model pricing assumptions and source.
Measured versus estimated values.
Any unavailable cost information.
Use actual provider usage metadata where available. Otherwise, use a documented tokenizer or provider-pricing estimate and label it accurately.

9. Measure cost and implement one improvement
The Week 11 build requires measuring cost per request and making at least one defensible cost improvement.

First establish a baseline using the same representative questions, model configuration, and evaluation conditions.

Then implement one targeted improvement justified by the measured baseline.

Possible options include:

Caching deterministic retrieval results.
Semantic caching with strict version and freshness controls.
Routing simple questions to a cheaper model.
Reducing unnecessary repeated context in agent handoffs.
Eliminating redundant model calls.
Improving prompt or context efficiency.
Select only an improvement that fits the actual codebase and available measurements.

Requirements:

Measure baseline cost per request.
Measure post-change cost per request using equivalent inputs.
Report absolute and percentage cost change.
Measure latency and correctness alongside cost.
Ensure cached results cannot silently bypass relevant version, document-freshness, or access constraints.
Do not claim savings from an unmeasured optimization.
If a valid comparison cannot be obtained because pricing or external usage metadata is unavailable, provide the available token-level comparison and explicitly document the limitation.

Do not make a costly external integration mandatory merely to complete the exercise.

10. Ten-times scaling analysis: tenx.md
Provide a production scaling analysis grounded in actual measurements.

Evaluate the likely first bottleneck among:

Cost.
p99 latency.
API rate limits.
Use the measured workload, model-provider limits where available, concurrency assumptions, and measured per-request resource consumption.

Estimate the consequences of increasing request volume by 10×.

Include:

Current measured or documented baseline.
Expected 10× workload.
The component most likely to saturate first.
The specific numeric evidence supporting the conclusion.
A practical mitigation.
If the evidence is insufficient to prove which bottleneck will fail first, identify the leading risk and state the assumptions instead of presenting an estimate as a measured fact.

Do not execute an uncontrolled 10× production load test. Use a documented calculation or a safe, bounded local simulation where necessary.

11. Deployment safety and rollback
Document the prompt or retrieval configuration version change.

Include a concise two-line deployment plan in the relevant artifact:

Canary: deploy the fix to a small traffic percentage, compare deprecated-method error rate, correctness, latency, and cost against the baseline.
Rollback: revert the prompt or retrieval configuration to the previous version if regression thresholds are breached, while preserving traces for investigation.
Make the thresholds explicit when reliable baseline measurements are available. Do not claim a canary deployment has occurred unless it actually has.

12. Required artifacts
Create the following files under artifacts/week11/, unless the repository already uses a more suitable artifact directory.

drill.md — Incident search procedure, measured time, log slice, root cause, and human verification status.
trace.json — Complete incident trace with per-span latency, tokens, costs, prompt version, and context IDs.
A permanent new evaluation case in the core test suite.
red_output.log — Captured pre-fix failing evaluation output.
green_output.log — Captured post-fix evaluation output.
cost_by_stage.md — Retrieval, generation, and tool-execution cost breakdown.
tenx.md — Numeric scaling analysis and first-bottleneck assessment.
A concise deployment and rollback plan, preferably included in drill.md or another clearly identified artifact.
Also add the minimal scripts or configuration required to reproduce trace lookup, export a trace, run the regression suite, and measure costs.

13. Bonus challenge
Wire the drill shut
Add the missing searchable field, index, or lookup mechanism that caused incident investigation to be slow.

Demonstrate that the improved lookup uses the actual trace data and returns the intended request. Do not claim instantaneous lookup without measuring it.

Second drill
If feasible, create a second clearly labeled incident fixture representing another bad developer-documentation response.

Repeat the search and record both timings.

Compare the first and second drills, identifying the field or index that improved lookup time.

If a human squadmate is required to plant the second incident or verify the drill, provide the reproducible procedure and report that verification as pending until performed.

14. Required execution sequence
Phase 1 — Inspect and implement
Inspect the current code, traces, schemas, test suite, and evaluation dataset.
Implement request tracing and searchable trace extraction.
Implement span-level timing and token/cost instrumentation.
Implement the selected cost improvement and measurement script.
Add the permanent regression case and prepare the failure-reproduction mechanism.
Prepare the artifact-generation workflow.
Finish all implementation changes before running verification.
Phase 2 — Initial verification
After all implementation work is complete:

Run the relevant existing tests.
Run the relevant new tests and available lint/type checks.
Fix implementation-related failures.
Rerun affected checks.
Do not run the incident evaluation before the fix during this phase. The explicit RED evaluation must happen in the next phase, against the original unfixed behavior.

Phase 3 — Incident and RED → GREEN experiment
Locate the selected historical trace or generate a clearly labeled deterministic fixture.
Perform the timed incident hunt and capture the actual duration.
Export the complete trace.
Run the evaluation suite with the new regression case before the targeted fix.
Save the actual RED output.
Apply the targeted prompt or retrieval fix and bump the prompt version when applicable.
Run the full suite after the fix.
Save the actual GREEN output and inspect regressions.
Phase 4 — Cost and scaling
Measure baseline cost per request.
Apply and measure one cost improvement.
Generate the stage cost breakdown.
Produce the 10× scaling analysis.
Generate all required artifacts using actual collected data.
Phase 5 — Final report
Summarize:

Implementation completed.
Files added and modified.
Incident search duration and findings.
Trace ID and root cause.
RED and GREEN pass/fail counts.
Prompt or retrieval version change.
Baseline and improved cost per request.
Largest scaling bottleneck and supporting numeric evidence.
Test and check results.
Any outstanding human verification or external dependency.
15. Acceptance criteria
The task is complete only when:

Every request can be correlated with its trace and individual spans.
Trace records include prompt version, context IDs, timing, token usage, and cost where measurable.
The selected incident can be found through the implemented search mechanism.
drill.md records the actual search time and exact lookup method.
trace.json contains the complete relevant execution path.
The new failure regression case is part of the permanent suite.
Actual pre-fix RED output is captured before applying the targeted fix.
Actual post-fix GREEN output is captured after the fix.
The targeted fix addresses the demonstrated cause without hiding the failure.
The cost-per-request improvement is measured against a comparable baseline.
cost_by_stage.md explains the three stage costs and attribution assumptions.
tenx.md contains a numeric, evidence-backed scaling assessment.
The deployment and rollback plan is documented.
Any uncompleted human drill or unavailable metric is clearly disclosed.
Final instruction: Implement first, verify after implementation, and run the deliberate RED → GREEN evaluation in the required order. Prioritize the smallest complete solution that meets the requirements. Do not fabricate evidence, stop at a plan, or claim a deliverable is complete when it is not.
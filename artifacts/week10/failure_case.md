# Week 10: Controlled HTTP 500 Failure Injection

Selected case: `gh-006`
Question: Does the current version of GET /repos/{owner}/{repo} still include a has_downloads field?

## Fault-Injection Configuration
- Fault enabled (env `RAGGER_FAULT_INJECTION`/`RAGGER_FAULT_CASE_ID`): `True`
- Target worker: `version_deprecation` (only this case, only this worker)
- Fault injected on this run: `True`
- Mechanism: the worker returns an HTTP 500 error result in-process (there is no network boundary between manager and workers in this build)

## Retry Behavior (from the recorded handoffs)
- Attempts: 3 (1 initial + 2 retries); configured limit: 3 attempts
- Retry limit respected: `True`
- Per-attempt timeout 30.0s respected: `True`
  - attempt 0: status=`failure` latency=0ms error='HTTP 500: version/deprecation check failed for Does the current version of GET /repos/{owner}/{repo} still include a has_downloads field?@2026-03-10'
  - attempt 1: status=`failure` latency=0ms error='HTTP 500: version/deprecation check failed for Does the current version of GET /repos/{owner}/{repo} still include a has_downloads field?@2026-03-10'
  - attempt 2: status=`failure` latency=0ms error='HTTP 500: version/deprecation check failed for Does the current version of GET /repos/{owner}/{repo} still include a has_downloads field?@2026-03-10'

## Actual HTTP 500 Response
```json
{
  "error": "HTTP 500: version/deprecation check failed for Does the current version of GET /repos/{owner}/{repo} still include a has_downloads field?@2026-03-10",
  "status": "failure",
  "fault_injected": true
}
```

## Actual Orchestrator Response
```
I could not find that information in the provided documents.

Note: version/deprecation evidence was unavailable (HTTP 500: version/deprecation check failed for Does the current version of GET /repos/{owner}/{repo} still include a has_downloads field?@2026-03-10); any version-specific statement above is unverified.
```

## Outcome
- Final task state: **completed: could not answer from evidence**
- Version verification state: `unavailable`
- Missing verification disclosed to the user: `True`
- Affirmative version/deprecation claims in the answer body: `none`
- Code sample presented as version-verified: `False`
- Graceful degradation: `True`
- Error details: `None`

## Execution Trace References
- Total latency: 73188ms
- Model-call tokens: 404 in / 122 out / 526 total
- Handoffs recorded: 5 (see `handoffs.log`)

## Worker Results
- version_deprecation: status=`failure` evidence=`False` attempts=`3`
- code_sample: status=`None` evidence=`False` attempts=`None`

## Conclusion
The orchestrator retried within its limits, did not present unverified version claims as verified, and told the user verification was unavailable.
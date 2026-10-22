# Week 10: Controlled HTTP 500 Failure Injection

Selected case: `gh-006`
Question: Does the current version of GET /repos/{owner}/{repo} still include a has_downloads field?

## Fault-Injection Configuration
- Fault enabled: `True`
- Worker: `version_deprecation`
- Fault injected: `True`
- Response: HTTP 500 (simulated via `InjectedHTTP500Error`)

## Retry Behavior
- Retries: not applicable (fault injection is deterministic, not transient)
- Retry limits respected: N/A (no retry attempted for injected faults)
- Timeouts respected: N/A

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
I’m sorry, but the provided evidence does not contain information about whether the current version of `GET /repos/{owner}/{repo}` includes a `has_downloads` field.

Note: version/deprecation verification was unavailable for this question.
```

## Graceful Degradation
- Complete answer: `yes`
- Missing version verification disclosed: `yes`
- Unsupported version claims/hallucinated code: `no`

## Execution Trace References
- Total latency: 3663ms
- Input tokens: 122 (estimated)
- Output tokens: 45 (estimated)
- Total tokens: 167 (estimated)
- Handoffs: 3

## Version/Deprecation Worker Result
- Status: `failure`
- Evidence: `False`
- Error: `HTTP 500: version/deprecation check failed for Does the current version of GET /repos/{owner}/{repo} still include a has_downloads field?@2026-03-10`

## Code-Sample Worker Result
- Status: `None`
- Evidence: `False`

## Final Conclusion
- Graceful degradation: `yes`
- No unsupported claims: `yes`

The orchestrator gracefully degraded and disclosed the missing verification the version/deprecation failure.
No hallucinated version-specific claims were detected.
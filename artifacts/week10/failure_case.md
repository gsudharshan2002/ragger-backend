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
No answer was generated (GROQ_API_KEY is not set. Please add it in Settings or set it in your .env file.)
```

## Graceful Degradation
- Complete answer: `yes`
- Missing version verification disclosed: `no`
- Unsupported version claims/hallucinated code: `no`

## Execution Trace References
- Total latency: 14ms
- Input tokens: 137 (estimated)
- Output tokens: 31 (estimated)
- Total tokens: 168 (estimated)
- Handoffs: 4

## Version/Deprecation Worker Result
- Status: `failure`
- Evidence: `False`
- Error: `HTTP 500: version/deprecation check failed for Does the current version of GET /repos/{owner}/{repo} still include a has_downloads field?@2026-03-10`

## Code-Sample Worker Result
- Status: `error`
- Evidence: `False`

## Final Conclusion
- Graceful degradation: `no`
- No unsupported claims: `yes`

The orchestrator did not adequately disclose the version/deprecation failure.
No hallucinated version-specific claims were detected.
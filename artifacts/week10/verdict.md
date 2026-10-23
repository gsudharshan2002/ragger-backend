# Week 10: KEEP or KILL Verdict
Decision: **KEEP**, because correctness is within one case of the single agent (noise at n=10) while the squad uses 46% fewer tokens and is not slower.
- Pass rate: single 60% (6/10) vs squad 70% (7/10).
- p50 latency: single 186746ms vs squad 50631ms.
- Tokens: single 27890 vs squad 15074 (squad/single = 0.5x); avg cost/question $0.001394 vs $0.000754.
- Trade-off: the squad adds a routing call and extra hop and must earn that with correctness, tokens or latency.
- Sunk cost: the build effort already spent is not a reason to keep it; the decision follows the measured numbers only.
- Data caveats: n=10 cases (p99 = max); keyword scoring is coarse; token counts include estimates; cost uses one flat rate.
- Re-run `python scripts/run_week10_experiment.py` after any change to re-derive this verdict.
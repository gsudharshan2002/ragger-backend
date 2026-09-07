# Judge disagreement analysis

Source labels: `labels_25.json` (source report `20260904T155318_hybrid-rerank-mmr_15580270_developer_docs_cases.json`, labeled_at `2026-09-04T15:54:31Z`).

## agreement_before -> agreement_after

- **agreement_before = 88.46%** (22/26) - judge_v1.txt, run `evals/results/20260904T160722_judge_validation.json` (reproduces the user's own same-day run at 15:55:43, `20260904T155543_judge_validation.json`, also 88.46%).
- **agreement_after = 88.46%** (22/26) - judge_v2.txt, run `evals/results/20260904T160907_judge_validation.json`.
- **No net change in this pair.** Same 3 cases disagree before and after: combine-002, combine-005, combine-013.

## The 3 disagreements, and who was actually right

1. **combine-005** - "AnyPublisher... performs type erasure by wrapping another publisher." Human: fail. Judge: pass ("correctly covers type erasure and wrapping"). **Judge is right** - the answer contains both expected keywords and is accurate. The human label looks anchored on this case's `regression: true` tag (the Hand Labeling UI shows a "Known failure" banner for regression cases while you're grading them), not on this specific answer's actual content.
2. **combine-002** - "The Publisher protocol declares a type that can deliver a sequence of values over time." Human: fail. Judge: pass (matches both expected keywords exactly). **Judge is right on the stated criterion** ("correct and helpful" text-only). The human's fail plausibly reflects real knowledge that this case's retrieval has historically failed (same regression-banner effect as above) - a legitimate concern, but out of scope for a judge that only sees question+answer+keywords, not retrieval state.
3. **combine-013** - "I couldn't find that information in the provided context." Human: pass. Judge: fail ("bare refusal without explanation or direction"). **Judge is right per the stated rubric** - the judge's own instructions say a refusal on a gradeable question should fail, and this refusal names no specific gap or pointer. The human's pass reflects a defensible but different standard (rewarding honesty over hallucination) that isn't what the single "correct and helpful" criterion asks for.

## Why judge_v2 didn't move the number here

judge_v2.txt's fix targets a 4th, older disagreement not present in this batch: in an earlier run (`20260904T150828_judge_validation.json`, agreement 84.62%, labels committed at `4def481`), **combine-012** was failed by the judge ("incorrectly mentions TopLevelEncoder and TopLevelDecoder instead of a protocol related to encoding") even though the answer explicitly self-corrected to the right protocol (TopLevelEncoder) as its final statement - a genuine judge comprehension bug, and the human's pass was correct. That exact wording doesn't recur in this run's freshly-generated answer for combine-012 (LLM answers vary run to run), so there was nothing for the fix to visibly correct in this specific before/after pair - the fix is real and grounded in an actual observed judge mistake, it just isn't the failure mode present in *this* snapshot's data.

## Prediction, scored honestly

`prediction.txt` predicted agreement would move from 84.62% (the older, since-superseded label set) to ~88%, with combine-012 flipping and combine-013 staying disagreed.

- **Right**: combine-013 stayed a disagreement, exactly as predicted, for the reason predicted.
- **Wrong**: the prediction anchored on the stale 84.62% baseline instead of this run's actual 88.46% baseline (the label set had already moved on by the time the fix was tested), and combine-012 never appeared as a disagreement in this pair at all - so it couldn't "flip." The prediction underestimated how much LLM answer non-determinism across runs would decouple the fix's target case from the actual before/after data.

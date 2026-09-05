<!-- Soft Suave · The AI Engineering League -->
# Week 4 Practical — Task Set E

## Label the failures, then buy back hit-rate@3 with exactly one change

| | |
|---|---|
| Domain | Developer documentation |
| Week | 4 — Debugging Retrieval — Hybrid, Reranking & Failure Separation |
| Module | M2 — Retrieval & RAG |
| Sat on | Week 5 · Monday |
| Marks | 100 |

> **This is an extension of the app you already built in Week 4.** It is not a build from scratch, and it tests only this week's concepts. Bring your numbers written down.


---

## 1. Problem statement

Your docs assistant answers 'how do I authenticate' well and then whiffs on 'what is the default for retry_backoff_ms' — it returns three plausible, semantically-adjacent retry guides, none of which contains the symbol. The team lead wants to swap the embedding model. Use your inspection view: prove where the failures actually live, make ONE retrieval change, and come back with a before and after number.


---

## 2. Requirements

1. Assemble a 12-question golden set from REAL developer questions, each tagged with the chunk_id you know is correct. At least 4 must contain an exact token dense retrieval is structurally bad at — a symbol name (retry_backoff_ms), an error code, or a version string (v3.2.0).
2. Measure hit-rate@3 on your current retriever over all 12. That is your baseline number, and you write it down before you change anything.
3. Run every miss through the inspection view and label each R (retrieval fetched bad context), G (model misused good context), or Not-In-Corpus, with ONE line of evidence per label. Paste the tally.
4. Make exactly ONE retrieval change, chosen from the tally and justified by it — BM25 + RRF fusion (k=60), or a cross-encoder rerank over the top 25. One change only.
5. Re-measure hit-rate@3 on the SAME 12 questions and report before -> after, plus the p50 latency per query before -> after. Both numbers, or the change doesn't count.
6. Name explicitly which of your original R-failures the change fixed and which it did not touch at all.


---

## 3. Expected output

A results.md containing: the 12-question golden set with known-correct chunk_ids, the baseline hit-rate@3, the R/G/Not-In-Corpus tally with one line of evidence per labelled failure, a one-paragraph justification of the single change you chose (referencing the tally), the after hit-rate@3, before/after p50 latency, a per-question fixed/unfixed/still-broken table, and a shipping decision with the number behind it. Plus the code diff.


---

## 4. Evaluation rubric

| Criterion | Points |
|---|---|
| hit-rate@3 before AND after on the SAME 12 questions, per-question record shown, with exactly ONE variable changed between runs | 35 |
| Every failure labelled R/G/Not-In-Corpus with one line of real evidence from the inspection view, and a tally | 25 |
| p50 latency measured before and after — the price of the change, stated honestly | 15 |
| Correct read of which specific failures the change fixed and which it left untouched, named per question | 15 |
| A shipping decision with the number behind it, including an honest 'not worth the latency' if that is what the data says | 10 |
| **Total** | **100** |

*Zero points for polish, UI, or "it works". This mirrors the House rubric: failure-finding and a number that moved are what score.*


---

## 5. Bonus challenge

Your top-3 for the retry query are the same method documented across v1, v2 and v3. Add MMR over the fused candidate list, tune lambda once, and report what it did to hit-rate@3 AND to the diversity of the top-3 — then say whether you would ship it, given that MMR can push the current version's page out of the top-3 in the name of variety.


---

## 6. Submission checklist

- [ ] golden_set.jsonl: 12 real developer questions, each with its known-correct chunk_id
- [ ] Baseline hit-rate@3, written down before any change was made
- [ ] The R/G/Not-In-Corpus tally with one line of evidence per failure
- [ ] Before -> after hit-rate@3 and before -> after p50 latency, in one table
- [ ] Per-question fixed / unfixed table
- [ ] Code diff showing exactly one retrieval change


---

## 7. Common mistakes

- **Adding BM25 AND a reranker in the same run, then reporting one delta — two changes means zero information about which one earned it. Ship one, measure, then consider the other.**
- **Fusing BM25 scores with cosine scores by adding or averaging them. They are not on the same scale and never were; RRF fuses RANKS for exactly this reason.**
- **Writing the golden set from questions you invented to make the retriever look good, instead of from real developer questions. A golden set that only contains questions you already pass measures nothing.**
- **Labelling a failure R because the answer was wrong, without opening the inspection view to check whether the parameter table was sitting in the top-3 all along. If it was there and the default is still wrong, it is G, and no retrieval change on earth will fix it.**
- **Agreeing to swap the embedding model because the lead suggested it. If the tally says most failures are exact-symbol R-failures, a denser embedding is the one thing that structurally cannot help — that is precisely what BM25 exists for.**


---

*Set E of 6. Sets A–F are equivalent in difficulty and objectives; only the domain differs.*

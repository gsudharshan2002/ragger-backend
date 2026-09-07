# Developer Documentation Evaluation

This is the Week 6 evaluation set for the developer-documentation track. `developer_docs_cases.json` ships with cases for `Combine | Apple Developer Documentation.pdf` (the document currently uploaded in this repo instance), but the eval isn't tied to it specifically - point `--cases` at any cases file to evaluate a different document.

## Run

Start with the configured environment and run:

```powershell
uv run python -m evals.run_developer_docs
```

Each run is saved as its own timestamped file under `evals/results/` (nothing gets overwritten), so history accumulates automatically. The `/api/v1/benchmark/developer-docs/results` endpoint - and the Week 6 card in the frontend - always compares the **two most recent runs**, whichever cases/document they used.

Make a run before a RAG change, then run again after the change - the second run's console output reports the score delta against the previous run automatically. `--label` is just an optional free-form tag stored with the run for your own reference (e.g. a strategy name); it no longer controls which file gets written.

To evaluate a different document, write a new cases file in the same shape as `developer_docs_cases.json` (question, expected sources, expected answer keywords, problem type, tags) and pass it explicitly:

```powershell
uv run python -m evals.run_developer_docs --cases evals/my_other_doc_cases.json
```

## Scoring

- **Retrieval score**: a free rule check - was the expected document and page actually retrieved?
- **Answer score**: whether the answer is correct and helpful. Graded by an LLM judge (Groq, binary pass/fail) by default - this catches paraphrased-but-correct answers that a keyword check would wrongly fail. Pass `--no-judge` to fall back to the cruder keyword-matching rule instead (no API key needed).
- **Combined score**: average of retrieval and answer scores.
- **Problem-type scores**: combined score grouped by `retrieval` and `answer_quality`.

Each case result also keeps `keyword_answer_score` (the old rule-based score) and `judge_reasoning` (the judge's one-line explanation) alongside whichever `answer_score` was actually used, so you can always see both.

## Validating the judge

An AI judge you never checked is just a confident number nobody trusts. Before relying on the judge's score, confirm it agrees with your own grading:

```powershell
uv run python -m evals.validate_judge
```

This grades the most recent run's answers with you (pass/fail, judge's verdict hidden until you've graded each one), then reports the agreement rate and saves a `judge_validation_*.json` report. 80%+ agreement is a reasonable bar to trust the judge; below that, read the printed mismatches and fix the judge's prompt (`evals/judge.py`) before trusting its number. Pass `--human-grades <file.json>` (a `{case_id: "pass"|"fail"}` map) to skip interactive prompting, e.g. for CI.

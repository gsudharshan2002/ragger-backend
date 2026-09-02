# Developer Documentation Evaluation

This is the Week 6 evaluation set for the developer-documentation track. It uses the SwiftUI and Core Graphics documents currently stored in this repository.

## Run

Start with the configured environment and run:

```powershell
uv run python -m evals.run_developer_docs
```

The command writes `evals/results/latest.json`. To create a baseline before a RAG change:

```powershell
uv run python -m evals.run_developer_docs --label baseline
```

After one improvement, run:

```powershell
uv run python -m evals.run_developer_docs --label improved
```

The second run reports the score delta against the baseline.

## Scoring

- Retrieval score: expected document and page were retrieved.
- Answer score: every expected keyword appears in the answer.
- Combined score: average of retrieval and answer scores.
- Problem-type scores: combined score grouped by `retrieval` and `answer_quality`.

The checks are deterministic and free. Before relying on an LLM judge, manually grade the same answers from `latest.json` and compare the result with the automatic answer score. A later version can add an LLM judge after that agreement check.

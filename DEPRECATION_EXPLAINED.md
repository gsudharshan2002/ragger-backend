# Why deprecation-checking exists in this RAG/agent system

You know what "deprecated" means (an API field/endpoint that's no longer used). The actual question is: **why does a documentation Q&A system need a dedicated concept and tool for this at all** — why isn't "search the docs well" enough? This doc answers that directly, using your own indexed data as evidence, not abstract theory.

## 1. The failure mode, stated plainly

Your knowledge base contains **two versions of the same API documented side by side**: `old-version.md` (API `2022-11-28`) and `current-version.md` (API `2026-03-10`). Both describe things like `GET /repos/{owner}/{repo}` — the old one says it *has* a `has_downloads` field; the current one says that field *was removed*.

Plain vector/BM25 retrieval has **no concept of "current" vs. "old."** It only measures *semantic similarity to the query*. If someone asks "does `GET /repos/{owner}/{repo}` still include `has_downloads`?", both the old-version chunk and the current-version chunk are **equally, highly relevant by similarity** — they're both directly about that exact field on that exact endpoint. Nothing about cosine similarity or BM25 scoring tells you one of them describes something that no longer exists.

So a plain RAG pipeline can retrieve the *old* chunk, and the LLM can generate a fluent, confident, well-cited answer that says "yes, it includes `has_downloads`" — **completely wrong, and with no visible sign anything is wrong.** That's the actual danger: this isn't a "no answer found" failure (which at least announces itself) — it's a *confidently wrong* answer, which is worse because the user has no reason to double-check it.

## 2. This isn't hypothetical — it's built into your own dataset

Look at `evals/developer_docs_cases.json`. Several cases are explicitly tagged `"mode": "wrong_source"`:

- `gh-002` — asks about `/rate_limit`'s `rate` field on the *current* version; the old version's chunk also mentions `rate` and could get retrieved instead.
- `gh-005` — asks what a submodule's `type` field reports on API `2022-11-28` specifically (answer: `"file"`) — while the current version reports `"submodule"`. Same question text pattern, different correct answer, depending entirely on which chunk gets retrieved.
- `gh-010` — same shape: workflow-dispatch status code is `204` on the old version, `200` on the current one.

These `"wrong_source"` labels are a **named failure category your course has been teaching you to recognize since it started tagging cases this way** — a retrieval that finds *a* real, on-topic chunk, but the *wrong version* of it. It sits in the same taxonomy as `missing_source` (found nothing) and `poor_ranking` (found the right thing, ranked low) — but it's a distinct problem, because the retrieved content isn't irrelevant or badly ranked, it's **relevant to the wrong point in time.**

## 3. Why "search better" doesn't fix this

Reranking, hybrid search, MMR — everything you built in earlier weeks — all optimize the same thing: *relevance to the query*. None of them have any signal for *temporal/version validity*, because that's not a property of text similarity at all. You could have a perfect embedding model and a perfect reranker, and the old-version chunk would *still* score just as well, because it genuinely *is* about the same field, on the same endpoint, in fluent English. The fix has to come from somewhere else entirely: an explicit, structured check that isn't about "how similar is this text" but "is this specific thing still true, **as of this specific version**."

That's precisely why `check_deprecation(endpoint_or_feature, api_version)` exists as its own tool with an **enum-typed version parameter**, separate from `retrieve`/`search_knowledge_base`. It isn't a better search — it's a *different kind of question* (a targeted fact-check on one axis: version), which is why Task Set E's tool-design requirement insists its description can't overlap with the general search tool's. They're solving different problems, not competing at the same one.

## 4. Why this is *the* example that justifies an agent over a fixed workflow

This is also not a coincidence — it's the best available illustration of the exact decision rule from Concept 6/7 ("does the correct path vary by input?"):

- A question about Combine (`combine-001`) needs **zero** version-checking — calling `check_deprecation` on it would be wasted work.
- A question about the *current* version (`gh-004`, `gh-009`) needs version-checking against `CURRENT` only.
- A question explicitly about the *old* version (`gh-005`, `gh-010`) needs version-checking against `OLD` — which your fixed workflow (Step 2), hard-coded to always check `CURRENT`, **cannot do** without a reasoning step first.
- A question spanning *both* versions (`gh-013`) arguably needs the tool called **twice** — something no fixed sequence of steps can express at all.

That's the whole reason this task set built a fixed workflow *and* an agent side by side: deprecation-checking is the one part of this domain where "what to do next" is genuinely determined by *the content of the question itself*, not by a rule you could have written in advance. That's precisely the condition under which Concept 6/7 said an agent earns its cost over a workflow — everything else in this task (retrieving generally-relevant docs, generating a final answer) is stable enough to hard-code.

## 5. The one-sentence version

**Deprecation-checking exists because RAG over documentation that changes over time has a failure mode retrieval quality alone cannot fix — retrieving a real, well-matched, but out-of-date answer — and it's the single clearest case in this whole task where whether an agent is worth using can be settled with real numbers instead of an opinion.**

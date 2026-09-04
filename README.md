# Ask My Papers — a RAG demo over my own research

A small retrieval-augmented generation (RAG) tool: ask natural-language
questions about my own published research and methodology docs, and get
answers grounded in (and citing) the actual source text — instead of an LLM
guessing from its training data.

## Architecture

```
[Offline, run once: ingest.py]
   PDFs in docs/  --> extract text --> strip known boilerplate lines
                  --> chunk on paragraph/sentence boundaries (~800 chars, ~100 overlap)
                  --> embed each chunk (OpenAI text-embedding-3-small)
                  --> store in local Chroma vector DB (chroma_db/)

[Online, every question: query.py / app.py]
   question --> hybrid search: vector similarity + BM25 keyword search, fused
            --> reranked down to the top-k by an LLM judge
            --> stuff those chunks + question into a prompt
            --> OpenAI generates an answer, grounded only in retrieved context
```

The LLM's weights never change — nothing here is "training." Retrieval and
generation both happen at query time; ingestion only builds the searchable
index ahead of time.

## Agentic RAG (agent_graph.py / agent_query.py)

`query.py` above is a single fixed pass: retrieve once, generate once. As an
upgrade, `agent_graph.py` wraps the same retrieve/generate building blocks
in a **LangGraph** agent that can decide retrieval wasn't good enough and
correct itself, and that remembers the conversation:

```
contextualize --> decompose --> retrieve --[decomposed: multi-query]--> END
                                          `-[single query]--> grade --[insufficient, retries left]--> rewrite --> retrieve   (loop)
                                                                    `-[sufficient, or out of retries]--> END
```

- **contextualize** — resolves a context-dependent follow-up ("what method
  do they use for that?") into a standalone search query using the
  conversation history, before the first retrieval attempt.
- **decompose** — checks whether the question needs pulling from multiple
  documents/topics and synthesizing across them (a comparison), and if so,
  splits it into up to `MAX_SUBQUERIES` focused sub-questions instead of one
  query trying to represent the whole comparison. See Query decomposition
  below.
- **retrieve** — runs `retrieve_reranked()` once per sub-query (just one,
  for a normal question) and merges the results, deduped by chunk id. A
  decomposed (multi-query) retrieval skips grading entirely and goes
  straight to generation — see below for why.
- **grade** — for a normal, single-query retrieval, an LLM call judges
  whether the retrieved chunks actually
  contain enough to answer the question, via **JSON-mode structured
  output** (`{"sufficient": ..., "confidence": ..., "reason": ...}`) rather
  than parsing a raw "SUFFICIENT"/"INSUFFICIENT" string — the old version's
  substring check also matched inside "INSUFFICIENT", which needed a second
  check just to work around itself. This pattern is sometimes called
  "Corrective RAG" / self-RAG.
- **rewrite** — if not, an LLM call rewrites the search query (also JSON
  mode, also told *why* the last attempt fell short) and the graph loops
  back to retrieve — capped at `MAX_RETRIES` so a genuinely unanswerable
  question can't loop forever.
- **memory** — the graph is compiled with a `MemorySaver` checkpointer, so
  calling `prepare_turn(question, thread_id=...)` with the same `thread_id`
  resumes that conversation's accumulated state automatically. Follow-ups
  like "what about the sample size?" resolve correctly because
  `chat_history` persists across turns without you managing it by hand.

**Generation happens outside the graph, streamed.** LangGraph's `.invoke()`
only returns once a run reaches `END`, so a node that calls the LLM can't
hand tokens back as they arrive — the whole answer would appear at once no
matter how it's written internally. So the graph above only decides *what
to answer with*; two functions handle the rest:

- `prepare_turn(question, thread_id)` runs the graph and returns the
  finalized `retrieved_chunks` (no LLM generation call yet).
- `stream_answer(question, state, thread_id)` streams the final answer
  token-by-token, then persists it (+ updated `chat_history`) back into the
  thread's checkpointed state via `AGENT.update_state()`.
- `run_turn(question, thread_id)` composes both, non-streaming, for callers
  that just want the final dict back (`eval.py`, scripts).

Run it:

```
python agent_query.py
```

## Query decomposition (decompose_node)

A single vector search over one query naturally biases toward whichever
document's terms dominate it — so "how does X in paper A compare to Y in
paper B?" tends to retrieve mostly one side of the comparison, no matter how
well-phrased the query is. `decompose_node` fixes this at the *question*
level rather than the retrieval level: it asks an LLM (JSON mode) whether
the question needs pulling from multiple documents/topics and synthesizing
across them, and if so, splits it into focused sub-questions — one retrieval
pass each, merged and deduped by chunk id.

Two things had to be true for this to actually work, both found by testing
it against the real failure case, not by inspection:

1. **Decomposed retrieval skips the grade/rewrite loop.** The first version
   routed a decomposed retrieval through the same grader as everything
   else — and the grader, judging "does this context state the answer,"
   reasonably called a correct multi-document retrieval "insufficient"
   (a comparison's source material rarely states the comparison verbatim).
   That triggered a rewrite, and rewrite_node collapses back to one query on
   retry (see its docstring) — silently undoing the decomposition that was
   actually working. Fix: a decomposed (multi sub-query) retrieval routes
   straight to generation; only a single-query retrieval is eligible for
   grading and retry.
2. **The generation prompt has to permit synthesis.** Even with the right
   chunks from both documents in context, the original prompt's "say you
   don't know if the context doesn't contain the answer" instruction made
   the model refuse to synthesize a comparison across passages that
   individually don't state one. `build_prompt()` now explicitly allows
   synthesizing across multiple passages as long as every individual claim
   stays grounded in the context.

Verified end-to-end: "How does the fiber bundle segmentation paper compare
to the diffusion MRI paper?" — previously always "I don't know" — now
decomposes into 3 sub-questions, retrieves from both papers, and generates
a real, correctly-cited comparison, with 0 retries.

## Hybrid search (rag_core.bm25_search / hybrid_retrieve)

Dense vector search can miss exact terms it doesn't treat as semantically
distinctive — specific numbers, acronyms, model names. `hybrid_retrieve()`
runs a BM25 keyword search (`rank_bm25`) alongside the existing vector
search and fuses both ranked lists with **Reciprocal Rank Fusion**: each
chunk's fused score is `sum(1 / (RRF_K + rank))` across whichever list(s) it
appears in, so a chunk both methods agree on outranks one only one method
liked — without needing to normalize two differently-scaled score types
against each other, since RRF only looks at rank position. `retrieve_reranked()`
now calls this instead of plain vector `retrieve()` for its candidate pool.

The BM25 index is built once (from every chunk currently in Chroma) and
cached in-process — rebuild it by restarting the process after re-running
`ingest.py`, the same caveat as the in-memory `MemorySaver` checkpointer.

## Reranking (rag_core.rerank / retrieve_reranked)

Vector similarity (`retrieve()`) is a bi-encoder: it embeds the question and
each chunk independently, so a chunk that's merely *topically similar* can
outrank one that actually *answers* the question. `retrieve_reranked()`
fixes this with the standard two-stage pattern:

1. Cast a wide net — `hybrid_retrieve()` for `RERANK_CANDIDATES` (20) candidates.
2. `rerank()` shows the question and all candidates *together* to an LLM
   judge, which scores each 0–10, and keeps only the top `k`.

This is what every entry point (`query.py`, `app.py`, and the agent's
`retrieve` node) calls now instead of raw `retrieve()`. It uses the same
OpenAI chat model as the judge rather than a dedicated cross-encoder (e.g.
via `sentence-transformers`), to avoid a second ML stack and keep the
project on one provider end to end — the tradeoff is one extra LLM call per
retrieval instead of a local model load.

Measured effect (via `eval.py`):

| | Before reranking | + reranking | + hybrid search &amp; chunking |
|---|---|---|---|
| Faithfulness | 5.00/5 | 5.00/5 | 5.00/5 |
| Relevancy | 3.67/5 | 4.33/5 | **4.33/5** |
| Questions needing a self-correction retry | 4/6 | 2/6 | **1/6** |

Two questions that previously got "I don't know" (institution affiliation,
shared authorship) now answer correctly. Retries kept dropping with each
retrieval-quality improvement — fewer retries means the agentic loop's
self-correction is doing less work, because the first retrieval pass is
simply better each time.

One gotcha worth knowing if you touch `rerank()`: chunks carry embedded
newlines from PDF extraction, so building the judge's numbered passage list
with `\n\n` between entries made passage boundaries ambiguous to the model
and silently corrupted the index→score mapping. Flattening each passage to
one line before numbering it fixed this — a good example of how a prompt
formatting bug can look exactly like a model-quality problem.

## Chunking (rag_core.chunk_text / strip_boilerplate)

Two independent problems, one fix each, both in `ingest.py`'s pipeline:

- **Structure-aware chunking.** The original `chunk_text()` sliced text at a
  raw fixed character offset, which can cut a sentence — or a table row —
  in half. It now splits on paragraph boundaries first, and only falls back
  to sentence boundaries for a paragraph bigger than a whole chunk, packing
  pieces greedily up to `chunk_size`. Overlap is the tail *sentences* of one
  chunk carried into the next, not an arbitrary character count that could
  itself land mid-sentence.
- **Boilerplate stripping.** pypdf extracts text in raw layout order, so on
  a multi-column preprint, a bioRxiv license footer can land *mid-paragraph*
  next to real content — this actually happened in testing: a chunk started
  with "...CC-BY-NC-ND 4.0 International license..." and, in the same
  800-character window, went straight into real acquisition parameters
  (b-values, TEs). `strip_boilerplate()` drops lines matching a handful of
  known preprint-server boilerplate patterns before chunking. It's a
  targeted fix for that specific, observed failure mode — not a general
  layout parser (a real one, e.g. PyMuPDF with column/footer detection,
  would generalize better but is a heavier dependency for what's currently
  a one-pattern-family problem).

Because chunk boundaries shift whenever `chunk_text()` changes, `ingest.py`
now deletes each file's old chunks (`collection.delete(where={"source": ...})`)
before re-inserting — otherwise `upsert()` only overwrites ids that still
exist, and a shrinking chunk count leaves orphaned stale chunks behind that
never show up in ingest.py's own printed count but still get retrieved.

## Source display (rag_core.summarize_sources)

A question fully answerable from one paper can legitimately retrieve several
*distinct* passages (different chunk ids, different text) from that same
file — e.g. 6 unique chunks, all from `2508.12942v2.pdf`. Printing one line
per chunk with a bare filename made that read as the same source
"duplicated" in the sources list, even though nothing was actually
duplicated (verified by checking chunk ids directly — no repeats, in the
vector store or in any retrieval path). `summarize_sources()` groups
retrieved chunks by file for display, so `query.py` and `agent_query.py`
print one row per document with a passage count (e.g. `2508.12942v2.pdf —
6 passages (rerank scores [10, 9, 9, 9, 8, 8])`), and `app.py`'s sources
expander groups the same way instead of repeating the filename as a header
once per chunk. This is purely a display fix — it doesn't touch retrieval,
reranking, or generation.

## Evaluation (eval.py)

`eval.py` runs the agent against a small question set
(`eval_questions.json`, grounded in the actual papers in `docs/`) and scores
each answer with an LLM judge on two reference-free axes — no hand-written
"gold answers" required:

- **faithfulness (1-5)** — is every claim in the answer actually supported
  by the retrieved context, including correctly saying "I don't know" when
  it isn't? This is the metric that catches hallucination.
- **relevancy (1-5)** — does the answer address what was actually asked?

It also reports how many questions needed a query rewrite, as a rough proxy
for how often naive single-shot retrieval would have failed on its own. One
question (`out-of-scope`) is deliberately unrelated to both papers — a
correctly-behaving system should score high on faithfulness (refuses rather
than fabricates) and low on relevancy (it can't answer something that
isn't there), which is itself evidence the system isn't hallucinating.

Run it:

```
python eval.py
```

Results are printed to the console and written to `eval_results.json`.

## Running it as a service (api.py)

The CLIs and Streamlit app import the pipeline directly. `api.py` puts it
behind HTTP so it can be deployed, monitored, and called by other clients.

```
GET  /health        liveness — touches nothing external (App Runner polls this)
GET  /ready         corpus size + build SHA
POST /ask           full answer as JSON
POST /ask/stream    Server-Sent Events: status -> sources -> token -> done
```

```bash
uvicorn api:app --port 8000
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -H 'x-api-key: $API_KEY' -d '{"question":"What gradient strength was used?"}'
```

Three design notes that aren't obvious from the code:

- **Streaming emits progress events, not just tokens.** Time-to-first-token is
  dominated by the *agent graph*, not generation: a turn makes 5–12 OpenAI
  calls (contextualize, decompose, embed, rerank, grade, possibly
  rewrite-and-retry) over 10–35s before the first answer token exists. The
  `status` events keep the connection warm ahead of any proxy idle timeout and
  give the user something truthful to watch.
- **Every SSE payload is JSON-encoded.** SSE frames are delimited by a blank
  line and answers are markdown full of newlines, so interpolating a raw token
  into `data: ...` shatters the stream into garbage frames. There's a test that
  fails if this regresses.
- **Handlers are plain `def`.** The pipeline is synchronous, so Starlette
  offloads them to its threadpool; the streaming endpoint runs its blocking
  work on one worker thread feeding an `asyncio.Queue`, rather than paying a
  threadpool round-trip per token.

## Conversation state (history_store.py)

History used to live in LangGraph's in-process `MemorySaver`. That made the
service **stateful**: turn 2 of a conversation landing on a different instance
or worker found no history, so `contextualize_node` had nothing to resolve the
follow-up against, retrieval ran on the raw pronoun, and the user got a
confidently wrong answer with nothing in the logs. It also grew without bound.

It now lives in DynamoDB (`thread_id` PK, JSON history, 24h TTL), passed into
the graph per call and written back by the caller. Instances are fungible, so
horizontal scaling actually works. `HISTORY_BACKEND=local` keeps an in-memory
version for tests and offline use.

## Infrastructure (infra/)

Terraform, flat layout — ~15 resources doesn't warrant modules. State lives in
S3 with native lockfile locking, because a local state file is worthless the
moment CI runs `terraform apply`.

| Resource | Why |
|---|---|
| ECR + lifecycle policy | Image registry; keeps only the 10 newest, or it grows ~1.2GB per deploy forever |
| DynamoDB table | Conversation history, on-demand billing, TTL cleanup |
| Secrets Manager ×2 | OpenAI key and client API key |
| IAM OIDC provider + roles | GitHub Actions auth with no long-lived keys |
| Budget alarm | Alerts at 50%, 100%, and forecast-to-exceed |

Two deliberate choices:

- **Secrets are created empty.** An `aws_secretsmanager_secret_version` would
  write the value into `terraform.tfstate`, which lives in S3. Values are set
  out-of-band: `aws secretsmanager put-secret-value --secret-id rag-api/openai`.
- **The OIDC trust policy scopes `sub`** to `repo:<owner>/<repo>:ref:refs/heads/main`.
  Unscoped, the provider only attests that a token came from GitHub Actions —
  not that it came from *this* repo — so any repository on GitHub could assume
  the role.

## CI/CD (.github/workflows/)

`ci.yml` on every PR: ruff, pytest, gitleaks over full history, `docker build`,
and `terraform fmt`/`validate`. The docker job asserts the security controls
actually held — it fails if `.env` reached the image or the container runs as
root — rather than trusting that `.dockerignore` did its job.

`deploy.yml` on main: OIDC assume-role → build → tag with the **git SHA** (not
`:latest`; you can't roll back or tell what's running from a floating tag) →
push → **run the eval gate inside the built image** → `terraform apply` →
deploy → poll until `RUNNING` → smoke-test `/health`, `/ready`, and a real
`/ask` against the live URL.

The eval gate is the interesting part: `eval.py` exits non-zero when mean
faithfulness or relevancy falls below threshold, so a retrieval regression
fails the build. It runs *inside the image being deployed*, so the artifact
measured is the exact one shipping. The key comes from Secrets Manager via the
same OIDC role, so it exists in one place and is never a GitHub secret.

Thresholds sit at 4.0/5 rather than 4.5 on purpose — the LLM judge still
varies run-to-run (observed 4.33 to 5.00 on identical answers) even at
`temperature=0`, and a gate that flakes is a gate people learn to bypass.

## Setup

1. `pip install -r requirements.txt`
2. Create a `.env` file and fill in your real `OPENAI_API_KEY` only.
3. Drop your PDFs (papers, methodology docs, etc.) into `docs/`.
4. Build the knowledge base: `python ingest.py`
5. Ask questions — every path below retrieves via `retrieve_reranked()` and
   streams its answer token-by-token:
   - CLI, single-shot: `python query.py`
   - CLI, agentic + memory: `python agent_query.py`
   - Web UI, agentic + memory + streaming (better for demos): `streamlit run app.py`
   - Evaluate answer quality: `python eval.py`

## Known limitations (worth naming yourself in an interview — it shows you understand the tradeoffs, not just that you called some libraries)

- **Boilerplate stripping is pattern-based, not a real layout parser.**
  `strip_boilerplate()` fixes the specific bioRxiv-footer-in-paragraph
  failure mode observed in testing, but it's a handful of regexes, not a
  general solution — a genuinely different journal template's boilerplate
  would sail right through until someone adds a pattern for it.
- **Reranking is LLM-based, not a dedicated cross-encoder.** `rerank()` asks
  a general chat model to score relevance rather than using a model built
  specifically for reranking — cheaper to integrate (no second ML stack,
  no new API key) but adds one more LLM call's worth of latency/cost per
  retrieval, and is one more place a prompt-formatting bug can silently
  degrade quality (see the newline gotcha in the Reranking section above).
- **BM25 index is in-memory and built lazily.** Same caveat as
  `MemorySaver`: it's cached per-process and built from whatever's in
  Chroma the first time it's needed, so it goes stale if you re-run
  `ingest.py` without restarting whatever process is holding the index.
- **Decomposition adds cost, and is itself an LLM call that can misjudge.**
  Every question now costs one extra JSON-mode call just to *decide*
  whether to decompose, and a wrongly-decomposed question fans out into
  several retrieval passes for no benefit — the same "one model's opinion,
  no cross-check" caveat as grading, just earlier in the pipeline.
- **The corpus is baked into the image.** Self-contained and hermetic, and it
  removes BM25 cache staleness by construction (the corpus can't change under
  a running process). The cost is that re-ingesting means rebuilding and
  redeploying the image — fine at 2 papers, wrong at 2,000.
- **One worker, and rate limiting is per-process.** With multiple instances
  the limit is effectively per-instance rather than global; a shared counter
  (Redis/DynamoDB) would be needed for a real limit.
- **BM25 index is rebuilt per instance.** Each one holds its own copy in
  memory — fine at 140 chunks, not at scale.
- **Grade/rewrite are still single LLM calls, no cross-check.** JSON mode
  fixed the brittle *parsing*, but the *judgment* itself is still one
  model's opinion with no second opinion or calibration against a labeled
  set — a confident wrong grade is still possible, just no longer a parsing
  bug when it happens.
- **Single knowledge source.** This only indexes PDFs; a hybrid version
  (like the "query my structured experiment data in plain English" idea we
  discussed) would combine this with a text-to-SQL path for numeric data.

## Suggested resume bullet — only once you've actually run this end-to-end

Something like:

> Built an agentic retrieval-augmented generation (RAG) system (OpenAI,
> ChromaDB, LangGraph, Streamlit) with hybrid (vector + BM25) retrieval,
> LLM reranking, query decomposition for cross-document synthesis, a
> self-correcting grade/rewrite loop with structured LLM output, streamed
> and persistent multi-turn memory, and an LLM-judge evaluation harness
> scoring faithfulness and relevancy — enabling grounded, citation-backed
> Q&A over a personal research corpus, in both a CLI and a chat UI.

Don't add this until you've actually gotten `ingest.py`, `agent_query.py`,
and `eval.py` running against your own PDFs — you want to be able to answer
any follow-up question about it without hesitation.

"""
rag_core.py
Shared building blocks for the RAG pipeline:
  chunk_text()        -> split long text into overlapping chunks
  embed_texts()        -> turn text chunks into vectors (OpenAI embeddings)
  get_collection()     -> open/create the local Chroma vector store
  retrieve()            -> semantic (vector) search: question -> top-k relevant chunks
  bm25_search()          -> keyword search over the same corpus
  hybrid_retrieve()      -> fuse vector + keyword search (reciprocal rank fusion)
  rerank()              -> re-score retrieved candidates for relevance, precisely
  retrieve_reranked()   -> hybrid_retrieve() a wide candidate set, then rerank to the best few
  generate_answer()    -> stuff retrieved chunks into a prompt, ask OpenAI to answer
  generate_answer_stream() -> same, but yields the answer token-by-token
  summarize_sources()   -> group retrieved chunks by source file, for display

Keeping this logic in one file (instead of copy-pasting into ingest.py /
query.py / app.py) is deliberate: it's the same pattern you'd use in a real
production RAG service, where ingestion and querying are separate jobs that
share a single retrieval/generation library.
"""

import json
import os
import re
import threading

import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()  # reads OPENAI_API_KEY from a local .env file (no-op in containers)

_API_KEY = os.getenv("OPENAI_API_KEY")
if not _API_KEY:
    # A bare KeyError here dies before anything useful is logged — in a
    # container that surfaces as an opaque startup crash and a rolled-back
    # deployment, with no hint that secret injection is what failed.
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Locally: put it in .env (see .env.example). "
        "In AWS: check the App Runner secret injection from Secrets Manager."
    )

# timeout: the SDK default is 600s — far past App Runner's fixed ~120s request
# cap, so a hung upstream call would blow the platform limit with no circuit
# breaker of our own.
OPENAI_CLIENT = OpenAI(api_key=_API_KEY, timeout=30.0, max_retries=2)

EMBEDDING_MODEL = "text-embedding-3-small"   # cheap + good enough for a demo
GENERATION_MODEL = "gpt-4o-mini"             # OpenAI chat model for grounded answers
# Absolute default: a relative path resolves against the process CWD, and if it
# resolves somewhere unexpected, get_or_create_collection() *silently creates an
# empty collection* rather than failing — every answer then becomes "I don't
# know" with no error anywhere. api.py fails startup if the count is 0.
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "papers")


# ---------------------------------------------------------------------------
# 1. Chunking
# ---------------------------------------------------------------------------
# Recurring bioRxiv/preprint-server boilerplate. pypdf extracts text in raw
# layout order, so on a multi-column preprint this footer text can land
# mid-paragraph, right next to real content — a concrete example turned up
# during testing: a license notice spliced into a sentence about scan
# parameters, in the same 800-char window. Stripping known boilerplate
# lines before chunking is a targeted fix for that observed failure mode —
# not a general layout parser (see the README's chunking notes).
_BOILERPLATE_PATTERNS = [
    re.compile(r"CC-BY.{0,30}International license", re.IGNORECASE),
    re.compile(r"author/funder.*granted bioRxiv", re.IGNORECASE),
    re.compile(r"copyright holder for this preprint", re.IGNORECASE),
    re.compile(r"which was not certified by peer review", re.IGNORECASE),
    re.compile(r"bioRxiv preprint\s*$", re.IGNORECASE),
    re.compile(r"^\s*https?://doi\.org/\S+\s*$"),
]


def strip_boilerplate(text: str) -> str:
    """Drop lines matching known preprint-server boilerplate (see above)."""
    lines = text.split("\n")
    kept = [ln for ln in lines if not any(p.search(ln) for p in _BOILERPLATE_PATTERNS)]
    return "\n".join(kept)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Split text into chunks along paragraph/sentence boundaries, instead of a
    raw fixed-size character window that can cut a sentence (or a table row)
    in half. Paragraphs are packed greedily up to chunk_size; any paragraph
    longer than chunk_size on its own is first split on sentence boundaries,
    so nothing ever forces a mid-sentence cut. Overlap is achieved by
    carrying the tail sentence(s) of one chunk into the start of the next,
    rather than an arbitrary character offset that could itself land
    mid-sentence.
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    pieces = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            pieces.append(para)
        else:
            pieces.extend(s for s in re.split(r"(?<=[.!?])\s+", para) if s)

    chunks = []
    current: list[str] = []
    current_len = 0
    for piece in pieces:
        piece_len = len(piece) + 1
        if current and current_len + piece_len > chunk_size:
            chunks.append(" ".join(current))
            # Carry the tail of this chunk forward as overlap, by whole
            # piece (sentence/paragraph), not by raw character count.
            tail: list[str] = []
            tail_len = 0
            for p in reversed(current):
                if tail_len + len(p) > overlap:
                    break
                tail.insert(0, p)
                tail_len += len(p)
            current, current_len = tail, tail_len
        current.append(piece)
        current_len += piece_len
    if current:
        chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------------------------
# 2. Embeddings
# ---------------------------------------------------------------------------
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of text chunks with OpenAI's embedding model."""
    response = OPENAI_CLIENT.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# 3. Vector store
# ---------------------------------------------------------------------------
_COLLECTION = None
_COLLECTION_LOCK = threading.Lock()


def get_collection():
    """
    Open (or create) the persistent local Chroma collection.

    Cached as a process-wide singleton: this used to construct a fresh
    PersistentClient on every retrieve() call, which under a threaded server
    means per-request object churn against the same SQLite file. Double-checked
    locking so concurrent first requests don't race to build it.
    """
    global _COLLECTION
    if _COLLECTION is None:
        with _COLLECTION_LOCK:
            if _COLLECTION is None:
                client = chromadb.PersistentClient(path=CHROMA_DIR)
                _COLLECTION = client.get_or_create_collection(name=COLLECTION_NAME)
    return _COLLECTION


# ---------------------------------------------------------------------------
# 4. Retrieval
# ---------------------------------------------------------------------------
def retrieve(question: str, k: int = 4) -> list[dict]:
    """
    Embed the question, run a similarity search against Chroma, and return
    the top-k matching chunks along with their source metadata.
    """
    collection = get_collection()
    query_embedding = embed_texts([question])[0]

    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    retrieved = []
    for doc_id, doc, meta, distance in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        retrieved.append({
            "id": doc_id,
            "text": doc,
            "source": meta.get("source", "unknown"),
            "distance": distance,
        })
    return retrieved


# ---------------------------------------------------------------------------
# 5. Hybrid (keyword) search
# ---------------------------------------------------------------------------
# Dense-vector search can miss exact terms it doesn't recognize as
# semantically distinctive — model names, acronyms, specific numbers. BM25
# (classic TF-IDF-family keyword ranking) catches those directly. Fusing
# both candidate lists gets the benefit of each without picking one.
_BM25_INDEX = None
_BM25_CHUNKS = None
_BM25_LOCK = threading.Lock()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _load_bm25_index():
    """
    Build (and cache) a BM25 index over every chunk currently in Chroma.

    Cached for the life of the process, so it goes stale if you re-run
    ingest.py without restarting. In the deployed container the corpus is
    baked into the image and therefore immutable for the life of the
    process, which removes that staleness by construction.

    Double-checked locking: without it, concurrent first requests each build
    the whole index (wasted work plus a transient memory spike) — a
    thundering herd on cold start. Call warm_caches() at startup to pay this
    cost once, before serving traffic.
    """
    global _BM25_INDEX, _BM25_CHUNKS
    if _BM25_INDEX is None:
        with _BM25_LOCK:
            if _BM25_INDEX is None:
                collection = get_collection()
                result = collection.get(include=["documents", "metadatas"])
                chunks = [
                    {"id": doc_id, "text": doc, "source": meta.get("source", "unknown")}
                    for doc_id, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
                ]
                # Assign the index last: _BM25_INDEX is the "is it ready?"
                # flag other threads check without the lock.
                _BM25_CHUNKS = chunks
                _BM25_INDEX = BM25Okapi([_tokenize(c["text"]) for c in chunks])
    return _BM25_INDEX, _BM25_CHUNKS


def warm_caches() -> int:
    """
    Open the Chroma collection and build the BM25 index up front, returning
    the chunk count. Called from the API's startup hook so the first real
    request doesn't pay for it — and so a misconfigured CHROMA_DIR (which
    silently yields an *empty* collection rather than an error) is caught at
    startup instead of turning every answer into "I don't know".
    """
    collection = get_collection()
    count = collection.count()
    if count:
        _load_bm25_index()
    return count


def bm25_search(question: str, k: int = 10) -> list[dict]:
    """Keyword search: top-k chunks by BM25 score against the question."""
    index, chunks = _load_bm25_index()
    if not chunks:
        return []
    scores = index.get_scores(_tokenize(question))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [{**chunks[i], "bm25_score": float(scores[i])} for i in ranked if scores[i] > 0]


RRF_K = 60  # standard reciprocal-rank-fusion damping constant


def hybrid_retrieve(question: str, k: int = 10) -> list[dict]:
    """
    Fuse dense vector search and BM25 keyword search with Reciprocal Rank
    Fusion: each chunk's fused score is the sum of 1/(RRF_K + rank) across
    whichever list(s) it appears in, so a chunk that both retrieval methods
    agree on outranks one only one method liked — without having to
    normalize or compare two differently-scaled score types directly (RRF
    only looks at rank position, sidestepping that entirely).

    Every returned chunk always has both a "distance" and a "bm25_score"
    key, one of which is None if that method didn't surface it — a chunk
    found only by keyword search (not in the vector top-k, or vice versa)
    otherwise silently carries just one method's field, and every caller
    downstream that assumes both keys always exist (there were several)
    breaks with a KeyError the first time a keyword-only hit survives
    reranking into the final results.
    """
    vector_hits = retrieve(question, k=k)
    keyword_hits = bm25_search(question, k=k)

    fused_scores: dict[str, float] = {}
    fused_chunks: dict[str, dict] = {}
    for hit_list in (vector_hits, keyword_hits):
        for rank, chunk in enumerate(hit_list):
            key = chunk["id"]
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            entry = fused_chunks.setdefault(key, {
                "id": chunk["id"],
                "text": chunk["text"],
                "source": chunk["source"],
                "distance": None,
                "bm25_score": None,
            })
            if chunk.get("distance") is not None:
                entry["distance"] = chunk["distance"]
            if chunk.get("bm25_score") is not None:
                entry["bm25_score"] = chunk["bm25_score"]

    ranked_keys = sorted(fused_scores, key=lambda key: fused_scores[key], reverse=True)[:k]
    return [{**fused_chunks[key], "rrf_score": fused_scores[key]} for key in ranked_keys]


# ---------------------------------------------------------------------------
# 6. Reranking
# ---------------------------------------------------------------------------
RERANK_CANDIDATES = 20   # cast a wide net with cheap vector search...
FINAL_K = 4              # ...then keep only this many after reranking


def rerank(question: str, chunks: list[dict], top_n: int = FINAL_K) -> list[dict]:
    """
    Re-score retrieval candidates for relevance with an LLM judge, and return
    the top_n — instead of trusting raw vector-similarity order.

    Vector similarity (what retrieve() uses) is a bi-encoder: it embeds the
    question and each chunk independently, so a chunk that's topically
    *similar* can outrank one that actually *answers* the question. A
    reranker looks at the question and each candidate *together*, which is
    slower but far more precise — hence the standard pattern: retrieve a
    wider candidate set than you need, then rerank down to a precise top-k.

    This uses the same OpenAI chat model as an LLM judge rather than a
    dedicated cross-encoder model (e.g. via sentence-transformers), to avoid
    pulling in a second ML stack (torch, etc.) just for reranking, and to
    keep the whole project on one provider (OpenAI) end to end.
    """
    if len(chunks) <= top_n:
        return chunks

    # Flatten each passage to a single line before numbering it. Chunks carry
    # embedded newlines from the source PDF; joining multi-line passages with
    # "\n\n" makes their boundaries ambiguous to the model (a mid-passage
    # blank-ish line reads like the start of the next entry), which silently
    # corrupts the index -> score mapping. One passage per line sidesteps that.
    numbered = "\n".join(f"[{i}] {' '.join(c['text'][:600].split())}" for i, c in enumerate(chunks))
    prompt = (
        "Score how relevant each passage is to answering the question, from "
        "0 (irrelevant) to 10 (directly answers it).\n\n"
        f"QUESTION: {question}\n\n"
        f"PASSAGES (one per line):\n{numbered}\n\n"
        "Reply with ONLY a JSON array, one object per passage, no other text: "
        '[{"index": <int>, "score": <int>}, ...]'
    )
    response = OPENAI_CLIENT.chat.completions.create(
        model=GENERATION_MODEL,
        max_tokens=600,
        temperature=0,  # grade_node/decompose_node already pin this; a judge
                        # that scores the same passages differently run-to-run
                        # makes the eval gate flaky
        messages=[{"role": "user", "content": prompt}],
    )
    raw = (response.choices[0].message.content or "").strip()

    try:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        scores = {int(item["index"]): item["score"] for item in json.loads(match.group(0))}
    except Exception:
        # Judge call failed or returned something unparsable — fall back to
        # the original vector-similarity order instead of crashing.
        return chunks[:top_n]

    ranked = sorted(enumerate(chunks), key=lambda pair: scores.get(pair[0], -1), reverse=True)
    return [{**chunk, "rerank_score": scores.get(i)} for i, chunk in ranked[:top_n]]


def retrieve_reranked(question: str, k: int = FINAL_K, candidates: int = RERANK_CANDIDATES) -> list[dict]:
    """Cast a wide hybrid (vector + BM25) net, then keep only the top-k after reranking."""
    wide = hybrid_retrieve(question, k=candidates)
    return rerank(question, wide, top_n=k)


# ---------------------------------------------------------------------------
# 7. Generation
# ---------------------------------------------------------------------------
def build_prompt(
    question: str,
    retrieved_chunks: list[dict],
    chat_history: list[dict] | None = None,
) -> str:
    context_block = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks
    )
    history_block = ""
    if chat_history:
        turns = "\n".join(f"{h['role'].capitalize()}: {h['content']}" for h in chat_history)
        history_block = (
            "Earlier turns in this conversation (for context on follow-up "
            f"questions like \"what about...\" — don't re-answer these):\n{turns}\n\n"
        )
    return (
        f"{history_block}"
        "Answer the QUESTION using ONLY the context below. You may synthesize "
        "across multiple passages — e.g. comparing or relating information "
        "from different sources — as long as every individual factual claim "
        "you make is grounded in the context; a comparison rarely appears as "
        "a single sentence someone already wrote, so don't require that. "
        "Only say you don't know if the context is missing the underlying "
        "facts needed, not merely because no single passage states the "
        "conclusion outright. Never make anything up. Cite the source file "
        "for each claim.\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {question}"
    )


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    chat_history: list[dict] | None = None,
) -> str:
    prompt = build_prompt(question, retrieved_chunks, chat_history)
    response = OPENAI_CLIENT.chat.completions.create(
        model=GENERATION_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def generate_answer_stream(
    question: str,
    retrieved_chunks: list[dict],
    chat_history: list[dict] | None = None,
):
    """
    Same as generate_answer(), but yields the answer incrementally as OpenAI
    generates it, instead of blocking until the whole response is done.
    Useful anywhere a human is watching in real time (a CLI, a chat UI) —
    the retrieve/grade/rewrite steps upstream already take a couple of
    seconds; streaming at least means the *answer* doesn't feel like it's
    hanging too.
    """
    prompt = build_prompt(question, retrieved_chunks, chat_history)
    stream = OPENAI_CLIENT.chat.completions.create(
        model=GENERATION_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ---------------------------------------------------------------------------
# 8. Display helpers
# ---------------------------------------------------------------------------
def summarize_sources(chunks: list[dict]) -> list[dict]:
    """
    Group retrieved chunks by source file for display. A question fully
    answerable from one paper can easily retrieve several distinct passages
    from that same file (different chunk ids, different text) — printed
    one-per-line with a bare filename, that reads as the same source
    "duplicated" even though nothing is actually duplicated. This groups
    them into one row per file with a passage count, so the display matches
    what's actually happening: several different citations, one document.
    """
    grouped: dict[str, dict] = {}
    order: list[str] = []
    for c in chunks:
        src = c["source"]
        if src not in grouped:
            grouped[src] = {"source": src, "count": 0, "distances": [], "rerank_scores": []}
            order.append(src)
        entry = grouped[src]
        entry["count"] += 1
        if c.get("distance") is not None:
            entry["distances"].append(c["distance"])
        if c.get("rerank_score") is not None:
            entry["rerank_scores"].append(c["rerank_score"])
    return [grouped[src] for src in order]

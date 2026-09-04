"""
api.py
HTTP service wrapping the agentic RAG pipeline.

Run locally:
    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /health       liveness — no OpenAI, no Chroma (App Runner polls this)
    GET  /ready        readiness — corpus size, build SHA
    POST /ask          full answer as JSON
    POST /ask/stream   Server-Sent Events: status -> sources -> token -> done

Two things here are less obvious than they look:

1. Time-to-first-token is dominated by the *agent graph*, not generation.
   A turn runs 5-12 OpenAI calls (contextualize, decompose, embed, rerank,
   grade, maybe rewrite-and-retry) before a single answer token exists —
   10-35s. So the streaming endpoint emits `status` events during that phase:
   it keeps the connection warm ahead of any proxy idle timeout, and it gives
   the user something truthful to look at instead of a dead spinner.

2. The pipeline is entirely synchronous (sync OpenAI client, sync generators).
   Rather than rewrite it async, blocking work runs in a worker thread and
   feeds an asyncio.Queue. Handlers that don't stream are plain `def`, which
   Starlette already offloads to its threadpool.
"""

import asyncio
import json
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import history_store
import rag_core
from agent_graph import prepare_turn, stream_answer
from observability import (
    BUILD_SHA,
    Timer,
    emit_metrics,
    log,
    new_request_id,
    request_id_var,
)

API_KEY = os.getenv("API_KEY", "")
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "500"))
CORS_ORIGINS = [o for o in os.getenv("CORS_ORIGINS", "*").split(",") if o]

_CORPUS_SIZE = 0


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Warm the Chroma collection and BM25 index before serving, and refuse to
    start on an empty corpus.

    That last part matters more than it sounds: CHROMA_DIR resolving to the
    wrong place doesn't raise — get_or_create_collection() silently creates an
    *empty* collection, and the service then answers "I don't know" to
    everything, with nothing in the logs to explain why. Failing loudly at
    startup turns a silent quality collapse into an obvious deploy failure.
    """
    global _CORPUS_SIZE
    with Timer() as t:
        _CORPUS_SIZE = rag_core.warm_caches()
    if _CORPUS_SIZE == 0:
        raise RuntimeError(
            f"Chroma collection '{rag_core.COLLECTION_NAME}' at {rag_core.CHROMA_DIR} is empty. "
            "Check CHROMA_DIR and that the index was built (python ingest.py)."
        )
    log.info("startup complete", extra={"fields": {
        "corpus_chunks": _CORPUS_SIZE,
        "warm_ms": t.ms,
        "chroma_dir": rag_core.CHROMA_DIR,
        "history_backend": history_store.HISTORY_BACKEND,
    }})
    yield


app = FastAPI(
    title="Ask My Papers",
    description="Agentic RAG over a personal research corpus.",
    version=BUILD_SHA,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: request id + structured access log
# ---------------------------------------------------------------------------
@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id_var.set(request.headers.get("x-request-id") or new_request_id())
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled error", extra={"fields": {"path": request.url.path}})
        raise
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    log.info("request", extra={"fields": {
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "duration_ms": duration_ms,
    }})
    response.headers["x-request-id"] = request_id_var.get()
    return response


# ---------------------------------------------------------------------------
# Auth + rate limiting
# ---------------------------------------------------------------------------
_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def _client_key(request: Request) -> str:
    # App Runner terminates TLS and forwards the caller in X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def guard(request: Request, x_api_key: str = Header(default="")) -> None:
    """
    Shared-secret auth + per-IP token bucket.

    This endpoint spends money on every call. Left open, a crawler finding it
    turns into an OpenAI bill, which is the real runaway-cost risk here —
    bigger than the AWS compute it runs on.
    """
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    key = _client_key(request)
    now = time.time()
    with _hits_lock:
        bucket = _hits[key]
        while bucket and now - bucket[0] > 60:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_PER_MIN:
            raise HTTPException(
                status_code=429,
                detail=f"rate limit: {RATE_LIMIT_PER_MIN} requests/min",
            )
        bucket.append(now)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)
    thread_id: str | None = Field(
        default=None,
        description="Omit to start a new conversation; pass the returned id to continue one.",
    )


class SourceOut(BaseModel):
    source: str
    count: int
    rerank_scores: list = []


class AskResponse(BaseModel):
    answer: str
    thread_id: str
    sources: list[SourceOut]
    search_query: str | None = None
    sub_queries: list = []
    retry_count: int = 0
    latency_ms: float


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """Liveness only — deliberately touches nothing external."""
    return {"status": "ok", "build": BUILD_SHA}


@app.get("/ready")
def ready() -> dict:
    return {
        "status": "ready" if _CORPUS_SIZE > 0 else "degraded",
        "corpus_chunks": _CORPUS_SIZE,
        "build": BUILD_SHA,
        "history_backend": history_store.HISTORY_BACKEND,
    }


# ---------------------------------------------------------------------------
# Ask (non-streaming)
# ---------------------------------------------------------------------------
@app.post("/ask", response_model=AskResponse, dependencies=[Depends(guard)])
def ask(payload: AskRequest) -> AskResponse:
    """Plain `def`: Starlette runs it in its threadpool, so the blocking
    OpenAI/Chroma calls never touch the event loop."""
    thread_id = payload.thread_id or uuid.uuid4().hex
    question = payload.question.strip()

    with Timer() as total:
        with Timer() as retrieval:
            state = prepare_turn(question, history_store.get_history(thread_id))
        answer_parts: list[str] = []
        with Timer() as generation:
            for token in stream_answer(question, state):
                answer_parts.append(token)
        answer = "".join(answer_parts)
        history_store.append_turn(thread_id, question, answer)

    _emit_turn_metrics(state, total.ms, retrieval.ms, generation.ms, streamed=False)
    return AskResponse(
        answer=answer,
        thread_id=thread_id,
        sources=[SourceOut(**s) for s in _sources_payload(state)],
        search_query=state.get("search_query"),
        sub_queries=state.get("sub_queries", []),
        retry_count=state.get("retry_count", 0),
        latency_ms=total.ms,
    )


# ---------------------------------------------------------------------------
# Ask (streaming, SSE)
# ---------------------------------------------------------------------------
def _sse(event: str, data: dict) -> str:
    """
    Frame one SSE event.

    json.dumps is load-bearing: SSE frames are delimited by a blank line, and
    answer tokens are markdown full of newlines. Interpolating a raw token
    into `data: ...` shatters the stream into garbage frames the moment the
    model emits a line break.
    """
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@app.post("/ask/stream", dependencies=[Depends(guard)])
async def ask_stream(payload: AskRequest) -> StreamingResponse:
    thread_id = payload.thread_id or uuid.uuid4().hex
    question = payload.question.strip()
    request_id = request_id_var.get()

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=512)
    DONE = object()

    def emit(event: str, data: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (event, data))

    def producer() -> None:
        """All blocking work happens here, on one worker thread."""
        request_id_var.set(request_id)
        try:
            emit("status", {"stage": "contextualizing"})
            with Timer() as retrieval:
                state = prepare_turn(question, history_store.get_history(thread_id))

            emit("status", {
                "stage": "retrieved",
                "retry_count": state.get("retry_count", 0),
                "sub_queries": state.get("sub_queries", []),
            })
            emit("sources", {
                "sources": _sources_payload(state),
                "search_query": state.get("search_query"),
            })

            with Timer() as generation:
                for token in stream_answer(
                    question,
                    state,
                    on_complete=lambda a: history_store.append_turn(thread_id, question, a),
                ):
                    emit("token", {"t": token})

            _emit_turn_metrics(state, retrieval.ms + generation.ms, retrieval.ms,
                               generation.ms, streamed=True)
            emit("done", {
                "thread_id": thread_id,
                "retrieval_ms": retrieval.ms,
                "generation_ms": generation.ms,
            })
        except Exception as exc:
            # The status code is already sent by the time this can happen, so
            # failures have to travel as an event, not an HTTP error.
            log.exception("stream failed")
            emit("error", {"message": str(exc)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, (DONE, None))

    async def event_source():
        yield _sse("status", {"stage": "accepted", "thread_id": thread_id})
        loop.run_in_executor(None, producer)
        while True:
            event, data = await queue.get()
            if event is DONE:
                break
            yield _sse(event, data)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # ask any nginx-ish proxy not to buffer
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sources_payload(state: dict) -> list[dict]:
    return [
        {
            "source": s["source"],
            "count": s["count"],
            "rerank_scores": sorted((x for x in s["rerank_scores"] if x is not None), reverse=True),
        }
        for s in rag_core.summarize_sources(state.get("retrieved_chunks", []))
    ]


def _emit_turn_metrics(state: dict, total_ms: float, retrieval_ms: float,
                       generation_ms: float, streamed: bool) -> None:
    emit_metrics(
        {
            "LatencyMs": total_ms,
            "RetrievalMs": retrieval_ms,
            "GenerationMs": generation_ms,
            "RetryCount": state.get("retry_count", 0),
            "ChunksRetrieved": len(state.get("retrieved_chunks", [])),
            "SubQueries": len(state.get("sub_queries", []) or []),
            "Turns": 1,
        },
        dimensions={"Endpoint": "stream" if streamed else "ask"},
    )

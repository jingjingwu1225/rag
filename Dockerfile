# Multi-stage: wheels are built in a throwaway stage so build toolchains never
# ship in the runtime image.
#
# Python 3.11 to match the environment this was developed and tested against
# (the host also has 3.12; pinning avoids debugging a version skew that only
# appears in the container).

# ---------- builder ----------
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip wheel --wheel-dir /wheels -r requirements-api.txt


# ---------- runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CHROMA_DIR=/app/chroma_db \
    PORT=8000

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements-api.txt .
RUN pip install --no-index --find-links=/wheels -r requirements-api.txt \
    && rm -rf /wheels

# Application code and the prebuilt vector index. Baking the index in makes
# the image self-contained and the corpus immutable for the life of the
# container — which also removes the BM25 cache-staleness problem by
# construction. Cost: re-ingesting means rebuilding the image.
COPY rag_core.py agent_graph.py history_store.py observability.py api.py ./
# eval.py ships too (~5 KB). CI gates on the eval run *inside this image*, so
# the artifact being measured is the exact one about to deploy — and the same
# eval can be re-run against a running container later.
COPY eval.py eval_questions.json ./
COPY chroma_db/ ./chroma_db/

# Non-root: nothing here needs to write to the filesystem.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# One worker on purpose. Rate limiting is per-process, and multiple workers
# would each hold their own BM25 index and Chroma handle — extra memory for
# no throughput gain on a workload that is almost entirely waiting on OpenAI.
# Horizontal scaling happens at the App Runner instance level instead, which
# works because conversation history lives in DynamoDB, not in the process.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers 1"]

"""
cache.py
Response caching, to cut repeated LLM spend and latency.

Where the money actually goes: a single turn makes 5-12 OpenAI calls
(contextualize, decompose, embed, rerank, grade, possibly rewrite-and-retry,
then generation). Asking the same question twice pays that twice. In a demo
that is exactly what happens — the same handful of questions get asked over
and over while showing the thing off.

Two backends, same shape as history_store:
  memory — process-local dict with TTL. Default, no dependencies.
  redis  — shared across instances, so a cache hit on one instance serves
           every instance. Set REDIS_URL to enable.

Degrades rather than fails: if Redis is configured but unreachable, calls
fall through to a miss and the request proceeds normally. A cache outage
should make the service slower, never broken.
"""

import hashlib
import json
import os
import threading
import time

CACHE_BACKEND = os.getenv("CACHE_BACKEND", "memory")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", str(60 * 60)))  # 1h
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
REDIS_URL = os.getenv("REDIS_URL", "")

# Counters, exposed on /ready so hit rate is visible without a metrics backend.
_stats = {"hits": 0, "misses": 0, "errors": 0}
_stats_lock = threading.Lock()


def _bump(key: str) -> None:
    with _stats_lock:
        _stats[key] += 1


def stats() -> dict:
    with _stats_lock:
        total = _stats["hits"] + _stats["misses"]
        return {
            **_stats,
            "hit_rate": round(_stats["hits"] / total, 3) if total else None,
            "backend": CACHE_BACKEND if CACHE_ENABLED else "disabled",
        }


def make_key(*parts: str) -> str:
    """
    Hash the inputs into a key.

    Hashed rather than raw because question text is unbounded, arrives from
    users, and would otherwise end up as a Redis key — awkward to inspect and
    a way to smuggle control characters into keyspace. The prefix keeps this
    app's keys distinguishable in a shared Redis.
    """
    digest = hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:32]
    return f"rag:{digest}"


# ---------------------------------------------------------------------------
# Memory backend
# ---------------------------------------------------------------------------
_MEM: dict[str, tuple[float, str]] = {}
_MEM_LOCK = threading.Lock()
_MEM_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "500"))


def _mem_get(key: str) -> str | None:
    with _MEM_LOCK:
        entry = _MEM.get(key)
        if not entry:
            return None
        expires_at, value = entry
        if expires_at < time.time():
            # Lazy expiry: nothing sweeps in the background, entries are
            # dropped when next touched.
            _MEM.pop(key, None)
            return None
        return value


def _mem_set(key: str, value: str, ttl: int) -> None:
    with _MEM_LOCK:
        if len(_MEM) >= _MEM_MAX_ENTRIES:
            # Crude bound: drop the soonest-to-expire entry. Not LRU, but this
            # exists to stop unbounded growth in a single process, not to be a
            # good eviction policy — that is what the Redis backend is for.
            oldest = min(_MEM, key=lambda k: _MEM[k][0])
            _MEM.pop(oldest, None)
        _MEM[key] = (time.time() + ttl, value)


# ---------------------------------------------------------------------------
# Redis backend
# ---------------------------------------------------------------------------
_REDIS = None
_REDIS_LOCK = threading.Lock()


def _redis():
    global _REDIS
    if _REDIS is None:
        with _REDIS_LOCK:
            if _REDIS is None:
                import redis  # imported lazily so the memory backend needs no dep

                _REDIS = redis.from_url(
                    REDIS_URL,
                    decode_responses=True,
                    # Short timeouts: a slow cache must not become the reason a
                    # request is slow. Better to miss than to wait.
                    socket_timeout=0.5,
                    socket_connect_timeout=0.5,
                )
    return _REDIS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get(key: str):
    """Return the cached value, or None on miss/disabled/error."""
    if not CACHE_ENABLED:
        return None
    try:
        raw = _redis().get(key) if CACHE_BACKEND == "redis" else _mem_get(key)
    except Exception:
        # Cache failures are never fatal — fall through to a miss.
        _bump("errors")
        return None

    if raw is None:
        _bump("misses")
        return None
    _bump("hits")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        _bump("errors")
        return None


def set(key: str, value, ttl: int = CACHE_TTL_SECONDS) -> None:
    """Store a value. Silently gives up on error."""
    if not CACHE_ENABLED:
        return
    try:
        raw = json.dumps(value)
        if CACHE_BACKEND == "redis":
            _redis().setex(key, ttl, raw)
        else:
            _mem_set(key, raw, ttl)
    except Exception:
        _bump("errors")


def clear() -> None:
    """Drop everything (tests, and the UI's 'new conversation')."""
    if CACHE_BACKEND == "redis":
        try:
            _redis().flushdb()
        except Exception:
            _bump("errors")
    else:
        with _MEM_LOCK:
            _MEM.clear()
    with _stats_lock:
        for k in _stats:
            _stats[k] = 0

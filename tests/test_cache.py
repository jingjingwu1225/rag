"""
Cache tests.

The behaviour that matters most here isn't hits and misses — it's that a
broken cache degrades into a miss rather than an exception. A cache outage
should make the service slower, never take it down.
"""

import cache


class TestKeys:
    def test_same_inputs_produce_the_same_key(self):
        assert cache.make_key("retrieve", "abc", "question") == cache.make_key("retrieve", "abc", "question")

    def test_different_inputs_produce_different_keys(self):
        assert cache.make_key("retrieve", "abc") != cache.make_key("retrieve", "abd")

    def test_key_component_boundaries_are_unambiguous(self):
        """
        Joining parts without a separator would make ("ab","c") and ("a","bc")
        collide — different queries silently sharing a cache entry.
        """
        assert cache.make_key("ab", "c") != cache.make_key("a", "bc")

    def test_key_is_bounded_and_prefixed(self):
        key = cache.make_key("retrieve", "x" * 5000)
        assert key.startswith("rag:")
        assert len(key) < 64, "unbounded user text must not become an unbounded key"


class TestMemoryBackend:
    def setup_method(self):
        cache.clear()

    def test_miss_then_hit(self):
        key = cache.make_key("t", "miss-then-hit")
        assert cache.get(key) is None
        cache.set(key, {"answer": 42})
        assert cache.get(key) == {"answer": 42}

    def test_round_trips_structured_values(self):
        key = cache.make_key("t", "structured")
        value = [{"id": "a::1", "distance": 0.5, "bm25_score": None, "text": "x\ny"}]
        cache.set(key, value)
        assert cache.get(key) == value

    def test_expired_entry_is_a_miss(self):
        key = cache.make_key("t", "expiry")
        cache.set(key, "value", ttl=-1)  # already expired
        assert cache.get(key) is None

    def test_stats_track_hits_and_misses(self):
        key = cache.make_key("t", "stats")
        cache.get(key)          # miss
        cache.set(key, "v")
        cache.get(key)          # hit
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_entries_are_bounded(self):
        """Unbounded growth in a long-lived process is a slow memory leak."""
        for i in range(cache._MEM_MAX_ENTRIES + 50):
            cache.set(cache.make_key("bulk", str(i)), i)
        assert len(cache._MEM) <= cache._MEM_MAX_ENTRIES


class TestDegradation:
    """A cache that cannot be reached must not break the request."""

    def setup_method(self):
        cache.clear()

    def test_get_returns_miss_when_backend_raises(self, monkeypatch):
        monkeypatch.setattr(cache, "CACHE_BACKEND", "redis")
        monkeypatch.setattr(cache, "_redis", lambda: (_ for _ in ()).throw(ConnectionError("down")))

        assert cache.get(cache.make_key("t", "down")) is None
        assert cache.stats()["errors"] >= 1

    def test_set_swallows_backend_errors(self, monkeypatch):
        monkeypatch.setattr(cache, "CACHE_BACKEND", "redis")
        monkeypatch.setattr(cache, "_redis", lambda: (_ for _ in ()).throw(ConnectionError("down")))

        cache.set(cache.make_key("t", "down"), "value")  # must not raise
        assert cache.stats()["errors"] >= 1

    def test_unparsable_payload_is_a_miss_not_a_crash(self, monkeypatch):
        key = cache.make_key("t", "corrupt")
        monkeypatch.setattr(cache, "_mem_get", lambda k: "{not json")
        assert cache.get(key) is None

    def test_disabled_cache_always_misses(self, monkeypatch):
        monkeypatch.setattr(cache, "CACHE_ENABLED", False)
        key = cache.make_key("t", "disabled")
        cache.set(key, "value")
        assert cache.get(key) is None


class TestRetrievalIntegration:
    """retrieve_reranked() is the expensive call the cache exists to avoid."""

    def setup_method(self):
        cache.clear()

    def test_second_identical_call_skips_the_work(self, monkeypatch):
        import rag_core

        calls = {"n": 0}

        def fake_hybrid(question, k):
            calls["n"] += 1
            return [{"id": "a::1", "text": "t", "source": "p.pdf",
                     "distance": 0.5, "bm25_score": None}]

        monkeypatch.setattr(rag_core, "hybrid_retrieve", fake_hybrid)
        monkeypatch.setattr(rag_core, "rerank", lambda q, c, top_n: c)

        first = rag_core.retrieve_reranked("same question", k=1)
        second = rag_core.retrieve_reranked("same question", k=1)

        assert first == second
        assert calls["n"] == 1, "second call should have been served from cache"

    def test_different_questions_do_not_share_an_entry(self, monkeypatch):
        import rag_core

        calls = {"n": 0}

        def fake_hybrid(question, k):
            calls["n"] += 1
            return [{"id": f"{question}::1", "text": question, "source": "p.pdf",
                     "distance": 0.5, "bm25_score": None}]

        monkeypatch.setattr(rag_core, "hybrid_retrieve", fake_hybrid)
        monkeypatch.setattr(rag_core, "rerank", lambda q, c, top_n: c)

        rag_core.retrieve_reranked("question one", k=1)
        rag_core.retrieve_reranked("question two", k=1)
        assert calls["n"] == 2

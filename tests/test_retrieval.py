"""
Tests for retrieval fusion and display grouping.

hybrid_retrieve()'s RRF math and its "every chunk carries both score keys"
invariant are tested here because that invariant was a real production bug:
a chunk found only by BM25 carried no `distance` key, and the UI crashed with
a KeyError the first time such a chunk survived reranking.
"""

import rag_core


def _fake_hits(prefix, ids, score_key, base=1.0):
    """Build a ranked hit list shaped like retrieve()/bm25_search() output."""
    hits = []
    for rank, chunk_id in enumerate(ids):
        hit = {
            "id": chunk_id,
            "text": f"{prefix} text for {chunk_id}",
            "source": "paper.pdf",
        }
        hit[score_key] = base + rank
        hits.append(hit)
    return hits


class TestRRFFusion:
    def test_chunk_found_by_both_methods_outranks_either_alone(self, monkeypatch):
        """The core claim of hybrid search: agreement between methods wins."""
        # "shared" is rank 1 in vectors and rank 1 in keywords; the others
        # appear in only one list.
        monkeypatch.setattr(rag_core, "retrieve",
                            lambda q, k: _fake_hits("vec", ["shared", "vec_only"], "distance"))
        monkeypatch.setattr(rag_core, "bm25_search",
                            lambda q, k: _fake_hits("kw", ["shared", "kw_only"], "bm25_score"))

        fused = rag_core.hybrid_retrieve("anything", k=10)
        assert fused[0]["id"] == "shared"
        assert fused[0]["rrf_score"] > fused[1]["rrf_score"]

    def test_rrf_score_matches_the_formula(self, monkeypatch):
        monkeypatch.setattr(rag_core, "retrieve",
                            lambda q, k: _fake_hits("vec", ["a"], "distance"))
        monkeypatch.setattr(rag_core, "bm25_search",
                            lambda q, k: _fake_hits("kw", ["a"], "bm25_score"))

        fused = rag_core.hybrid_retrieve("anything", k=10)
        # Rank 0 in both lists: 1/(K+1) + 1/(K+1)
        expected = 2 * (1.0 / (rag_core.RRF_K + 1))
        assert fused[0]["rrf_score"] == expected

    def test_deduplicates_by_chunk_id(self, monkeypatch):
        monkeypatch.setattr(rag_core, "retrieve",
                            lambda q, k: _fake_hits("vec", ["a", "b"], "distance"))
        monkeypatch.setattr(rag_core, "bm25_search",
                            lambda q, k: _fake_hits("kw", ["b", "a"], "bm25_score"))

        fused = rag_core.hybrid_retrieve("anything", k=10)
        assert len(fused) == 2
        assert len({c["id"] for c in fused}) == 2


class TestBothScoreKeysInvariant:
    """
    Regression guard for the KeyError crash: a keyword-only hit used to carry
    bm25_score but no distance key at all, so any consumer doing
    chunk["distance"] blew up as soon as one survived into the results.
    """

    def test_every_chunk_has_both_keys(self, monkeypatch):
        monkeypatch.setattr(rag_core, "retrieve",
                            lambda q, k: _fake_hits("vec", ["vec_only"], "distance"))
        monkeypatch.setattr(rag_core, "bm25_search",
                            lambda q, k: _fake_hits("kw", ["kw_only"], "bm25_score"))

        fused = rag_core.hybrid_retrieve("anything", k=10)
        assert len(fused) == 2
        for chunk in fused:
            assert "distance" in chunk, f"missing distance: {chunk['id']}"
            assert "bm25_score" in chunk, f"missing bm25_score: {chunk['id']}"

    def test_missing_method_yields_none_not_absent_key(self, monkeypatch):
        monkeypatch.setattr(rag_core, "retrieve",
                            lambda q, k: _fake_hits("vec", ["vec_only"], "distance"))
        monkeypatch.setattr(rag_core, "bm25_search", lambda q, k: [])

        fused = rag_core.hybrid_retrieve("anything", k=10)
        assert fused[0]["bm25_score"] is None
        assert fused[0]["distance"] is not None


class TestSummarizeSources:
    def test_groups_multiple_passages_from_one_file(self):
        """
        Several distinct passages from the same paper should read as one
        source with a count, not the same filename repeated N times.
        """
        chunks = [
            {"id": "p::1", "source": "paper.pdf", "distance": 0.5, "rerank_score": 10},
            {"id": "p::2", "source": "paper.pdf", "distance": 0.6, "rerank_score": 9},
            {"id": "p::3", "source": "paper.pdf", "distance": 0.7, "rerank_score": 8},
        ]
        summary = rag_core.summarize_sources(chunks)
        assert len(summary) == 1
        assert summary[0]["source"] == "paper.pdf"
        assert summary[0]["count"] == 3
        assert sorted(summary[0]["rerank_scores"], reverse=True) == [10, 9, 8]

    def test_keeps_distinct_sources_separate_in_retrieval_order(self):
        chunks = [
            {"id": "b::1", "source": "b.pdf", "distance": 0.5, "rerank_score": 9},
            {"id": "a::1", "source": "a.pdf", "distance": 0.6, "rerank_score": 8},
            {"id": "b::2", "source": "b.pdf", "distance": 0.7, "rerank_score": 7},
        ]
        summary = rag_core.summarize_sources(chunks)
        assert [s["source"] for s in summary] == ["b.pdf", "a.pdf"]
        assert summary[0]["count"] == 2

    def test_tolerates_none_scores(self):
        """A keyword-only chunk has distance=None; grouping must not crash."""
        chunks = [{"id": "p::1", "source": "paper.pdf", "distance": None, "rerank_score": None}]
        summary = rag_core.summarize_sources(chunks)
        assert summary[0]["count"] == 1
        assert summary[0]["distances"] == []
        assert summary[0]["rerank_scores"] == []

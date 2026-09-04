"""
Tests for the ingestion-side pure functions: chunking and boilerplate
stripping. No network, no API keys, no Chroma — these are the parts that
decide what ends up in the index, so they're worth pinning down.
"""

import itertools

import rag_core


class TestStripBoilerplate:
    def test_removes_bioarxiv_license_footer(self):
        """
        The concrete failure this was written for: pypdf extracts multi-column
        preprints in raw layout order, so the bioRxiv license footer landed
        mid-paragraph next to real acquisition parameters, inside one chunk.
        """
        text = (
            "We acquired dMRI data at 0.4 mm isotropic resolution.\n"
            ".CC-BY-NC-ND 4.0 International licenseavailable under a\n"
            "(which was not certified by peer review) is the author/funder, "
            "who has granted bioRxiv a license to display the preprint\n"
            "b = 43,000 s/mm2, at TEs of 33, 45, 60, and 78 ms."
        )
        cleaned = rag_core.strip_boilerplate(text)

        assert "CC-BY" not in cleaned
        assert "author/funder" not in cleaned
        assert "peer review" not in cleaned
        # Real content on both sides of the footer must survive.
        assert "0.4 mm isotropic" in cleaned
        assert "43,000 s/mm2" in cleaned

    def test_matches_license_across_version_number(self):
        """Regression: the original regex failed to span the '4.0' version."""
        assert "CC-BY" not in rag_core.strip_boilerplate(
            ".CC-BY-NC-ND 4.0 International licenseavailable under a"
        )

    def test_leaves_ordinary_text_untouched(self):
        text = "Diffusion MRI resolves axonal architecture.\nWe used 500 mT/m gradients."
        assert rag_core.strip_boilerplate(text) == text


class TestChunkText:
    def test_empty_input_returns_no_chunks(self):
        assert rag_core.chunk_text("") == []
        assert rag_core.chunk_text("   \n  ") == []

    def test_short_text_is_a_single_chunk(self):
        assert rag_core.chunk_text("One short paragraph.") == ["One short paragraph."]

    def test_does_not_split_mid_sentence(self):
        """
        The whole point of the paragraph/sentence-aware rewrite: a fixed
        character window used to cut sentences (and table rows) in half.
        """
        sentences = [f"This is sentence number {i} and it carries real content." for i in range(40)]
        chunks = rag_core.chunk_text(" ".join(sentences), chunk_size=200, overlap=40)

        assert len(chunks) > 1
        for chunk in chunks:
            # Every chunk should end at a sentence boundary, not mid-word.
            assert chunk.rstrip().endswith("."), f"chunk ends mid-sentence: {chunk[-40:]!r}"

    def test_respects_paragraph_boundaries(self):
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
        chunks = rag_core.chunk_text(text, chunk_size=1000, overlap=50)
        # Comfortably under chunk_size, so it stays whole.
        assert len(chunks) == 1

    def test_chunks_respect_size_budget(self):
        text = " ".join(f"Sentence {i} with some filler words in it." for i in range(100))
        chunk_size = 300
        chunks = rag_core.chunk_text(text, chunk_size=chunk_size, overlap=50)
        for chunk in chunks:
            # Allow one sentence of slack: a chunk is only closed once adding
            # the *next* piece would exceed the budget.
            assert len(chunk) < chunk_size * 2

    def test_overlap_carries_context_forward(self):
        sentences = [f"Fact {i} is stated plainly." for i in range(30)]
        chunks = rag_core.chunk_text(" ".join(sentences), chunk_size=150, overlap=60)
        assert len(chunks) > 2
        # Consecutive chunks should share some text, so a fact spanning a
        # boundary is still retrievable from at least one chunk.
        overlaps = [
            bool(set(a.split()) & set(b.split()))
            for a, b in itertools.pairwise(chunks)
        ]
        assert all(overlaps)

    def test_oversized_paragraph_is_split_on_sentences(self):
        para = " ".join(f"Long sentence number {i} about diffusion imaging." for i in range(50))
        chunks = rag_core.chunk_text(para, chunk_size=200, overlap=40)
        assert len(chunks) > 1

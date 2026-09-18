from src.chunking import chunk_text
from src.cost_router import should_use_local


class TestChunkText:
    def test_produces_overlapping_chunks_of_expected_size(self):
        text = " ".join(["word"] * 1200)
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 3
        assert [len(c.split()) for c in chunks] == [500, 500, 300]

    def test_no_text_lost_across_chunk_boundaries(self):
        words = [f"w{i}" for i in range(1000)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=200, overlap=20)
        # every word must appear in at least one chunk
        seen = set()
        for c in chunks:
            seen.update(c.split())
        assert seen == set(words)

    def test_empty_text_returns_no_chunks(self):
        assert chunk_text("", chunk_size=500, overlap=50) == []

    def test_overlap_must_be_smaller_than_chunk_size(self):
        import pytest
        with pytest.raises(ValueError):
            chunk_text("some text", chunk_size=100, overlap=100)


class TestCostRouter:
    def test_short_simple_question_routes_local(self):
        assert should_use_local("What is RAG?") is True

    def test_complexity_keyword_forces_cloud_even_if_short(self):
        assert should_use_local("Compare A and B") is False

    def test_long_question_routes_cloud(self):
        long_q = " ".join(["word"] * 20)
        assert should_use_local(long_q) is False

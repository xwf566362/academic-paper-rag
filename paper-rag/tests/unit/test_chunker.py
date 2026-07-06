# -*- coding: utf-8 -*-
"""Tests for chunker -- paper chunking strategies."""

from unittest.mock import patch


def make_parsed(text, title="Test Paper"):
    """Helper: build a parse_pdf return value from raw text."""
    return dict(
        file_name="test.pdf",
        file_path="/tmp/test.pdf",
        title=title,
        page_count=1,
        metadata=dict(author="", subject="", keywords=""),
        sections=[],
        full_text=text,
    )


class TestBuildChunksFromPdf:

    def test_basic_chunking(self, mock_config):
        """Paper with abstract, introduction, and references produces at least 2 chunks."""
        text = "Abstract\nThis is the abstract.\nIntroduction\nSome intro content.\nReferences\n[1] Ref."
        with patch("app.processing.chunker.parse_pdf", return_value=make_parsed(text)):
            from app.processing.chunker import build_chunks_from_pdf
            chunks = build_chunks_from_pdf("test.pdf")
        assert len(chunks) >= 2
        for c in chunks:
            assert "text" in c
            assert "metadata" in c
            assert c["metadata"]["file_name"] == "test.pdf"

    def test_no_abstract(self, mock_config):
        """Paper without Abstract keyword still chunks correctly."""
        text = "Introduction\nJust starts with intro.\nConclusion\nDone."
        with patch("app.processing.chunker.parse_pdf", return_value=make_parsed(text)):
            from app.processing.chunker import build_chunks_from_pdf
            chunks = build_chunks_from_pdf("test.pdf")
        assert len(chunks) >= 1

    def test_no_references(self, mock_config):
        """Paper without References section is still valid."""
        text = "Abstract\nAbstract text.\nConclusion\nFinal remarks."
        with patch("app.processing.chunker.parse_pdf", return_value=make_parsed(text)):
            from app.processing.chunker import build_chunks_from_pdf
            chunks = build_chunks_from_pdf("test.pdf")
        assert len(chunks) >= 1

    def test_each_chunk_has_unique_id(self, mock_config):
        text = "Abstract\nAbstract text.\nIntroduction\nIntro text."
        with patch("app.processing.chunker.parse_pdf", return_value=make_parsed(text)):
            from app.processing.chunker import build_chunks_from_pdf
            chunks = build_chunks_from_pdf("test.pdf")
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "chunk IDs must be unique"

    def test_metadata_has_section(self, mock_config):
        text = "Abstract\nAbstract.\nIntroduction\nIntro.\nConclusion\nDone."
        with patch("app.processing.chunker.parse_pdf", return_value=make_parsed(text)):
            from app.processing.chunker import build_chunks_from_pdf
            chunks = build_chunks_from_pdf("test.pdf")
        sections = {c["metadata"]["section"] for c in chunks}
        assert "Introduction" in sections or "Methodology" in sections
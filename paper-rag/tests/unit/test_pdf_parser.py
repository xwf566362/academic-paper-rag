# -*- coding: utf-8 -*-
"""Tests for pdf_parser."""

import os
import pytest
from unittest.mock import MagicMock, patch
from app.processing.pdf_parser import is_section_header, extract_references


class TestIsSectionHeader:
    def test_abstract_matches(self):
        assert is_section_header("Abstract") is True
        assert is_section_header("abstract") is True

    def test_introduction_matches(self):
        assert is_section_header("Introduction") is True
        assert is_section_header("1. Introduction") is True

    def test_methodology_matches(self):
        assert is_section_header("Methodology") is True

    def test_references_matches(self):
        assert is_section_header("References") is True

    def test_short_lines_rejected(self):
        assert is_section_header("") is False

    def test_sentence_with_period_rejected(self):
        assert is_section_header("This is a sentence.") is False

    def test_long_lines_rejected(self):
        assert is_section_header("A" * 101) is False


class TestExtractReferences:
    def test_extract_numbered_refs(self):
        text = "Some text.\nReferences\n[1] Smith. 2023.\n[2] Jones. 2022."
        refs = extract_references(text)
        assert len(refs) == 2
        assert "Smith" in refs[0]

    def test_no_references_returns_empty(self):
        assert extract_references("Just text.") == []


class TestParsePdf:
    def test_basic_structure(self, tmp_path):
        import fitz
        mock_page = MagicMock(spec=fitz.Page)
        mock_page.get_text.return_value = "Introduction\nSome text."
        mock_doc = MagicMock(spec=fitz.Document)
        mock_doc.metadata = {"author": "A", "subject": "", "keywords": ""}
        mock_doc.page_count = 1
        mock_doc.__getitem__.return_value = mock_page
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("")  # create the file so exists() returns True
        with patch("app.processing.pdf_parser.fitz.open", return_value=mock_doc):
            from app.processing.pdf_parser import parse_pdf
            result = parse_pdf(str(pdf_path))
        assert result["page_count"] == 1
        assert "full_text" in result

    def test_file_not_found(self):
        from app.processing.pdf_parser import parse_pdf
        with pytest.raises(FileNotFoundError):
            parse_pdf("/nonexistent/file.pdf")

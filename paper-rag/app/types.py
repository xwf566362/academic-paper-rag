# -*- coding: utf-8 -*-
"""Shared type definitions for the paper-rag application."""

from typing import Dict, List, Optional, TypedDict


class ChunkMetadata(TypedDict, total=False):
    """Metadata embedded in each chunk."""
    file_name: str
    paper_title: str
    section: str
    chunk_index: int
    page_range: str
    is_abstract: bool
    is_reference: bool


class ChunkDict(TypedDict, total=False):
    """A single chunk produced by the chunker."""
    chunk_id: str
    text: str
    metadata: ChunkMetadata


class SearchResult(TypedDict):
    """A single search result from VectorStore."""
    text: str
    metadata: ChunkMetadata
    score: float


class PaperInfo(TypedDict):
    """Paper listing entry from VectorStore.list_papers()."""
    file_name: str
    paper_title: str
    chunk_count: int


class ParsedPDF(TypedDict):
    """Return type of parse_pdf()."""
    file_name: str
    file_path: str
    title: str
    page_count: int
    metadata: Dict[str, str]
    sections: List[dict]
    full_text: str

# -*- coding: utf-8 -*-
"""Unit tests for VectorStore -- CRUD, search, and index alignment."""

import numpy as np
import pytest


class TestInit:
    def test_empty_store_has_no_papers(self, vector_store):
        assert vector_store.paper_count == 0
        assert vector_store.total_chunks == 0
        assert vector_store.list_papers() == []

    def test_empty_store_search_returns_empty(self, vector_store):
        assert vector_store.search("anything") == []


class TestAddChunks:
    def test_add_single_paper(self, vector_store, sample_chunks):
        paper_a = [c for c in sample_chunks if c["metadata"]["file_name"] == "paper_a.pdf"]
        n = vector_store.add_paper_chunks(paper_a)
        assert n == 3
        assert vector_store.total_chunks == 3
        assert vector_store.paper_count == 1

    def test_add_empty_chunks_returns_zero(self, vector_store):
        assert vector_store.add_paper_chunks([]) == 0

    def test_add_two_papers(self, vector_store, sample_chunks):
        vector_store.add_paper_chunks(sample_chunks)
        assert vector_store.total_chunks == 5
        assert vector_store.paper_count == 2

    def test_add_updates_embeddings(self, vector_store, sample_chunks):
        vector_store.add_paper_chunks(sample_chunks)
        assert vector_store._embeddings is not None
        assert vector_store._embeddings.shape[0] == 5
        assert vector_store._embeddings.shape[1] == 4

    def test_ids_are_unique(self, vector_store, sample_chunks):
        vector_store.add_paper_chunks(sample_chunks)
        ids = vector_store._ids
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_metadata_aligns_with_embeddings(self, vector_store, sample_chunks):
        vector_store.add_paper_chunks(sample_chunks)
        n = len(vector_store._ids)
        assert len(vector_store._documents) == n
        assert len(vector_store._metadatas) == n
        assert vector_store._embeddings.shape[0] == n


class TestListPapers:
    def test_list_returns_all_papers(self, vector_store, sample_chunks):
        vector_store.add_paper_chunks(sample_chunks)
        papers = vector_store.list_papers()
        names = {p["file_name"] for p in papers}
        assert names == {"paper_a.pdf", "paper_b.pdf"}

    def test_list_includes_chunk_counts(self, vector_store, sample_chunks):
        vector_store.add_paper_chunks(sample_chunks)
        papers = {p["file_name"]: p["chunk_count"] for p in vector_store.list_papers()}
        assert papers["paper_a.pdf"] == 3
        assert papers["paper_b.pdf"] == 2


class TestSearch:
    def test_search_returns_results(self, vector_store, sample_chunks):
        vector_store.add_paper_chunks(sample_chunks)
        results = vector_store.search("transformers")
        assert len(results) > 0
        for r in results:
            assert "text" in r
            assert "metadata" in r
            assert "score" in r

    def test_search_with_filter(self, vector_store, sample_chunks):
        vector_store.add_paper_chunks(sample_chunks)
        results = vector_store.search("test", filter_paper="paper_a.pdf")
        for r in results:
            assert r["metadata"]["file_name"] == "paper_a.pdf"

    def test_search_with_nonexistent_filter(self, vector_store, sample_chunks):
        vector_store.add_paper_chunks(sample_chunks)
        assert vector_store.search("test", filter_paper="nonexistent.pdf") == []


class TestDeletePaper:
    def test_delete_removes_chunks(self, vector_store, sample_chunks):
        vector_store.add_paper_chunks(sample_chunks)
        deleted = vector_store.delete_paper("paper_a.pdf")
        assert deleted == 3
        assert vector_store.paper_count == 1
        assert vector_store.total_chunks == 2

    def test_delete_keeps_remaining_data_aligned(self, vector_store, sample_chunks):
        """Regression: embeddings must align 1:1 with ids/docs/metas after delete."""
        vector_store.add_paper_chunks(sample_chunks)
        vector_store.delete_paper("paper_a.pdf")
        ids = vector_store._ids
        docs = vector_store._documents
        metas = vector_store._metadatas
        embs = vector_store._embeddings
        assert len(ids) == len(docs) == len(metas) == (embs.shape[0] if embs is not None else 0)
        for m in metas:
            assert m["file_name"] == "paper_b.pdf"

    def test_delete_nonexistent(self, vector_store, sample_chunks):
        vector_store.add_paper_chunks(sample_chunks)
        assert vector_store.delete_paper("nonexistent.pdf") == 0

    def test_delete_last_paper_empties_store(self, vector_store, sample_chunks):
        vector_store.add_paper_chunks(sample_chunks)
        vector_store.delete_paper("paper_a.pdf")
        vector_store.delete_paper("paper_b.pdf")
        assert vector_store.paper_count == 0
        assert vector_store.total_chunks == 0
        assert vector_store._embeddings is None

    def test_delete_from_empty(self, vector_store):
        assert vector_store.delete_paper("anything.pdf") == 0

    def test_lifecycle_add_delete_add(self, vector_store, sample_chunks):
        paper_a = [c for c in sample_chunks if c["metadata"]["file_name"] == "paper_a.pdf"]
        vector_store.add_paper_chunks(paper_a)
        assert vector_store.total_chunks == 3
        vector_store.delete_paper("paper_a.pdf")
        assert vector_store.total_chunks == 0
        vector_store.add_paper_chunks(paper_a)
        assert vector_store.total_chunks == 3
        assert vector_store._embeddings is not None
        assert vector_store._embeddings.shape[0] == 3


class TestClearAll:
    def test_clear_all_empties_store(self, vector_store, sample_chunks):
        vector_store.add_paper_chunks(sample_chunks)
        vector_store.clear_all()
        assert vector_store.paper_count == 0
        assert vector_store.total_chunks == 0
        assert vector_store._embeddings is None
        assert vector_store._ids == []

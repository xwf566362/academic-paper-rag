# -*- coding: utf-8 -*-
"""pytest shared fixtures — mocks for embedder, reranker, config, and vector store"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

# Ensure the project root is on sys.path so imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Session-level: patch DATA_DIR / CHROMA_DIR to a temp path so tests never
# touch real data.  Done via pytest_configure so it fires before any fixture.
# ---------------------------------------------------------------------------

_TMP_ROOT = None  # cleaned up in pytest_unconfigure


def pytest_configure():
    global _TMP_ROOT
    _TMP_ROOT = Path(tempfile.mkdtemp(prefix="paper_rag_test_"))

    # Patch module-level path constants in app.config
    import app.config as cfg_mod

    test_data = _TMP_ROOT / "data"
    (test_data / "papers").mkdir(parents=True, exist_ok=True)
    (test_data / "chroma").mkdir(parents=True, exist_ok=True)

    cfg_mod.DATA_DIR = test_data
    cfg_mod.PAPERS_DIR = test_data / "papers"
    cfg_mod.CHROMA_DIR = test_data / "chroma"
    cfg_mod.LOGS_DIR = _TMP_ROOT / "logs"


def pytest_unconfigure():
    global _TMP_ROOT
    if _TMP_ROOT is not None and _TMP_ROOT.exists():
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset all module-level singletons and close old VectorStore connections."""
    import app.config as cfg_mod
    cfg_mod._config = None

    import app.processing.embeddings as emb_mod
    emb_mod._embedder = None
    emb_mod._reranker = None
    EmbeddingModel = emb_mod.EmbeddingModel
    RerankerModel = emb_mod.RerankerModel
    if EmbeddingModel._instance is not None:
        EmbeddingModel._instance = None
    if RerankerModel._instance is not None:
        RerankerModel._instance = None

    import app.database.vector_store as vs_mod
    if vs_mod._vector_store is not None:
        try:
            vs_mod._vector_store.close()
        except Exception:
            pass
        # Remove the DB file so the next test starts clean
        import sqlite3, pathlib
        db = pathlib.Path(vs_mod._vector_store._db_path)
        vs_mod._vector_store = None
        try:
            db.unlink(missing_ok=True)
        except Exception:
            pass
    else:
        vs_mod._vector_store = None

    import app.api.llm_client as llm_mod
    llm_mod._clients = {}


@pytest.fixture
def mock_config():
    """Return a patched AppConfig with safe test values and stub it into
    app.config.get_config()."""
    import app.config as cfg_mod
    from app.config import AppConfig

    cfg = AppConfig(
        default_provider="test",
        request_timeout=5,
        max_retries=1,
        retry_delay=0.01,
        top_k_initial=10,
        top_k_rerank=3,
        chunk_score_threshold=0.0,
        chunk_size=512,
        chunk_overlap=64,
        max_section_chars=60000,
        server_host="127.0.0.1",
        server_port=0,
        server_reload=False,
    )
    cfg.llm_providers["test"] = cfg_mod.LLMProviderConfig(
        api_key="test-key",
        base_url="http://localhost:0/v1",
        model="test-model",
    )
    cfg_mod._config = cfg
    return cfg


@pytest.fixture
def mock_embedder():
    """Return a MockEmbedder and register it via get_embedder()."""
    import app.processing.embeddings as emb_mod

    class MockEmbedder:
        def encode(self, texts, **kw):
            # Return a deterministic Nx4 float32 array
            return np.array([[0.1, 0.2, 0.3, 0.4]] * len(texts), dtype=np.float32)

        def encode_query(self, query):
            return [0.1, 0.2, 0.3, 0.4]

    instance = MockEmbedder()
    emb_mod._embedder = instance
    return instance


@pytest.fixture
def mock_reranker():
    """Return a MockReranker and register it via get_reranker()."""
    import app.processing.embeddings as emb_mod

    class MockReranker:
        def rerank(self, query, candidates, top_k=5):
            scored = [(t, m, 0.9) for t, m in candidates[:top_k]]
            return scored

    instance = MockReranker()
    emb_mod._reranker = instance
    return instance


@pytest.fixture
def vector_store(mock_config, mock_embedder, mock_reranker):
    """Return a clean VectorStore ready for tests."""
    from app.database.vector_store import VectorStore

    vs = VectorStore()
    # Clear any pre-loaded state
    vs.clear_all()
    return vs


@pytest.fixture
def sample_chunks():
    """Return two papers' worth of chunk dicts for insertion tests."""
    return [
        # Paper A — 3 chunks
        dict(text="Introduction to transformers.",
             metadata=dict(file_name="paper_a.pdf", paper_title="Paper A", section="Introduction", chunk_index=0)),
        dict(text="Methodology details.",
             metadata=dict(file_name="paper_a.pdf", paper_title="Paper A", section="Methodology", chunk_index=1)),
        dict(text="Experimental results.",
             metadata=dict(file_name="paper_a.pdf", paper_title="Paper A", section="Experiments", chunk_index=2)),
        # Paper B — 2 chunks
        dict(text="Background of NLP.",
             metadata=dict(file_name="paper_b.pdf", paper_title="Paper B", section="Introduction", chunk_index=0)),
        dict(text="Conclusion and future work.",
             metadata=dict(file_name="paper_b.pdf", paper_title="Paper B", section="Conclusion", chunk_index=1)),
    ]

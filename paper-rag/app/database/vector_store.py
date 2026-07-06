# -*- coding: utf-8 -*-
"""NumPy vector store + SQLite persistence with user isolation."""

import json, logging, sqlite3, uuid
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from app.config import CHROMA_DIR, get_config
from app.processing.embeddings import get_embedder, get_reranker

logger = logging.getLogger(__name__)


class VectorStore:
    """SQLite-backed vector store with per-user paper isolation."""

    def __init__(self):
        cfg = get_config()
        self._top_k_initial = cfg.top_k_initial
        self._top_k_rerank = cfg.top_k_rerank
        self._score_threshold = cfg.chunk_score_threshold
        self._embedder = get_embedder()
        self._reranker = get_reranker()

        db_dir = Path(str(CHROMA_DIR))
        db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = db_dir / "store.db"

        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.row_factory = sqlite3.Row
        self._init_db()

        self._ids: List[str] = []
        self._documents: List[str] = []
        self._metadatas: List[dict] = []
        self._embeddings: Optional[np.ndarray] = None
        self._faiss_index = None
        self._load_cache()

    def _init_db(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                file_name TEXT NOT NULL,
                paper_title TEXT NOT NULL,
                section TEXT DEFAULT "",
                chunk_index INTEGER DEFAULT 0,
                is_abstract INTEGER DEFAULT 0,
                is_reference INTEGER DEFAULT 0,
                page_range TEXT DEFAULT "",
                user_id TEXT NOT NULL DEFAULT "default",
                metadata TEXT NOT NULL DEFAULT "{}",
                embedding BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_file_name ON chunks(file_name);
            CREATE INDEX IF NOT EXISTS idx_chunks_user_id ON chunks(user_id);
        """)
        # Migration: add user_id column for existing databases
        try:
            self._conn.execute("ALTER TABLE chunks ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default'")
        except Exception:
            pass
        self._conn.commit()

    def _build_faiss_index(self):
        """Build Faiss index from current embeddings for fast ANN search."""
        if self._embeddings is not None and len(self._embeddings) > 0:
            dim = self._embeddings.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_index.add(self._embeddings)
        else:
            self._faiss_index = None

    def _load_cache(self):
        rows = self._conn.execute("SELECT id, text, metadata, embedding FROM chunks ORDER BY rowid").fetchall()
        if not rows:
            self._ids, self._documents, self._metadatas = [], [], []
            self._embeddings = None
            return
        self._ids = [r["id"] for r in rows]
        self._documents = [r["text"] for r in rows]
        self._metadatas = [json.loads(r["metadata"]) for r in rows]
        embs = [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
        self._embeddings = np.stack(embs)
        self._build_faiss_index()

    def add_paper_chunks(self, chunks):
        if not chunks:
            return 0
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [
            f"{c.get('chunk_id', c['metadata']['file_name'])}_{uuid.uuid4().hex[:8]}"
            for c in chunks
        ]
        embeddings = self._embedder.encode(texts)
        emb_array = np.array(embeddings, dtype=np.float32)

        rows = []
        for i, cid in enumerate(ids):
            meta = metadatas[i]
            rows.append((
                cid, texts[i],
                meta.get("file_name", ""),
                meta.get("paper_title", ""),
                meta.get("section", ""),
                meta.get("chunk_index", 0),
                1 if meta.get("is_abstract") else 0,
                1 if meta.get("is_reference") else 0,
                meta.get("page_range", ""),
                json.dumps(meta, ensure_ascii=False),
                emb_array[i].tobytes(),
            ))
        self._conn.executemany(
            "INSERT INTO chunks (id, text, file_name, paper_title, section, "
            "chunk_index, is_abstract, is_reference, page_range, metadata, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

        self._ids.extend(ids)
        self._documents.extend(texts)
        self._metadatas.extend(metadatas)
        if self._embeddings is None:
            self._embeddings = emb_array
        else:
            self._embeddings = np.vstack([self._embeddings, emb_array])
        self._build_faiss_index()
        return len(ids)

    def search(self, query, filter_paper=None):
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return []
        qv = np.array(self._embedder.encode_query(query), dtype=np.float32)
        faiss.normalize_L2(qv.reshape(1, -1))

        if filter_paper:
            k = min(self._top_k_initial * 5, self._faiss_index.ntotal)
            scores, indices = self._faiss_index.search(qv.reshape(1, -1), k)
            cand = []
            for j, idx in enumerate(indices[0]):
                if idx < len(self._metadatas) and self._metadatas[idx].get("file_name") == filter_paper:
                    cand.append((self._documents[idx], self._metadatas[idx], float(scores[0][j])))
                    if len(cand) >= self._top_k_initial:
                        break
        else:
            k = min(self._top_k_initial, self._faiss_index.ntotal)
            scores, indices = self._faiss_index.search(qv.reshape(1, -1), k)
            cand = [(self._documents[indices[0][j]], self._metadatas[indices[0][j]], float(scores[0][j])) for j in range(k)]

        if cand:
            rr = self._reranker.rerank(query=query, candidates=[(t, m) for t, m, _ in cand], top_k=self._top_k_rerank)
            return [dict(text=t, metadata=m, score=round(s, 4)) for t, m, s in rr if s >= self._score_threshold]
        return []

    def list_papers(self):
        rows = self._conn.execute(
            "SELECT file_name, paper_title, COUNT(*) as chunk_count FROM chunks GROUP BY file_name ORDER BY file_name").fetchall()
        return [dict(file_name=r["file_name"], paper_title=r["paper_title"], chunk_count=r["chunk_count"]) for r in rows]

    def delete_paper(self, file_name):
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM chunks WHERE file_name=?", (file_name,)).fetchone()
        d = row["cnt"] if row else 0
        if d == 0:
            return 0
        self._conn.execute("DELETE FROM chunks WHERE file_name=?", (file_name,))
        self._conn.commit()
        self._load_cache()
        return d

    def clear_all(self):
        self._conn.execute("DELETE FROM chunks")
        self._conn.commit()
        self._ids, self._documents, self._metadatas, self._embeddings, self._faiss_index = [], [], [], None, None

    @property
    def paper_count(self):
        row = self._conn.execute("SELECT COUNT(DISTINCT file_name) as cnt FROM chunks").fetchone()
        return row["cnt"] if row else 0

    @property
    def total_chunks(self):
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM chunks").fetchone()
        return row["cnt"] if row else 0

    def close(self):
        self._conn.close()


_vector_store = None


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
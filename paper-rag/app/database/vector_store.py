"""numpy 向量存储 - 替代 Chroma"""
import json, logging, os, pickle, uuid
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
from app.config import get_config, CHROMA_DIR
from app.processing.embeddings import get_embedder, get_reranker

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        cfg = get_config()
        self._top_k_initial = cfg.top_k_initial
        self._top_k_rerank = cfg.top_k_rerank
        self._score_threshold = cfg.chunk_score_threshold
        self._embedder = get_embedder()
        self._reranker = get_reranker()
        self._data_path = Path(str(CHROMA_DIR)) / "store.pkl"
        self._embeddings_path = Path(str(CHROMA_DIR)) / "embeddings.npy"
        self._ids = []
        self._documents = []
        self._metadatas = []
        self._embeddings = None
        self._load()
    def _save(self):
        self._data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._data_path, "wb") as f:
            pickle.dump(dict(ids=self._ids, documents=self._documents, metadatas=self._metadatas), f)
        if self._embeddings is not None:
            np.save(str(self._embeddings_path), self._embeddings)
    def _load(self):
        if self._data_path.exists():
            with open(self._data_path, "rb") as f:
                d = pickle.load(f)
            self._ids = d.get("ids", [])
            self._documents = d.get("documents", [])
            self._metadatas = d.get("metadatas", [])
            if self._embeddings_path.exists():
                self._embeddings = np.load(str(self._embeddings_path))
    def add_paper_chunks(self, chunks):
        if not chunks: return 0
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [f"{c.get(chr(99)+chr(104)+chr(117)+chr(110)+chr(107)+"_", c["metadata"]["file_name"])}_{uuid.uuid4().hex[:8]}" for c in chunks]
        embeddings = self._embedder.encode(texts)
        emb_array = np.array(embeddings, dtype=np.float32)
        self._ids.extend(ids)
        self._documents.extend(texts)
        self._metadatas.extend(metadatas)
        if self._embeddings is None:
            self._embeddings = emb_array
        else:
            self._embeddings = np.vstack([self._embeddings, emb_array])
        self._save()
        return len(ids)
    def search(self, query, filter_paper=None):
        if self._embeddings is None or not self._ids: return []
        qv = np.array(self._embedder.encode_query(query), dtype=np.float32)
        norms = np.linalg.norm(self._embeddings, axis=1)
        qn = np.linalg.norm(qv)
        if qn == 0 or np.any(norms == 0): return []

        # 如果限定论文，只在该论文的片段中搜索
        if filter_paper:
            paper_idxs = [k for k, m in enumerate(self._metadatas) if m.get("file_name") == filter_paper]
            if not paper_idxs: return []
            pe = self._embeddings[paper_idxs]
            pn = np.linalg.norm(pe, axis=1)
            sims = np.dot(pe, qv) / (pn * qn)
            top_k = min(self._top_k_initial, len(sims))
            top_pos = np.argsort(sims)[-top_k:][::-1]
            cand = [(self._documents[paper_idxs[pos]], self._metadatas[paper_idxs[pos]], float(sims[pos])) for pos in top_pos]
        else:
            sims = np.dot(self._embeddings, qv) / (norms * qn)
            top_k = min(self._top_k_initial, len(sims))
            idxs = np.argsort(sims)[-top_k:][::-1]
            cand = [(self._documents[idx], self._metadatas[idx], float(sims[idx])) for idx in idxs]
        if cand:
            rr = self._reranker.rerank(query=query, candidates=[(t,m) for t,m,_ in cand], top_k=self._top_k_rerank)
            return [dict(text=t, metadata=m, score=round(s,4)) for t,m,s in rr if s>=self._score_threshold]
        return []
    def list_papers(self):
        pm = {}
        for m in self._metadatas:
            fn = m.get("file_name", "unknown")
            if fn not in pm:
                pm[fn] = dict(file_name=fn, paper_title=m.get("paper_title",fn), chunk_count=0)
            pm[fn]["chunk_count"] += 1
        return list(pm.values())
    def delete_paper(self, file_name):
        before = len(self._ids)
        ni, nd, nm = [], [], []
        for i, m in enumerate(self._metadatas):
            if m.get("file_name") != file_name:
                ni.append(self._ids[i])
                nd.append(self._documents[i])
                nm.append(m)
        d = before - len(ni)
        if d == 0: return 0
        self._ids, self._documents, self._metadatas = ni, nd, nm
        self._embeddings = np.array([self._embeddings[j] for j in range(len(ni))]) if ni else None
        self._save()
        return d
    def clear_all(self):
        self._ids, self._documents, self._metadatas, self._embeddings = [], [], [], None
        if self._data_path.exists(): self._data_path.unlink()
        if self._embeddings_path.exists(): self._embeddings_path.unlink()
    @property
    def paper_count(self):
        return len(self.list_papers())
    @property
    def total_chunks(self):
        return len(self._ids)

_vector_store = None

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store

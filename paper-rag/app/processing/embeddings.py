"""BGE嵌入与重排模块"""
import logging
from typing import List, Optional, Tuple
import numpy as np
from app.config import get_config

logger = logging.getLogger(__name__)


class EmbeddingModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._initialized = False
        return cls._instance

    def initialize(self):
        if self._initialized:
            return
        cfg = get_config()
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"加载嵌入模型: {cfg.embedding_model_name} ...")
            self._model = SentenceTransformer(cfg.embedding_model_name, device=cfg.embedding_device)
            self._normalize = cfg.embedding_normalize
            self._initialized = True
            logger.info("嵌入模型加载完成")
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}")
            raise

    @property
    def model(self):
        if not self._initialized:
            self.initialize()
        return self._model

    def encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        self.initialize()
        embeddings = self._model.encode(texts, show_progress_bar=False)
        if self._normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings.tolist()

    def encode_query(self, query: str) -> List[float]:
        self.initialize()
        prefixed = f"为这个句子生成表示以用于检索相关文章：{query}"
        emb = self._model.encode([prefixed], show_progress_bar=False)
        if self._normalize:
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb[0].tolist()


class RerankerModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._initialized = False
        return cls._instance

    def initialize(self):
        if self._initialized:
            return
        cfg = get_config()
        if not cfg.reranker_model_name:
            logger.info("重排模型未配置，跳过加载")
            self._initialized = True
            return
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"加载重排模型: {cfg.reranker_model_name} ...")
            self._model = CrossEncoder(cfg.reranker_model_name, device=cfg.reranker_device)
            self._initialized = True
            logger.info("重排模型加载完成")
        except Exception as e:
            logger.warning(f"重排模型加载失败，跳过重排: {e}")
            self._initialized = True

    def rerank(self, query: str, candidates: List[Tuple[str, dict]], top_k: int = 5) -> List[Tuple[str, dict, float]]:
        if not candidates or not self._model:
            return [(t, m, 1.0) for t, m in candidates[:top_k]]
        self.initialize()
        if not self._model:
            return [(t, m, 1.0) for t, m in candidates[:top_k]]
        pairs = [(query, text) for text, _ in candidates]
        scores = self._model.predict(pairs)
        scored = [(text, meta, float(scores[i])) for i, (text, meta) in enumerate(candidates)]
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]


_embedder: Optional[EmbeddingModel] = None
_reranker: Optional[RerankerModel] = None


def get_embedder() -> EmbeddingModel:
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingModel()
    return _embedder


def get_reranker() -> RerankerModel:
    global _reranker
    if _reranker is None:
        _reranker = RerankerModel()
    return _reranker
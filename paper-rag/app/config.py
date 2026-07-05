"""配置加载模块"""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, field

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
DATA_DIR = ROOT_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
CHROMA_DIR = DATA_DIR / "chroma"
LOGS_DIR = ROOT_DIR / "logs"


@dataclass
class LLMProviderConfig:
    """单个LLM厂商配置"""
    api_key: str
    base_url: str
    model: str
    max_tokens: int = 2048
    temperature: float = 0.3


@dataclass
class AppConfig:
    """全局应用配置"""
    default_provider: str = "tongyi"
    llm_providers: Dict[str, LLMProviderConfig] = field(default_factory=dict)
    request_timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 2.0
    embedding_model_name: str = "BAAI/bge-large-zh-v1.5"
    embedding_device: str = "cpu"
    embedding_normalize: bool = True
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"
    chroma_persist_dir: str = "data/chroma"
    chroma_collection: str = "papers"
    chroma_distance: str = "cosine"
    top_k_initial: int = 20
    top_k_rerank: int = 5
    chunk_score_threshold: float = 0.3
    chunk_size: int = 512
    chunk_overlap: int = 64
    max_section_chars: int = 60000
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    server_reload: bool = True
    frontend_host: str = "0.0.0.0"
    frontend_port: int = 7860
    frontend_share: bool = False


_config: Optional[AppConfig] = None


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    global _config
    path = config_path or CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"配置文件未找到: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = AppConfig()
    providers_raw = raw.get("llm_providers", {})
    for name, p in providers_raw.items():
        cfg.llm_providers[name] = LLMProviderConfig(
            api_key=str(p.get("api_key", "")),
            base_url=str(p.get("base_url", "")),
            model=str(p.get("model", "")),
            max_tokens=int(p.get("max_tokens", 2048)),
            temperature=float(p.get("temperature", 0.3)),
        )
    cfg.default_provider = raw.get("default_provider", "tongyi")
    cfg.request_timeout = int(raw.get("request_timeout", 60))
    cfg.max_retries = int(raw.get("max_retries", 3))
    cfg.retry_delay = float(raw.get("retry_delay", 2.0))
    emb = raw.get("embedding", {})
    cfg.embedding_model_name = emb.get("model_name", cfg.embedding_model_name)
    cfg.embedding_device = emb.get("device", cfg.embedding_device)
    cfg.embedding_normalize = emb.get("normalize_embeddings", True)
    rr = raw.get("reranker", {})
    cfg.reranker_model_name = rr.get("model_name", cfg.reranker_model_name)
    cfg.reranker_device = rr.get("device", cfg.reranker_device)
    ch = raw.get("chroma", {})
    cfg.chroma_persist_dir = ch.get("persist_directory", cfg.chroma_persist_dir)
    cfg.chroma_collection = ch.get("collection_name", cfg.chroma_collection)
    cfg.chroma_distance = ch.get("distance", cfg.chroma_distance)
    rt = raw.get("retrieval", {})
    cfg.top_k_initial = int(rt.get("top_k_initial", cfg.top_k_initial))
    cfg.top_k_rerank = int(rt.get("top_k_rerank", cfg.top_k_rerank))
    cfg.chunk_score_threshold = float(rt.get("chunk_score_threshold", cfg.chunk_score_threshold))
    ck = raw.get("chunking", {})
    cfg.chunk_size = int(ck.get("chunk_size", cfg.chunk_size))
    cfg.chunk_overlap = int(ck.get("chunk_overlap", cfg.chunk_overlap))
    cfg.max_section_chars = int(ck.get("max_section_chars", cfg.max_section_chars))
    sv = raw.get("server", {})
    cfg.server_host = sv.get("host", cfg.server_host)
    cfg.server_port = int(sv.get("port", cfg.server_port))
    cfg.server_reload = bool(sv.get("reload", cfg.server_reload))
    fe = raw.get("frontend", {})
    cfg.frontend_host = fe.get("server_name", cfg.frontend_host)
    cfg.frontend_port = int(fe.get("server_port", cfg.frontend_port))
    cfg.frontend_share = bool(fe.get("share", cfg.frontend_share))
    _config = cfg
    return cfg


def get_config() -> AppConfig:
    if _config is None:
        return load_config()
    return _config


def get_provider_config(provider_name: Optional[str] = None) -> LLMProviderConfig:
    cfg = get_config()
    name = provider_name or cfg.default_provider
    if name not in cfg.llm_providers:
        available = list(cfg.llm_providers.keys())
        raise ValueError(f"未找到厂商 '{name}' 的配置，可用厂商: {available}")
    return cfg.llm_providers[name]


def ensure_dirs():
    for d in [DATA_DIR, PAPERS_DIR, CHROMA_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()

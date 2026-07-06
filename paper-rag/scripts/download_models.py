"""Download ML models on first run (cached to HuggingFace cache volume)."""
import logging, sys, time

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

models = [
    ("embedding", "BAAI/bge-large-zh-v1.5", "SentenceTransformer"),
    ("reranker", "BAAI/bge-reranker-v2-m3", "CrossEncoder"),
]

for name, model_id, cls_name in models:
    logger.info(f"Downloading {name} model: {model_id} ...")
    t0 = time.time()
    try:
        if "reranker" in model_id:
            from sentence_transformers import CrossEncoder
            CrossEncoder(model_id)
        else:
            from sentence_transformers import SentenceTransformer
            SentenceTransformer(model_id)
        elapsed = time.time() - t0
        logger.info(f"  Done ({elapsed:.1f}s)")
    except Exception as e:
        logger.error(f"  Failed: {e}")
        sys.exit(1)

logger.info("All models ready.")

"""FastAPI 应用主入口"""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_config
from app.api.routers import router as api_router
from app.api.analysis_routes import router as analysis_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    logger.info("=" * 50)
    logger.info("论文RAG知识库服务启动")
    logger.info(f"默认LLM厂商: {cfg.default_provider}")
    logger.info(f"可用厂商: {list(cfg.llm_providers.keys())}")
    logger.info(f"向量库目录: {cfg.chroma_persist_dir}")
    logger.info("=" * 50)
    yield
    logger.info("论文RAG知识库服务关闭")


app = FastAPI(title="论文RAG知识库", description="基于本地嵌入+远程LLM的学术论文智能问答系统", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(api_router)
app.include_router(analysis_router)


@app.get("/health", tags=["系统"])
async def health_check():
    return {"status": "ok", "service": "paper-rag"}


if __name__ == "__main__":
    cfg = get_config()
    uvicorn.run("app.main:app", host=cfg.server_host, port=cfg.server_port, reload=cfg.server_reload)
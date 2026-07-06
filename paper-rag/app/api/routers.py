"""FastAPI 路由模块"""
import logging
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from app.config import PAPERS_DIR, get_config
from app.database.vector_store import get_vector_store
from app.processing.chunker import build_chunks_from_pdf
from app.api.paper_analyzer import cache_paper
from app.api.llm_client import ACADEMIC_SYSTEM_PROMPT, AuthError, LLMAPIError, NetworkError, QuotaError, RateLimitError, get_llm_client
from app.database.papers import get_paper_meta, upsert_paper_meta, list_papers_meta, delete_paper_meta
import asyncio
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["论文RAG知识库"])
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    paper_filter: Optional[str] = Field(None, description="限定单篇论文")
    provider: Optional[str] = Field(None, description="LLM厂商")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="引用片段数")
class CompareRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    paper_names: List[str] = Field(..., min_length=2, max_length=10, description="论文文件名列表")
    provider: Optional[str] = Field(None)
class ChunkSource(BaseModel):
    text: str
    paper: str
    section: str
    score: float
class AskResponse(BaseModel):
    answer: str
    sources: List[ChunkSource]
    provider: str
    model: str
@router.post("/upload", summary="上传论文PDF")
async def upload_paper(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持PDF格式")
    save_path = PAPERS_DIR / file.filename
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")
    try:
        chunks = build_chunks_from_pdf(str(save_path))
    except Exception as e:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=400, detail=f"PDF解析失败: {e}")
    if not chunks:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=400, detail="未提取到有效文本")
    try:
        vs = get_vector_store()
        count = vs.add_paper_chunks(chunks)
    except Exception as e:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=500, detail=f"向量入库失败: {e}")
        # 缓存论文全文以供分析功能使用
    from app.processing.pdf_parser import parse_pdf
    parsed = parse_pdf(str(save_path))
    cache_paper(file.filename, parsed["full_text"])
    return {"message": f"论文 '{file.filename}' 上传成功", "file_name": file.filename, "paper_title": chunks[0]["metadata"]["paper_title"], "chunks": count}
@router.post("/upload/batch", summary="Batch upload PDFs")
async def upload_papers_batch(files: List[UploadFile] = File(...)):
    """Upload multiple PDFs, process each independently."""
    results = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            results.append({"file": file.filename, "status": "skipped", "reason": "Not a PDF"})
            continue
        save_path = PAPERS_DIR / file.filename
        try:
            with open(save_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        except Exception as e:
            results.append({"file": file.filename, "status": "error", "reason": f"Save failed: {e}"})
            continue
        try:
            chunks = build_chunks_from_pdf(str(save_path))
        except Exception as e:
            if save_path.exists(): save_path.unlink()
            results.append({"file": file.filename, "status": "error", "reason": f"Parse failed: {e}"})
            continue
        if not chunks:
            if save_path.exists(): save_path.unlink()
            results.append({"file": file.filename, "status": "error", "reason": "No text"})
            continue
        try:
            vs = get_vector_store()
            count = vs.add_paper_chunks(chunks)
        except Exception as e:
            if save_path.exists(): save_path.unlink()
            results.append({"file": file.filename, "status": "error", "reason": f"Store failed: {e}"})
            continue
        parsed = parse_pdf(str(save_path))
        cache_paper(file.filename, parsed["full_text"])
        upsert_paper_meta(file_name=file.filename, title=chunks[0]["metadata"]["paper_title"])
        results.append({"file": file.filename, "status": "ok", "chunks": count, "title": chunks[0]["metadata"]["paper_title"]})
    return {"results": results, "total": len(results), "success": sum(1 for r in results if r["status"] == "ok")}
@router.get("/papers", summary="获取知识库论文列表")
async def list_papers():
    return get_vector_store().list_papers()
@router.delete("/papers/{file_name}", summary="删除论文")
async def delete_paper(file_name: str):
    vs = get_vector_store()
    deleted = vs.delete_paper(file_name)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"未找到论文 '{file_name}'")
    return {"message": f"已删除 '{file_name}' 的 {deleted} 个分片"}
@router.post("/ask", response_model=AskResponse, summary="RAG问答")
async def ask_question(req: AskRequest):
    vs = get_vector_store()
    try:
        chunks = vs.search(req.question, filter_paper=req.paper_filter)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {e}")
    if not chunks:
        return AskResponse(answer="未检索到相关内容。", sources=[], provider="", model="")
    top_k = min(req.top_k or 5, len(chunks))
    chunks = chunks[:top_k]
    try:
        client = get_llm_client(req.provider)
        answer = await client.chat(system_prompt=ACADEMIC_SYSTEM_PROMPT, context_chunks=chunks, user_query=req.question)
    except (NetworkError, AuthError, RateLimitError, QuotaError, LLMAPIError) as e:
        status = {AuthError: 401, RateLimitError: 429, QuotaError: 402, NetworkError: 503}.get(type(e), 500)
        raise HTTPException(status_code=status, detail=str(e))
    sources = [ChunkSource(text=c["text"][:500], paper=c["metadata"].get("file_name", "未知"), section=c["metadata"].get("section", ""), score=c["score"]) for c in chunks]
    provider_cfg = get_config().llm_providers.get(req.provider or get_config().default_provider)
    return AskResponse(answer=answer, sources=sources, provider=req.provider or get_config().default_provider, model=provider_cfg.model if provider_cfg else "")
@router.post("/ask/compare", response_model=AskResponse, summary="多论文对比问答")
async def compare_papers(req: CompareRequest):
    vs = get_vector_store()
    all_chunks = []
    for paper_name in req.paper_names:
        try:
            paper_chunks = vs.search(req.question, filter_paper=paper_name)
            all_chunks.extend(paper_chunks)
        except Exception as e:
            logger.warning(f"检索论文 '{paper_name}' 失败: {e}")
    if not all_chunks:
        return AskResponse(answer="未在指定论文中检索到相关内容。", sources=[], provider="", model="")
    compare_query = f"请对比以下论文中关于该问题的不同观点、方法或结论。\n问题: {req.question}\n\n注意：标注每篇论文的观点差异，如有共识也请指出。"
    try:
        client = get_llm_client(req.provider)
        answer = await client.chat(system_prompt=ACADEMIC_SYSTEM_PROMPT, context_chunks=all_chunks, user_query=compare_query)
    except (NetworkError, AuthError, RateLimitError, QuotaError, LLMAPIError) as e:
        status = {AuthError: 401, RateLimitError: 429, QuotaError: 402, NetworkError: 503}.get(type(e), 500)
        raise HTTPException(status_code=status, detail=str(e))
    sources = [ChunkSource(text=c["text"][:500], paper=c["metadata"].get("file_name", "未知"), section=c["metadata"].get("section", ""), score=c["score"]) for c in all_chunks]
    provider_cfg = get_config().llm_providers.get(req.provider or get_config().default_provider)
    return AskResponse(answer=answer, sources=sources, provider=req.provider or get_config().default_provider, model=provider_cfg.model if provider_cfg else "")
@router.post("/ask/stream", response_class=StreamingResponse, summary="RAG流式问答（SSE）")
async def ask_question_stream(req: AskRequest):
    """流式问答接口, 返回 Server-Sent Events 格式."""
    vs = get_vector_store()
    try:
        chunks = vs.search(req.question, filter_paper=req.paper_filter)
    except Exception as e:
        async def err_gen():
            yield f"data: 检索失败: {e}\n\ndata: [DONE]\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")
    if not chunks:
        async def empty_gen():
            yield "data: 未检索到相关内容。\n\ndata: [DONE]\n"
        return StreamingResponse(empty_gen(), media_type="text/event-stream")
    top_k = min(req.top_k or 5, len(chunks))
    chunks = chunks[:top_k]
    async def generate():
        try:
            client = get_llm_client(req.provider)
            async for token in client.stream_chat(ACADEMIC_SYSTEM_PROMPT, chunks, req.question):
                yield f"data: {token}\n\n"
        except Exception as e:
            yield f"data: [错误: {e}]\n\n"
        yield "data: [DONE]\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
@router.get("/papers/{file_name}/meta", summary="Get paper metadata")
async def get_paper_meta_route(file_name: str):
    meta = get_paper_meta(file_name)
    return meta or {"file_name": file_name, "summary": "", "keywords": [], "bibtex": ""}
@router.post("/papers/{file_name}/summary", summary="Generate summary + keywords via LLM")
async def generate_summary_route(file_name: str, provider: Optional[str] = None):
    vs = get_vector_store()
    chunks = vs.search("summary", filter_paper=file_name)
    if not chunks:
        raise HTTPException(status_code=404, detail="Paper not found")
    context = "\n".join([c["text"][:500] for c in chunks[:5]])
    client = get_llm_client(provider)
    kw_prompt = f"Extract 5-8 keywords from this paper (comma-separated):\n\n{context}"
    prompt_sum = f"Summarize this paper in Chinese (200 chars max):\n\n{context}"
    try:
        summary = await client.chat(
            system_prompt="You are an academic assistant.",
            context_chunks=[],
            user_query=prompt_sum,
        )
        kw_text = await client.chat(
            system_prompt="",
            context_chunks=[],
            user_query=kw_prompt,
        )
        keywords = [k.strip() for k in kw_text.split(",") if k.strip()]
        upsert_paper_meta(file_name=file_name, summary=summary, keywords=keywords)
        return {"file_name": file_name, "summary": summary, "keywords": keywords}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("""/papers/{file_name}/bibtex""", summary="Export BibTeX as BibTeX format using LLM")
async def export_bibtex_route(file_name: str, provider: Optional[str] = None):
    """Export references as BibTeX. Only uses LLM, no fallback."""
    vs = get_vector_store()
    ref_texts = []
    for j, m in enumerate(vs._metadatas):
        if m.get("file_name") == file_name:
            txt = vs._documents[j].strip()
            if txt and txt[0] == "[" and txt.split("]")[0][1:].isdigit():
                ref_texts.append(txt)
    full_text = chr(92).join(ref_texts[:60]) if ref_texts else ""
    if not full_text:
        return {"bibtex": "", "error": "No reference text found in paper chunks"}
    cfg = get_config()
    name = provider or cfg.default_provider
    pc = cfg.llm_providers.get(name)
    if not pc or not pc.api_key or pc.api_key.startswith("your-"):
        return {"bibtex": "", "error": "API key not configured - go to Settings tab first"}
    import httpx
    try:
        resp = await httpx.AsyncClient(timeout=90).post(
            f"{pc.base_url.rstrip(chr(47))}/chat/completions",
            headers={"Authorization": f"Bearer {pc.api_key}", "Content-Type": "application/json"},
            json={"model": pc.model, "messages": [
                {"role": "system", "content": "You are a BibTeX formatter. Output ONLY valid BibTeX entries."},
                {"role": "user", "content": "Convert these references to BibTeX @article format with author, title, journal, year. ONLY BibTeX:\n\n" + full_text}
            ], "temperature": 0.1, "max_tokens": 4096},
        )
        if resp.status_code == 200:
            bibtex = resp.json()["choices"][0]["message"]["content"]
            return {"bibtex": bibtex}
        return {"bibtex": "", "error": f"API error {resp.status_code}"}
    except Exception as e:
        return {"bibtex": "", "error": str(e)}
@router.delete("/papers/{file_name}", summary="删除论文")
async def delete_paper(file_name: str):
    vs = get_vector_store()
    deleted = vs.delete_paper(file_name)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"未找到论文 '{file_name}'")
    return {"message": f"已删除 '{file_name}' 的 {deleted} 个分片"}
@router.post("/ask", response_model=AskResponse, summary="RAG问答")
async def ask_question(req: AskRequest):
    vs = get_vector_store()
    try:
        chunks = vs.search(req.question, filter_paper=req.paper_filter)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {e}")
    if not chunks:
        return AskResponse(answer="未检索到相关内容。", sources=[], provider="", model="")
    top_k = min(req.top_k or 5, len(chunks))
    chunks = chunks[:top_k]
    try:
        client = get_llm_client(req.provider)
        answer = await client.chat(system_prompt=ACADEMIC_SYSTEM_PROMPT, context_chunks=chunks, user_query=req.question)
    except (NetworkError, AuthError, RateLimitError, QuotaError, LLMAPIError) as e:
        status = {AuthError: 401, RateLimitError: 429, QuotaError: 402, NetworkError: 503}.get(type(e), 500)
        raise HTTPException(status_code=status, detail=str(e))
    sources = [ChunkSource(text=c["text"][:500], paper=c["metadata"].get("file_name", "未知"), section=c["metadata"].get("section", ""), score=c["score"]) for c in chunks]
    provider_cfg = get_config().llm_providers.get(req.provider or get_config().default_provider)
    return AskResponse(answer=answer, sources=sources, provider=req.provider or get_config().default_provider, model=provider_cfg.model if provider_cfg else "")
@router.post("/ask/compare", response_model=AskResponse, summary="多论文对比问答")
async def compare_papers(req: CompareRequest):
    vs = get_vector_store()
    all_chunks = []
    for paper_name in req.paper_names:
        try:
            paper_chunks = vs.search(req.question, filter_paper=paper_name)
            all_chunks.extend(paper_chunks)
        except Exception as e:
            logger.warning(f"检索论文 '{paper_name}' 失败: {e}")
    if not all_chunks:
        return AskResponse(answer="未在指定论文中检索到相关内容。", sources=[], provider="", model="")
    compare_query = f"请对比以下论文中关于该问题的不同观点、方法或结论。\n问题: {req.question}\n\n注意：标注每篇论文的观点差异，如有共识也请指出。"
    try:
        client = get_llm_client(req.provider)
        answer = await client.chat(system_prompt=ACADEMIC_SYSTEM_PROMPT, context_chunks=all_chunks, user_query=compare_query)
    except (NetworkError, AuthError, RateLimitError, QuotaError, LLMAPIError) as e:
        status = {AuthError: 401, RateLimitError: 429, QuotaError: 402, NetworkError: 503}.get(type(e), 500)
        raise HTTPException(status_code=status, detail=str(e))
    sources = [ChunkSource(text=c["text"][:500], paper=c["metadata"].get("file_name", "未知"), section=c["metadata"].get("section", ""), score=c["score"]) for c in all_chunks]
    provider_cfg = get_config().llm_providers.get(req.provider or get_config().default_provider)
    return AskResponse(answer=answer, sources=sources, provider=req.provider or get_config().default_provider, model=provider_cfg.model if provider_cfg else "")
@router.post("/ask/stream", response_class=StreamingResponse, summary="RAG流式问答（SSE）")
async def ask_question_stream(req: AskRequest):
    """流式问答接口, 返回 Server-Sent Events 格式."""
    vs = get_vector_store()
    try:
        chunks = vs.search(req.question, filter_paper=req.paper_filter)
    except Exception as e:
        async def err_gen():
            yield f"data: 检索失败: {e}\n\ndata: [DONE]\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")
    if not chunks:
        async def empty_gen():
            yield "data: 未检索到相关内容。\n\ndata: [DONE]\n"
        return StreamingResponse(empty_gen(), media_type="text/event-stream")
    top_k = min(req.top_k or 5, len(chunks))
    chunks = chunks[:top_k]
    async def generate():
        try:
            client = get_llm_client(req.provider)
            async for token in client.stream_chat(ACADEMIC_SYSTEM_PROMPT, chunks, req.question):
                yield f"data: {token}\n\n"
        except Exception as e:
            yield f"data: [错误: {e}]\n\n"
        yield "data: [DONE]\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
@router.get("/papers/{file_name}/meta", summary="Get paper metadata")
async def get_paper_meta_route(file_name: str):
    meta = get_paper_meta(file_name)
    return meta or {"file_name": file_name, "summary": "", "keywords": [], "bibtex": ""}
@router.post("/papers/{file_name}/summary", summary="Generate summary + keywords via LLM")
async def generate_summary_route(file_name: str, provider: Optional[str] = None):
    vs = get_vector_store()
    chunks = vs.search("summary", filter_paper=file_name)
    if not chunks:
        raise HTTPException(status_code=404, detail="Paper not found")
    context = "\n".join([c["text"][:500] for c in chunks[:5]])
    from app.processing.pdf_parser import extract_references
    refs = extract_references(full_text) if full_text else []
    entries = [f"@article{{ref{i},\n  title = {{{ref[:500]}}},\n  year = {{2024}}\n}}" for i, ref in enumerate(refs)]
    if not entries:
        title = ""
        for m in vs._metadatas:
            if m.get("file_name") == file_name and m.get("paper_title"):
                title = m["paper_title"]
                break
        if not title:
            title = file_name.replace(".pdf", "")
        cid = file_name.replace(".pdf","").replace(" ","_")
        entries = [f"@misc{{{cid},\n  title = {{{title}}},\n  year = {{2024}}\n}}"]
    bibtex = "\n".join(entries)
    upsert_paper_meta(file_name=file_name, bibtex=bibtex)
    return {"bibtex": bibtex}
@router.delete("/papers/{file_name}/meta", summary="Delete paper metadata")
async def delete_paper_meta_route(file_name: str):
    delete_paper_meta(file_name)
    return {"status": "deleted"}
@router.get("/stats", summary="知识库统计")
async def get_stats():
    vs = get_vector_store()
    return {"paper_count": vs.paper_count, "total_chunks": vs.total_chunks, "default_provider": get_config().default_provider, "available_providers": list(get_config().llm_providers.keys())}
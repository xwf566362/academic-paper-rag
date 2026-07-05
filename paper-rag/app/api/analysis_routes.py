"""论文分析专用路由 - 方法解析 + 流程图生成"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.llm_client import LLMAPIError, NetworkError, AuthError, RateLimitError, QuotaError
from app.api.paper_analyzer import (
    get_cached_paper, analyze_method, generate_flowchart,
    METHOD_ANALYSIS_PROMPT, FLOWCHART_PROMPT
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analyze", tags=["论文分析"])


class AnalyzeRequest(BaseModel):
    file_name: str = Field(..., description="论文文件名")
    provider: Optional[str] = Field(None, description="LLM厂商")


class MethodResponse(BaseModel):
    file_name: str
    analysis: str
    has_method_section: bool
    is_ct_font: bool


class FlowchartResponse(BaseModel):
    file_name: str
    steps: str
    mermaid_code: str
    has_method_section: bool


@router.post("/method", response_model=MethodResponse, summary="一键解析研究方法")
async def analyze_method_endpoint(req: AnalyzeRequest):
    """对已解析论文进行结构化方法分析"""
    full_text = get_cached_paper(req.file_name)
    if not full_text:
        raise HTTPException(status_code=400, detail="论文全文未找到，请先完成上传与解析")
    try:
        result = await analyze_method(req.file_name, full_text, req.provider)
        return result
    except (NetworkError, AuthError, RateLimitError, QuotaError, LLMAPIError) as e:
        status = {AuthError: 401, RateLimitError: 429, QuotaError: 402, NetworkError: 503}.get(type(e), 500)
        raise HTTPException(status_code=status, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {e}")


@router.post("/flowchart", response_model=FlowchartResponse, summary="一键生成算法流程图")
async def generate_flowchart_endpoint(req: AnalyzeRequest):
    """从论文方法描述生成算法流程图"""
    full_text = get_cached_paper(req.file_name)
    if not full_text:
        raise HTTPException(status_code=400, detail="论文全文未找到，请先完成上传与解析")
    try:
        result = await generate_flowchart(req.file_name, full_text, req.provider)
        return result
    except (NetworkError, AuthError, RateLimitError, QuotaError, LLMAPIError) as e:
        status = {AuthError: 401, RateLimitError: 429, QuotaError: 402, NetworkError: 503}.get(type(e), 500)
        raise HTTPException(status_code=status, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {e}")
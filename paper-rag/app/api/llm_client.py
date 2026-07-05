"""多厂商LLM API客户端"""
import logging
import time
from typing import Optional
import httpx
from app.config import get_config, get_provider_config

logger = logging.getLogger(__name__)


class LLMAPIError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__(message)


class NetworkError(LLMAPIError): pass
class AuthError(LLMAPIError): pass
class RateLimitError(LLMAPIError): pass
class QuotaError(LLMAPIError): pass


class LLMClient:
    def __init__(self, provider: Optional[str] = None):
        cfg = get_config()
        provider_cfg = get_provider_config(provider)
        self.provider = provider or cfg.default_provider
        self.api_key = provider_cfg.api_key
        self.base_url = provider_cfg.base_url.rstrip("/")
        self.model = provider_cfg.model
        self.max_tokens = provider_cfg.max_tokens
        self.temperature = provider_cfg.temperature
        self.timeout = cfg.request_timeout
        self.max_retries = cfg.max_retries
        self.retry_delay = cfg.retry_delay
        if not self.api_key or self.api_key.startswith("your-"):
            logger.warning(f"[{self.provider}] API密钥未配置")

    def _get_headers(self) -> dict:
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

    def _build_chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _build_messages(self, system_prompt: str, context_chunks: list, user_query: str, max_chunk_size: int = 800) -> list:
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            text = chunk.get("text", "")
            meta = chunk.get("metadata", {})
            source = meta.get("file_name", "未知论文")
            section = meta.get("section", "")
            snippet = text[:max_chunk_size]
            context_parts.append(f"[来源 {i+1}] 论文: {source}" + (f" | 章节: {section}" if section else "") + f"\n{snippet}")
        context_str = "\n\n---\n\n".join(context_parts)
        user_content = f"请基于以下论文片段回答问题。\n\n论文片段:\n{context_str}\n\n问题: {user_query}"
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

    async def chat(self, system_prompt: str, context_chunks: list, user_query: str, max_chunk_size: int = 800, max_tokens: Optional[int] = None) -> str:
        messages = self._build_messages(system_prompt, context_chunks, user_query, max_chunk_size)
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens or self.max_tokens, "temperature": self.temperature, "stream": False}
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(self._build_chat_url(), headers=self._get_headers(), json=payload)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                elif response.status_code == 401:
                    raise AuthError(f"[{self.provider}] API密钥认证失败", status_code=401)
                elif response.status_code == 429:
                    raise RateLimitError(f"[{self.provider}] 接口限流", status_code=429)
                elif response.status_code == 402:
                    raise QuotaError(f"[{self.provider}] 额度不足", status_code=402)
                elif 500 <= response.status_code < 600:
                    raise LLMAPIError(f"[{self.provider}] 服务端错误 (HTTP {response.status_code})", status_code=response.status_code)
                else:
                    raise LLMAPIError(f"[{self.provider}] 未知错误 (HTTP {response.status_code}): {response.text[:200]}", status_code=response.status_code)
            except httpx.ConnectError as e:
                last_exception = NetworkError(f"[{self.provider}] 网络连接失败: {e}")
            except httpx.TimeoutException as e:
                last_exception = NetworkError(f"[{self.provider}] 请求超时 ({self.timeout}秒): {e}")
            except (AuthError, RateLimitError, QuotaError, LLMAPIError):
                raise
            except Exception as e:
                last_exception = LLMAPIError(f"[{self.provider}] 异常: {e}")
            if attempt < self.max_retries:
                wait = self.retry_delay * (2 ** attempt)
                logger.info(f"重试 {attempt+1}/{self.max_retries}，等待 {wait:.1f}秒...")
                time.sleep(wait)
        raise last_exception or LLMAPIError(f"[{self.provider}] 所有重试失败")


ACADEMIC_SYSTEM_PROMPT = (
    "你是一个严谨的学术研究助手。\n"
    "核心原则：\n"
    "1. 严格基于提供的论文片段回答，不编造内容。\n"
    "2. 如果片段不足以回答，明确说明无法确定。\n"
    "3. 区分论文原文内容与自己的推断。\n"
    "4. 每次引用标注论文名和章节。\n"
    "5. 禁止编造引文、数据或实验结果。\n"
    "6. 禁止将不同论文的结论混为一谈。"
)


_clients: dict = {}


def get_llm_client(provider: Optional[str] = None) -> LLMClient:
    key = provider or get_config().default_provider
    if key not in _clients:
        _clients[key] = LLMClient(key)
    return _clients[key]
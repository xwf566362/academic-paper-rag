# -*- coding: utf-8 -*-
"""Tests for LLMClient -- retry, error handling."""

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestLLMClientErrorHandling:

    @pytest.fixture(autouse=True)
    def setup(self, mock_config):
        from app.api.llm_client import LLMClient
        self.client = LLMClient("test")

    def _mock_response(self, status=200, body=None):
        mock = MagicMock()
        mock.status_code = status
        mock.json.return_value = body or {
            "choices": [{"message": {"content": "Test answer"}}]
        }
        mock.text = "mock response"
        return mock

    async def test_successful_chat(self):
        mock_resp = self._mock_response(200)
        with patch.object(self.client, "_get_headers", return_value={}):
            with patch("httpx.AsyncClient.post", AsyncMock(return_value=mock_resp)):
                answer = await self.client.chat("sp", [], "q")
        assert answer == "Test answer"

    async def test_auth_error(self):
        from app.api.llm_client import AuthError
        mock_resp = self._mock_response(401)
        with patch.object(self.client, "_get_headers", return_value={}):
            with patch("httpx.AsyncClient.post", AsyncMock(return_value=mock_resp)):
                with pytest.raises(AuthError):
                    await self.client.chat("sp", [], "q")

    async def test_rate_limit_error(self):
        from app.api.llm_client import RateLimitError
        mock_resp = self._mock_response(429)
        with patch.object(self.client, "_get_headers", return_value={}):
            with patch("httpx.AsyncClient.post", AsyncMock(return_value=mock_resp)):
                with pytest.raises(RateLimitError):
                    await self.client.chat("sp", [], "q")

    async def test_quota_error(self):
        from app.api.llm_client import QuotaError
        mock_resp = self._mock_response(402)
        with patch.object(self.client, "_get_headers", return_value={}):
            with patch("httpx.AsyncClient.post", AsyncMock(return_value=mock_resp)):
                with pytest.raises(QuotaError):
                    await self.client.chat("sp", [], "q")

    async def test_connection_error_retries(self):
        """ConnectError -> retry -> success"""
        suc_resp = self._mock_response(200)
        mock_post = AsyncMock(side_effect=[
            httpx.ConnectError("connection refused"),
            suc_resp,
        ])
        with patch.object(self.client, "_get_headers", return_value={}):
            with patch("httpx.AsyncClient.post", mock_post):
                answer = await self.client.chat("sp", [], "q")
        assert answer == "Test answer"
        assert mock_post.call_count == 2
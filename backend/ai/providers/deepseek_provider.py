"""DeepSeek provider implementation."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from ai.provider import AIProvider
from core.config import get_settings

logger = logging.getLogger(__name__)


class DeepSeekProvider(AIProvider):
    """DeepSeek API provider (OpenAI-compatible API)."""

    BASE_URL = "https://api.deepseek.com/v1"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.deepseek_api_key
        self.model = model or settings.deepseek_model
        if not self.api_key:
            raise ValueError("DeepSeek API key is required")

    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate chat completion via DeepSeek API."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 4096),
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
        return {
            "content": data["choices"][0]["message"]["content"],
            "model": data.get("model", self.model),
            "usage": data.get("usage", {}),
        }

    async def validate_credentials(self) -> bool:
        """Verify DeepSeek API key."""
        if not self.api_key:
            return False
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.BASE_URL}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=10.0,
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"DeepSeek validation failed: {e}")
            return False

    async def health_check(self) -> bool:
        return await self.validate_credentials()

    def get_supported_models(self) -> List[str]:
        return ["deepseek-chat", "deepseek-coder"]

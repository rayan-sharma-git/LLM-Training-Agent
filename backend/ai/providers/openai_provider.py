"""OpenAI provider implementation."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from ai.provider import AIProvider
from core.config import get_settings

logger = logging.getLogger(__name__)


class OpenAIProvider(AIProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        self.base_url = "https://api.openai.com/v1"
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate chat completion."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "messages": messages, **kwargs},
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return {
                "content": data["choices"][0]["message"]["content"],
                "model": data["model"],
                "usage": data.get("usage", {}),
            }

    async def validate_credentials(self) -> bool:
        """Verify API key."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=10.0,
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"OpenAI validation failed: {e}")
            return False

    async def health_check(self) -> bool:
        """Check provider availability."""
        return await self.validate_credentials()

    def get_supported_models(self) -> List[str]:
        """Return available models."""
        return ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
"""Anthropic provider implementation."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from ai.provider import AIProvider
from core.config import get_settings

logger = logging.getLogger(__name__)


class AnthropicProvider(AIProvider):
    """Anthropic API provider."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.anthropic_model
        self.base_url = "https://api.anthropic.com/v1"
        if not self.api_key:
            raise ValueError("Anthropic API key is required")

    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate chat completion."""
        system = kwargs.pop("system", None)
        payload = {"model": self.model, "max_tokens": 1024, "messages": messages, **kwargs}
        if system:
            payload["system"] = system
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{self.base_url}/messages", headers=headers, json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            return {
                "content": data["content"][0]["text"],
                "model": data["model"],
                "usage": {"prompt_tokens": data["usage"]["input_tokens"], "completion_tokens": data["usage"]["output_tokens"]},
            }

    async def validate_credentials(self) -> bool:
        """Verify API key."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/messages",
                    headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                    json={"model": self.model, "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]},
                    timeout=10.0,
                )
                return response.status_code in (200, 400)  # 400 means auth is valid but bad request
        except Exception as e:
            logger.error(f"Anthropic validation failed: {e}")
            return False

    async def health_check(self) -> bool:
        """Check provider availability."""
        return await self.validate_credentials()

    def get_supported_models(self) -> List[str]:
        """Return available models."""
        return ["claude-3-5-haiku-20241022", "claude-3-opus-20240229"]
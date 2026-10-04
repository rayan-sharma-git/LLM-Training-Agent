"""OpenAI-compatible provider (for LMStudio, Text Generation WebUI, etc.)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from ai.provider import AIProvider
from core.config import get_settings
from core.security import validate_provider_url

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(AIProvider):
    """
    OpenAI-compatible API provider.

    Works with any local or remote server that implements the
    OpenAI Chat Completions API format, such as:
      - LM Studio
      - Text Generation WebUI (OpenAI extension)
      - vLLM
      - Together AI
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.openai_compatible_api_key or "dummy-key"
        self.base_url = validate_provider_url(base_url or settings.openai_compatible_base_url, allow_remote=True)
        self.model = model or settings.openai_compatible_model

    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate chat completion via OpenAI-compatible API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
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
        """Verify the endpoint is reachable."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=10.0,
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"OpenAI-compatible validation failed: {e}")
            return False

    async def health_check(self) -> bool:
        return await self.validate_credentials()

    def get_supported_models(self) -> List[str]:
        """Query the endpoint's /models for available models."""
        import asyncio

        try:
            async def _fetch():
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.base_url}/models",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        timeout=10.0,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        return [m["id"] for m in data.get("data", [])]
                    return []

            loop = asyncio.get_event_loop()
            models = loop.run_until_complete(_fetch())
            return models if models else ["gpt-3.5-turbo"]
        except Exception:
            return ["gpt-3.5-turbo"]

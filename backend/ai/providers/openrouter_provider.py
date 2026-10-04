"""OpenRouter provider implementation."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from ai.provider import AIProvider
from core.config import get_settings

logger = logging.getLogger(__name__)

# OpenRouter API documentation: https://openrouter.ai/docs
# Free models regularly available:
#   - meta-llama/llama-3-8b-instruct:free
#   - mistralai/mistral-7b-instruct:free
#   - google/gemma-7b-it:free
# These change over time; we query the API for the current list.


class OpenRouterProvider(AIProvider):
    """OpenRouter API provider (gateway to many models)."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.openrouter_api_key
        self.model = model or settings.openrouter_model
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")

    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate chat completion via OpenRouter API."""
        system = kwargs.pop("system", None)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/rayan-sharma-git/llm-training-agent",
            "X-Title": "LLM Training Agent",
        }

        # If a system prompt is provided separately, prepend it as a system message
        if system:
            messages = [{"role": "system", "content": system}] + list(messages)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/chat/completions",
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
        """Verify OpenRouter API key."""
        if not self.api_key:
            return False
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.BASE_URL}/models",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": "https://github.com/rayan-sharma-git/llm-training-agent",
                        "X-Title": "LLM Training Agent",
                    },
                    timeout=10.0,
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"OpenRouter validation failed: {e}")
            return False

    async def health_check(self) -> bool:
        return await self.validate_credentials()

    def get_supported_models(self) -> List[str]:
        """
        Query the OpenRouter API for currently available models.
        Returns at least a fallback if the API is unreachable.
        """
        import asyncio

        try:
            async def _fetch():
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.BASE_URL}/models",
                        headers={
                            "Authorization": f"Bearer {self.api_key}" if self.api_key else None,
                            "HTTP-Referer": "https://github.com/rayan-sharma-git/llm-training-agent",
                        },
                        timeout=10.0,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        return [m["id"] for m in data.get("data", [])]
                    return []

            loop = asyncio.get_event_loop()
            models = loop.run_until_complete(_fetch())
            return models if models else ["meta-llama/llama-3-8b-instruct:free"]
        except Exception:
            return ["meta-llama/llama-3-8b-instruct:free"]

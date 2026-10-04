"""Cohere provider implementation."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from ai.provider import AIProvider
from core.config import get_settings

logger = logging.getLogger(__name__)


class CohereProvider(AIProvider):
    """Cohere API provider."""

    BASE_URL = "https://api.cohere.com/v2"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.cohere_api_key
        self.model = model or settings.cohere_model
        if not self.api_key:
            raise ValueError("Cohere API key is required")

    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate chat completion via Cohere API."""
        # Cohere uses a different message format: chat_history + message
        chat_history = []
        last_message = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                continue  # Cohere v2 doesn't support system in the same way
            chat_history.append({"role": role, "content": content})

        if chat_history:
            last = chat_history.pop()
            last_message = last.get("content", "")

        payload: Dict[str, Any] = {
            "model": self.model,
            "chat_history": chat_history,
            "message": last_message,
            "temperature": kwargs.get("temperature", 0.7),
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/chat",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()

        # Cohere v2 returns text in different places depending on response
        text = data.get("text", "")
        if not text and "message" in data:
            content = data["message"].get("content", [])
            if isinstance(content, list) and len(content) > 0:
                text = content[0].get("text", "")

        return {
            "content": text,
            "model": self.model,
            "usage": {
                "prompt_tokens": data.get("usage", {}).get("read_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("generated_tokens", 0),
            },
        }

    async def validate_credentials(self) -> bool:
        """Verify Cohere API key."""
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
            logger.error(f"Cohere validation failed: {e}")
            return False

    async def health_check(self) -> bool:
        return await self.validate_credentials()

    def get_supported_models(self) -> List[str]:
        return ["command-r", "command-r-plus", "command"]

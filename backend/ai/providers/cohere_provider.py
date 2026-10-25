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

    def build_payload(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Build a Cohere **v2** chat request.

        v2 takes one ``messages`` list (system, user and assistant turns in
        order). The earlier code sent the v1 shape (``message`` +
        ``chat_history``) to the v2 URL, which v2 rejects, and it discarded the
        system prompt.
        """
        turns: List[Dict[str, str]] = []
        system = kwargs.get("system")
        if system and not any(m.get("role") == "system" for m in messages):
            turns.append({"role": "system", "content": system})
        for msg in messages:
            role = msg.get("role", "user")
            if role not in ("system", "user", "assistant"):
                role = "user"
            turns.append({"role": role, "content": msg.get("content", "")})
        return {
            "model": self.model,
            "messages": turns,
            "temperature": kwargs.get("temperature", 0.7),
        }

    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate chat completion via Cohere API."""
        payload = self.build_payload(messages, **kwargs)

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

        # v2: {"message": {"content": [{"type": "text", "text": "..."}]}}
        text = ""
        content = (data.get("message") or {}).get("content", [])
        if isinstance(content, list):
            text = "".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type", "text") == "text"
            )
        elif isinstance(content, str):
            text = content

        return {
            "content": text,
            "model": self.model,
            "usage": {
                "prompt_tokens": ((data.get("usage") or {}).get("tokens") or {}).get("input_tokens", 0),
                "completion_tokens": ((data.get("usage") or {}).get("tokens") or {}).get("output_tokens", 0),
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
        return ["command-a-03-2025", "command-r-plus", "command-r"]

"""Google Gemini provider implementation."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from ai.provider import AIProvider
from core.config import get_settings

logger = logging.getLogger(__name__)


class GeminiProvider(AIProvider):
    """Google Gemini API provider."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        if not self.api_key:
            raise ValueError("Gemini API key is required")

    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate chat completion via Gemini API."""
        system = kwargs.pop("system", None)
        temperature = kwargs.pop("temperature", 0.7)

        # Gemini uses a single "contents" array; system instructions go in system_instruction
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Convert OpenAI-style roles to Gemini format
            if role == "system":
                continue  # Handled via system_instruction
            contents.append({
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": content}],
            })

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": kwargs.get("max_tokens", 4096),
            },
        }
        if system:
            payload["system_instruction"] = {"parts": [{"text": system}]}

        url = f"{self.BASE_URL}/models/{self.model}:generateContent"
        headers = {"Content-Type": "application/json"}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                params={"key": self.api_key},
                headers=headers,
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()

        # Extract text from Gemini response
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)

        return {
            "content": text,
            "model": self.model,
            "usage": {
                "prompt_tokens": data.get("usageMetadata", {}).get("promptTokenCount", 0),
                "completion_tokens": data.get("usageMetadata", {}).get("candidatesTokenCount", 0),
            },
        }

    async def validate_credentials(self) -> bool:
        """Check that API key is set and valid."""
        if not self.api_key:
            return False
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.BASE_URL}/models/{self.model}:generateContent"
                response = await client.post(
                    url,
                    params={"key": self.api_key},
                    json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
                    timeout=10.0,
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Gemini validation failed: {e}")
            return False

    async def health_check(self) -> bool:
        return await self.validate_credentials()

    def get_supported_models(self) -> List[str]:
        return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]

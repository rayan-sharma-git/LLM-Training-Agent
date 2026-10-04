"""Ollama provider implementation."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from ai.provider import AIProvider
from core.config import get_settings
from core.security import validate_provider_url

logger = logging.getLogger(__name__)


class OllamaProvider(AIProvider):
    """Ollama local LLM provider."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.base_url = validate_provider_url(base_url or settings.ollama_base_url, allow_remote=False)
        self.model = model or settings.ollama_model

    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate chat completion.

        Callers pass ``system=...`` and ``temperature=...`` (see
        ``ai.llm.LLMHelper`` and ``chat.engine.ChatEngine``). Ollama's
        ``/api/chat`` endpoint does not accept these as top-level JSON fields,
        so they are mapped to the correct Ollama request format:
          - ``system`` becomes a leading system message.
          - ``temperature`` (and other sampling params) go inside ``options``.
        Any remaining kwargs are forwarded as-is (e.g. ``format``).
        """
        payload_kwargs = dict(kwargs)

        system_prompt = payload_kwargs.pop("system", None)
        if system_prompt and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": system_prompt}, *messages]

        options: Dict[str, Any] = payload_kwargs.pop("options", None) or {}
        for param in ("temperature", "max_tokens", "top_p", "num_predict"):
            if param in payload_kwargs:
                value = payload_kwargs.pop(param)
                if value is not None:
                    ollama_param = "num_predict" if param == "max_tokens" else param
                    options.setdefault(ollama_param, value)
        if options:
            payload_kwargs["options"] = options

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False, **payload_kwargs},
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
            return {
                "content": data.get("message", {}).get("content", ""),
                "model": self.model,
                "usage": data.get("usage", {}),
            }

    async def validate_credentials(self) -> bool:
        """Verify Ollama is running."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/tags", timeout=5.0)
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama validation failed: {e}")
            return False

    async def health_check(self) -> bool:
        """Check provider availability."""
        return await self.validate_credentials()

    def get_supported_models(self) -> List[str]:
        """Query the local Ollama API for actually installed models."""
        models = self._fetch_installed_models()
        if models:
            return models
        # Fallback to common models if Ollama is not reachable
        return ["llama3.2", "llama3.1", "llama3", "mistral", "codellama", "phi3", "gemma2"]

    def _fetch_installed_models(self) -> List[str]:
        """Make a synchronous HTTP request to Ollama's /api/tags endpoint.

        A synchronous client is used intentionally: this method is called from
        sync contexts (and from within a running asyncio loop via FastAPI), so
        ``loop.run_until_complete`` would raise ``RuntimeError`` and silently
        fall back to a hardcoded model list.
        """
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
            return []
        except Exception as e:
            logger.debug(f"Ollama model fetch failed: {e}")
            return []
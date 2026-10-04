"""AI provider package with factory function."""
from __future__ import annotations

import logging

from core.config import get_settings, get_active_provider, get_active_model, get_api_key_for_provider
from core import runtime_config
from ai.provider import AIProvider

logger = logging.getLogger(__name__)


def get_provider(provider_name: str = None) -> AIProvider:
    """
    Get an AI provider instance by name.

    Supported providers:
        openai, anthropic, gemini, deepseek, cohere,
        openrouter, ollama, openai_compatible

    Falls back to the configured default if provider_name is None.
    """
    settings = get_settings()
    name = (provider_name or get_active_provider()).lower()

    # Lazy imports to avoid loading provider SDKs that aren't needed
    if name == "openai":
        from ai.providers.openai_provider import OpenAIProvider
        api_key = get_api_key_for_provider("openai")
        if not api_key:
            raise ValueError("OpenAI API key is not configured. Set it in VS Code Settings.")
        return OpenAIProvider(
            api_key=api_key,
            model=get_active_model("openai") or settings.openai_model,
        )

    elif name == "anthropic":
        from ai.providers.anthropic_provider import AnthropicProvider
        api_key = get_api_key_for_provider("anthropic")
        if not api_key:
            raise ValueError("Anthropic API key is not configured. Set it in VS Code Settings.")
        return AnthropicProvider(
            api_key=api_key,
            model=get_active_model("anthropic") or settings.anthropic_model,
        )

    elif name == "gemini":
        from ai.providers.gemini_provider import GeminiProvider
        api_key = get_api_key_for_provider("gemini")
        if not api_key:
            raise ValueError("Gemini API key is not configured. Set it in VS Code Settings.")
        return GeminiProvider(
            api_key=api_key,
            model=get_active_model("gemini") or settings.gemini_model,
        )

    elif name == "deepseek":
        from ai.providers.deepseek_provider import DeepSeekProvider
        api_key = get_api_key_for_provider("deepseek")
        if not api_key:
            raise ValueError("DeepSeek API key is not configured. Set it in VS Code Settings.")
        return DeepSeekProvider(
            api_key=api_key,
            model=get_active_model("deepseek") or settings.deepseek_model,
        )

    elif name == "cohere":
        from ai.providers.cohere_provider import CohereProvider
        api_key = get_api_key_for_provider("cohere")
        if not api_key:
            raise ValueError("Cohere API key is not configured. Set it in VS Code Settings.")
        return CohereProvider(
            api_key=api_key,
            model=get_active_model("cohere") or settings.cohere_model,
        )

    elif name == "openrouter":
        from ai.providers.openrouter_provider import OpenRouterProvider
        api_key = get_api_key_for_provider("openrouter")
        if not api_key:
            raise ValueError("OpenRouter API key is not configured. Set it in VS Code Settings.")
        return OpenRouterProvider(
            api_key=api_key,
            model=get_active_model("openrouter") or settings.openrouter_model,
        )

    elif name == "openai_compatible":
        from ai.providers.openai_compatible_provider import OpenAICompatibleProvider
        return OpenAICompatibleProvider(
            api_key=get_api_key_for_provider("openai_compatible"),
            base_url=runtime_config.get_runtime_config("openai_compatible_base_url") or settings.openai_compatible_base_url,
            model=get_active_model("openai_compatible") or settings.openai_compatible_model,
        )

    elif name == "ollama":
        from ai.providers.ollama_provider import OllamaProvider
        return OllamaProvider(
            base_url=runtime_config.get_runtime_config("ollama_base_url") or settings.ollama_base_url,
            model=get_active_model("ollama") or settings.ollama_model,
        )

    else:
        raise ValueError(
            f"Unknown provider '{name}'. "
            f"Supported: openai, anthropic, gemini, deepseek, cohere, "
            f"openrouter, ollama, openai_compatible"
        )


def list_providers() -> dict[str, dict]:
    """
    Return metadata about all available providers.

    Returns a dict mapping provider name to:
        - requires_key: whether an API key is needed
        - models: list of known model names (may be empty if dynamic)
        - description: human-readable description
    """
    return {
        "openai": {
            "requires_key": True,
            "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-3.5-turbo"],
            "description": "OpenAI API (paid, no free tier for chat)",
        },
        "anthropic": {
            "requires_key": True,
            "models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
            "description": "Anthropic Claude (paid, limited free tier)",
        },
        "gemini": {
            "requires_key": True,
            "models": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
            "description": "Google Gemini (free tier with rate limits)",
        },
        "deepseek": {
            "requires_key": True,
            "models": ["deepseek-chat", "deepseek-coder"],
            "description": "DeepSeek (free tier: 50M tokens/month)",
        },
        "cohere": {
            "requires_key": True,
            "models": ["command-r-plus", "command-r", "command"],
            "description": "Cohere (free tier with limited requests)",
        },
        "openrouter": {
            "requires_key": True,
            "models": [],  # Dynamic - fetched from OpenRouter API
            "description": "OpenRouter (free + paid models, API key required)",
        },
        "ollama": {
            "requires_key": False,
            "models": [],  # Dynamic - fetched from local Ollama
            "description": "Ollama - fully local, no API key required",
        },
        "openai_compatible": {
            "requires_key": True,
            "models": [],  # User-defined
            "description": "OpenAI-compatible custom endpoint (e.g., LMStudio, Text Generation WebUI)",
        },
    }


def get_models(provider_name: str) -> list[str]:
    """
    Get available models for a provider.
    For dynamic providers (ollama, openrouter), queries their API.
    For static providers, returns the known model list.
    """
    name = provider_name.lower()

    if name == "ollama":
        try:
            provider: AIProvider = get_provider("ollama")
            return provider.get_supported_models()
        except Exception:
            return ["llama3.2", "llama3.1", "mistral", "codellama"]

    if name == "openrouter":
        try:
            provider = get_provider("openrouter")
            return provider.get_supported_models()
        except Exception:
            return []

    if name == "openai_compatible":
        try:
            provider = get_provider("openai_compatible")
            return provider.get_supported_models()
        except Exception:
            return []

    # Static model lists
    providers_info = list_providers()
    return providers_info.get(name, {}).get("models", [])
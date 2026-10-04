"""Application configuration."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from core import runtime_config


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env files."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "LLM Training Agent"
    app_version: str = "1.0.0"
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    debug: bool = False
    cors_origins: list[str] = ["http://127.0.0.1", "http://localhost"]

    # --- AI Provider Configuration ---
    default_provider: str = "ollama"

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"

    # Anthropic
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-5-haiku-20241022"

    # Google Gemini
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-1.5-flash"

    # DeepSeek
    deepseek_api_key: Optional[str] = None
    deepseek_model: str = "deepseek-chat"

    # Cohere
    cohere_api_key: Optional[str] = None
    cohere_model: str = "command-r"

    # OpenRouter
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "meta-llama/llama-3.1-8b-instruct:free"

    # OpenAI-Compatible (custom endpoint)
    openai_compatible_api_key: Optional[str] = None
    openai_compatible_base_url: str = "http://localhost:1234/v1"
    openai_compatible_model: str = "gpt-3.5-turbo"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # --- Storage ---
    database_url: str = "sqlite+aiosqlite:///./llm_training_agent.db"

    # --- Logging ---
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def reload_settings() -> Settings:
    """Clear the cache and re-read settings from environment."""
    get_settings.cache_clear()
    return get_settings()


def get_active_provider() -> str:
    """Return the active provider: runtime override first, then env default."""
    return runtime_config.get_runtime_config("provider") or get_settings().default_provider


def get_active_model(provider: str | None = None) -> str:
    """Return the active model for a provider.

    Precedence:
      1. Runtime config (set by the VS Code extension via API).
      2. Settings model for the selected provider.
    """
    provider = provider or get_active_provider()
    # If a model was explicitly set for this provider in runtime config,
    # use it. The runtime "model" key stores the model for the active provider.
    runtime_model = runtime_config.get_runtime_config("model")
    if runtime_model:
        return runtime_model

    settings = get_settings()
    model_field = f"{provider}_model"
    return str(getattr(settings, model_field, ""))


def get_api_key_for_provider(provider: str) -> Optional[str]:
    """Return the API key for a provider.

    Precedence:
      1. Runtime in-memory key (sent securely by VS Code SecretStorage bridge).
      2. Environment variable / .env file.
    """
    runtime_key = runtime_config.get_api_key(provider)
    if runtime_key:
        return runtime_key

    settings = get_settings()
    key_field = f"{provider}_api_key"
    if provider == "openai_compatible":
        key_field = "openai_compatible_api_key"
    return getattr(settings, key_field, None)
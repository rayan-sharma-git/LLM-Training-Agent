"""Abstract AI provider interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List



class AIProvider(ABC):
    """Abstract base for AI providers."""

    @abstractmethod
    async def chat_completion(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """Generate chat completion."""
        pass

    @abstractmethod
    async def validate_credentials(self) -> bool:
        """Verify provider credentials."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check provider availability."""
        pass

    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """Return available models."""
        pass
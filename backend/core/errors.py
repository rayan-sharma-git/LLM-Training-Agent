"""Application error classes."""
from __future__ import annotations

from typing import Any, Dict, Optional

from core.redaction import scrub_obj


class AppError(Exception):
    """Base application error."""
    
    def __init__(self, message: str, error_code: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for JSON responses."""
        return {
            "errorCode": self.error_code,
            "message": scrub_obj(self.message),
            "details": scrub_obj(self.details),
        }


class AnalysisError(AppError):
    """Analysis pipeline error."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, error_code="ANALYSIS_FAILED", details=details)


class ProviderError(AppError):
    """AI provider error."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, error_code="PROVIDER_UNAVAILABLE", details=details)


class ValidationError(AppError):
    """Validation error."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, error_code="VALIDATION_ERROR", details=details)


class StorageError(AppError):
    """Persistent storage failure (database missing, corrupted, locked, ...).

    Raised by the ``storage`` layer so API routes can translate it into a
    meaningful HTTP error (503) instead of silently returning fake empty
    results.  The message is already human-readable and credential-scrubbed.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "STORAGE_UNAVAILABLE",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message=message, error_code=error_code, details=details)
"""
Custom exception classes for multilingual LLM services.

Provides standardized error handling with proper error codes, messages, and recovery hints.
"""

from typing import Any, Dict, List, Optional


class MultilingualBaseException(Exception):
    """Base exception for all multilingual service errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "MULTILINGUAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
        recovery_hint: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.recovery_hint = recovery_hint

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "recovery_hint": self.recovery_hint
        }


class LanguageNotSupportedException(MultilingualBaseException):
    """Raised when an unsupported language is requested."""

    def __init__(
        self,
        language: str,
        supported_languages: List[str],
        details: Optional[Dict[str, Any]] = None
    ):
        message = f"Language '{language}' is not supported. Supported languages: {', '.join(supported_languages)}"
        super().__init__(
            message=message,
            error_code="LANGUAGE_NOT_SUPPORTED",
            details={
                "requested_language": language,
                "supported_languages": supported_languages,
                **(details or {})
            },
            recovery_hint="Use one of the supported languages or request language support to be added"
        )


class TranslationServiceException(MultilingualBaseException):
    """Raised when translation service encounters an error."""

    def __init__(
        self,
        message: str,
        source_language: str = "unknown",
        target_language: str = "unknown",
        translation_model: str = "unknown",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Translation failed from {source_language} to {target_language}: {message}",
            error_code="TRANSLATION_FAILED",
            details={
                "source_language": source_language,
                "target_language": target_language,
                "translation_model": translation_model,
                **(details or {})
            },
            recovery_hint="Try with different language pair or check translation service status"
        )


class LLMGenerationException(MultilingualBaseException):
    """Raised when LLM generation fails."""

    def __init__(
        self,
        message: str,
        model_name: str = "unknown",
        language: str = "unknown",
        detail_level: str = "unknown",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"LLM generation failed for model {model_name} in {language}: {message}",
            error_code="LLM_GENERATION_FAILED",
            details={
                "model_name": model_name,
                "language": language,
                "detail_level": detail_level,
                **(details or {})
            },
            recovery_hint="Try with different model or language, or check model availability"
        )


class CacheException(MultilingualBaseException):
    """Raised when cache operations fail."""

    def __init__(
        self,
        message: str,
        operation: str = "unknown",
        cache_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Cache {operation} failed: {message}",
            error_code="CACHE_OPERATION_FAILED",
            details={
                "operation": operation,
                "cache_key": cache_key,
                **(details or {})
            },
            recovery_hint="Service will continue without cache, but performance may be degraded"
        )


class ParallelProcessingException(MultilingualBaseException):
    """Raised when parallel processing fails."""

    def __init__(
        self,
        message: str,
        failed_languages: List[str],
        successful_languages: List[str],
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Parallel processing failed: {message}",
            error_code="PARALLEL_PROCESSING_FAILED",
            details={
                "failed_languages": failed_languages,
                "successful_languages": successful_languages,
                "total_languages": len(failed_languages) + len(successful_languages),
                **(details or {})
            },
            recovery_hint="Try sequential processing or reduce the number of languages"
        )


class ResourceExhaustedException(MultilingualBaseException):
    """Raised when system resources are exhausted."""

    def __init__(
        self,
        message: str,
        resource_type: str,
        current_usage: float,
        limit: float,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Resource exhausted ({resource_type}): {message}",
            error_code="RESOURCE_EXHAUSTED",
            details={
                "resource_type": resource_type,
                "current_usage": current_usage,
                "limit": limit,
                "utilization_percent": (current_usage / limit) * 100 if limit > 0 else 0,
                **(details or {})
            },
            recovery_hint="Reduce concurrent requests or increase resource limits"
        )


class QualityThresholdException(MultilingualBaseException):
    """Raised when generated content doesn't meet quality thresholds."""

    def __init__(
        self,
        message: str,
        quality_score: float,
        threshold: float,
        language: str,
        content_type: str = "explanation",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Quality threshold not met for {content_type} in {language}: {message}",
            error_code="QUALITY_THRESHOLD_NOT_MET",
            details={
                "quality_score": quality_score,
                "threshold": threshold,
                "language": language,
                "content_type": content_type,
                "quality_gap": threshold - quality_score,
                **(details or {})
            },
            recovery_hint="Try regenerating content or use fallback language"
        )


class ModelUnavailableException(MultilingualBaseException):
    """Raised when required LLM model is not available."""

    def __init__(
        self,
        model_name: str,
        service_name: str = "unknown",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Model '{model_name}' is unavailable in service '{service_name}'",
            error_code="MODEL_UNAVAILABLE",
            details={
                "model_name": model_name,
                "service_name": service_name,
                **(details or {})
            },
            recovery_hint="Check model service status or use fallback model"
        )


class ConfigurationException(MultilingualBaseException):
    """Raised when configuration is invalid or missing."""

    def __init__(
        self,
        message: str,
        config_key: str,
        config_value: Any = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Configuration error for '{config_key}': {message}",
            error_code="CONFIGURATION_ERROR",
            details={
                "config_key": config_key,
                "config_value": config_value,
                **(details or {})
            },
            recovery_hint="Check configuration settings and ensure all required values are set"
        )


def handle_multilingual_exception(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    logger=None
) -> Dict[str, Any]:
    """
    Convert any exception to standardized multilingual error response.

    Args:
        exception: The exception to handle
        context: Additional context information
        logger: Logger instance for error recording

    Returns:
        Standardized error dictionary
    """
    if isinstance(exception, MultilingualBaseException):
        error_dict = exception.to_dict()
    else:
        # Convert generic exceptions to MultilingualBaseException
        error_dict = MultilingualBaseException(
            message=str(exception),
            error_code="UNKNOWN_ERROR",
            details={"original_exception_type": type(exception).__name__},
            recovery_hint="Please try again or contact support if the issue persists"
        ).to_dict()

    # Add context if provided
    if context:
        error_dict["details"].update(context)

    # Log the error if logger provided
    if logger:
        logger.error(f"Multilingual service error: {error_dict['error_code']} - {error_dict['message']}",
                    extra={"error_details": error_dict["details"]})

    return error_dict
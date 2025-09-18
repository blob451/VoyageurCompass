"""
Health check endpoints for multilingual LLM services.

Provides comprehensive health monitoring, model availability checks,
and system status reporting for production monitoring.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List

from django.conf import settings
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from Analytics.services.circuit_breaker import get_all_circuit_breaker_stats
from Analytics.services.feature_flags import get_feature_flags
from Analytics.services.local_llm_service import get_local_llm_service
from Analytics.services.multilingual_metrics import get_multilingual_metrics
from Analytics.services.multilingual_optimizer import get_multilingual_optimizer
from Analytics.services.translation_service import get_translation_service

logger = logging.getLogger(__name__)


@extend_schema(
    summary="Multilingual system health check",
    description="Comprehensive health check for multilingual LLM services including model availability, feature flags, and performance metrics",
    responses={
        200: {
            "description": "Health check results",
            "example": {
                "status": "healthy",
                "timestamp": "2025-01-17T12:00:00Z",
                "multilingual_enabled": True,
                "supported_languages": ["en", "fr", "es"],
                "model_status": {
                    "fr": "healthy",
                    "es": "healthy"
                },
                "quality_scores": {
                    "fr": 0.85,
                    "es": 0.87
                },
                "feature_flags": {
                    "multilingual_enabled": True,
                    "french_enabled": True,
                    "spanish_enabled": True
                },
                "circuit_breakers": {
                    "fr": "closed",
                    "es": "closed"
                },
                "performance_metrics": {
                    "avg_response_time": 2.3,
                    "cache_hit_rate": 0.75,
                    "error_rate": 0.02
                }
            }
        }
    }
)
@api_view(["GET"])
@permission_classes([AllowAny])
def multilingual_health(request):
    """
    Comprehensive health check for multilingual LLM services.

    Returns detailed status information about:
    - Feature flag states
    - Model availability per language
    - Circuit breaker states
    - Performance metrics
    - Quality scores
    """
    start_time = time.time()

    try:
        health_status = {
            "status": "unknown",
            "timestamp": datetime.now().isoformat(),
            "check_duration_ms": 0,
            "multilingual_enabled": False,
            "supported_languages": ["en", "fr", "es"],
            "model_status": {},
            "quality_scores": {},
            "feature_flags": {},
            "circuit_breakers": {},
            "performance_metrics": {},
            "translation_service": "unknown",
            "cache_status": "unknown",
            "errors": [],
            "warnings": []
        }

        # Check feature flags
        try:
            feature_flags = get_feature_flags()
            flags_status = feature_flags.get_all_flags_status()

            health_status["multilingual_enabled"] = feature_flags.is_enabled("multilingual_llm_enabled")
            health_status["feature_flags"] = {
                "multilingual_enabled": flags_status["flags"]["multilingual_llm_enabled"]["enabled"],
                "french_enabled": flags_status["flags"]["french_generation_enabled"]["enabled"],
                "spanish_enabled": flags_status["flags"]["spanish_generation_enabled"]["enabled"],
                "parallel_processing_enabled": flags_status["flags"]["parallel_processing_enabled"]["enabled"],
                "emergency_fallback": flags_status["emergency_status"]["emergency_fallback_enabled"]
            }

            if flags_status["emergency_status"]["emergency_fallback_enabled"]:
                health_status["warnings"].append("Emergency fallback is enabled")

        except Exception as e:
            health_status["errors"].append(f"Feature flags check failed: {str(e)}")
            logger.error(f"Feature flags health check failed: {str(e)}")

        # Check circuit breakers
        try:
            circuit_stats = get_all_circuit_breaker_stats()
            if "circuit_breakers" in circuit_stats and "multilingual" in circuit_stats["circuit_breakers"]:
                multilingual_breaker = circuit_stats["circuit_breakers"]["multilingual"]
                health_status["circuit_breakers"] = {}

                for lang, lang_stats in multilingual_breaker["languages"].items():
                    health_status["circuit_breakers"][lang] = lang_stats["state"]
                    if lang_stats["state"] == "open":
                        health_status["errors"].append(f"Circuit breaker open for {lang}")
                    elif lang_stats["state"] == "half_open":
                        health_status["warnings"].append(f"Circuit breaker half-open for {lang}")

        except Exception as e:
            health_status["errors"].append(f"Circuit breaker check failed: {str(e)}")
            logger.error(f"Circuit breaker health check failed: {str(e)}")

        # Check model availability for each language
        test_languages = ["fr", "es"] if health_status["multilingual_enabled"] else []

        for lang in test_languages:
            try:
                # Quick model availability test
                test_result = _test_model_availability(lang)
                health_status["model_status"][lang] = test_result["status"]

                if test_result["status"] == "healthy":
                    health_status["quality_scores"][lang] = test_result.get("quality_score", 0.0)
                else:
                    health_status["errors"].append(f"Model unhealthy for {lang}: {test_result.get('error', 'Unknown error')}")

            except Exception as e:
                health_status["model_status"][lang] = "unhealthy"
                health_status["errors"].append(f"Model check failed for {lang}: {str(e)}")
                logger.error(f"Model health check failed for {lang}: {str(e)}")

        # Check translation service
        try:
            translation_service = get_translation_service()
            if hasattr(translation_service, 'is_healthy') and translation_service.is_healthy():
                health_status["translation_service"] = "healthy"
            else:
                health_status["translation_service"] = "degraded"
                health_status["warnings"].append("Translation service may be degraded")
        except Exception as e:
            health_status["translation_service"] = "unhealthy"
            health_status["errors"].append(f"Translation service check failed: {str(e)}")

        # Get performance metrics
        try:
            metrics_service = get_multilingual_metrics()
            current_metrics = metrics_service.get_real_time_dashboard()

            if "current_rates" in current_metrics:
                rates = current_metrics["current_rates"]
                health_status["performance_metrics"] = {
                    "avg_response_time": rates.get("avg_response_time", 0),
                    "error_rate": rates.get("error_rate", 0),
                    "requests_per_minute": rates.get("requests_per_minute", 0),
                    "quality_score": rates.get("quality_score", 0)
                }

                # Add warnings for poor performance
                if rates.get("avg_response_time", 0) > 10:
                    health_status["warnings"].append("High response time detected")
                if rates.get("error_rate", 0) > 0.1:
                    health_status["warnings"].append("High error rate detected")

        except Exception as e:
            health_status["warnings"].append(f"Performance metrics unavailable: {str(e)}")

        # Check cache status
        try:
            from django.core.cache import cache
            cache.set("health_check_test", "ok", 30)
            if cache.get("health_check_test") == "ok":
                health_status["cache_status"] = "healthy"
            else:
                health_status["cache_status"] = "degraded"
                health_status["warnings"].append("Cache may be degraded")
        except Exception as e:
            health_status["cache_status"] = "unhealthy"
            health_status["warnings"].append(f"Cache check failed: {str(e)}")

        # Determine overall status
        if health_status["errors"]:
            health_status["status"] = "unhealthy"
        elif health_status["warnings"]:
            health_status["status"] = "degraded"
        else:
            health_status["status"] = "healthy"

        # Calculate check duration
        health_status["check_duration_ms"] = round((time.time() - start_time) * 1000, 2)

        # Set appropriate HTTP status
        if health_status["status"] == "healthy":
            response_status = status.HTTP_200_OK
        elif health_status["status"] == "degraded":
            response_status = status.HTTP_200_OK  # Still operational
        else:
            response_status = status.HTTP_503_SERVICE_UNAVAILABLE

        return Response(health_status, status=response_status)

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)

        error_response = {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "check_duration_ms": round((time.time() - start_time) * 1000, 2),
            "error": str(e),
            "multilingual_enabled": False
        }

        return Response(error_response, status=status.HTTP_503_SERVICE_UNAVAILABLE)


def _test_model_availability(language: str) -> Dict[str, Any]:
    """
    Test model availability for a specific language.

    Args:
        language: Language code to test

    Returns:
        Dictionary with test results
    """
    try:
        # Simple test data for model availability check
        test_data = {
            "symbol": "AAPL",
            "score_0_10": 7.5,
            "weighted_scores": {"sma50vs200": 0.7},
            "indicators": {"sma50": 150.0, "sma200": 145.0}
        }

        # Quick generation test with short timeout
        llm_service = get_local_llm_service()

        start_time = time.time()

        # Test with minimal content to check model availability
        test_result = llm_service.generate_multilingual_explanation(
            test_data,
            detail_level="summary",
            explanation_type="technical_analysis",
            target_language=language
        )

        generation_time = time.time() - start_time

        if test_result and isinstance(test_result, dict):
            # Extract quality score if available
            quality_score = test_result.get('quality_score', 0.8)  # Default if not present

            return {
                "status": "healthy",
                "quality_score": quality_score,
                "response_time": round(generation_time, 2),
                "content_length": len(test_result.get('content', '')),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "unhealthy",
                "error": "No valid response from model",
                "response_time": round(generation_time, 2),
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@extend_schema(
    summary="Detailed multilingual metrics",
    description="Get detailed performance and quality metrics for multilingual services",
    responses={200: {"description": "Detailed metrics data"}}
)
@api_view(["GET"])
@permission_classes([AllowAny])
def multilingual_metrics(request):
    """Get detailed multilingual metrics for monitoring dashboards."""
    try:
        metrics_service = get_multilingual_metrics()

        # Get comprehensive metrics
        real_time_metrics = metrics_service.get_real_time_dashboard()
        quality_metrics = metrics_service.get_quality_dashboard()
        performance_report = metrics_service.get_performance_report()

        # Get optimizer statistics
        try:
            optimizer = get_multilingual_optimizer()
            optimizer_stats = optimizer.get_pool_statistics()
        except Exception as e:
            logger.warning(f"Could not get optimizer stats: {str(e)}")
            optimizer_stats = {"error": str(e)}

        # Combine all metrics
        combined_metrics = {
            "timestamp": datetime.now().isoformat(),
            "real_time": real_time_metrics,
            "quality": quality_metrics,
            "performance": performance_report,
            "optimizer": optimizer_stats,
            "feature_flags": get_feature_flags().get_all_flags_status(),
            "circuit_breakers": get_all_circuit_breaker_stats()
        }

        return Response(combined_metrics)

    except Exception as e:
        logger.error(f"Metrics endpoint failed: {str(e)}")
        return Response(
            {"error": str(e), "timestamp": datetime.now().isoformat()},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Quick health ping",
    description="Lightweight health check for load balancer health checks",
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unhealthy"}
    }
)
@api_view(["GET"])
@permission_classes([AllowAny])
def health_ping(request):
    """Lightweight health check for load balancers."""
    try:
        # Quick checks only
        feature_flags = get_feature_flags()
        multilingual_enabled = feature_flags.is_enabled("multilingual_llm_enabled")
        emergency_fallback = feature_flags.is_enabled("emergency_fallback_enabled")

        if emergency_fallback:
            return Response(
                {"status": "degraded", "reason": "emergency_fallback_enabled"},
                status=status.HTTP_200_OK
            )

        return Response({
            "status": "healthy",
            "multilingual_enabled": multilingual_enabled,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Health ping failed: {str(e)}")
        return Response(
            {"status": "unhealthy", "error": str(e)},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )


@extend_schema(
    summary="Feature flags status",
    description="Get current status of all feature flags",
    responses={200: {"description": "Feature flags status"}}
)
@api_view(["GET"])
@permission_classes([AllowAny])
def feature_flags_status(request):
    """Get current feature flags status."""
    try:
        feature_flags = get_feature_flags()
        flags_status = feature_flags.get_all_flags_status(request.user if request.user.is_authenticated else None)

        return Response(flags_status)

    except Exception as e:
        logger.error(f"Feature flags status failed: {str(e)}")
        return Response(
            {"error": str(e), "timestamp": datetime.now().isoformat()},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
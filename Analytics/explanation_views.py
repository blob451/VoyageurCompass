"""
Explanation API views for Analytics app.
Provides endpoints for generating and managing natural language explanations using local LLaMA model.
"""

import logging
import random
import statistics
import time
from datetime import datetime

from django.db import transaction
from django.db.utils import OperationalError
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from Analytics.services.explanation_service import get_explanation_service
from Analytics.services.translation_service import get_translation_service
from Analytics.services.multilingual_pipeline import get_multilingual_pipeline
from Analytics.services.language_detector import get_language_detector, detect_request_language
from Analytics.views import AnalysisThrottle
from Data.models import AnalyticsResults

logger = logging.getLogger(__name__)


def retry_database_operation(operation, max_retries=3, base_delay=0.1):
    """
    Retry database operation with exponential backoff to handle concurrency issues.

    Args:
        operation: Callable that performs the database operation
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff

    Returns:
        Result of the operation

    Raises:
        Exception: If all retries are exhausted
    """
    for attempt in range(max_retries + 1):
        try:
            return operation()
        except (OperationalError, Exception) as e:
            error_msg = str(e).lower()

            # Check if it's a concurrency-related error
            if any(
                keyword in error_msg
                for keyword in [
                    "could not serialize access due to concurrent update",
                    "deadlock detected",
                    "database is locked",
                    "concurrent update",
                ]
            ):
                if attempt < max_retries:
                    # Exponential backoff with jitter
                    delay = base_delay * (2**attempt) + random.uniform(0, 0.1)
                    logger.warning(
                        f"[DB RETRY] Concurrency issue on attempt {attempt + 1}/{max_retries + 1}: {error_msg}"
                    )
                    logger.info(f"[DB RETRY] Retrying in {delay:.3f} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Database operation failed after {max_retries + 1} attempts: {error_msg}")
                    raise
            else:
                # Non-concurrency error, don't retry
                logger.error(f"Non-retryable database error: {error_msg}")
                raise

    # This should never be reached
    raise Exception("Unexpected retry logic error")


@extend_schema(
    summary="Generate explanation for analysis result",
    description="Generate natural language explanation for a specific analysis result using local LLaMA model",
    parameters=[
        OpenApiParameter(
            name="analysis_id",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.PATH,
            required=True,
            description="Analysis result ID",
        ),
        OpenApiParameter(
            name="detail_level",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            required=False,
            description="Explanation detail level: summary (Standard), standard (Enhanced), detailed (Premium) - default: standard",
        ),
        OpenApiParameter(
            name="language",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            required=False,
            description="Target language for explanation: en, fr, es (default: en)",
        ),
    ],
    responses={
        200: {
            "description": "Explanation generated successfully",
            "example": {
                "success": True,
                "analysis_id": 123,
                "symbol": "AAPL",
                "explanation": {
                    "content": "AAPL receives a 7.5/10 analysis score...",
                    "confidence_score": 0.85,
                    "detail_level": "standard",
                    "method": "llm",
                    "language": "en",
                    "model_used": "phi3:3.8b",
                    "generation_time": 3.2,
                    "word_count": 245,
                    "translated_content": "AAPL reçoit un score d'analyse de 7.5/10...",
                    "translation_quality": 0.91,
                    "translation_model": "qwen2:latest",
                    "translation_time": 1.8,
                    "target_language": "fr",
                    "total_generation_time": 5.0,
                },
                "multilingual": {
                    "requested_language": "fr",
                    "translation_available": True,
                    "supported_languages": ["en", "fr", "es"],
                },
            },
        },
        404: {"description": "Analysis not found"},
        403: {"description": "Not authorized to view this analysis"},
        400: {"description": "Invalid parameters"},
    },
)
@api_view(["POST"])
@permission_classes([IsAuthenticated])
@throttle_classes([AnalysisThrottle])
def generate_explanation(request, analysis_id):
    """
    Generate explanation for a specific analysis result.

    Path Parameters:
        analysis_id: Analysis result ID

    Query Parameters:
        detail_level: Explanation detail level (summary/standard/detailed, default: standard)
                      Note: summary=Standard, standard=Enhanced, detailed=Premium
        language: Target language (en/fr/es, default: en)

    Returns:
        Generated explanation with multilingual metadata
    """
    detail_level = request.query_params.get("detail_level", "standard")
    explicit_language = request.query_params.get("language", None)
    force_regenerate = request.query_params.get("force_regenerate", "false").lower() == "true"

    # Validate detail level
    if detail_level not in ["summary", "standard", "detailed"]:
        return Response(
            {"error": "detail_level must be one of: summary, standard, detailed"}, status=status.HTTP_400_BAD_REQUEST
        )

    # Smart language detection with fallback hierarchy
    language = detect_request_language(request, request.user, explicit_language)

    # Get language detector for validation and metadata
    language_detector = get_language_detector()
    language_preference = language_detector.detect_user_language(request, request.user, explicit_language)

    logger.info(f"Language detected for explanation: {language} (confidence: {language_preference.confidence:.2f}, source: {language_preference.source})")

    try:
        # Get the analysis result and ensure it belongs to the user
        analysis_result = (
            AnalyticsResults.objects.filter(id=analysis_id, user=request.user).select_related("stock").first()
        )

        if not analysis_result:
            return Response({"error": "Analysis not found"}, status=status.HTTP_404_NOT_FOUND)

        logger.info(f"Generating explanation for analysis {analysis_id} ({analysis_result.stock.symbol})")

        # Get explanation service
        explanation_service = get_explanation_service()

        # Detailed logging for debugging
        logger.info(f"[EXPLAIN] Processing explanation request for analysis {analysis_id}")
        logger.info(f"[EXPLAIN] Detail level: {detail_level}")
        logger.info(f"[EXPLAIN] Service enabled flag: {explanation_service.enabled}")

        try:
            llm_available = explanation_service.llm_service.is_available()
            logger.info(f"[EXPLAIN] LLM service available: {llm_available}")
        except Exception as e:
            logger.error(f"[EXPLAIN] Error checking LLM availability: {str(e)}", exc_info=True)
            llm_available = False

        if not explanation_service.is_enabled():
            logger.error(
                f"[EXPLAIN] Service not enabled - enabled: {explanation_service.enabled}, llm_available: {llm_available}"
            )
            return Response({"error": "Explanation service not available"}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        # Check if explanation already exists for this detail level
        existing_explanations = analysis_result.explanations_json or {}
        existing_levels = existing_explanations.get("levels", {})

        if detail_level in existing_levels and not force_regenerate:
            # Return existing explanation for this detail level
            logger.info(f"[EXPLAIN] Returning existing {detail_level} explanation for analysis {analysis_id}")
            existing_explanation = existing_levels[detail_level]
            explanation_result = {
                "content": existing_explanation.get("content", ""),
                "confidence_score": existing_explanation.get("confidence", 0.0),
                "method": analysis_result.explanation_method or "llama",
                "word_count": existing_explanation.get(
                    "word_count", len(existing_explanation.get("content", "").split())
                ),
                "indicators_explained": existing_explanations.get("indicators_explained", []),
                "risk_factors": existing_explanations.get("risk_factors", []),
                "recommendation": existing_explanations.get("recommendation", "HOLD"),
                "generation_time": 0.0,  # Retrieved from cache/DB
            }
        else:
            # Generate new explanation for this detail level
            regeneration_reason = "force regeneration requested" if force_regenerate else "no existing explanation"
            logger.info(f"[EXPLAIN] Starting {detail_level} explanation generation for {analysis_result.stock.symbol} ({regeneration_reason})")
            try:
                explanation_result = explanation_service.explain_prediction_single(
                    analysis_result, detail_level=detail_level, language=language, user=request.user, force_regenerate=force_regenerate
                )

                if explanation_result:
                    logger.info(
                        f"[EXPLAIN] Explanation generated successfully - method: {explanation_result.get('method')}, length: {len(explanation_result.get('content', ''))}"
                    )
                else:
                    logger.error(f"[EXPLAIN] Explanation generation returned None for analysis {analysis_id}")

            except Exception as exp_error:
                logger.error(f"[EXPLAIN] Exception during explanation generation: {str(exp_error)}", exc_info=True)
                explanation_result = None

        if not explanation_result:
            logger.error(f"[EXPLAIN] Failed to generate explanation for analysis {analysis_id}")
            return Response({"error": "Failed to generate explanation"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Update the database with explanation using retry mechanism for concurrency
        def save_explanation():
            # Refresh the analysis_result to get latest state
            analysis_result.refresh_from_db()

            # Check if explanation for this detail level needs to be saved
            existing_explanations = analysis_result.explanations_json or {}
            levels = existing_explanations.get("levels", {})

            # Generate new explanation if this detail level doesn't exist or method changed
            needs_new_explanation = (
                detail_level not in levels or analysis_result.explanation_method != explanation_result.get("method")
            )

            if needs_new_explanation:
                with transaction.atomic():
                    # Use select_for_update to prevent concurrent modifications
                    locked_result = AnalyticsResults.objects.select_for_update().get(id=analysis_id, user=request.user)

                    # Refresh existing explanations and double-check after acquiring lock
                    current_explanations = locked_result.explanations_json or {}
                    current_levels = current_explanations.get("levels", {})

                    if (
                        detail_level not in current_levels
                        or locked_result.explanation_method != explanation_result.get("method")
                    ):
                        # Update the levels with new explanation
                        current_levels[detail_level] = {
                            "content": explanation_result.get("content", ""),
                            "confidence": explanation_result.get("confidence_score", 0.0),
                            "generated_at": datetime.now().isoformat(),
                            "word_count": explanation_result.get("word_count", 0),
                        }

                        # Update the full explanations_json structure
                        locked_result.explanations_json = {
                            "levels": current_levels,
                            "indicators_explained": explanation_result.get("indicators_explained", []),
                            "risk_factors": explanation_result.get("risk_factors", []),
                            "recommendation": explanation_result.get("recommendation", "HOLD"),
                            "current_level": detail_level,
                        }

                        # Update other fields (keep narrative_text for backward compatibility with latest)
                        locked_result.explanation_method = explanation_result.get("method", "unknown")
                        locked_result.explanation_version = "1.0"
                        locked_result.narrative_text = explanation_result.get("content", "")
                        locked_result.explanation_confidence = explanation_result.get("confidence_score", 0.0)
                        locked_result.explained_at = datetime.now()
                        locked_result.save()
                        logger.info(f"[DB SAVE] Explanation saved for analysis {analysis_id} ({detail_level} level)")
                    else:
                        logger.info(
                            f"[DB SAVE] Explanation for {detail_level} level already exists for analysis {analysis_id}"
                        )
            else:
                logger.info(f"[DB SAVE] Explanation for {detail_level} level already exists for analysis {analysis_id}")

        # Execute the save operation with retry logic
        try:
            retry_database_operation(save_explanation, max_retries=3, base_delay=0.1)
        except Exception as e:
            logger.error(f"Failed to save explanation after retries: {str(e)}")
            # Continue execution - explanation generation was successful, only save failed

        # Handle multilingual generation if requested language is not English
        multilingual_result = None
        if language != "en":
            logger.info(f"[MULTILINGUAL] Generating explanation in {language} using multilingual pipeline")
            try:
                multilingual_pipeline = get_multilingual_pipeline()

                # Prepare analysis data for multilingual pipeline
                analysis_data = {
                    "symbol": analysis_result.stock.symbol,
                    "currentPrice": getattr(analysis_result, 'currentPrice', 0) or 0,
                    "score_0_10": analysis_result.score_0_10,
                    "analysis_id": analysis_id,
                    "timestamp": analysis_result.created_at.isoformat(),
                }

                # User preferences from request (could be expanded)
                user_preferences = {
                    "detail_level": detail_level,
                    "preferred_format": request.query_params.get("format", "standard"),
                }

                multilingual_result = multilingual_pipeline.generate_explanation(
                    analysis_data=analysis_data,
                    target_language=language,
                    detail_level=detail_level,
                    explanation_type="technical_analysis",
                    user_preferences=user_preferences
                )

                if multilingual_result:
                    logger.info(
                        f"[MULTILINGUAL] Generation successful - quality: {multilingual_result.get('quality_metrics', {}).get('overall_score', 0):.2f}, "
                        f"method: {multilingual_result.get('generation_method', 'unknown')}"
                    )
                else:
                    logger.warning(f"[MULTILINGUAL] Pipeline failed, falling back to translation service")

                    # Fallback to translation service
                    translation_service = get_translation_service()
                    translation_context = {
                        "symbol": analysis_result.stock.symbol,
                        "score": analysis_result.score_0_10,
                        "analysis_id": analysis_id
                    }

                    translation_result = translation_service.translate_explanation(
                        english_text=explanation_result.get("content", ""),
                        target_language=language,
                        context=translation_context
                    )

                    if translation_result:
                        # Convert translation result to multilingual format
                        multilingual_result = {
                            "explanation": translation_result.get("translated_text", ""),
                            "language": language,
                            "generation_method": "translated_fallback",
                            "model_used": translation_result.get("translation_model", "unknown"),
                            "quality_metrics": {
                                "overall_score": translation_result.get("quality_score", 0.0),
                                "translation_quality": translation_result.get("quality_score", 0.0),
                            },
                            "cultural_formatting_applied": False,
                            "timestamp": datetime.now().isoformat(),
                        }

            except Exception as multilingual_error:
                logger.error(f"[MULTILINGUAL] Pipeline error: {str(multilingual_error)}")
                multilingual_result = None

        # Prepare response data with multilingual support
        explanation_data = {
            "content": explanation_result.get("content", ""),
            "confidence_score": explanation_result.get("confidence_score", 0.0),
            "detail_level": detail_level,
            "method": explanation_result.get("method", "unknown"),
            "generation_time": explanation_result.get("generation_time", 0.0),
            "word_count": explanation_result.get("word_count", 0),
            "indicators_explained": explanation_result.get("indicators_explained", []),
            "risk_factors": explanation_result.get("risk_factors", []),
            "recommendation": explanation_result.get("recommendation", "HOLD"),
            "language": "en",  # Original language
            "model_used": explanation_result.get("model_used", "unknown"),
        }

        # Add multilingual data if available
        if multilingual_result:
            quality_metrics = multilingual_result.get("quality_metrics", {})
            explanation_data.update({
                "multilingual_content": multilingual_result.get("explanation", ""),
                "multilingual_quality": quality_metrics.get("overall_score", 0.0),
                "multilingual_model": multilingual_result.get("model_used", "unknown"),
                "generation_method": multilingual_result.get("generation_method", "unknown"),
                "target_language": language,
                "cultural_formatting_applied": multilingual_result.get("cultural_formatting_applied", False),
                "pipeline_version": multilingual_result.get("pipeline_version", "1.0"),
                "quality_breakdown": {
                    "terminology_score": quality_metrics.get("terminology_score", 0.0),
                    "completeness_score": quality_metrics.get("completeness_score", 0.0),
                    "cultural_appropriateness": quality_metrics.get("cultural_appropriateness", 0.0),
                },
            })
            # Use multilingual content as primary content for non-English requests
            explanation_data["content"] = multilingual_result.get("explanation", explanation_data["content"])
            explanation_data["language"] = language
            explanation_data["word_count"] = len(multilingual_result.get("explanation", "").split())
        
        response_data = {
            "success": True,
            "analysis_id": analysis_id,
            "symbol": analysis_result.stock.symbol,
            "explanation": explanation_data,
            "multilingual": {
                "requested_language": language,
                "pipeline_available": multilingual_result is not None,
                "supported_languages": language_detector.supported_codes,
                "pipeline_enabled": get_multilingual_pipeline().enabled,
                "quality_metrics_available": multilingual_result is not None and "quality_metrics" in multilingual_result,
                "cultural_formatting_applied": multilingual_result.get("cultural_formatting_applied", False) if multilingual_result else False,
                "language_detection": {
                    "detected_language": language_preference.language,
                    "confidence": language_preference.confidence,
                    "source": language_preference.source,
                    "explicit_request": explicit_language is not None,
                    "fallback_applied": language != language_preference.language,
                },
            },
        }

        logger.info(f"Explanation generated successfully for analysis {analysis_id}")
        return Response(response_data)

    except Exception as e:
        logger.error(f"Error in generate_explanation: {str(e)}", exc_info=True)
        logger.error(f"Explanation generation failed for analysis {analysis_id}: {str(e)}")
        return Response(
            {"error": f"Explanation generation failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Get explanation service status",
    description="Get current status of the explanation service and LLM availability",
    responses={
        200: {
            "description": "Service status retrieved",
            "example": {
                "success": True,
                "status": {
                    "enabled": True,
                    "llm_available": True,
                    "models": {
                        "summary": {"name": "phi3:3.8b", "healthy": True, "last_check": "2025-01-17T10:30:00Z"},
                        "standard": {"name": "phi3:3.8b", "healthy": True, "last_check": "2025-01-17T10:30:00Z"},
                        "detailed": {"name": "llama3.1:8b", "healthy": True, "last_check": "2025-01-17T10:30:00Z"},
                        "translation": {"name": "qwen2:latest", "healthy": True, "last_check": "2025-01-17T10:30:00Z"}
                    },
                    "performance": {
                        "total_requests": 1250,
                        "cache_hits": 340,
                        "cache_misses": 910,
                        "avg_generation_time": 4.2,
                        "error_count": 12
                    },
                    "circuit_breaker": {
                        "state": "CLOSED",
                        "failure_count": 0,
                        "last_failure": None
                    },
                    "cache_ttl": 300,
                    "supported_languages": ["en", "fr", "es"]
                }
            }
        }
    },
)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def explanation_service_status(request):
    """
    Get explanation service status.

    Returns:
        Current status of explanation service and LLM availability
    """
    try:
        explanation_service = get_explanation_service()
        status_data = explanation_service.get_service_status()

        return Response({"success": True, "status": status_data})

    except Exception as e:
        logger.error(f"Error getting explanation service status: {str(e)}")
        return Response(
            {"error": f"Failed to get service status: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Get explanation for existing analysis",
    description="Retrieve previously generated explanation for an analysis result",
    parameters=[
        OpenApiParameter(
            name="analysis_id",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.PATH,
            required=True,
            description="Analysis result ID",
        ),
        OpenApiParameter(
            name="detail_level",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            required=False,
            description="Explanation detail level: summary, standard, detailed (default: latest)",
        ),
    ],
    responses={
        200: {
            "description": "Explanation retrieved successfully",
            "example": {
                "success": True,
                "has_explanation": True,
                "explanation": {
                    "content": "Previously generated explanation...",
                    "confidence_score": 0.85,
                    "method": "llm",
                    "explained_at": "2025-01-17T10:30:00Z",
                },
            },
        },
        404: {"description": "Analysis not found"},
        403: {"description": "Not authorized to view this analysis"},
    },
)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_explanation(request, analysis_id):
    """
    Get explanation for an existing analysis result.

    Path Parameters:
        analysis_id: Analysis result ID

    Query Parameters:
        detail_level: Detail level to retrieve (summary/standard/detailed, default: latest)

    Returns:
        Existing explanation data if available
    """
    detail_level = request.query_params.get("detail_level", None)
    language = request.query_params.get("language", "en")

    # Validate language
    if language not in ["en", "fr", "es"]:
        return Response(
            {"error": "language must be one of: en, fr, es"}, status=status.HTTP_400_BAD_REQUEST
        )

    try:
        # Get the analysis result and ensure it belongs to the user
        analysis_result = (
            AnalyticsResults.objects.filter(id=analysis_id, user=request.user).select_related("stock").first()
        )

        if not analysis_result:
            return Response({"error": "Analysis not found"}, status=status.HTTP_404_NOT_FOUND)

        # Get explanations data
        explanations_json = analysis_result.explanations_json or {}
        levels = explanations_json.get("levels", {})

        # Determine which explanation content to return
        explanation_content = None
        used_level = None

        if detail_level and detail_level in levels:
            # Return specific detail level if requested and available
            level_data = levels[detail_level]
            explanation_content = level_data.get("content", "")
            used_level = detail_level
        elif detail_level and detail_level not in levels:
            # Specific level requested but doesn't exist - return empty to trigger generation
            explanation_content = None
            used_level = None
        elif levels:
            # If no specific level requested, return the latest generated level
            # Priority: detailed -> standard -> summary (most comprehensive first)
            for preferred_level in ["detailed", "standard", "summary"]:
                if preferred_level in levels:
                    level_data = levels[preferred_level]
                    explanation_content = level_data.get("content", "")
                    used_level = preferred_level
                    break
        else:
            # Fallback to narrative_text for backward compatibility
            explanation_content = analysis_result.narrative_text
            used_level = "legacy"

        has_explanation = bool(explanation_content)

        response_data = {
            "success": True,
            "analysis_id": analysis_id,
            "symbol": analysis_result.stock.symbol,
            "has_explanation": has_explanation,
            "detail_level_returned": used_level,
            "detail_level_requested": detail_level,
        }

        if has_explanation:
            if used_level != "legacy" and used_level in levels:
                level_data = levels[used_level]

                # Check if on-demand translation is needed
                stored_language = analysis_result.narrative_language or "en"
                final_content = explanation_content
                multilingual_data = None

                if language != stored_language and language in ["fr", "es"]:
                    # Generate multilingual explanation on-demand
                    try:
                        from Analytics.services.explanation_service import get_explanation_service
                        explanation_service = get_explanation_service()

                        if explanation_service.enabled:
                            # Create analysis data for multilingual generation
                            analysis_data = {
                                "symbol": analysis_result.stock.symbol,
                                "score_0_10": analysis_result.score_0_10,
                                "weighted_scores": {
                                    "w_sma50vs200": float(analysis_result.w_sma50vs200) if analysis_result.w_sma50vs200 else None,
                                    "w_rsi14": float(analysis_result.w_rsi14) if analysis_result.w_rsi14 else None,
                                    "w_macd12269": float(analysis_result.w_macd12269) if analysis_result.w_macd12269 else None,
                                },
                                "components": analysis_result.components or {}
                            }

                            # Generate multilingual explanation
                            multilingual_result = explanation_service.explain_prediction_single(
                                analysis_result,
                                detail_level=used_level,
                                language=language,
                                user=request.user
                            )

                            if multilingual_result and multilingual_result.get("content"):
                                final_content = multilingual_result["content"]
                                multilingual_data = {
                                    "translated": True,
                                    "target_language": language,
                                    "source_language": stored_language,
                                    "translation_confidence": multilingual_result.get("confidence_score", 0.0),
                                    "word_count": multilingual_result.get("word_count", len(final_content.split())),
                                    "generation_method": "on_demand_translation"
                                }
                                logger.info(f"[GET TRANSLATION] Successfully translated {used_level} explanation from {stored_language} to {language} for analysis {analysis_id}")
                            else:
                                logger.warning(f"[GET TRANSLATION] Failed to translate {used_level} explanation for analysis {analysis_id}, using original content")
                                multilingual_data = {
                                    "translated": False,
                                    "target_language": language,
                                    "source_language": stored_language,
                                    "error": "Translation failed",
                                    "generation_method": "fallback_original"
                                }
                    except Exception as e:
                        logger.error(f"[GET TRANSLATION] Error during on-demand translation for analysis {analysis_id}: {str(e)}")
                        multilingual_data = {
                            "translated": False,
                            "target_language": language,
                            "source_language": stored_language,
                            "error": str(e),
                            "generation_method": "fallback_original"
                        }

                response_data["explanation"] = {
                    "content": final_content,
                    "confidence_score": level_data.get("confidence", analysis_result.explanation_confidence),
                    "method": analysis_result.explanation_method,
                    "version": analysis_result.explanation_version,
                    "explained_at": level_data.get(
                        "generated_at",
                        analysis_result.explained_at.isoformat() if analysis_result.explained_at else None,
                    ),
                    "language": language if multilingual_data and multilingual_data.get("translated") else stored_language,
                    "word_count": multilingual_data.get("word_count") if multilingual_data else level_data.get("word_count", len(explanation_content.split())),
                    "detail_level": used_level,
                    "structured_data": explanations_json,
                    "available_levels": list(levels.keys()),
                    "multilingual": multilingual_data
                }
            else:
                # Legacy format for backward compatibility
                response_data["explanation"] = {
                    "content": explanation_content,
                    "confidence_score": analysis_result.explanation_confidence,
                    "method": analysis_result.explanation_method,
                    "version": analysis_result.explanation_version,
                    "explained_at": analysis_result.explained_at.isoformat() if analysis_result.explained_at else None,
                    "language": analysis_result.narrative_language,
                    "word_count": len(explanation_content.split()) if explanation_content else 0,
                    "detail_level": "legacy",
                    "structured_data": explanations_json,
                    "available_levels": list(levels.keys()),
                }

        return Response(response_data)

    except Exception as e:
        logger.error(f"Error retrieving explanation for analysis {analysis_id}: {str(e)}")
        return Response(
            {"error": f"Failed to retrieve explanation: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Get multilingual pipeline status",
    description="Get current status of the multilingual explanation pipeline and supported features",
    responses={
        200: {
            "description": "Pipeline status retrieved",
            "example": {
                "success": True,
                "pipeline_status": {
                    "enabled": True,
                    "supported_languages": ["en", "fr", "es"],
                    "default_language": "en",
                    "llm_service_available": True,
                    "cultural_formatter_enabled": True,
                    "quality_threshold": 0.8,
                    "validation_enabled": True,
                    "models_by_language": {
                        "en": "llama3.1:8b",
                        "fr": "qwen2:latest",
                        "es": "qwen2:latest"
                    },
                    "cultural_formatting": {
                        "en": {"currency_symbol": "$", "currency_position": "before"},
                        "fr": {"currency_symbol": "€", "currency_position": "after"},
                        "es": {"currency_symbol": "€", "currency_position": "after"}
                    }
                }
            }
        }
    },
)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def multilingual_pipeline_status(request):
    """
    Get multilingual pipeline status and configuration.

    Returns:
        Current status of multilingual pipeline and supported features
    """
    try:
        multilingual_pipeline = get_multilingual_pipeline()
        pipeline_status = multilingual_pipeline.get_pipeline_status()

        # Add additional configuration details
        from django.conf import settings

        pipeline_status.update({
            "models_by_language": getattr(settings, "LLM_MODELS_BY_LANGUAGE", {}),
            "cultural_formatting": getattr(settings, "FINANCIAL_FORMATTING", {}),
            "financial_terminology_mapping": getattr(settings, "FINANCIAL_TERMINOLOGY_MAPPING", {}),
        })

        return Response({"success": True, "pipeline_status": pipeline_status})

    except Exception as e:
        logger.error(f"Error getting multilingual pipeline status: {str(e)}")
        return Response(
            {"error": f"Failed to get pipeline status: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Generate bulk multilingual explanations",
    description="Generate explanations for multiple analysis results in multiple languages simultaneously",
    request={
        "application/json": {
            "type": "object",
            "properties": {
                "analysis_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of analysis IDs to generate explanations for",
                },
                "target_languages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of target languages (en, fr, es)",
                    "default": ["en", "fr", "es"]
                },
                "detail_level": {
                    "type": "string",
                    "enum": ["summary", "standard", "detailed"],
                    "description": "Explanation detail level",
                    "default": "standard",
                },
                "force_regenerate": {
                    "type": "boolean",
                    "description": "Force regeneration even if cached explanations exist",
                    "default": False
                }
            },
            "required": ["analysis_ids"],
        }
    },
    responses={
        200: {
            "description": "Bulk explanations generated successfully",
            "example": {
                "success": True,
                "results": {
                    "123": {
                        "symbol": "AAPL",
                        "explanations": {
                            "en": {"content": "...", "quality_score": 0.92},
                            "fr": {"content": "...", "quality_score": 0.89},
                            "es": {"content": "...", "quality_score": 0.87}
                        },
                        "processing_time": 4.2,
                        "success": True
                    }
                },
                "summary": {
                    "total_requests": 1,
                    "successful_requests": 1,
                    "total_explanations": 3,
                    "total_processing_time": 4.2,
                    "average_quality_score": 0.89
                }
            },
        },
        400: {"description": "Invalid request parameters"},
    },
)
@api_view(["POST"])
@permission_classes([IsAuthenticated])
@throttle_classes([AnalysisThrottle])
def generate_bulk_multilingual_explanations(request):
    """
    Generate explanations for multiple analysis results in multiple languages.

    Request Body:
        analysis_ids: List of analysis IDs
        target_languages: List of target languages (default: ["en", "fr", "es"])
        detail_level: Explanation detail level (default: "standard")
        force_regenerate: Force regeneration (default: False)

    Returns:
        Bulk explanation results with performance metrics
    """
    from Analytics.services.multilingual_optimizer import get_multilingual_optimizer, OptimizationRequest
    from Analytics.services.multilingual_metrics import get_multilingual_metrics, PerformanceMetric, UsageMetric
    from Data.models import AnalyticsResults

    try:
        # Parse request data
        analysis_ids = request.data.get("analysis_ids", [])
        target_languages = request.data.get("target_languages", ["en", "fr", "es"])
        detail_level = request.data.get("detail_level", "standard")
        force_regenerate = request.data.get("force_regenerate", False)

        # Validate request
        if not analysis_ids:
            return Response({"error": "analysis_ids is required"}, status=status.HTTP_400_BAD_REQUEST)

        if len(analysis_ids) > 20:
            return Response({"error": "Maximum 20 analyses allowed per bulk request"}, status=status.HTTP_400_BAD_REQUEST)

        if not all(lang in ["en", "fr", "es"] for lang in target_languages):
            return Response({"error": "Supported languages: en, fr, es"}, status=status.HTTP_400_BAD_REQUEST)

        if detail_level not in ["summary", "standard", "detailed"]:
            return Response({"error": "detail_level must be one of: summary, standard, detailed"}, status=status.HTTP_400_BAD_REQUEST)

        # Get optimizer and metrics services
        optimizer = get_multilingual_optimizer()
        metrics_service = get_multilingual_metrics()

        # Get analysis results for user
        analysis_results = AnalyticsResults.objects.filter(
            id__in=analysis_ids,
            user=request.user
        ).select_related("stock")

        if len(analysis_results) != len(analysis_ids):
            found_ids = [ar.id for ar in analysis_results]
            missing_ids = [aid for aid in analysis_ids if aid not in found_ids]
            return Response(
                {"error": f"Analyses not found or not accessible: {missing_ids}"},
                status=status.HTTP_404_NOT_FOUND
            )

        # Process bulk requests
        results = {}
        total_processing_time = 0
        successful_requests = 0
        total_explanations = 0
        quality_scores = []

        start_time = time.time()

        for analysis_result in analysis_results:
            analysis_start = time.time()

            try:
                # Prepare analysis data
                analysis_data = {
                    "symbol": analysis_result.stock.symbol,
                    "score_0_10": analysis_result.score_0_10 or 0,
                    "weighted_scores": analysis_result.weighted_scores or {},
                    "indicators": analysis_result.indicators or {},
                    "recommendation": analysis_result.recommendation or "HOLD",
                    "confidence": analysis_result.confidence_score or 0.0
                }

                # Create optimization request
                opt_request = OptimizationRequest(
                    analysis_id=analysis_result.id,
                    symbol=analysis_result.stock.symbol,
                    analysis_data=analysis_data,
                    target_languages=target_languages,
                    detail_level=detail_level,
                    user_id=request.user.id,
                    force_regenerate=force_regenerate
                )

                # Process request
                opt_result = optimizer.process_multilingual_request(opt_request)
                analysis_processing_time = time.time() - analysis_start
                total_processing_time += analysis_processing_time

                if opt_result.success:
                    successful_requests += 1
                    total_explanations += len(opt_result.explanations)

                    # Collect quality scores
                    for lang, explanation in opt_result.explanations.items():
                        quality_score = explanation.get('quality_score', 0.8)
                        quality_scores.append(quality_score)

                        # Record usage metric
                        usage_metric = UsageMetric(
                            user_id=request.user.id,
                            symbol=analysis_result.stock.symbol,
                            language=lang,
                            detail_level=detail_level,
                            feature_type="bulk_explanation",
                            timestamp=datetime.now(),
                            processing_time=analysis_processing_time / len(opt_result.explanations),
                            success=True,
                            cache_hit=opt_result.cache_stats.get('hits', 0) > 0
                        )
                        metrics_service.record_usage_metric(usage_metric)

                    results[str(analysis_result.id)] = {
                        "symbol": analysis_result.stock.symbol,
                        "explanations": opt_result.explanations,
                        "processing_time": analysis_processing_time,
                        "success": True,
                        "cache_stats": opt_result.cache_stats,
                        "performance_metrics": opt_result.performance_metrics
                    }

                    # Record performance metric
                    perf_metric = PerformanceMetric(
                        operation_id=f"bulk_{analysis_result.id}",
                        symbol=analysis_result.stock.symbol,
                        languages_requested=target_languages,
                        languages_completed=list(opt_result.explanations.keys()),
                        processing_time=analysis_processing_time,
                        memory_usage_mb=opt_result.performance_metrics.get('memory_usage', 0),
                        cache_hit_rate=opt_result.cache_stats.get('hits', 0) / max(1, opt_result.cache_stats.get('hits', 0) + opt_result.cache_stats.get('misses', 0)),
                        parallel_efficiency=opt_result.performance_metrics.get('parallel_efficiency', 0),
                        timestamp=datetime.now(),
                        success=True
                    )
                    metrics_service.record_performance_metric(perf_metric)

                else:
                    results[str(analysis_result.id)] = {
                        "symbol": analysis_result.stock.symbol,
                        "explanations": {},
                        "processing_time": analysis_processing_time,
                        "success": False,
                        "errors": opt_result.errors
                    }

            except Exception as e:
                logger.error(f"Error processing bulk explanation for analysis {analysis_result.id}: {str(e)}")
                results[str(analysis_result.id)] = {
                    "symbol": analysis_result.stock.symbol,
                    "explanations": {},
                    "processing_time": time.time() - analysis_start,
                    "success": False,
                    "errors": [str(e)]
                }

        # Calculate summary statistics
        avg_quality_score = statistics.mean(quality_scores) if quality_scores else 0.0

        return Response({
            "success": True,
            "results": results,
            "summary": {
                "total_requests": len(analysis_ids),
                "successful_requests": successful_requests,
                "total_explanations": total_explanations,
                "total_processing_time": total_processing_time,
                "average_processing_time": total_processing_time / len(analysis_ids),
                "average_quality_score": avg_quality_score,
                "languages_processed": target_languages,
                "detail_level": detail_level
            }
        })

    except Exception as e:
        logger.error(f"Error in bulk multilingual explanation generation: {str(e)}")
        return Response(
            {"error": f"Failed to generate bulk explanations: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Generate parallel multilingual explanation",
    description="Generate explanation for a single analysis in multiple languages with parallel processing",
    parameters=[
        OpenApiParameter(
            name="analysis_id",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.PATH,
            required=True,
            description="Analysis result ID",
        ),
    ],
    request={
        "application/json": {
            "type": "object",
            "properties": {
                "target_languages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of target languages",
                    "default": ["en", "fr", "es"]
                },
                "detail_level": {
                    "type": "string",
                    "enum": ["summary", "standard", "detailed"],
                    "description": "Explanation detail level",
                    "default": "standard",
                },
                "force_regenerate": {
                    "type": "boolean",
                    "description": "Force regeneration",
                    "default": False
                }
            },
        }
    },
    responses={
        200: {
            "description": "Parallel explanations generated successfully",
            "example": {
                "success": True,
                "analysis_id": 123,
                "symbol": "AAPL",
                "explanations": {
                    "en": {"content": "...", "quality_score": 0.92},
                    "fr": {"content": "...", "quality_score": 0.89},
                    "es": {"content": "...", "quality_score": 0.87}
                },
                "performance": {
                    "processing_time": 4.2,
                    "parallel_efficiency": 0.85,
                    "cache_hit_rate": 0.67,
                    "languages_completed": 3
                }
            },
        },
        404: {"description": "Analysis not found"},
        400: {"description": "Invalid parameters"},
    },
)
@api_view(["POST"])
@permission_classes([IsAuthenticated])
@throttle_classes([AnalysisThrottle])
def generate_parallel_multilingual_explanation(request, analysis_id):
    """
    Generate explanation for a single analysis in multiple languages using parallel processing.

    Path Parameters:
        analysis_id: Analysis result ID

    Request Body:
        target_languages: List of target languages (default: ["en", "fr", "es"])
        detail_level: Explanation detail level (default: "standard")
        force_regenerate: Force regeneration (default: False)

    Returns:
        Parallel explanation results with performance metrics
    """
    from Analytics.services.multilingual_optimizer import get_multilingual_optimizer, OptimizationRequest
    from Analytics.services.multilingual_metrics import get_multilingual_metrics, PerformanceMetric, QualityMetric
    from Data.models import AnalyticsResults

    try:
        # Parse request data
        target_languages = request.data.get("target_languages", ["en", "fr", "es"])
        detail_level = request.data.get("detail_level", "standard")
        force_regenerate = request.data.get("force_regenerate", False)

        # Validate request
        if not all(lang in ["en", "fr", "es"] for lang in target_languages):
            return Response({"error": "Supported languages: en, fr, es"}, status=status.HTTP_400_BAD_REQUEST)

        if detail_level not in ["summary", "standard", "detailed"]:
            return Response({"error": "detail_level must be one of: summary, standard, detailed"}, status=status.HTTP_400_BAD_REQUEST)

        # Get analysis result
        analysis_result = AnalyticsResults.objects.filter(
            id=analysis_id,
            user=request.user
        ).select_related("stock").first()

        if not analysis_result:
            return Response({"error": "Analysis not found"}, status=status.HTTP_404_NOT_FOUND)

        # Get services
        optimizer = get_multilingual_optimizer()
        metrics_service = get_multilingual_metrics()

        # Prepare analysis data
        analysis_data = {
            "symbol": analysis_result.stock.symbol,
            "score_0_10": analysis_result.score_0_10 or 0,
            "weighted_scores": analysis_result.weighted_scores or {},
            "indicators": analysis_result.indicators or {},
            "recommendation": analysis_result.recommendation or "HOLD",
            "confidence": analysis_result.confidence_score or 0.0
        }

        # Create optimization request
        opt_request = OptimizationRequest(
            analysis_id=analysis_result.id,
            symbol=analysis_result.stock.symbol,
            analysis_data=analysis_data,
            target_languages=target_languages,
            detail_level=detail_level,
            user_id=request.user.id,
            force_regenerate=force_regenerate
        )

        # Process request
        start_time = time.time()
        opt_result = optimizer.process_multilingual_request(opt_request)
        processing_time = time.time() - start_time

        if not opt_result.success:
            return Response(
                {"error": "Failed to generate parallel explanations", "details": opt_result.errors},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Record metrics
        for lang, explanation in opt_result.explanations.items():
            quality_score = explanation.get('quality_score', 0.8)

            # Record quality metric
            quality_metric = QualityMetric(
                content_id=f"{analysis_id}_{lang}_{detail_level}",
                language=lang,
                quality_score=quality_score,
                fluency_score=quality_score * 0.95,
                accuracy_score=quality_score * 1.02,
                completeness_score=quality_score * 0.98,
                cultural_appropriateness=quality_score * 0.97,
                technical_accuracy=quality_score * 1.01,
                timestamp=datetime.now(),
                model_used=explanation.get('model_used', 'unknown'),
                detail_level=detail_level
            )
            metrics_service.record_quality_metric(quality_metric)

        # Record performance metric
        perf_metric = PerformanceMetric(
            operation_id=f"parallel_{analysis_id}",
            symbol=analysis_result.stock.symbol,
            languages_requested=target_languages,
            languages_completed=list(opt_result.explanations.keys()),
            processing_time=processing_time,
            memory_usage_mb=opt_result.performance_metrics.get('memory_usage', 0),
            cache_hit_rate=opt_result.cache_stats.get('hits', 0) / max(1, opt_result.cache_stats.get('hits', 0) + opt_result.cache_stats.get('misses', 0)),
            parallel_efficiency=opt_result.performance_metrics.get('parallel_efficiency', 0),
            timestamp=datetime.now(),
            success=True
        )
        metrics_service.record_performance_metric(perf_metric)

        return Response({
            "success": True,
            "analysis_id": analysis_id,
            "symbol": analysis_result.stock.symbol,
            "explanations": opt_result.explanations,
            "performance": {
                "processing_time": processing_time,
                "parallel_efficiency": opt_result.performance_metrics.get('parallel_efficiency', 0),
                "cache_hit_rate": opt_result.cache_stats.get('hits', 0) / max(1, opt_result.cache_stats.get('hits', 0) + opt_result.cache_stats.get('misses', 0)),
                "languages_completed": len(opt_result.explanations),
                "cache_stats": opt_result.cache_stats,
                "memory_efficient": opt_result.performance_metrics.get('memory_efficient', False)
            },
            "quality": {
                "average_quality_score": statistics.mean([exp.get('quality_score', 0.8) for exp in opt_result.explanations.values()]),
                "quality_by_language": {lang: exp.get('quality_score', 0.8) for lang, exp in opt_result.explanations.items()}
            }
        })

    except Exception as e:
        logger.error(f"Error in parallel multilingual explanation generation: {str(e)}")
        return Response(
            {"error": f"Failed to generate parallel explanations: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Get multilingual metrics dashboard",
    description="Get comprehensive metrics and monitoring data for multilingual operations",
    parameters=[
        OpenApiParameter(
            name="time_range_hours",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            required=False,
            description="Time range in hours for metrics (default: 24)",
        ),
        OpenApiParameter(
            name="language",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            required=False,
            description="Filter by specific language",
        ),
    ],
    responses={
        200: {
            "description": "Metrics dashboard data",
            "example": {
                "success": True,
                "dashboard": {
                    "real_time": {
                        "system_health": "healthy",
                        "requests_per_minute": 15.2,
                        "avg_response_time": 2.1,
                        "cache_hit_rate": 0.73
                    },
                    "quality": {
                        "avg_quality": 0.89,
                        "assessments_count": 145
                    },
                    "performance": {
                        "parallel_efficiency": 0.85,
                        "total_requests": 58
                    }
                }
            },
        },
    },
)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_multilingual_metrics_dashboard(request):
    """
    Get comprehensive metrics dashboard for multilingual operations.

    Query Parameters:
        time_range_hours: Time range in hours (default: 24)
        language: Filter by specific language

    Returns:
        Comprehensive metrics dashboard data
    """
    from Analytics.services.multilingual_metrics import get_multilingual_metrics
    from Analytics.services.multilingual_optimizer import get_multilingual_optimizer

    try:
        time_range_hours = int(request.query_params.get("time_range_hours", 24))
        language_filter = request.query_params.get("language", None)

        metrics_service = get_multilingual_metrics()
        optimizer = get_multilingual_optimizer()

        # Get various reports
        quality_report = metrics_service.get_quality_report(language=language_filter, time_range_hours=time_range_hours)
        performance_report = metrics_service.get_performance_report(time_range_hours=time_range_hours)
        usage_analytics = metrics_service.get_usage_analytics(time_range_hours=time_range_hours)
        real_time_dashboard = metrics_service.get_real_time_dashboard()
        optimizer_metrics = optimizer.get_performance_metrics()

        dashboard_data = {
            "real_time": real_time_dashboard,
            "quality": quality_report,
            "performance": performance_report,
            "usage": usage_analytics,
            "optimizer": optimizer_metrics,
            "system_info": {
                "time_range_hours": time_range_hours,
                "language_filter": language_filter,
                "supported_languages": ["en", "fr", "es"],
                "timestamp": datetime.now().isoformat()
            }
        }

        return Response({
            "success": True,
            "dashboard": dashboard_data
        })

    except Exception as e:
        logger.error(f"Error getting multilingual metrics dashboard: {str(e)}")
        return Response(
            {"error": f"Failed to get metrics dashboard: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Warm multilingual cache",
    description="Pre-warm cache for common symbols and languages to improve response times",
    request={
        "application/json": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of symbols to warm cache for",
                    "default": ["AAPL", "MSFT", "GOOGL"]
                },
                "languages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of languages to warm cache for",
                    "default": ["en", "fr", "es"]
                },
                "detail_levels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of detail levels to warm cache for",
                    "default": ["summary", "standard"]
                }
            },
        }
    },
    responses={
        200: {
            "description": "Cache warming completed",
            "example": {
                "success": True,
                "warming_result": {
                    "warmed_explanations": 18,
                    "warming_time": 45.2,
                    "symbols_processed": 3,
                    "languages_processed": 3
                }
            },
        },
    },
)
@api_view(["POST"])
@permission_classes([IsAuthenticated])
@throttle_classes([AnalysisThrottle])
def warm_multilingual_cache(request):
    """
    Pre-warm cache for common symbols and languages.

    Request Body:
        symbols: List of symbols to warm (default: ["AAPL", "MSFT", "GOOGL"])
        languages: List of languages to warm (default: ["en", "fr", "es"])
        detail_levels: List of detail levels to warm (default: ["summary", "standard"])

    Returns:
        Cache warming results and statistics
    """
    from Analytics.services.multilingual_optimizer import get_multilingual_optimizer

    try:
        symbols = request.data.get("symbols", ["AAPL", "MSFT", "GOOGL"])
        languages = request.data.get("languages", ["en", "fr", "es"])
        detail_levels = request.data.get("detail_levels", ["summary", "standard"])

        # Validate input
        if len(symbols) > 10:
            return Response({"error": "Maximum 10 symbols allowed for cache warming"}, status=status.HTTP_400_BAD_REQUEST)

        if not all(lang in ["en", "fr", "es"] for lang in languages):
            return Response({"error": "Supported languages: en, fr, es"}, status=status.HTTP_400_BAD_REQUEST)

        if not all(level in ["summary", "standard", "detailed"] for level in detail_levels):
            return Response({"error": "Supported detail levels: summary, standard, detailed"}, status=status.HTTP_400_BAD_REQUEST)

        optimizer = get_multilingual_optimizer()

        # Start cache warming
        warming_result = optimizer.warm_cache(
            symbols=symbols,
            languages=languages,
            detail_levels=detail_levels
        )

        return Response({
            "success": True,
            "warming_result": warming_result,
            "request_info": {
                "symbols": symbols,
                "languages": languages,
                "detail_levels": detail_levels,
                "total_combinations": len(symbols) * len(languages) * len(detail_levels)
            }
        })

    except Exception as e:
        logger.error(f"Error warming multilingual cache: {str(e)}")
        return Response(
            {"error": f"Failed to warm cache: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

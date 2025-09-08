"""
Explanation API views for Analytics app.
Provides endpoints for generating and managing natural language explanations using local LLaMA model.
"""

from datetime import datetime
from django.core.cache import cache
from django.db import transaction
from django.db.utils import OperationalError
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes
import logging
import time
import random

from Analytics.services.explanation_service import get_explanation_service
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
            if any(keyword in error_msg for keyword in [
                'could not serialize access due to concurrent update',
                'deadlock detected',
                'database is locked',
                'concurrent update'
            ]):
                if attempt < max_retries:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                    logger.warning(f"[DB RETRY] Concurrency issue on attempt {attempt + 1}/{max_retries + 1}: {error_msg}")
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
            name='analysis_id',
            type=OpenApiTypes.INT,
            location=OpenApiParameter.PATH,
            required=True,
            description='Analysis result ID'
        ),
        OpenApiParameter(
            name='detail_level',
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            required=False,
            description='Explanation detail level: summary, standard, detailed (default: standard)'
        ),
    ],
    responses={
        200: {
            'description': 'Explanation generated successfully',
            'example': {
                'success': True,
                'explanation': {
                    'content': 'AAPL receives a 7.5/10 analysis score...',
                    'confidence_score': 0.85,
                    'detail_level': 'standard',
                    'method': 'llm'
                }
            }
        },
        404: {'description': 'Analysis not found'},
        403: {'description': 'Not authorized to view this analysis'},
        400: {'description': 'Invalid parameters'}
    }
)
@api_view(['POST'])
@permission_classes([IsAuthenticated])
@throttle_classes([AnalysisThrottle])
def generate_explanation(request, analysis_id):
    """
    Generate explanation for a specific analysis result.

    Path Parameters:
        analysis_id: Analysis result ID

    Query Parameters:
        detail_level: Explanation detail level (summary/standard/detailed, default: standard)

    Returns:
        Generated explanation with metadata
    """
    detail_level = request.query_params.get('detail_level', 'standard')

    # Validate detail level
    if detail_level not in ['summary', 'standard', 'detailed']:
        return Response(
            {'error': 'detail_level must be one of: summary, standard, detailed'},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        # Get the analysis result and ensure it belongs to the user
        analysis_result = AnalyticsResults.objects.filter(
            id=analysis_id,
            user=request.user
        ).select_related('stock').first()

        if not analysis_result:
            return Response(
                {'error': 'Analysis not found'},
                status=status.HTTP_404_NOT_FOUND
            )

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
            logger.error(f"[EXPLAIN] Service not enabled - enabled: {explanation_service.enabled}, llm_available: {llm_available}")
            return Response(
                {'error': 'Explanation service not available'},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        # Check if explanation already exists for this detail level
        existing_explanations = analysis_result.explanations_json or {}
        existing_levels = existing_explanations.get('levels', {})

        if detail_level in existing_levels:
            # Return existing explanation for this detail level
            logger.info(f"[EXPLAIN] Returning existing {detail_level} explanation for analysis {analysis_id}")
            existing_explanation = existing_levels[detail_level]
            explanation_result = {
                'content': existing_explanation.get('content', ''),
                'confidence_score': existing_explanation.get('confidence', 0.0),
                'method': analysis_result.explanation_method or 'llm',
                'word_count': existing_explanation.get('word_count', len(existing_explanation.get('content', '').split())),
                'indicators_explained': existing_explanations.get('indicators_explained', []),
                'risk_factors': existing_explanations.get('risk_factors', []),
                'recommendation': existing_explanations.get('recommendation', 'HOLD'),
                'generation_time': 0.0  # Retrieved from cache/DB
            }
        else:
            # Generate new explanation for this detail level
            logger.info(f"[EXPLAIN] Starting {detail_level} explanation generation for {analysis_result.stock.symbol}")
            try:
                explanation_result = explanation_service.explain_prediction_single(
                    analysis_result,
                    detail_level=detail_level,
                    user=request.user
                )

                if explanation_result:
                    logger.info(f"[EXPLAIN] Explanation generated successfully - method: {explanation_result.get('method')}, length: {len(explanation_result.get('content', ''))}")
                else:
                    logger.error(f"[EXPLAIN] Explanation generation returned None for analysis {analysis_id}")

            except Exception as exp_error:
                logger.error(f"[EXPLAIN] Exception during explanation generation: {str(exp_error)}", exc_info=True)
                explanation_result = None

        if not explanation_result:
            logger.error(f"[EXPLAIN] Failed to generate explanation for analysis {analysis_id}")
            return Response(
                {'error': 'Failed to generate explanation'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Update the database with explanation using retry mechanism for concurrency
        def save_explanation():
            # Refresh the analysis_result to get latest state
            analysis_result.refresh_from_db()

            # Check if explanation for this detail level needs to be saved
            existing_explanations = analysis_result.explanations_json or {}
            levels = existing_explanations.get('levels', {})

            # Generate new explanation if this detail level doesn't exist or method changed
            needs_new_explanation = (
                detail_level not in levels or 
                analysis_result.explanation_method != explanation_result.get('method')
            )

            if needs_new_explanation:
                with transaction.atomic():
                    # Use select_for_update to prevent concurrent modifications
                    locked_result = AnalyticsResults.objects.select_for_update().get(id=analysis_id, user=request.user)

                    # Refresh existing explanations and double-check after acquiring lock
                    current_explanations = locked_result.explanations_json or {}
                    current_levels = current_explanations.get('levels', {})

                    if detail_level not in current_levels or locked_result.explanation_method != explanation_result.get('method'):
                        # Update the levels with new explanation
                        current_levels[detail_level] = {
                            'content': explanation_result.get('content', ''),
                            'confidence': explanation_result.get('confidence_score', 0.0),
                            'generated_at': datetime.now().isoformat(),
                            'word_count': explanation_result.get('word_count', 0)
                        }

                        # Update the full explanations_json structure
                        locked_result.explanations_json = {
                            'levels': current_levels,
                            'indicators_explained': explanation_result.get('indicators_explained', []),
                            'risk_factors': explanation_result.get('risk_factors', []),
                            'recommendation': explanation_result.get('recommendation', 'HOLD'),
                            'current_level': detail_level
                        }

                        # Update other fields (keep narrative_text for backward compatibility with latest)
                        locked_result.explanation_method = explanation_result.get('method', 'unknown')
                        locked_result.explanation_version = '1.0'
                        locked_result.narrative_text = explanation_result.get('content', '')
                        locked_result.explanation_confidence = explanation_result.get('confidence_score', 0.0)
                        locked_result.explained_at = datetime.now()
                        locked_result.save()
                        logger.info(f"[DB SAVE] Explanation saved for analysis {analysis_id} ({detail_level} level)")
                    else:
                        logger.info(f"[DB SAVE] Explanation for {detail_level} level already exists for analysis {analysis_id}")
            else:
                logger.info(f"[DB SAVE] Explanation for {detail_level} level already exists for analysis {analysis_id}")

        # Execute the save operation with retry logic
        try:
            retry_database_operation(save_explanation, max_retries=3, base_delay=0.1)
        except Exception as e:
            logger.error(f"Failed to save explanation after retries: {str(e)}")
            # Continue execution - explanation generation was successful, only save failed

        response_data = {
            'success': True,
            'analysis_id': analysis_id,
            'symbol': analysis_result.stock.symbol,
            'explanation': {
                'content': explanation_result.get('content', ''),
                'confidence_score': explanation_result.get('confidence_score', 0.0),
                'detail_level': detail_level,
                'method': explanation_result.get('method', 'unknown'),
                'generation_time': explanation_result.get('generation_time', 0.0),
                'word_count': explanation_result.get('word_count', 0),
                'indicators_explained': explanation_result.get('indicators_explained', []),
                'risk_factors': explanation_result.get('risk_factors', []),
                'recommendation': explanation_result.get('recommendation', 'HOLD')
            }
        }

        logger.info(f"Explanation generated successfully for analysis {analysis_id}")
        return Response(response_data)

    except Exception as e:
        logger.error(f"Error in generate_explanation: {str(e)}", exc_info=True)
        logger.error(f"Explanation generation failed for analysis {analysis_id}: {str(e)}")
        return Response(
            {'error': f'Explanation generation failed: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Get explanation service status",
    description="Get current status of the explanation service and LLM availability",
    responses={
        200: {
            'description': 'Service status retrieved',
            'example': {
                'enabled': True,
                'llm_available': True,
                'model_name': 'llama3.1:70b',
                'cache_ttl': 300
            }
        }
    }
)
@api_view(['GET'])
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

        return Response({
            'success': True,
            'status': status_data
        })

    except Exception as e:
        logger.error(f"Error getting explanation service status: {str(e)}")
        return Response(
            {'error': f'Failed to get service status: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Get explanation for existing analysis",
    description="Retrieve previously generated explanation for an analysis result",
    parameters=[
        OpenApiParameter(
            name='analysis_id',
            type=OpenApiTypes.INT,
            location=OpenApiParameter.PATH,
            required=True,
            description='Analysis result ID'
        ),
        OpenApiParameter(
            name='detail_level',
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            required=False,
            description='Explanation detail level: summary, standard, detailed (default: latest)'
        ),
    ],
    responses={
        200: {
            'description': 'Explanation retrieved successfully',
            'example': {
                'success': True,
                'has_explanation': True,
                'explanation': {
                    'content': 'Previously generated explanation...',
                    'confidence_score': 0.85,
                    'method': 'llm',
                    'explained_at': '2025-01-17T10:30:00Z'
                }
            }
        },
        404: {'description': 'Analysis not found'},
        403: {'description': 'Not authorized to view this analysis'}
    }
)
@api_view(['GET'])
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
    detail_level = request.query_params.get('detail_level', None)

    try:
        # Get the analysis result and ensure it belongs to the user
        analysis_result = AnalyticsResults.objects.filter(
            id=analysis_id,
            user=request.user
        ).select_related('stock').first()

        if not analysis_result:
            return Response(
                {'error': 'Analysis not found'},
                status=status.HTTP_404_NOT_FOUND
            )

        # Get explanations data
        explanations_json = analysis_result.explanations_json or {}
        levels = explanations_json.get('levels', {})

        # Determine which explanation content to return
        explanation_content = None
        used_level = None

        if detail_level and detail_level in levels:
            # Return specific detail level if requested and available
            level_data = levels[detail_level]
            explanation_content = level_data.get('content', '')
            used_level = detail_level
        elif detail_level and detail_level not in levels:
            # Specific level requested but doesn't exist - return empty to trigger generation
            explanation_content = None
            used_level = None
        elif levels:
            # If no specific level requested, return the latest generated level
            # Priority: detailed -> standard -> summary (most comprehensive first)
            for preferred_level in ['detailed', 'standard', 'summary']:
                if preferred_level in levels:
                    level_data = levels[preferred_level]
                    explanation_content = level_data.get('content', '')
                    used_level = preferred_level
                    break
        else:
            # Fallback to narrative_text for backward compatibility
            explanation_content = analysis_result.narrative_text
            used_level = 'legacy'

        has_explanation = bool(explanation_content)

        response_data = {
            'success': True,
            'analysis_id': analysis_id,
            'symbol': analysis_result.stock.symbol,
            'has_explanation': has_explanation,
            'detail_level_returned': used_level,
            'detail_level_requested': detail_level
        }

        if has_explanation:
            if used_level != 'legacy' and used_level in levels:
                level_data = levels[used_level]
                response_data['explanation'] = {
                    'content': explanation_content,
                    'confidence_score': level_data.get('confidence', analysis_result.explanation_confidence),
                    'method': analysis_result.explanation_method,
                    'version': analysis_result.explanation_version,
                    'explained_at': level_data.get('generated_at', 
                                     analysis_result.explained_at.isoformat() if analysis_result.explained_at else None),
                    'language': analysis_result.narrative_language,
                    'word_count': level_data.get('word_count', len(explanation_content.split())),
                    'detail_level': used_level,
                    'structured_data': explanations_json,
                    'available_levels': list(levels.keys())
                }
            else:
                # Legacy format for backward compatibility
                response_data['explanation'] = {
                    'content': explanation_content,
                    'confidence_score': analysis_result.explanation_confidence,
                    'method': analysis_result.explanation_method,
                    'version': analysis_result.explanation_version,
                    'explained_at': analysis_result.explained_at.isoformat() if analysis_result.explained_at else None,
                    'language': analysis_result.narrative_language,
                    'word_count': len(explanation_content.split()) if explanation_content else 0,
                    'detail_level': 'legacy',
                    'structured_data': explanations_json,
                    'available_levels': list(levels.keys())
                }

        return Response(response_data)

    except Exception as e:
        logger.error(f"Error retrieving explanation for analysis {analysis_id}: {str(e)}")
        return Response(
            {'error': f'Failed to retrieve explanation: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

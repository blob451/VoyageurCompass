"""
Async analysis status checking endpoints.
"""

from django.core.cache import cache
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response


@extend_schema(
    summary="Check analysis status",
    description="Check the status of async stock analysis (backfill and analysis progress)",
    parameters=[
        OpenApiParameter(
            name="symbol",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.PATH,
            required=True,
            description="Stock ticker symbol to check status for",
        ),
    ],
    responses={
        200: {
            "description": "Status retrieved successfully",
            "example": {
                "status": "completed", 
                "symbol": "AAPL", 
                "progress": "Analysis complete",
                "analysis_result": {}
            }
        },
        404: {"description": "No status found for symbol"},
    },
)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def check_analysis_status(request, symbol):
    """
    Check the current status of async analysis for a stock symbol.
    
    Returns the status of backfill and/or analysis tasks, including progress
    and results when complete.
    """
    import logging
    
    logger = logging.getLogger(__name__)
    symbol = symbol.upper()
    user_id = request.user.id
    
    try:
        # Check both backfill and analysis status
        backfill_status_key = f"backfill_status_{symbol}_{user_id}"
        analysis_status_key = f"analysis_status_{symbol}_{user_id}"
        
        backfill_status = cache.get(backfill_status_key)
        analysis_status = cache.get(analysis_status_key)
        
        # Determine overall status
        if analysis_status and analysis_status.get("status") == "completed":
            # Analysis is complete - return results
            return Response({
                "status": "completed",
                "symbol": symbol,
                "stage": "analysis_complete",
                "progress": "Analysis completed successfully",
                "analysis_result": analysis_status.get("analysis_result"),
                "completed_at": analysis_status.get("completed_at"),
                "task_id": analysis_status.get("task_id")
            })
        
        elif analysis_status and analysis_status.get("status") == "running":
            # Analysis is in progress
            return Response({
                "status": "running",
                "symbol": symbol,
                "stage": "analysis_running",
                "progress": "Technical analysis in progress...",
                "started_at": analysis_status.get("started_at"),
                "task_id": analysis_status.get("task_id"),
                "estimated_remaining": "30-60 seconds"
            })
        
        elif backfill_status and backfill_status.get("status") == "analysis_triggered":
            # Backfill complete, analysis triggered
            return Response({
                "status": "running", 
                "symbol": symbol,
                "stage": "analysis_starting",
                "progress": "Data backfill complete, starting technical analysis...",
                "backfill_result": backfill_status.get("backfill_result"),
                "analysis_task_id": backfill_status.get("analysis_task_id"),
                "estimated_remaining": "1-2 minutes"
            })
        
        elif backfill_status and backfill_status.get("status") == "backfill_complete":
            # Backfill complete, no analysis triggered
            return Response({
                "status": "completed",
                "symbol": symbol,
                "stage": "backfill_complete",
                "progress": "Data backfill completed (analysis not requested)",
                "backfill_result": backfill_status.get("backfill_result"),
                "completed_at": backfill_status.get("completed_at")
            })
        
        elif backfill_status and backfill_status.get("status") == "running":
            # Backfill is in progress
            return Response({
                "status": "running",
                "symbol": symbol,
                "stage": "backfill_running", 
                "progress": "Downloading historical data...",
                "started_at": backfill_status.get("started_at"),
                "task_id": backfill_status.get("task_id"),
                "estimated_remaining": "2-3 minutes"
            })
        
        elif (backfill_status and backfill_status.get("status") == "failed") or \
             (analysis_status and analysis_status.get("status") == "failed"):
            # Either backfill or analysis failed
            failed_status = analysis_status if analysis_status and analysis_status.get("status") == "failed" else backfill_status
            return Response({
                "status": "failed",
                "symbol": symbol,
                "stage": "failed",
                "error": failed_status.get("error"),
                "failed_at": failed_status.get("failed_at"),
                "task_id": failed_status.get("task_id")
            })
        
        else:
            # No status found
            return Response({
                "status": "not_found",
                "symbol": symbol,
                "message": f"No async analysis status found for {symbol}. Use ?async=true to start async analysis."
            }, status=status.HTTP_404_NOT_FOUND)
            
    except Exception as e:
        logger.error(f"Error checking analysis status for {symbol}: {str(e)}")
        return Response({
            "error": f"Failed to check status: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
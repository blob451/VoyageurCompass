"""
Async Processing Views for Analytics app.
Provides endpoints for concurrent batch processing of analysis and explanation requests.
"""

from datetime import datetime

from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from Analytics.engine.ta_engine import TechnicalAnalysisEngine
from Analytics.services.async_processing_pipeline import get_async_processing_pipeline
from Analytics.services.hybrid_analysis_coordinator import (
    get_hybrid_analysis_coordinator,
)
from Analytics.views import AnalysisThrottle


@extend_schema(
    summary="Process batch analysis",
    description="Process multiple stock analyses concurrently using async processing pipeline",
    request={
        "application/json": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock symbols to analyze",
                },
                "detail_level": {
                    "type": "string",
                    "enum": ["summary", "standard", "detailed"],
                    "description": "Level of analysis detail",
                    "default": "standard",
                },
                "months": {"type": "integer", "description": "Number of months for analysis", "default": 6},
            },
            "required": ["symbols"],
        }
    },
)
@api_view(["POST"])
@permission_classes([IsAuthenticated])
@throttle_classes([AnalysisThrottle])
def batch_analyze(request):
    """
    Process multiple stock analyses concurrently.
    """
    try:
        symbols = request.data.get("symbols", [])
        detail_level = request.data.get("detail_level", "standard")
        months = request.data.get("months", 6)

        if not symbols:
            return Response({"error": "No symbols provided"}, status=status.HTTP_400_BAD_REQUEST)

        if len(symbols) > 20:
            return Response({"error": "Maximum 20 symbols allowed per batch"}, status=status.HTTP_400_BAD_REQUEST)

        # Prepare analysis requests
        analysis_requests = []
        for symbol in symbols:
            analysis_requests.append({"symbol": symbol.upper(), "months": months, "user": request.user})

        # Get async processing pipeline
        async_pipeline = get_async_processing_pipeline()

        # Define processing function
        def process_single_analysis(req):
            symbol = req["symbol"]
            months = req["months"]
            user = req["user"]

            # Run technical analysis
            ta_engine = TechnicalAnalysisEngine()
            analysis_result = ta_engine.analyze_stock(symbol, months=months, user=user)

            if not analysis_result:
                return None

            # Get enhanced explanation if requested
            if detail_level != "summary":
                hybrid_coordinator = get_hybrid_analysis_coordinator()
                explanation_result = hybrid_coordinator.generate_enhanced_explanation(
                    analysis_data=analysis_result, detail_level=detail_level
                )

                if explanation_result:
                    analysis_result["explanation"] = explanation_result

            return analysis_result

        # Process batch
        batch_result = async_pipeline.process_batch_analysis(
            analysis_requests,
            process_single_analysis,
            f"batch_analyze_{request.user.id}_{int(datetime.now().timestamp())}",
        )

        return Response(
            {
                "batch_id": batch_result["batch_id"],
                "results": batch_result["results"],
                "processing_time": batch_result["processing_time"],
                "total_requests": batch_result["total_requests"],
                "successful_requests": batch_result["successful_requests"],
                "failed_requests": batch_result["failed_requests"],
                "success_rate": batch_result["success_rate"],
                "completed_at": batch_result["completed_at"],
            }
        )

    except Exception as e:
        return Response({"error": f"Batch analysis failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@extend_schema(
    summary="Process batch explanations",
    description="Generate enhanced explanations for multiple analyses concurrently",
    request={
        "application/json": {
            "type": "object",
            "properties": {
                "analysis_data_list": {
                    "type": "array",
                    "items": {"type": "object", "description": "Analysis data object"},
                    "description": "List of analysis data objects",
                },
                "detail_level": {
                    "type": "string",
                    "enum": ["summary", "standard", "detailed"],
                    "description": "Level of explanation detail",
                    "default": "standard",
                },
            },
            "required": ["analysis_data_list"],
        }
    },
)
@api_view(["POST"])
@permission_classes([IsAuthenticated])
@throttle_classes([AnalysisThrottle])
def batch_explain(request):
    """
    Generate enhanced explanations for multiple analyses concurrently.
    """
    try:
        analysis_data_list = request.data.get("analysis_data_list", [])
        detail_level = request.data.get("detail_level", "standard")

        if not analysis_data_list:
            return Response({"error": "No analysis data provided"}, status=status.HTTP_400_BAD_REQUEST)

        if len(analysis_data_list) > 15:
            return Response({"error": "Maximum 15 analyses allowed per batch"}, status=status.HTTP_400_BAD_REQUEST)

        # Get async processing pipeline
        async_pipeline = get_async_processing_pipeline()

        # Process batch explanations
        batch_result = async_pipeline.process_sentiment_explanation_batch(analysis_data_list, detail_level)

        return Response(
            {
                "batch_id": batch_result["batch_id"],
                "explanations": batch_result["results"],
                "processing_time": batch_result["processing_time"],
                "total_requests": batch_result["total_requests"],
                "successful_requests": batch_result["successful_requests"],
                "failed_requests": batch_result["failed_requests"],
                "success_rate": batch_result["success_rate"],
                "completed_at": batch_result["completed_at"],
            }
        )

    except Exception as e:
        return Response({"error": f"Batch explanation failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@extend_schema(
    summary="Get task status",
    description="Get the status of an async task",
    parameters=[
        OpenApiParameter(
            name="task_id", type=OpenApiTypes.STR, location=OpenApiParameter.PATH, required=True, description="Task ID"
        )
    ],
)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_task_status(request, task_id):
    """
    Get the status of a specific async task.
    """
    try:
        async_pipeline = get_async_processing_pipeline()
        task_status_data = async_pipeline.get_task_status(task_id)

        if not task_status_data:
            return Response({"error": "Task not found"}, status=status.HTTP_404_NOT_FOUND)

        return Response(task_status_data)

    except Exception as e:
        return Response({"error": f"Failed to get task status: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@extend_schema(
    summary="Get batch status",
    description="Get the status of an async batch",
    parameters=[
        OpenApiParameter(
            name="batch_id",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.PATH,
            required=True,
            description="Batch ID",
        )
    ],
)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_batch_status(request, batch_id):
    """
    Get the status of a specific async batch.
    """
    try:
        async_pipeline = get_async_processing_pipeline()
        batch_status_data = async_pipeline.get_batch_status(batch_id)

        if "error" in batch_status_data:
            return Response(batch_status_data, status=status.HTTP_404_NOT_FOUND)

        return Response(batch_status_data)

    except Exception as e:
        return Response(
            {"error": f"Failed to get batch status: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Get async pipeline performance", description="Get performance metrics for the async processing pipeline"
)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def async_performance(request):
    """
    Get performance summary for the async processing pipeline.
    """
    try:
        async_pipeline = get_async_processing_pipeline()
        performance_summary = async_pipeline.get_performance_summary()

        return Response(performance_summary)

    except Exception as e:
        return Response(
            {"error": f"Failed to get performance metrics: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

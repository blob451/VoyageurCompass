"""
Data Quality API Views

Provides REST endpoints for accessing data quality metrics and monitoring information.
"""

from django.core.cache import cache
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from Data.services.data_quality_monitor import data_quality_monitor


@extend_schema(
    summary="Get data quality dashboard",
    description="Retrieve comprehensive data quality metrics and monitoring information",
    responses={
        200: {
            "description": "Data quality metrics retrieved successfully",
            "example": {
                "overall_quality_score": 7.5,
                "timestamp": "2025-09-12T01:00:00Z",
                "stock_data_quality": {"quality_score": 8.2},
                "recommendations": []
            }
        },
    },
)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def data_quality_dashboard(request):
    """
    Get comprehensive data quality dashboard information.
    
    Returns cached quality metrics or triggers a new quality check
    if no recent data is available.
    """
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Get cached dashboard data
        dashboard_data = data_quality_monitor.get_quality_dashboard()
        
        return Response({
            "success": True,
            "data_quality": dashboard_data
        })
        
    except Exception as e:
        logger.error(f"Error retrieving data quality dashboard: {str(e)}")
        return Response({
            "error": f"Failed to retrieve data quality metrics: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@extend_schema(
    summary="Trigger data quality check",
    description="Run comprehensive data quality analysis and return results",
    responses={
        200: {
            "description": "Data quality check completed successfully"
        },
        202: {
            "description": "Data quality check initiated (async)"
        },
    },
)
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def run_data_quality_check(request):
    """
    Trigger a comprehensive data quality check.
    
    This may take several minutes to complete for large datasets.
    """
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Data quality check requested by user: {request.user.username}")
        
        # Run comprehensive quality check
        result = data_quality_monitor.run_comprehensive_check()
        
        if 'error' in result:
            return Response({
                "success": False,
                "error": result['error']
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Log key results
        overall_score = result.get('overall_quality_score', 0)
        issues_count = len(result.get('recommendations', []))
        logger.info(f"Data quality check completed: Score {overall_score:.1f}/10, {issues_count} issues")
        
        return Response({
            "success": True,
            "message": "Data quality check completed successfully",
            "overall_score": overall_score,
            "issues_found": issues_count,
            "timestamp": result.get('timestamp'),
            "full_results": result
        })
        
    except Exception as e:
        logger.error(f"Error running data quality check: {str(e)}")
        return Response({
            "error": f"Failed to run data quality check: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@extend_schema(
    summary="Get data quality trends",
    description="Retrieve historical data quality trends for dashboard charts",
    parameters=[
        OpenApiParameter(
            name="days",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            required=False,
            description="Number of days of history to retrieve (default: 7)",
        ),
    ],
    responses={
        200: {
            "description": "Data quality trends retrieved successfully"
        },
    },
)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def data_quality_trends(request):
    """
    Get historical data quality trends for charting and analysis.
    """
    import logging
    from datetime import datetime, timedelta
    
    logger = logging.getLogger(__name__)
    days = int(request.query_params.get('days', 7))
    
    try:
        trends = []
        
        # Collect historical data from cache
        for i in range(days):
            check_date = (datetime.now() - timedelta(days=i)).date()
            cache_key = f"data_quality_history_{check_date.isoformat()}"
            
            historical_data = cache.get(cache_key)
            if historical_data:
                trends.append(historical_data)
        
        # Sort by date
        trends.sort(key=lambda x: x.get('date', ''))
        
        return Response({
            "success": True,
            "days_requested": days,
            "data_points": len(trends),
            "trends": trends,
            "message": f"Retrieved {len(trends)} data points over {days} days"
        })
        
    except Exception as e:
        logger.error(f"Error retrieving data quality trends: {str(e)}")
        return Response({
            "error": f"Failed to retrieve trends: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@extend_schema(
    summary="Get data quality alerts",
    description="Retrieve current data quality alerts and warnings",
    responses={
        200: {
            "description": "Data quality alerts retrieved successfully"
        },
    },
)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def data_quality_alerts(request):
    """
    Get current data quality alerts for immediate attention.
    """
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Get cached quality data
        dashboard_data = data_quality_monitor.get_quality_dashboard()
        
        # Extract alerts from recommendations
        recommendations = dashboard_data.get('recommendations', [])
        
        # Categorize by severity
        critical_alerts = [r for r in recommendations if r.get('severity') == 'high']
        warnings = [r for r in recommendations if r.get('severity') == 'medium'] 
        info_alerts = [r for r in recommendations if r.get('severity') == 'low']
        
        # Check for system errors
        error_status = cache.get("data_quality_monitor_error")
        system_errors = [error_status] if error_status else []
        
        return Response({
            "success": True,
            "alert_summary": {
                "critical_count": len(critical_alerts),
                "warning_count": len(warnings),
                "info_count": len(info_alerts),
                "system_error_count": len(system_errors)
            },
            "critical_alerts": critical_alerts,
            "warnings": warnings,
            "info_alerts": info_alerts,
            "system_errors": system_errors,
            "last_check": dashboard_data.get('timestamp'),
            "overall_status": dashboard_data.get('summary', {}).get('overall_status', 'unknown')
        })
        
    except Exception as e:
        logger.error(f"Error retrieving data quality alerts: {str(e)}")
        return Response({
            "error": f"Failed to retrieve alerts: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
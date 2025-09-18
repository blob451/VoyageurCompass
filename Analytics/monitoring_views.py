"""
Monitoring Views for Prometheus Metrics Endpoint and Production Monitoring

Provides HTTP endpoints for metrics collection, system health monitoring,
and production monitoring with alerting capabilities.
"""

import logging
from django.http import HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from drf_spectacular.utils import extend_schema
import json

from Analytics.monitoring import get_metrics_collector
from Analytics.services.production_monitoring import get_monitoring_service

logger = logging.getLogger(__name__)


@require_http_methods(["GET"])
def metrics_endpoint(request):
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus text format for scraping.
    """
    collector = get_metrics_collector()
    metrics_data = collector.export_metrics()
    
    return HttpResponse(
        metrics_data,
        content_type='text/plain; version=0.0.4; charset=utf-8'
    )


@require_http_methods(["GET"])
def health_metrics(request):
    """
    Health check endpoint with basic metrics summary.
    
    Returns JSON with cache hit rates and system status.
    """
    collector = get_metrics_collector()
    
    health_data = {
        'status': 'healthy',
        'metrics_enabled': collector.enabled,
        'cache_performance': {},
        'system_info': {
            'monitoring_active': True,
            'prometheus_available': collector.enabled
        }
    }
    
    # Add cache hit rates if metrics are enabled
    if collector.enabled:
        services = ['sentiment_analyzer', 'local_llm_service', 'explanation_service']
        for service in services:
            try:
                hit_rate = collector.get_cache_hit_rate(service)
                health_data['cache_performance'][service] = {
                    'hit_rate': round(hit_rate, 3),
                    'status': 'optimal' if hit_rate > 0.8 else 'degraded' if hit_rate > 0.5 else 'poor'
                }
            except Exception:
                health_data['cache_performance'][service] = {
                    'hit_rate': 0.0,
                    'status': 'unknown'
                }
    
    return HttpResponse(
        json.dumps(health_data, indent=2),
        content_type='application/json'
    )


@extend_schema(
    summary="Production monitoring status",
    description="Get current production monitoring service status and configuration",
    responses={200: {"description": "Monitoring status information"}}
)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def production_monitoring_status(request):
    """Get production monitoring service status."""
    try:
        monitoring_service = get_monitoring_service()
        status_info = monitoring_service.get_monitoring_status()

        return Response(status_info)

    except Exception as e:
        logger.error(f"Error getting monitoring status: {str(e)}")
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Recent production alerts",
    description="Get recent production alerts and incidents",
    responses={200: {"description": "List of recent alerts"}}
)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def production_alerts(request):
    """Get recent production alerts."""
    try:
        monitoring_service = get_monitoring_service()
        limit = int(request.GET.get('limit', 50))

        alerts = monitoring_service.get_recent_alerts(limit)

        return Response({
            "alerts": alerts,
            "count": len(alerts),
            "limit": limit
        })

    except Exception as e:
        logger.error(f"Error getting production alerts: {str(e)}")
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Force health check",
    description="Force an immediate comprehensive health check",
    responses={200: {"description": "Health check results"}}
)
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def force_health_check(request):
    """Force an immediate health check."""
    try:
        monitoring_service = get_monitoring_service()
        results = monitoring_service.force_health_check()

        return Response(results)

    except Exception as e:
        logger.error(f"Error forcing health check: {str(e)}")
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Start production monitoring",
    description="Start the production monitoring background service",
    responses={200: {"description": "Monitoring service started"}}
)
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def start_monitoring(request):
    """Start production monitoring service."""
    try:
        monitoring_service = get_monitoring_service()
        monitoring_service.start_monitoring()

        return Response({
            "status": "started",
            "message": "Production monitoring service started"
        })

    except Exception as e:
        logger.error(f"Error starting monitoring: {str(e)}")
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Stop production monitoring",
    description="Stop the production monitoring background service",
    responses={200: {"description": "Monitoring service stopped"}}
)
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def stop_monitoring(request):
    """Stop production monitoring service."""
    try:
        monitoring_service = get_monitoring_service()
        monitoring_service.stop_monitoring()

        return Response({
            "status": "stopped",
            "message": "Production monitoring service stopped"
        })

    except Exception as e:
        logger.error(f"Error stopping monitoring: {str(e)}")
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
"""
Monitoring Views for Analytics app.
Provides endpoints for system health, performance metrics, and operational analytics.
"""

from datetime import datetime, timedelta
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes

# Conditional import ensures monitoring service CI compatibility
try:
    from Analytics.services.advanced_monitoring_service import get_monitoring_service
    MONITORING_SERVICE_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Monitoring service unavailable: {e}")
    get_monitoring_service = None
    MONITORING_SERVICE_AVAILABLE = False

import logging
logger = logging.getLogger(__name__)


def monitoring_service_required(func):
    """Decorator to handle monitoring service availability."""
    def wrapper(request, *args, **kwargs):
        if not MONITORING_SERVICE_AVAILABLE:
            return Response({
                'status': 'unavailable',
                'message': 'Monitoring service unavailable in CI environment',
                'error': 'psutil dependency missing'
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        return func(request, *args, **kwargs)
    return wrapper


@extend_schema(
    summary="Get system health status",
    description="Get overall system health and component status"
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
@monitoring_service_required
def system_health(request):
    """
    Get current system health status.
    """
    try:
        monitoring_service = get_monitoring_service()
        health_data = monitoring_service.get_system_health()

        return Response(health_data)

    except Exception as e:
        logger.error(f"Error getting system health: {str(e)}")
        return Response(
            {'error': f'Failed to get system health: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Get performance dashboard",
    description="Get comprehensive performance dashboard with metrics and alerts"
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
@monitoring_service_required
def performance_dashboard(request):
    """
    Get comprehensive performance dashboard data.
    """
    try:
        monitoring_service = get_monitoring_service()
        dashboard_data = monitoring_service.get_performance_dashboard()

        return Response(dashboard_data)

    except Exception as e:
        logger.error(f"Error getting performance dashboard: {str(e)}")
        return Response(
            {'error': f'Failed to get performance dashboard: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Get metric history",
    description="Get historical data for a specific metric",
    parameters=[
        OpenApiParameter(
            name='metric_name',
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            required=True,
            description='Name of the metric to retrieve'
        ),
        OpenApiParameter(
            name='hours',
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            required=False,
            description='Number of hours of history to retrieve (default: 24)'
        )
    ]
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def metric_history(request):
    """
    Get historical data for a specific metric.
    """
    try:
        metric_name = request.GET.get('metric_name')
        hours = int(request.GET.get('hours', 24))

        if not metric_name:
            return Response(
                {'error': 'metric_name parameter is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if hours < 1 or hours > 168:  # Max 1 week
            return Response(
                {'error': 'hours must be between 1 and 168'},
                status=status.HTTP_400_BAD_REQUEST
            )

        monitoring_service = get_monitoring_service()

        # Get metric history
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        history = monitoring_service.metrics_collector.get_metric_history(
            metric_name, start_time, end_time
        )

        # Get summary statistics
        summary = monitoring_service.metrics_collector.get_metric_summary(
            metric_name, hours
        )

        return Response({
            'metric_name': metric_name,
            'time_range': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'hours': hours
            },
            'history': history,
            'summary': summary
        })

    except ValueError as e:
        return Response(
            {'error': f'Invalid parameter: {str(e)}'},
            status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        logger.error(f"Error getting metric history: {str(e)}")
        return Response(
            {'error': f'Failed to get metric history: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Get recent alerts",
    description="Get recent monitoring alerts",
    parameters=[
        OpenApiParameter(
            name='severity',
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            required=False,
            description='Filter by alert severity (warning, critical)'
        ),
        OpenApiParameter(
            name='hours',
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            required=False,
            description='Number of hours of alerts to retrieve (default: 24)'
        )
    ]
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def recent_alerts(request):
    """
    Get recent monitoring alerts.
    """
    try:
        severity = request.GET.get('severity')
        hours = int(request.GET.get('hours', 24))

        if severity and severity not in ['warning', 'critical']:
            return Response(
                {'error': 'severity must be warning or critical'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if hours < 1 or hours > 168:  # Max 1 week
            return Response(
                {'error': 'hours must be between 1 and 168'},
                status=status.HTTP_400_BAD_REQUEST
            )

        monitoring_service = get_monitoring_service()
        alerts = monitoring_service.alert_manager.get_recent_alerts(
            severity=severity, hours=hours
        )

        # Count alerts by severity
        alert_counts = {'warning': 0, 'critical': 0}
        for alert in alerts:
            alert_counts[alert['severity']] += 1

        return Response({
            'alerts': alerts,
            'total_count': len(alerts),
            'alert_counts': alert_counts,
            'time_range': {
                'hours': hours,
                'end_time': datetime.now().isoformat()
            },
            'filters': {
                'severity': severity
            }
        })

    except ValueError as e:
        return Response(
            {'error': f'Invalid parameter: {str(e)}'},
            status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        logger.error(f"Error getting recent alerts: {str(e)}")
        return Response(
            {'error': f'Failed to get recent alerts: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Get performance profiles",
    description="Get performance profiling data for operations",
    parameters=[
        OpenApiParameter(
            name='operation_type',
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            required=False,
            description='Filter by operation type'
        ),
        OpenApiParameter(
            name='hours',
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            required=False,
            description='Number of hours of profiles to analyze (default: 24)'
        )
    ]
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def performance_profiles(request):
    """
    Get performance profiling data and statistics.
    """
    try:
        operation_type = request.GET.get('operation_type')
        hours = int(request.GET.get('hours', 24))

        if hours < 1 or hours > 168:  # Max 1 week
            return Response(
                {'error': 'hours must be between 1 and 168'},
                status=status.HTTP_400_BAD_REQUEST
            )

        monitoring_service = get_monitoring_service()
        profile_summary = monitoring_service.performance_profiler.get_profile_summary(
            operation_type=operation_type, hours=hours
        )

        return Response({
            'profile_summary': profile_summary,
            'filters': {
                'operation_type': operation_type,
                'hours': hours
            },
            'generated_at': datetime.now().isoformat()
        })

    except ValueError as e:
        return Response(
            {'error': f'Invalid parameter: {str(e)}'},
            status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        logger.error(f"Error getting performance profiles: {str(e)}")
        return Response(
            {'error': f'Failed to get performance profiles: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Get available metrics",
    description="Get list of available metrics and their definitions"
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def available_metrics(request):
    """
    Get list of available metrics and their definitions.
    """
    try:
        monitoring_service = get_monitoring_service()
        metric_definitions = monitoring_service.metrics_collector.metric_definitions

        # Get recent metric counts
        metric_counts = {}
        for metric_name in metric_definitions.keys():
            summary = monitoring_service.metrics_collector.get_metric_summary(metric_name, hours=1)
            if 'error' not in summary:
                metric_counts[metric_name] = summary['count']
            else:
                metric_counts[metric_name] = 0

        return Response({
            'metric_definitions': metric_definitions,
            'metric_counts': metric_counts,
            'total_metrics': len(metric_definitions),
            'active_metrics': sum(1 for count in metric_counts.values() if count > 0)
        })

    except Exception as e:
        logger.error(f"Error getting available metrics: {str(e)}")
        return Response(
            {'error': f'Failed to get available metrics: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Record custom metric",
    description="Record a custom metric value (admin only)",
    request={
        'application/json': {
            'type': 'object',
            'properties': {
                'metric_name': {
                    'type': 'string',
                    'description': 'Name of the metric'
                },
                'value': {
                    'type': 'number',
                    'description': 'Metric value'
                },
                'labels': {
                    'type': 'object',
                    'description': 'Optional metric labels'
                }
            },
            'required': ['metric_name', 'value']
        }
    }
)
@api_view(['POST'])
@permission_classes([IsAdminUser])
def record_metric(request):
    """
    Record a custom metric value (admin only).
    """
    try:
        metric_name = request.data.get('metric_name')
        value = request.data.get('value')
        labels = request.data.get('labels', {})

        if not metric_name:
            return Response(
                {'error': 'metric_name is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if value is None:
            return Response(
                {'error': 'value is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            value = float(value)
        except (ValueError, TypeError):
            return Response(
                {'error': 'value must be a number'},
                status=status.HTTP_400_BAD_REQUEST
            )

        monitoring_service = get_monitoring_service()
        monitoring_service.metrics_collector.record_metric(
            metric_name=metric_name,
            value=value,
            labels=labels
        )

        return Response({
            'message': 'Metric recorded successfully',
            'metric_name': metric_name,
            'value': value,
            'labels': labels,
            'recorded_by': request.user.username,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error recording custom metric: {str(e)}")
        return Response(
            {'error': f'Failed to record metric: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Get monitoring service status",
    description="Get status and configuration of the monitoring service"
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def monitoring_status(request):
    """
    Get monitoring service status and configuration.
    """
    try:
        monitoring_service = get_monitoring_service()

        return Response({
            'service_status': 'active',
            'background_monitoring': monitoring_service.monitoring_active,
            'components': {
                'metrics_collector': {
                    'active': True,
                    'retention_hours': monitoring_service.metrics_collector.retention_hours,
                    'metric_count': len(monitoring_service.metrics_collector.metric_definitions)
                },
                'performance_profiler': {
                    'active': True,
                    'active_profiles': len(monitoring_service.performance_profiler.active_profiles),
                    'completed_profiles': len(monitoring_service.performance_profiler.completed_profiles)
                },
                'alert_manager': {
                    'active': True,
                    'alert_rules': len(monitoring_service.alert_manager.alert_rules),
                    'recent_alerts': len(monitoring_service.alert_manager.alerts)
                }
            },
            'capabilities': [
                'system_metrics_collection',
                'performance_profiling',
                'alert_management',
                'custom_metrics',
                'real_time_monitoring'
            ],
            'uptime': 'N/A',  # Would track actual uptime in production
            'last_updated': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting monitoring status: {str(e)}")
        return Response(
            {'error': f'Failed to get monitoring status: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

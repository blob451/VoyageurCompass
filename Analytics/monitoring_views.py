"""
Monitoring Views for Prometheus Metrics Endpoint

Provides HTTP endpoints for metrics collection and system health monitoring.
"""

from django.http import HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import json

from Analytics.monitoring import get_metrics_collector


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
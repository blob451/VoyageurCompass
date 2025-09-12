"""
Prometheus Metrics Collection for VoyageurCompass Analytics Platform

Provides comprehensive performance monitoring and observability metrics
for caching, model inference, and API endpoint performance.
"""

import time
from collections import defaultdict
from functools import wraps
from typing import Dict, Any, Optional, List, Callable
import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("Prometheus client not available - metrics collection disabled")
    PROMETHEUS_AVAILABLE = False


class MetricsCollector:
    """Centralised metrics collection for Analytics platform."""
    
    def __init__(self):
        """Initialise metrics collector with Prometheus metrics."""
        self.enabled = PROMETHEUS_AVAILABLE
        
        if not self.enabled:
            return
            
        # Create custom registry for clean metrics management
        self.registry = CollectorRegistry()
        
        # Cache performance metrics
        self.cache_hits = Counter(
            'voyageur_cache_hits_total',
            'Total cache hits by service and cache type',
            ['service', 'cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'voyageur_cache_misses_total', 
            'Total cache misses by service and cache type',
            ['service', 'cache_type'],
            registry=self.registry
        )
        
        # Model inference metrics
        self.model_inference_duration = Histogram(
            'voyageur_model_inference_duration_seconds',
            'Model inference time in seconds',
            ['model_type', 'model_name'],
            registry=self.registry
        )
        
        self.model_inference_total = Counter(
            'voyageur_model_inference_total',
            'Total model inference requests',
            ['model_type', 'model_name', 'status'],
            registry=self.registry
        )
        
        # API endpoint performance
        self.api_request_duration = Histogram(
            'voyageur_api_request_duration_seconds',
            'API request duration in seconds',
            ['endpoint', 'method', 'status_code'],
            registry=self.registry
        )
        
        self.api_request_total = Counter(
            'voyageur_api_request_total',
            'Total API requests',
            ['endpoint', 'method', 'status_code'],
            registry=self.registry
        )
        
        # System resource metrics
        self.active_connections = Gauge(
            'voyageur_active_database_connections',
            'Number of active database connections',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'voyageur_memory_usage_bytes',
            'Memory usage in bytes by component',
            ['component'],
            registry=self.registry
        )
        
        # Fallback usage metrics
        self.fallback_usage = Counter(
            'voyageur_fallback_usage_total',
            'Total fallback usage by service and fallback type',
            ['service', 'fallback_type', 'reason'],
            registry=self.registry
        )
        
        # Business logic metrics
        self.analysis_requests = Counter(
            'voyageur_analysis_requests_total',
            'Total analysis requests by type',
            ['analysis_type', 'symbol'],
            registry=self.registry
        )
        
        self.prediction_accuracy = Histogram(
            'voyageur_prediction_accuracy_ratio',
            'Prediction accuracy ratio (0-1)',
            ['model_type', 'timeframe'],
            registry=self.registry
        )
        
        # Application info
        self.app_info = Info(
            'voyageur_application_info',
            'Application information',
            registry=self.registry
        )
        self.app_info.info({
            'version': '1.0.0',
            'component': 'analytics_platform',
            'environment': 'production'
        })
    
    def record_cache_hit(self, service: str, cache_type: str = 'default') -> None:
        """Record a cache hit event."""
        if self.enabled:
            self.cache_hits.labels(service=service, cache_type=cache_type).inc()
    
    def record_cache_miss(self, service: str, cache_type: str = 'default') -> None:
        """Record a cache miss event."""
        if self.enabled:
            self.cache_misses.labels(service=service, cache_type=cache_type).inc()
    
    def record_model_inference(self, model_type: str, model_name: str, 
                              duration: float, success: bool = True) -> None:
        """Record model inference metrics."""
        if not self.enabled:
            return
            
        status = 'success' if success else 'error'
        self.model_inference_duration.labels(
            model_type=model_type, 
            model_name=model_name
        ).observe(duration)
        self.model_inference_total.labels(
            model_type=model_type,
            model_name=model_name,
            status=status
        ).inc()
    
    def record_api_request(self, endpoint: str, method: str, 
                          status_code: int, duration: float) -> None:
        """Record API request metrics."""
        if not self.enabled:
            return
            
        self.api_request_duration.labels(
            endpoint=endpoint,
            method=method, 
            status_code=str(status_code)
        ).observe(duration)
        self.api_request_total.labels(
            endpoint=endpoint,
            method=method,
            status_code=str(status_code)
        ).inc()
    
    def record_fallback_usage(self, service: str, fallback_type: str, 
                             reason: str = 'unknown') -> None:
        """Record fallback mechanism usage."""
        if self.enabled:
            self.fallback_usage.labels(
                service=service,
                fallback_type=fallback_type,
                reason=reason
            ).inc()
    
    def record_analysis_request(self, analysis_type: str, symbol: str) -> None:
        """Record business analysis request."""
        if self.enabled:
            self.analysis_requests.labels(
                analysis_type=analysis_type,
                symbol=symbol
            ).inc()
    
    def record_prediction_accuracy(self, model_type: str, timeframe: str, 
                                  accuracy: float) -> None:
        """Record model prediction accuracy."""
        if self.enabled:
            self.prediction_accuracy.labels(
                model_type=model_type,
                timeframe=timeframe
            ).observe(accuracy)
    
    def update_active_connections(self, count: int) -> None:
        """Update active database connections gauge."""
        if self.enabled:
            self.active_connections.set(count)
    
    def update_memory_usage(self, component: str, bytes_used: int) -> None:
        """Update memory usage gauge."""
        if self.enabled:
            self.memory_usage.labels(component=component).set(bytes_used)
    
    def get_cache_hit_rate(self, service: str, cache_type: str = 'default') -> float:
        """Calculate cache hit rate for a service."""
        if not self.enabled:
            return 0.0
            
        try:
            hits = self.cache_hits.labels(service=service, cache_type=cache_type)._value._value
            misses = self.cache_misses.labels(service=service, cache_type=cache_type)._value._value
            total = hits + misses
            return (hits / total) if total > 0 else 0.0
        except Exception:
            return 0.0
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        if not self.enabled:
            return "# Prometheus metrics not available\n"
        return generate_latest(self.registry).decode('utf-8')


def timed_metric(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to automatically time function execution and record metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                
                # Extract service name from function or class
                service_name = getattr(func, '__qualname__', func.__name__)
                if hasattr(func, '__self__'):
                    service_name = func.__self__.__class__.__name__
                
                # Record metrics if collector is available
                if hasattr(wrapper, '_metrics_collector'):
                    wrapper._metrics_collector.record_model_inference(
                        model_type=metric_name,
                        model_name=service_name,
                        duration=duration,
                        success=success
                    )
        
        return wrapper
    return decorator


# Global metrics collector instance
metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return metrics_collector
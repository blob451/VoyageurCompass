"""
LLM Monitoring System

Comprehensive monitoring and logging for LLM operations including:
- Structured request/response logging  
- Performance metrics collection
- Error tracking and alerting
- Quality metrics monitoring
- Resource utilisation tracking
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from contextlib import contextmanager
from functools import wraps
import threading

from django.conf import settings
from django.core.cache import cache
from django.db import models
from django.utils import timezone

# Configure structured logging
logger = logging.getLogger('llm_monitoring')


class LLMMetrics:
    """Thread-safe metrics collector for LLM operations."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._metrics = {
            'requests_total': 0,
            'requests_successful': 0,
            'requests_failed': 0,
            'response_times': [],
            'model_usage': {},
            'detail_level_usage': {},
            'errors': [],
            'quality_scores': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        self._reset_time = time.time()
    
    def increment_counter(self, counter_name: str, value: int = 1):
        """Thread-safe counter increment."""
        with self._lock:
            if counter_name in self._metrics:
                self._metrics[counter_name] += value
    
    def record_response_time(self, response_time: float):
        """Record response time with automatic cleanup of old data."""
        with self._lock:
            self._metrics['response_times'].append({
                'timestamp': time.time(),
                'duration': response_time
            })
            
            # Keep only last 1000 response times
            if len(self._metrics['response_times']) > 1000:
                self._metrics['response_times'] = self._metrics['response_times'][-1000:]
    
    def record_model_usage(self, model_name: str):
        """Record usage of specific model."""
        with self._lock:
            self._metrics['model_usage'][model_name] = self._metrics['model_usage'].get(model_name, 0) + 1
    
    def record_detail_level_usage(self, detail_level: str):
        """Record usage of detail level."""
        with self._lock:
            self._metrics['detail_level_usage'][detail_level] = self._metrics['detail_level_usage'].get(detail_level, 0) + 1
    
    def record_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """Record error with context."""
        with self._lock:
            error_record = {
                'timestamp': time.time(),
                'error_type': error_type,
                'error_message': error_message,
                'context': context or {}
            }
            self._metrics['errors'].append(error_record)
            
            # Keep only last 100 errors
            if len(self._metrics['errors']) > 100:
                self._metrics['errors'] = self._metrics['errors'][-100:]
    
    def record_quality_score(self, score: float):
        """Record quality score."""
        with self._lock:
            if 0.0 <= score <= 1.0:
                self._metrics['quality_scores'].append({
                    'timestamp': time.time(),
                    'score': score
                })
                
                # Keep only last 500 quality scores
                if len(self._metrics['quality_scores']) > 500:
                    self._metrics['quality_scores'] = self._metrics['quality_scores'][-500:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        with self._lock:
            current_time = time.time()
            uptime = current_time - self._reset_time
            
            # Calculate response time statistics
            recent_response_times = [
                r['duration'] for r in self._metrics['response_times'] 
                if current_time - r['timestamp'] < 3600  # Last hour
            ]
            
            response_stats = {}
            if recent_response_times:
                response_stats = {
                    'count': len(recent_response_times),
                    'avg': sum(recent_response_times) / len(recent_response_times),
                    'min': min(recent_response_times),
                    'max': max(recent_response_times),
                    'p95': sorted(recent_response_times)[int(len(recent_response_times) * 0.95)] if len(recent_response_times) > 20 else max(recent_response_times)
                }
            
            # Calculate quality statistics
            recent_quality_scores = [
                q['score'] for q in self._metrics['quality_scores']
                if current_time - q['timestamp'] < 3600  # Last hour
            ]
            
            quality_stats = {}
            if recent_quality_scores:
                quality_stats = {
                    'count': len(recent_quality_scores),
                    'avg': sum(recent_quality_scores) / len(recent_quality_scores),
                    'min': min(recent_quality_scores),
                    'max': max(recent_quality_scores)
                }
            
            # Calculate error statistics
            recent_errors = [
                e for e in self._metrics['errors']
                if current_time - e['timestamp'] < 3600  # Last hour
            ]
            
            error_summary = {}
            if recent_errors:
                error_types = {}
                for error in recent_errors:
                    error_type = error['error_type']
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                
                error_summary = {
                    'total_errors': len(recent_errors),
                    'error_types': error_types,
                    'error_rate': len(recent_errors) / max(self._metrics['requests_total'], 1) * 100
                }
            
            # Calculate success rate
            success_rate = 0
            if self._metrics['requests_total'] > 0:
                success_rate = (self._metrics['requests_successful'] / self._metrics['requests_total']) * 100
            
            # Calculate cache hit rate
            total_cache_requests = self._metrics['cache_hits'] + self._metrics['cache_misses']
            cache_hit_rate = 0
            if total_cache_requests > 0:
                cache_hit_rate = (self._metrics['cache_hits'] / total_cache_requests) * 100
            
            return {
                'uptime_seconds': uptime,
                'timestamp': current_time,
                'requests': {
                    'total': self._metrics['requests_total'],
                    'successful': self._metrics['requests_successful'],
                    'failed': self._metrics['requests_failed'],
                    'success_rate': success_rate,
                    'requests_per_second': self._metrics['requests_total'] / max(uptime, 1)
                },
                'response_times': response_stats,
                'quality': quality_stats,
                'errors': error_summary,
                'models': dict(self._metrics['model_usage']),
                'detail_levels': dict(self._metrics['detail_level_usage']),
                'cache': {
                    'hits': self._metrics['cache_hits'],
                    'misses': self._metrics['cache_misses'],
                    'hit_rate': cache_hit_rate
                }
            }
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self._metrics = {
                'requests_total': 0,
                'requests_successful': 0,
                'requests_failed': 0,
                'response_times': [],
                'model_usage': {},
                'detail_level_usage': {},
                'errors': [],
                'quality_scores': [],
                'cache_hits': 0,
                'cache_misses': 0
            }
            self._reset_time = time.time()


# Global metrics instance
llm_metrics = LLMMetrics()


class StructuredLLMLogger:
    """Structured logger for LLM operations."""
    
    def __init__(self, logger_name: str = 'llm_operations'):
        self.logger = logging.getLogger(logger_name)
        self.correlation_id = None
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for request tracing."""
        self.correlation_id = correlation_id
    
    def log_request(self, analysis_data: Dict[str, Any], detail_level: str, 
                   correlation_id: str = None, sanitize: bool = True):
        """Log LLM request with structured data."""
        if correlation_id:
            self.correlation_id = correlation_id
        
        # Sanitise sensitive data
        sanitized_data = self._sanitize_data(analysis_data) if sanitize else analysis_data
        
        log_entry = {
            'event_type': 'llm_request',
            'timestamp': datetime.now().isoformat(),
            'correlation_id': self.correlation_id,
            'detail_level': detail_level,
            'symbol': sanitized_data.get('symbol'),
            'recommendation': sanitized_data.get('recommendation'),
            'technical_score': sanitized_data.get('technical_score'),
            'request_size': len(str(sanitized_data)),
            'indicator_count': len(sanitized_data.get('indicators', {}))
        }
        
        self.logger.info(json.dumps(log_entry))
        
        # Update metrics
        llm_metrics.increment_counter('requests_total')
        llm_metrics.record_detail_level_usage(detail_level)
    
    def log_response(self, result: Dict[str, Any], response_time: float, 
                    model_used: str = None, quality_score: float = None,
                    sanitize: bool = True):
        """Log LLM response with structured data."""
        
        # Sanitise response data
        sanitized_result = self._sanitize_data(result) if sanitize else result
        
        log_entry = {
            'event_type': 'llm_response',
            'timestamp': datetime.now().isoformat(),
            'correlation_id': self.correlation_id,
            'response_time_seconds': response_time,
            'model_used': model_used,
            'quality_score': quality_score,
            'response_size': len(str(sanitized_result)),
            'explanation_word_count': len(sanitized_result.get('explanation', '').split()) if sanitized_result.get('explanation') else 0,
            'success': 'explanation' in (sanitized_result or {})
        }
        
        self.logger.info(json.dumps(log_entry))
        
        # Update metrics
        if log_entry['success']:
            llm_metrics.increment_counter('requests_successful')
        else:
            llm_metrics.increment_counter('requests_failed')
        
        llm_metrics.record_response_time(response_time)
        
        if model_used:
            llm_metrics.record_model_usage(model_used)
        
        if quality_score is not None:
            llm_metrics.record_quality_score(quality_score)
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None,
                 error_type: str = None):
        """Log LLM operation error with context."""
        
        error_entry = {
            'event_type': 'llm_error',
            'timestamp': datetime.now().isoformat(),
            'correlation_id': self.correlation_id,
            'error_type': error_type or type(error).__name__,
            'error_message': str(error),
            'context': self._sanitize_data(context or {})
        }
        
        self.logger.error(json.dumps(error_entry))
        
        # Update metrics
        llm_metrics.increment_counter('requests_failed')
        llm_metrics.record_error(
            error_type=error_entry['error_type'],
            error_message=error_entry['error_message'],
            context=error_entry['context']
        )
    
    def log_cache_event(self, event_type: str, cache_key: str, hit: bool = None):
        """Log cache-related events."""
        
        cache_entry = {
            'event_type': f'cache_{event_type}',
            'timestamp': datetime.now().isoformat(),
            'correlation_id': self.correlation_id,
            'cache_key_hash': hashlib.md5(cache_key.encode()).hexdigest()[:16],  # Hash for privacy
            'cache_hit': hit
        }
        
        self.logger.debug(json.dumps(cache_entry))
        
        # Update cache metrics
        if hit is True:
            llm_metrics.increment_counter('cache_hits')
        elif hit is False:
            llm_metrics.increment_counter('cache_misses')
    
    def _sanitize_data(self, data: Any) -> Any:
        """Sanitise sensitive data for logging."""
        import hashlib
        
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if key.lower() in ['user_id', 'username', 'email', 'password', 'token']:
                    # Hash sensitive fields
                    sanitized[key] = hashlib.md5(str(value).encode()).hexdigest()[:8]
                elif key == 'explanation' and isinstance(value, str) and len(value) > 500:
                    # Truncate long explanations for logging
                    sanitized[key] = value[:500] + "... [truncated]"
                else:
                    sanitized[key] = self._sanitize_data(value)
            return sanitized
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data[:10]]  # Limit list size
        else:
            return data


class LLMPerformanceAlert:
    """Performance alerting system for LLM operations."""
    
    def __init__(self):
        self.alert_thresholds = {
            'error_rate_percent': 10.0,
            'avg_response_time_seconds': 20.0,
            'success_rate_percent': 90.0,
            'quality_score_min': 0.7
        }
        self.alert_cooldown = 300  # 5 minutes cooldown between similar alerts
        self.last_alerts = {}
    
    def check_and_alert(self, metrics_summary: Dict[str, Any]) -> List[str]:
        """Check metrics against thresholds and generate alerts."""
        alerts = []
        current_time = time.time()
        
        # Error rate alert
        error_rate = metrics_summary.get('errors', {}).get('error_rate', 0)
        if error_rate > self.alert_thresholds['error_rate_percent']:
            alert_key = 'error_rate'
            if self._should_alert(alert_key, current_time):
                alerts.append(f"HIGH ERROR RATE: {error_rate:.1f}% (threshold: {self.alert_thresholds['error_rate_percent']}%)")
                self.last_alerts[alert_key] = current_time
        
        # Response time alert
        avg_response_time = metrics_summary.get('response_times', {}).get('avg', 0)
        if avg_response_time > self.alert_thresholds['avg_response_time_seconds']:
            alert_key = 'response_time'
            if self._should_alert(alert_key, current_time):
                alerts.append(f"SLOW RESPONSE TIME: {avg_response_time:.2f}s (threshold: {self.alert_thresholds['avg_response_time_seconds']}s)")
                self.last_alerts[alert_key] = current_time
        
        # Success rate alert
        success_rate = metrics_summary.get('requests', {}).get('success_rate', 100)
        if success_rate < self.alert_thresholds['success_rate_percent']:
            alert_key = 'success_rate'
            if self._should_alert(alert_key, current_time):
                alerts.append(f"LOW SUCCESS RATE: {success_rate:.1f}% (threshold: {self.alert_thresholds['success_rate_percent']}%)")
                self.last_alerts[alert_key] = current_time
        
        # Quality score alert
        avg_quality = metrics_summary.get('quality', {}).get('avg', 1.0)
        if avg_quality < self.alert_thresholds['quality_score_min']:
            alert_key = 'quality_score'
            if self._should_alert(alert_key, current_time):
                alerts.append(f"LOW QUALITY SCORE: {avg_quality:.2f} (threshold: {self.alert_thresholds['quality_score_min']})")
                self.last_alerts[alert_key] = current_time
        
        return alerts
    
    def _should_alert(self, alert_key: str, current_time: float) -> bool:
        """Check if enough time has passed since last alert."""
        last_alert_time = self.last_alerts.get(alert_key, 0)
        return current_time - last_alert_time > self.alert_cooldown


# Global instances
llm_logger = StructuredLLMLogger()
llm_alerting = LLMPerformanceAlert()


@contextmanager
def llm_monitoring_context(analysis_data: Dict[str, Any], detail_level: str,
                          correlation_id: str = None):
    """Context manager for comprehensive LLM operation monitoring."""
    
    # Generate correlation ID if not provided
    if not correlation_id:
        correlation_id = str(uuid.uuid4())[:8]
    
    llm_logger.set_correlation_id(correlation_id)
    start_time = time.time()
    
    # Log request
    llm_logger.log_request(analysis_data, detail_level, correlation_id)
    
    try:
        yield correlation_id
    except Exception as e:
        # Log error
        llm_logger.log_error(e, context={
            'detail_level': detail_level,
            'symbol': analysis_data.get('symbol')
        })
        raise
    finally:
        # Final response logging will be handled by the service layer
        pass


def llm_operation_monitor(func):
    """Decorator for monitoring LLM operations."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract monitoring context from arguments
        analysis_data = kwargs.get('analysis_data') or (args[1] if len(args) > 1 else {})
        detail_level = kwargs.get('detail_level') or (args[2] if len(args) > 2 else 'unknown')
        
        correlation_id = str(uuid.uuid4())[:8]
        
        with llm_monitoring_context(analysis_data, detail_level, correlation_id):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                response_time = time.time() - start_time
                
                # Log successful response
                model_used = result.get('model_used') if result else None
                quality_score = result.get('quality_score') if result else None
                
                llm_logger.log_response(
                    result=result or {},
                    response_time=response_time,
                    model_used=model_used,
                    quality_score=quality_score
                )
                
                return result
                
            except Exception as e:
                response_time = time.time() - start_time
                
                # Log error response
                llm_logger.log_error(e, context={
                    'function': func.__name__,
                    'detail_level': detail_level,
                    'response_time': response_time
                })
                
                raise
    
    return wrapper


def get_llm_monitoring_dashboard_data() -> Dict[str, Any]:
    """Get comprehensive monitoring data for dashboard display."""
    
    # Get current metrics summary
    metrics_summary = llm_metrics.get_summary()
    
    # Check for alerts
    current_alerts = llm_alerting.check_and_alert(metrics_summary)
    
    # Get system health indicators
    health_indicators = {
        'status': 'healthy',
        'issues': []
    }
    
    # Determine overall health status
    if current_alerts:
        health_indicators['status'] = 'warning' if len(current_alerts) <= 2 else 'critical'
        health_indicators['issues'] = current_alerts
    
    # Add trend analysis
    trends = _calculate_trends(metrics_summary)
    
    dashboard_data = {
        'timestamp': datetime.now().isoformat(),
        'health': health_indicators,
        'metrics': metrics_summary,
        'trends': trends,
        'alerts': current_alerts,
        'recommendations': _generate_recommendations(metrics_summary)
    }
    
    # Cache dashboard data for 30 seconds
    cache.set('llm_monitoring_dashboard', dashboard_data, 30)
    
    return dashboard_data


def _calculate_trends(metrics_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate performance trends."""
    
    # Get cached historical data for comparison
    historical_data = cache.get('llm_metrics_history', [])
    current_timestamp = time.time()
    
    # Add current metrics to history
    historical_data.append({
        'timestamp': current_timestamp,
        'metrics': metrics_summary
    })
    
    # Keep only last 24 hours of data
    cutoff_time = current_timestamp - (24 * 3600)
    historical_data = [h for h in historical_data if h['timestamp'] > cutoff_time]
    
    # Cache updated history
    cache.set('llm_metrics_history', historical_data, 86400)  # 24 hours
    
    trends = {}
    
    if len(historical_data) >= 2:
        # Calculate trends
        recent_metrics = historical_data[-1]['metrics']
        older_metrics = historical_data[0]['metrics']
        
        # Response time trend
        recent_avg_time = recent_metrics.get('response_times', {}).get('avg', 0)
        older_avg_time = older_metrics.get('response_times', {}).get('avg', 0)
        
        if older_avg_time > 0:
            response_time_trend = ((recent_avg_time - older_avg_time) / older_avg_time) * 100
            trends['response_time_change_percent'] = response_time_trend
        
        # Success rate trend
        recent_success_rate = recent_metrics.get('requests', {}).get('success_rate', 0)
        older_success_rate = older_metrics.get('requests', {}).get('success_rate', 0)
        
        success_rate_change = recent_success_rate - older_success_rate
        trends['success_rate_change_percent'] = success_rate_change
        
        # Request volume trend
        recent_total = recent_metrics.get('requests', {}).get('total', 0)
        older_total = older_metrics.get('requests', {}).get('total', 0)
        
        request_volume_change = recent_total - older_total
        trends['request_volume_change'] = request_volume_change
    
    return trends


def _generate_recommendations(metrics_summary: Dict[str, Any]) -> List[str]:
    """Generate optimisation recommendations based on metrics."""
    
    recommendations = []
    
    # Cache hit rate recommendations
    cache_hit_rate = metrics_summary.get('cache', {}).get('hit_rate', 0)
    if cache_hit_rate < 30:
        recommendations.append("Consider increasing cache timeout to improve hit rate")
    
    # Response time recommendations
    avg_response_time = metrics_summary.get('response_times', {}).get('avg', 0)
    if avg_response_time > 10:
        recommendations.append("High response times detected - consider model optimisation")
    
    # Error rate recommendations
    error_rate = metrics_summary.get('errors', {}).get('error_rate', 0)
    if error_rate > 5:
        recommendations.append("Elevated error rate - review error logs for patterns")
    
    # Model usage recommendations
    model_usage = metrics_summary.get('models', {})
    if model_usage:
        most_used_model = max(model_usage, key=model_usage.get)
        usage_percent = (model_usage[most_used_model] / sum(model_usage.values())) * 100
        
        if usage_percent > 80:
            recommendations.append(f"Model {most_used_model} handles {usage_percent:.0f}% of requests - consider load balancing")
    
    # Quality score recommendations
    avg_quality = metrics_summary.get('quality', {}).get('avg', 1.0)
    if avg_quality < 0.8:
        recommendations.append("Quality scores below optimal - consider prompt optimisation")
    
    return recommendations
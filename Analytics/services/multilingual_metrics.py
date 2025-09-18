"""
MultilingualMetrics service for VoyageurCompass.

Comprehensive metrics tracking and monitoring for multilingual LLM operations,
including quality assessment, performance monitoring, and usage analytics.
"""

import json
import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from django.conf import settings
from django.core.cache import cache
from django.utils import timezone

logger = logging.getLogger(__name__)


@dataclass
class QualityMetric:
    """Quality assessment for a translation/explanation."""
    content_id: str
    language: str
    quality_score: float
    fluency_score: float
    accuracy_score: float
    completeness_score: float
    cultural_appropriateness: float
    technical_accuracy: float
    timestamp: datetime
    model_used: str
    detail_level: str


@dataclass
class PerformanceMetric:
    """Performance metric for multilingual operations."""
    operation_id: str
    symbol: str
    languages_requested: List[str]
    languages_completed: List[str]
    processing_time: float
    memory_usage_mb: float
    cache_hit_rate: float
    parallel_efficiency: float
    timestamp: datetime
    success: bool
    error_details: Optional[str] = None


@dataclass
class UsageMetric:
    """Usage analytics for multilingual features."""
    user_id: Optional[int]
    symbol: str
    language: str
    detail_level: str
    feature_type: str  # 'explanation', 'translation', 'batch'
    timestamp: datetime
    processing_time: float
    success: bool
    cache_hit: bool


class MultilingualMetrics:
    """Comprehensive metrics tracking for multilingual operations."""

    def __init__(self):
        """Initialize the multilingual metrics system."""
        # Configuration
        self.enabled = getattr(settings, 'MULTILINGUAL_METRICS_ENABLED', True)
        self.retention_days = getattr(settings, 'METRICS_RETENTION_DAYS', 30)
        self.max_memory_samples = getattr(settings, 'METRICS_MAX_MEMORY_SAMPLES', 1000)

        # Thread safety
        self._lock = threading.RLock()

        # In-memory storage for recent metrics
        self._quality_metrics = deque(maxlen=self.max_memory_samples)
        self._performance_metrics = deque(maxlen=self.max_memory_samples)
        self._usage_metrics = deque(maxlen=self.max_memory_samples)

        # Aggregated statistics with bounded collections
        self._language_stats = defaultdict(lambda: {
            'requests': 0,
            'successes': 0,
            'total_time': 0.0,
            'quality_scores': deque(maxlen=100),  # Bounded collection
            'cache_hits': 0,
            'last_cleanup': datetime.now()
        })

        self._model_stats = defaultdict(lambda: {
            'usage_count': 0,
            'avg_quality': 0.0,
            'avg_time': 0.0,
            'success_rate': 0.0,
            'quality_history': deque(maxlen=50),  # Bounded collection
            'performance_history': deque(maxlen=50),  # Bounded collection
            'last_used': datetime.now()
        })

        # Cleanup scheduling
        self.cleanup_interval_minutes = getattr(settings, 'METRICS_CLEANUP_INTERVAL_MINUTES', 60)
        self.max_inactive_days = getattr(settings, 'METRICS_MAX_INACTIVE_DAYS', 7)
        self._last_cleanup = datetime.now()
        self._cleanup_counter = 0

        # Real-time monitoring
        self._current_session = {
            'start_time': datetime.now(),
            'requests_processed': 0,
            'errors_encountered': 0,
            'peak_memory_usage': 0.0,
            'cache_effectiveness': 0.0
        }

        # Alert thresholds
        self.quality_threshold = getattr(settings, 'QUALITY_ALERT_THRESHOLD', 0.7)
        self.performance_threshold = getattr(settings, 'PERFORMANCE_ALERT_THRESHOLD', 10.0)
        self.error_rate_threshold = getattr(settings, 'ERROR_RATE_ALERT_THRESHOLD', 0.1)

        logger.info("MultilingualMetrics initialized successfully")

    def record_quality_metric(self, metric: QualityMetric) -> None:
        """Record a quality assessment metric."""
        if not self.enabled:
            return

        with self._lock:
            self._quality_metrics.append(metric)

            # Update language stats
            lang_stats = self._language_stats[metric.language]
            lang_stats['quality_scores'].append(metric.quality_score)

            # Update model stats
            model_stats = self._model_stats[metric.model_used]
            model_stats['usage_count'] += 1

            # Update running averages
            self._update_model_averages(metric.model_used)

            # Check for quality alerts
            self._check_quality_alerts(metric)

        logger.debug(f"Recorded quality metric for {metric.language}: {metric.quality_score:.2f}")

    def record_performance_metric(self, metric: PerformanceMetric) -> None:
        """Record a performance metric."""
        if not self.enabled:
            return

        with self._lock:
            self._performance_metrics.append(metric)

            # Update session statistics
            self._current_session['requests_processed'] += 1
            if not metric.success:
                self._current_session['errors_encountered'] += 1

            self._current_session['peak_memory_usage'] = max(
                self._current_session['peak_memory_usage'],
                metric.memory_usage_mb
            )

            # Update language performance stats
            for lang in metric.languages_completed:
                lang_stats = self._language_stats[lang]
                lang_stats['requests'] += 1
                if metric.success:
                    lang_stats['successes'] += 1
                lang_stats['total_time'] += metric.processing_time

                if metric.cache_hit_rate > 0:
                    lang_stats['cache_hits'] += 1

            # Check for performance alerts
            self._check_performance_alerts(metric)

        logger.debug(f"Recorded performance metric for {metric.symbol}: {metric.processing_time:.2f}s")

    def record_usage_metric(self, metric: UsageMetric) -> None:
        """Record a usage analytics metric."""
        if not self.enabled:
            return

        with self._lock:
            self._usage_metrics.append(metric)

            # Update cache effectiveness
            if metric.cache_hit:
                self._current_session['cache_effectiveness'] += 1

        logger.debug(f"Recorded usage metric: {metric.feature_type} for {metric.language}")

    def get_quality_report(
        self,
        language: Optional[str] = None,
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """Generate a comprehensive quality report."""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=time_range_hours)

            # Filter metrics by time and language
            relevant_metrics = [
                m for m in self._quality_metrics
                if m.timestamp >= cutoff_time and (language is None or m.language == language)
            ]

            if not relevant_metrics:
                return {
                    'summary': 'No quality data available',
                    'metrics_count': 0
                }

            # Calculate quality statistics
            quality_scores = [m.quality_score for m in relevant_metrics]
            fluency_scores = [m.fluency_score for m in relevant_metrics]
            accuracy_scores = [m.accuracy_score for m in relevant_metrics]

            # Group by language
            by_language = defaultdict(list)
            for metric in relevant_metrics:
                by_language[metric.language].append(metric)

            language_quality = {}
            for lang, metrics in by_language.items():
                scores = [m.quality_score for m in metrics]
                language_quality[lang] = {
                    'avg_quality': statistics.mean(scores),
                    'min_quality': min(scores),
                    'max_quality': max(scores),
                    'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    'sample_count': len(scores),
                    'below_threshold_count': sum(1 for s in scores if s < self.quality_threshold)
                }

            # Group by model
            by_model = defaultdict(list)
            for metric in relevant_metrics:
                by_model[metric.model_used].append(metric)

            model_quality = {}
            for model, metrics in by_model.items():
                scores = [m.quality_score for m in metrics]
                model_quality[model] = {
                    'avg_quality': statistics.mean(scores),
                    'usage_count': len(metrics),
                    'reliability': sum(1 for s in scores if s >= self.quality_threshold) / len(scores)
                }

            return {
                'summary': {
                    'total_assessments': len(relevant_metrics),
                    'avg_quality': statistics.mean(quality_scores),
                    'avg_fluency': statistics.mean(fluency_scores),
                    'avg_accuracy': statistics.mean(accuracy_scores),
                    'quality_distribution': self._calculate_quality_distribution(quality_scores),
                    'time_range_hours': time_range_hours
                },
                'by_language': language_quality,
                'by_model': model_quality,
                'trends': self._calculate_quality_trends(relevant_metrics),
                'alerts': self._get_quality_alerts()
            }

    def get_performance_report(
        self,
        symbol: Optional[str] = None,
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=time_range_hours)

            # Filter metrics
            relevant_metrics = [
                m for m in self._performance_metrics
                if m.timestamp >= cutoff_time and (symbol is None or m.symbol == symbol)
            ]

            if not relevant_metrics:
                return {
                    'summary': 'No performance data available',
                    'metrics_count': 0
                }

            # Calculate performance statistics
            processing_times = [m.processing_time for m in relevant_metrics]
            memory_usage = [m.memory_usage_mb for m in relevant_metrics]
            cache_hit_rates = [m.cache_hit_rate for m in relevant_metrics]
            parallel_efficiency = [m.parallel_efficiency for m in relevant_metrics]

            success_count = sum(1 for m in relevant_metrics if m.success)
            error_count = len(relevant_metrics) - success_count

            # Performance by language count
            by_language_count = defaultdict(list)
            for metric in relevant_metrics:
                lang_count = len(metric.languages_requested)
                by_language_count[lang_count].append(metric)

            performance_by_complexity = {}
            for lang_count, metrics in by_language_count.items():
                times = [m.processing_time for m in metrics]
                performance_by_complexity[f"{lang_count}_languages"] = {
                    'avg_time': statistics.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'request_count': len(metrics)
                }

            return {
                'summary': {
                    'total_requests': len(relevant_metrics),
                    'success_rate': success_count / len(relevant_metrics),
                    'error_rate': error_count / len(relevant_metrics),
                    'avg_processing_time': statistics.mean(processing_times),
                    'avg_memory_usage': statistics.mean(memory_usage),
                    'avg_cache_hit_rate': statistics.mean(cache_hit_rates),
                    'avg_parallel_efficiency': statistics.mean(parallel_efficiency),
                    'time_range_hours': time_range_hours
                },
                'performance_distribution': {
                    'processing_time_percentiles': self._calculate_percentiles(processing_times),
                    'memory_usage_percentiles': self._calculate_percentiles(memory_usage),
                    'cache_effectiveness': statistics.mean(cache_hit_rates)
                },
                'by_complexity': performance_by_complexity,
                'trends': self._calculate_performance_trends(relevant_metrics),
                'alerts': self._get_performance_alerts(),
                'resource_usage': {
                    'peak_memory': max(memory_usage) if memory_usage else 0,
                    'avg_memory': statistics.mean(memory_usage) if memory_usage else 0,
                    'memory_efficiency': 'good' if max(memory_usage) < 500 else 'needs_attention'
                }
            }

    def get_usage_analytics(
        self,
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """Generate usage analytics report."""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=time_range_hours)

            relevant_metrics = [
                m for m in self._usage_metrics
                if m.timestamp >= cutoff_time
            ]

            if not relevant_metrics:
                return {
                    'summary': 'No usage data available',
                    'metrics_count': 0
                }

            # Language popularity
            language_usage = defaultdict(int)
            for metric in relevant_metrics:
                language_usage[metric.language] += 1

            # Feature usage
            feature_usage = defaultdict(int)
            for metric in relevant_metrics:
                feature_usage[metric.feature_type] += 1

            # Detail level preference
            detail_level_usage = defaultdict(int)
            for metric in relevant_metrics:
                detail_level_usage[metric.detail_level] += 1

            # Peak usage times
            hourly_usage = defaultdict(int)
            for metric in relevant_metrics:
                hour = metric.timestamp.hour
                hourly_usage[hour] += 1

            return {
                'summary': {
                    'total_operations': len(relevant_metrics),
                    'unique_symbols': len(set(m.symbol for m in relevant_metrics)),
                    'unique_users': len(set(m.user_id for m in relevant_metrics if m.user_id)),
                    'avg_processing_time': statistics.mean([m.processing_time for m in relevant_metrics]),
                    'success_rate': sum(1 for m in relevant_metrics if m.success) / len(relevant_metrics),
                    'cache_hit_rate': sum(1 for m in relevant_metrics if m.cache_hit) / len(relevant_metrics)
                },
                'language_popularity': dict(sorted(language_usage.items(), key=lambda x: x[1], reverse=True)),
                'feature_usage': dict(feature_usage),
                'detail_level_preference': dict(detail_level_usage),
                'peak_hours': dict(sorted(hourly_usage.items(), key=lambda x: x[1], reverse=True)[:5]),
                'trends': self._calculate_usage_trends(relevant_metrics)
            }

    def get_real_time_dashboard(self) -> Dict[str, Any]:
        """Get real-time dashboard data."""
        with self._lock:
            session_duration = (datetime.now() - self._current_session['start_time']).total_seconds()

            # Recent metrics (last 5 minutes)
            recent_cutoff = datetime.now() - timedelta(minutes=5)
            recent_performance = [
                m for m in self._performance_metrics
                if m.timestamp >= recent_cutoff
            ]

            recent_quality = [
                m for m in self._quality_metrics
                if m.timestamp >= recent_cutoff
            ]

            # Calculate current rates
            requests_per_minute = 0
            avg_response_time = 0
            current_error_rate = 0

            if recent_performance:
                requests_per_minute = len(recent_performance) / 5  # per minute in last 5 minutes
                avg_response_time = statistics.mean([m.processing_time for m in recent_performance])
                errors = sum(1 for m in recent_performance if not m.success)
                current_error_rate = errors / len(recent_performance)

            return {
                'session': {
                    'uptime_seconds': session_duration,
                    'total_requests': self._current_session['requests_processed'],
                    'total_errors': self._current_session['errors_encountered'],
                    'peak_memory_mb': self._current_session['peak_memory_usage'],
                    'cache_effectiveness': self._current_session['cache_effectiveness']
                },
                'current_rates': {
                    'requests_per_minute': requests_per_minute,
                    'avg_response_time': avg_response_time,
                    'error_rate': current_error_rate,
                    'quality_score': statistics.mean([m.quality_score for m in recent_quality]) if recent_quality else 0
                },
                'system_health': {
                    'status': self._determine_system_health(),
                    'active_languages': list(set(m.language for m in recent_quality)),
                    'active_models': list(set(m.model_used for m in recent_quality)),
                    'cache_hit_rate': statistics.mean([m.cache_hit_rate for m in recent_performance]) if recent_performance else 0
                },
                'alerts': {
                    'quality_alerts': len(self._get_quality_alerts()),
                    'performance_alerts': len(self._get_performance_alerts()),
                    'critical_alerts': self._get_critical_alerts()
                }
            }

    def export_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        format_type: str = 'json'
    ) -> str:
        """Export metrics data for external analysis."""
        with self._lock:
            # Filter all metrics by time range
            quality_data = [
                asdict(m) for m in self._quality_metrics
                if start_time <= m.timestamp <= end_time
            ]

            performance_data = [
                asdict(m) for m in self._performance_metrics
                if start_time <= m.timestamp <= end_time
            ]

            usage_data = [
                asdict(m) for m in self._usage_metrics
                if start_time <= m.timestamp <= end_time
            ]

            export_data = {
                'export_metadata': {
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'generated_at': datetime.now().isoformat(),
                    'version': 'v4.0'
                },
                'quality_metrics': quality_data,
                'performance_metrics': performance_data,
                'usage_metrics': usage_data,
                'summary_statistics': {
                    'total_quality_assessments': len(quality_data),
                    'total_performance_records': len(performance_data),
                    'total_usage_records': len(usage_data)
                }
            }

            if format_type == 'json':
                return json.dumps(export_data, indent=2, default=str)
            else:
                # Could add CSV, XML formats in the future
                return json.dumps(export_data, default=str)

    def _update_model_averages(self, model_name: str) -> None:
        """Update running averages for a model."""
        model_metrics = [m for m in self._quality_metrics if m.model_used == model_name]

        if model_metrics:
            quality_scores = [m.quality_score for m in model_metrics]
            self._model_stats[model_name]['avg_quality'] = statistics.mean(quality_scores)

    def _check_quality_alerts(self, metric: QualityMetric) -> None:
        """Check for quality-related alerts."""
        if metric.quality_score < self.quality_threshold:
            logger.warning(
                f"Quality alert: {metric.language} explanation quality ({metric.quality_score:.2f}) "
                f"below threshold ({self.quality_threshold})"
            )

    def _check_performance_alerts(self, metric: PerformanceMetric) -> None:
        """Check for performance-related alerts."""
        if metric.processing_time > self.performance_threshold:
            logger.warning(
                f"Performance alert: {metric.symbol} processing time ({metric.processing_time:.2f}s) "
                f"exceeds threshold ({self.performance_threshold}s)"
            )

        if not metric.success:
            logger.error(f"Error alert: Failed to process {metric.symbol}: {metric.error_details}")

    def _calculate_quality_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Calculate quality score distribution."""
        distribution = {
            'excellent': 0,  # 0.9+
            'good': 0,       # 0.8-0.89
            'acceptable': 0, # 0.7-0.79
            'poor': 0        # <0.7
        }

        for score in scores:
            if score >= 0.9:
                distribution['excellent'] += 1
            elif score >= 0.8:
                distribution['good'] += 1
            elif score >= 0.7:
                distribution['acceptable'] += 1
            else:
                distribution['poor'] += 1

        return distribution

    def _calculate_percentiles(self, values: List[float]) -> Dict[str, float]:
        """Calculate percentile distribution."""
        if not values:
            return {}

        sorted_values = sorted(values)
        length = len(sorted_values)

        return {
            'p50': sorted_values[int(length * 0.5)],
            'p75': sorted_values[int(length * 0.75)],
            'p90': sorted_values[int(length * 0.9)],
            'p95': sorted_values[int(length * 0.95)],
            'p99': sorted_values[int(length * 0.99)] if length > 100 else sorted_values[-1]
        }

    def _calculate_quality_trends(self, metrics: List[QualityMetric]) -> Dict[str, Any]:
        """Calculate quality trends over time."""
        if len(metrics) < 2:
            return {'trend': 'insufficient_data'}

        # Sort by timestamp
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)

        # Calculate moving average
        window_size = min(10, len(sorted_metrics) // 2)
        moving_averages = []

        for i in range(window_size, len(sorted_metrics)):
            window = sorted_metrics[i-window_size:i]
            avg_quality = statistics.mean([m.quality_score for m in window])
            moving_averages.append(avg_quality)

        if len(moving_averages) < 2:
            return {'trend': 'insufficient_data'}

        # Calculate trend direction
        recent_avg = statistics.mean(moving_averages[-3:])
        earlier_avg = statistics.mean(moving_averages[:3])

        if recent_avg > earlier_avg * 1.05:
            trend = 'improving'
        elif recent_avg < earlier_avg * 0.95:
            trend = 'declining'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'recent_average': recent_avg,
            'earlier_average': earlier_avg,
            'improvement_rate': (recent_avg - earlier_avg) / earlier_avg if earlier_avg > 0 else 0
        }

    def _calculate_performance_trends(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Calculate performance trends over time."""
        if len(metrics) < 2:
            return {'trend': 'insufficient_data'}

        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)

        # Calculate moving averages for processing time
        window_size = min(10, len(sorted_metrics) // 2)
        moving_times = []

        for i in range(window_size, len(sorted_metrics)):
            window = sorted_metrics[i-window_size:i]
            avg_time = statistics.mean([m.processing_time for m in window])
            moving_times.append(avg_time)

        if len(moving_times) < 2:
            return {'trend': 'insufficient_data'}

        recent_time = statistics.mean(moving_times[-3:])
        earlier_time = statistics.mean(moving_times[:3])

        if recent_time < earlier_time * 0.95:
            trend = 'improving'
        elif recent_time > earlier_time * 1.05:
            trend = 'declining'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'recent_avg_time': recent_time,
            'earlier_avg_time': earlier_time,
            'performance_change': (earlier_time - recent_time) / earlier_time if earlier_time > 0 else 0
        }

    def _calculate_usage_trends(self, metrics: List[UsageMetric]) -> Dict[str, Any]:
        """Calculate usage trends over time."""
        if len(metrics) < 2:
            return {'trend': 'insufficient_data'}

        # Group by hour
        hourly_counts = defaultdict(int)
        for metric in metrics:
            hour_key = metric.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_counts[hour_key] += 1

        if len(hourly_counts) < 2:
            return {'trend': 'insufficient_data'}

        sorted_hours = sorted(hourly_counts.keys())
        recent_hours = sorted_hours[-3:] if len(sorted_hours) >= 3 else sorted_hours
        earlier_hours = sorted_hours[:3] if len(sorted_hours) >= 6 else sorted_hours[:len(sorted_hours)//2]

        recent_avg = statistics.mean([hourly_counts[h] for h in recent_hours])
        earlier_avg = statistics.mean([hourly_counts[h] for h in earlier_hours])

        if recent_avg > earlier_avg * 1.1:
            trend = 'increasing'
        elif recent_avg < earlier_avg * 0.9:
            trend = 'decreasing'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'recent_hourly_avg': recent_avg,
            'earlier_hourly_avg': earlier_avg,
            'growth_rate': (recent_avg - earlier_avg) / earlier_avg if earlier_avg > 0 else 0
        }

    def _get_quality_alerts(self) -> List[Dict[str, Any]]:
        """Get current quality alerts."""
        alerts = []
        recent_cutoff = datetime.now() - timedelta(hours=1)

        recent_quality = [
            m for m in self._quality_metrics
            if m.timestamp >= recent_cutoff
        ]

        # Check for low quality scores
        for metric in recent_quality:
            if metric.quality_score < self.quality_threshold:
                alerts.append({
                    'type': 'low_quality',
                    'severity': 'warning',
                    'message': f"Low quality score for {metric.language}: {metric.quality_score:.2f}",
                    'timestamp': metric.timestamp,
                    'details': {
                        'language': metric.language,
                        'quality_score': metric.quality_score,
                        'model': metric.model_used
                    }
                })

        return alerts

    def _get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get current performance alerts."""
        alerts = []
        recent_cutoff = datetime.now() - timedelta(hours=1)

        recent_performance = [
            m for m in self._performance_metrics
            if m.timestamp >= recent_cutoff
        ]

        # Check for slow processing times
        for metric in recent_performance:
            if metric.processing_time > self.performance_threshold:
                alerts.append({
                    'type': 'slow_processing',
                    'severity': 'warning',
                    'message': f"Slow processing for {metric.symbol}: {metric.processing_time:.2f}s",
                    'timestamp': metric.timestamp,
                    'details': {
                        'symbol': metric.symbol,
                        'processing_time': metric.processing_time,
                        'memory_usage': metric.memory_usage_mb
                    }
                })

        # Check error rate
        if recent_performance:
            error_rate = sum(1 for m in recent_performance if not m.success) / len(recent_performance)
            if error_rate > self.error_rate_threshold:
                alerts.append({
                    'type': 'high_error_rate',
                    'severity': 'critical',
                    'message': f"High error rate: {error_rate:.1%}",
                    'timestamp': datetime.now(),
                    'details': {
                        'error_rate': error_rate,
                        'total_requests': len(recent_performance),
                        'failed_requests': sum(1 for m in recent_performance if not m.success)
                    }
                })

        return alerts

    def _get_critical_alerts(self) -> List[Dict[str, Any]]:
        """Get critical system alerts."""
        alerts = []

        # Check memory usage
        if self._current_session['peak_memory_usage'] > 1024:  # 1GB
            alerts.append({
                'type': 'high_memory_usage',
                'severity': 'critical',
                'message': f"High memory usage: {self._current_session['peak_memory_usage']:.1f}MB"
            })

        # Check error rate
        total_requests = self._current_session['requests_processed']
        if total_requests > 0:
            error_rate = self._current_session['errors_encountered'] / total_requests
            if error_rate > 0.2:  # 20% error rate
                alerts.append({
                    'type': 'critical_error_rate',
                    'severity': 'critical',
                    'message': f"Critical error rate: {error_rate:.1%}"
                })

        return alerts

    def _determine_system_health(self) -> str:
        """Determine overall system health status."""
        critical_alerts = self._get_critical_alerts()
        performance_alerts = self._get_performance_alerts()
        quality_alerts = self._get_quality_alerts()

        if critical_alerts:
            return 'critical'
        elif len(performance_alerts) > 5 or len(quality_alerts) > 10:
            return 'degraded'
        elif len(performance_alerts) > 0 or len(quality_alerts) > 0:
            return 'warning'
        else:
            return 'healthy'

    def reset_session_metrics(self) -> None:
        """Reset current session metrics."""
        with self._lock:
            self._current_session = {
                'start_time': datetime.now(),
                'requests_processed': 0,
                'errors_encountered': 0,
                'peak_memory_usage': 0.0,
                'cache_effectiveness': 0.0
            }
            logger.info("Session metrics reset")


    def cleanup_old_metrics(self, force: bool = False) -> Dict[str, int]:
        """Clean up old metrics data to prevent memory leaks."""
        if not self.enabled:
            return {'cleaned': 0, 'reason': 'metrics_disabled'}

        now = datetime.now()
        should_cleanup = (
            force or
            (now - self._last_cleanup).total_seconds() > (self.cleanup_interval_minutes * 60)
        )

        if not should_cleanup:
            return {'cleaned': 0, 'reason': 'not_time_yet'}

        cleaned_count = 0

        with self._lock:
            try:
                # Clean up inactive language stats
                inactive_languages = []
                for language, stats in self._language_stats.items():
                    last_cleanup = stats.get('last_cleanup', datetime.min)
                    if (now - last_cleanup).days > self.max_inactive_days:
                        inactive_languages.append(language)

                for language in inactive_languages:
                    del self._language_stats[language]
                    cleaned_count += 1

                # Clean up inactive model stats
                inactive_models = []
                for model, stats in self._model_stats.items():
                    last_used = stats.get('last_used', datetime.min)
                    if (now - last_used).days > self.max_inactive_days:
                        inactive_models.append(model)

                for model in inactive_models:
                    del self._model_stats[model]
                    cleaned_count += 1

                # Clean up old quality scores and history data
                for stats in self._language_stats.values():
                    # Quality scores are already bounded by deque maxlen
                    stats['last_cleanup'] = now

                for stats in self._model_stats.values():
                    # History collections are already bounded by deque maxlen
                    stats['last_used'] = now

                # Update cleanup tracking
                self._last_cleanup = now
                self._cleanup_counter += 1

                logger.info(f"Metrics cleanup completed: removed {cleaned_count} inactive entries")

                return {
                    'cleaned': cleaned_count,
                    'cleanup_count': self._cleanup_counter,
                    'language_stats_count': len(self._language_stats),
                    'model_stats_count': len(self._model_stats),
                    'quality_metrics_count': len(self._quality_metrics),
                    'performance_metrics_count': len(self._performance_metrics),
                    'usage_metrics_count': len(self._usage_metrics)
                }

            except Exception as e:
                logger.error(f"Error during metrics cleanup: {str(e)}")
                return {'cleaned': 0, 'error': str(e)}

    def get_memory_usage_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics for monitoring."""
        try:
            import sys

            return {
                'collections': {
                    'quality_metrics': {
                        'current_size': len(self._quality_metrics),
                        'max_size': self._quality_metrics.maxlen,
                        'memory_estimate_kb': sys.getsizeof(self._quality_metrics) / 1024
                    },
                    'performance_metrics': {
                        'current_size': len(self._performance_metrics),
                        'max_size': self._performance_metrics.maxlen,
                        'memory_estimate_kb': sys.getsizeof(self._performance_metrics) / 1024
                    },
                    'usage_metrics': {
                        'current_size': len(self._usage_metrics),
                        'max_size': self._usage_metrics.maxlen,
                        'memory_estimate_kb': sys.getsizeof(self._usage_metrics) / 1024
                    },
                    'language_stats': {
                        'count': len(self._language_stats),
                        'memory_estimate_kb': sys.getsizeof(self._language_stats) / 1024
                    },
                    'model_stats': {
                        'count': len(self._model_stats),
                        'memory_estimate_kb': sys.getsizeof(self._model_stats) / 1024
                    }
                },
                'cleanup': {
                    'last_cleanup': self._last_cleanup.isoformat(),
                    'cleanup_count': self._cleanup_counter,
                    'cleanup_interval_minutes': self.cleanup_interval_minutes,
                    'max_inactive_days': self.max_inactive_days
                },
                'configuration': {
                    'max_memory_samples': self.max_memory_samples,
                    'retention_days': self.retention_days,
                    'metrics_enabled': self.enabled
                }
            }

        except Exception as e:
            logger.warning(f"Error getting memory usage stats: {str(e)}")
            return {
                'error': str(e),
                'basic_counts': {
                    'quality_metrics': len(self._quality_metrics),
                    'performance_metrics': len(self._performance_metrics),
                    'usage_metrics': len(self._usage_metrics),
                    'language_stats': len(self._language_stats),
                    'model_stats': len(self._model_stats)
                }
            }

    def trigger_periodic_cleanup(self) -> None:
        """Trigger periodic cleanup if needed (called during metric recording)."""
        try:
            # Only check cleanup every 100 metric recordings to avoid overhead
            if self._cleanup_counter % 100 == 0:
                self.cleanup_old_metrics()
        except Exception as e:
            logger.warning(f"Error in periodic cleanup trigger: {str(e)}")

    def reset_all_metrics(self) -> Dict[str, Any]:
        """Reset all metrics (for testing or emergency cleanup)."""
        if not self.enabled:
            return {'status': 'metrics_disabled'}

        old_counts = {
            'quality_metrics': len(self._quality_metrics),
            'performance_metrics': len(self._performance_metrics),
            'usage_metrics': len(self._usage_metrics),
            'language_stats': len(self._language_stats),
            'model_stats': len(self._model_stats)
        }

        with self._lock:
            try:
                self._quality_metrics.clear()
                self._performance_metrics.clear()
                self._usage_metrics.clear()
                self._language_stats.clear()
                self._model_stats.clear()

                # Reset session data
                self._current_session = {
                    'start_time': datetime.now(),
                    'requests_processed': 0,
                    'errors_encountered': 0,
                    'peak_memory_usage': 0.0,
                    'cache_effectiveness': 0.0
                }

                # Reset cleanup tracking
                self._last_cleanup = datetime.now()
                self._cleanup_counter = 0

                logger.info("All metrics have been reset")

                return {
                    'status': 'success',
                    'reset_time': datetime.now().isoformat(),
                    'old_counts': old_counts,
                    'new_counts': {
                        'quality_metrics': 0,
                        'performance_metrics': 0,
                        'usage_metrics': 0,
                        'language_stats': 0,
                        'model_stats': 0
                    }
                }

            except Exception as e:
                logger.error(f"Error resetting metrics: {str(e)}")
                return {'status': 'error', 'error': str(e)}


# Global metrics instance
_metrics_instance = None
_metrics_lock = threading.Lock()


def get_multilingual_metrics() -> MultilingualMetrics:
    """Get the global multilingual metrics instance."""
    global _metrics_instance

    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = MultilingualMetrics()

    return _metrics_instance
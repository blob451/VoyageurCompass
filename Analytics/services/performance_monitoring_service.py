"""
Comprehensive performance monitoring and observability service.
Aggregates metrics from all system components for centralized monitoring.
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

from django.core.cache import cache
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric with metadata."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    service: str
    category: str
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata or {}
        }


@dataclass
class SystemHealthStatus:
    """Overall system health status."""
    overall_health: str  # healthy, degraded, unhealthy
    services_status: Dict[str, str]
    critical_issues: List[str]
    warnings: List[str]
    last_updated: datetime
    uptime: float
    total_requests: int
    error_rate: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            **asdict(self),
            'last_updated': self.last_updated.isoformat()
        }


class PerformanceMonitoringService:
    """Centralized performance monitoring and observability service."""

    def __init__(self):
        self.start_time = timezone.now()
        self.metrics_buffer: List[PerformanceMetric] = []
        self.buffer_size = getattr(settings, "MONITORING_BUFFER_SIZE", 1000)
        self.metrics_lock = threading.Lock()

        # Health thresholds
        self.health_thresholds = {
            "response_time_ms": 5000,  # 5 seconds max response time
            "error_rate_percent": 5.0,  # 5% max error rate
            "memory_usage_percent": 85.0,  # 85% max memory usage
            "cpu_usage_percent": 80.0,  # 80% max CPU usage
            "cache_hit_rate_percent": 70.0,  # 70% min cache hit rate
        }

        # Service registry for monitoring
        self.monitored_services = {
            "llm_service": self._get_llm_metrics,
            "batch_analysis": self._get_batch_analysis_metrics,
            "async_pipeline": self._get_async_pipeline_metrics,
            "connection_pools": self._get_connection_pool_metrics,
            "model_warmup": self._get_model_warmup_metrics,
            "cache_manager": self._get_cache_metrics,
            "resource_manager": self._get_resource_manager_metrics,
            "translation_service": self._get_translation_metrics,
        }

        # Performance alerts configuration
        self.alerts_enabled = getattr(settings, "MONITORING_ALERTS_ENABLED", True)
        self.alert_cooldown = getattr(settings, "MONITORING_ALERT_COOLDOWN", 300)  # 5 minutes
        self.last_alerts = {}

        # Metrics retention
        self.metrics_retention_hours = getattr(settings, "MONITORING_RETENTION_HOURS", 24)

        logger.info("Performance monitoring service initialized")

    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all monitored services."""
        current_time = timezone.now()
        all_metrics = {
            "timestamp": current_time.isoformat(),
            "system_uptime": (current_time - self.start_time).total_seconds(),
            "services": {}
        }

        # Collect metrics from each service
        for service_name, collector_func in self.monitored_services.items():
            try:
                service_metrics = collector_func()
                all_metrics["services"][service_name] = service_metrics

                # Add to metrics buffer
                self._add_service_metrics_to_buffer(service_name, service_metrics, current_time)

            except Exception as e:
                logger.error(f"Failed to collect metrics from {service_name}: {str(e)}")
                all_metrics["services"][service_name] = {"error": str(e)}

        # Add system-level metrics
        all_metrics["system"] = self._get_system_metrics()

        return all_metrics

    def get_health_status(self) -> SystemHealthStatus:
        """Get comprehensive system health status."""
        current_metrics = self.collect_all_metrics()
        critical_issues = []
        warnings = []
        services_status = {}

        # Analyze each service health
        for service_name, service_metrics in current_metrics.get("services", {}).items():
            if "error" in service_metrics:
                services_status[service_name] = "unhealthy"
                critical_issues.append(f"{service_name}: {service_metrics['error']}")
                continue

            # Check service-specific health indicators
            service_health = self._assess_service_health(service_name, service_metrics)
            services_status[service_name] = service_health["status"]

            if service_health["status"] == "unhealthy":
                critical_issues.extend(service_health["issues"])
            elif service_health["status"] == "degraded":
                warnings.extend(service_health["issues"])

        # Determine overall health
        unhealthy_services = [s for s in services_status.values() if s == "unhealthy"]
        degraded_services = [s for s in services_status.values() if s == "degraded"]

        if unhealthy_services:
            overall_health = "unhealthy"
        elif degraded_services:
            overall_health = "degraded"
        else:
            overall_health = "healthy"

        # Calculate system-wide metrics
        total_requests = self._calculate_total_requests(current_metrics)
        error_rate = self._calculate_error_rate(current_metrics)

        return SystemHealthStatus(
            overall_health=overall_health,
            services_status=services_status,
            critical_issues=critical_issues,
            warnings=warnings,
            last_updated=timezone.now(),
            uptime=(timezone.now() - self.start_time).total_seconds(),
            total_requests=total_requests,
            error_rate=error_rate
        )

    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data."""
        health_status = self.get_health_status()
        recent_metrics = self.get_recent_metrics(hours=1)

        # Calculate performance trends
        trends = self._calculate_performance_trends(recent_metrics)

        # Get top performance issues
        issues = self._identify_performance_issues(recent_metrics)

        return {
            "health_status": health_status.to_dict(),
            "performance_trends": trends,
            "recent_metrics": recent_metrics,
            "performance_issues": issues,
            "service_overview": self._get_service_overview(),
            "recommendations": self._generate_performance_recommendations(health_status, trends)
        }

    def get_recent_metrics(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get metrics from the last N hours."""
        cutoff_time = timezone.now() - timedelta(hours=hours)

        with self.metrics_lock:
            recent_metrics = [
                metric.to_dict()
                for metric in self.metrics_buffer
                if metric.timestamp > cutoff_time
            ]

        return sorted(recent_metrics, key=lambda x: x['timestamp'], reverse=True)

    def record_custom_metric(
        self,
        name: str,
        value: float,
        unit: str,
        service: str,
        category: str = "custom",
        metadata: Dict[str, Any] = None
    ):
        """Record a custom performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=timezone.now(),
            service=service,
            category=category,
            metadata=metadata or {}
        )

        with self.metrics_lock:
            self.metrics_buffer.append(metric)
            # Trim buffer if it exceeds size limit
            if len(self.metrics_buffer) > self.buffer_size:
                self.metrics_buffer = self.metrics_buffer[-self.buffer_size:]

        # Store in cache for persistence
        self._store_metric_in_cache(metric)

    def _get_llm_metrics(self) -> Dict[str, Any]:
        """Get LLM service metrics."""
        try:
            from Analytics.services.local_llm_service import get_local_llm_service
            llm_service = get_local_llm_service()

            status = llm_service.get_service_status()

            return {
                "current_model": status.get("current_model"),
                "available_models": status.get("available_models", []),
                "circuit_breaker_state": status.get("circuit_breaker", {}).get("state"),
                "total_requests": status.get("performance_monitor", {}).get("total_requests", 0),
                "successful_requests": status.get("performance_monitor", {}).get("successful_requests", 0),
                "failed_requests": status.get("performance_monitor", {}).get("failed_requests", 0),
                "average_response_time": status.get("performance_monitor", {}).get("average_response_time", 0),
                "ollama_available": status.get("ollama_available", False),
                "health_status": "healthy" if status.get("ollama_available") else "unhealthy"
            }
        except Exception as e:
            logger.error(f"Failed to get LLM metrics: {str(e)}")
            return {"error": str(e)}

    def _get_batch_analysis_metrics(self) -> Dict[str, Any]:
        """Get batch analysis service metrics."""
        try:
            from Analytics.services.batch_analysis_service import get_batch_analysis_service
            batch_service = get_batch_analysis_service()

            stats = batch_service.get_batch_performance_stats()

            return {
                "total_batches": stats.get("total_batches", 0),
                "successful_analyses": stats.get("successful_analyses", 0),
                "failed_analyses": stats.get("failed_analyses", 0),
                "average_time_per_stock": stats.get("average_time_per_stock", 0),
                "cache_hits": stats.get("cache_hits", 0),
                "cache_hit_rate": stats.get("cache_hit_rate", 0),
                "throughput_stocks_per_second": stats.get("throughput_stocks_per_second", 0),
                "health_status": "healthy"
            }
        except Exception as e:
            logger.error(f"Failed to get batch analysis metrics: {str(e)}")
            return {"error": str(e)}

    def _get_async_pipeline_metrics(self) -> Dict[str, Any]:
        """Get async processing pipeline metrics."""
        try:
            from Analytics.services.async_processing_pipeline import get_async_processing_pipeline
            pipeline = get_async_processing_pipeline()

            performance = pipeline.get_performance_summary()

            return {
                "max_workers": performance.get("max_workers", 0),
                "total_batch_requests": performance.get("total_batch_requests", 0),
                "successful_batch_requests": performance.get("successful_batch_requests", 0),
                "batch_success_rate": performance.get("batch_success_rate", 0),
                "average_batch_time": performance.get("average_batch_time", 0),
                "current_tasks": performance.get("current_tasks", 0),
                "executor_active": performance.get("executor_active", False),
                "health_status": "healthy" if performance.get("executor_active") else "degraded"
            }
        except Exception as e:
            logger.error(f"Failed to get async pipeline metrics: {str(e)}")
            return {"error": str(e)}

    def _get_connection_pool_metrics(self) -> Dict[str, Any]:
        """Get connection pool metrics."""
        try:
            from Analytics.services.connection_pool_manager import get_connection_pool_manager
            connection_manager = get_connection_pool_manager()

            stats = connection_manager.get_comprehensive_stats()

            return {
                "database_stats": stats.get("database", {}),
                "redis_stats": stats.get("redis", {}),
                "http_stats": stats.get("http", {}),
                "health_status": "healthy"
            }
        except Exception as e:
            logger.error(f"Failed to get connection pool metrics: {str(e)}")
            return {"error": str(e)}

    def _get_model_warmup_metrics(self) -> Dict[str, Any]:
        """Get model warmup service metrics."""
        try:
            from Analytics.services.model_warmup_service import get_model_warmup_service
            warmup_service = get_model_warmup_service()

            metrics = warmup_service.get_performance_metrics()
            status = warmup_service.get_warmup_status()

            return {
                **metrics,
                "warmup_status": status,
                "health_status": "healthy"
            }
        except Exception as e:
            logger.error(f"Failed to get model warmup metrics: {str(e)}")
            return {"error": str(e)}

    def _get_cache_metrics(self) -> Dict[str, Any]:
        """Get cache manager metrics."""
        try:
            from Analytics.services.cache_manager import get_cache_manager
            cache_manager = get_cache_manager()

            performance = cache_manager.get_performance_metrics()

            return {
                **performance,
                "health_status": "healthy" if performance.get("hit_rate", 0) > 0.5 else "degraded"
            }
        except Exception as e:
            logger.error(f"Failed to get cache metrics: {str(e)}")
            return {"error": str(e)}

    def _get_resource_manager_metrics(self) -> Dict[str, Any]:
        """Get resource manager metrics."""
        try:
            from Analytics.services.model_resource_manager import get_model_resource_manager
            resource_manager = get_model_resource_manager()

            if resource_manager:
                status = resource_manager.get_resource_status()
                return {
                    **status,
                    "health_status": "healthy" if status.get("healthy", False) else "degraded"
                }
            else:
                return {"health_status": "not_available"}
        except Exception as e:
            logger.error(f"Failed to get resource manager metrics: {str(e)}")
            return {"error": str(e)}

    def _get_translation_metrics(self) -> Dict[str, Any]:
        """Get translation service metrics."""
        try:
            from Analytics.services.translation_service import get_translation_service
            translation_service = get_translation_service()

            metrics = translation_service.get_performance_metrics()

            return {
                **metrics,
                "health_status": "healthy"
            }
        except Exception as e:
            logger.error(f"Failed to get translation metrics: {str(e)}")
            return {"error": str(e)}

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics."""
        import psutil

        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
                "process_count": len(psutil.pids()),
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {str(e)}")
            return {"error": str(e)}

    def _assess_service_health(self, service_name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the health of a specific service."""
        if "error" in metrics:
            return {
                "status": "unhealthy",
                "issues": [f"{service_name} error: {metrics['error']}"]
            }

        issues = []
        warnings = []

        # Check service-specific health indicators
        if service_name == "llm_service":
            if not metrics.get("ollama_available", False):
                issues.append("Ollama service not available")

            error_rate = self._calculate_service_error_rate(metrics)
            if error_rate > 10:  # 10% error rate threshold
                issues.append(f"High error rate: {error_rate:.1f}%")

        elif service_name == "async_pipeline":
            if not metrics.get("executor_active", False):
                warnings.append("Thread executor not active")

        elif service_name == "cache_manager":
            hit_rate = metrics.get("hit_rate", 0)
            if hit_rate < 0.5:  # 50% hit rate threshold
                warnings.append(f"Low cache hit rate: {hit_rate:.1%}")

        # Determine status
        if issues:
            status = "unhealthy"
        elif warnings:
            status = "degraded"
        else:
            status = "healthy"

        return {
            "status": status,
            "issues": issues + warnings
        }

    def _calculate_service_error_rate(self, metrics: Dict[str, Any]) -> float:
        """Calculate error rate for a service."""
        total = metrics.get("total_requests", 0) or metrics.get("total_batches", 0)
        failed = metrics.get("failed_requests", 0) or metrics.get("failed_analyses", 0)

        if total == 0:
            return 0.0

        return (failed / total) * 100

    def _calculate_total_requests(self, metrics: Dict[str, Any]) -> int:
        """Calculate total requests across all services."""
        total = 0
        for service_metrics in metrics.get("services", {}).values():
            if isinstance(service_metrics, dict):
                total += service_metrics.get("total_requests", 0)
                total += service_metrics.get("total_batches", 0)
        return total

    def _calculate_error_rate(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall system error rate."""
        total_requests = 0
        total_errors = 0

        for service_metrics in metrics.get("services", {}).values():
            if isinstance(service_metrics, dict):
                requests = service_metrics.get("total_requests", 0) or service_metrics.get("total_batches", 0)
                errors = service_metrics.get("failed_requests", 0) or service_metrics.get("failed_analyses", 0)

                total_requests += requests
                total_errors += errors

        if total_requests == 0:
            return 0.0

        return (total_errors / total_requests) * 100

    def _add_service_metrics_to_buffer(self, service_name: str, metrics: Dict[str, Any], timestamp: datetime):
        """Add service metrics to the buffer for trend analysis."""
        if "error" in metrics:
            return

        # Extract key metrics and add to buffer
        key_metrics = self._extract_key_metrics(service_name, metrics)

        with self.metrics_lock:
            for metric_name, value in key_metrics.items():
                if isinstance(value, (int, float)):
                    metric = PerformanceMetric(
                        name=metric_name,
                        value=float(value),
                        unit=self._get_metric_unit(metric_name),
                        timestamp=timestamp,
                        service=service_name,
                        category="performance"
                    )
                    self.metrics_buffer.append(metric)

    def _extract_key_metrics(self, service_name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics for trend analysis."""
        key_metrics = {}

        # Common metrics
        for key in ["total_requests", "successful_requests", "failed_requests",
                   "average_response_time", "throughput", "hit_rate"]:
            if key in metrics:
                key_metrics[key] = metrics[key]

        # Service-specific metrics
        if service_name == "llm_service":
            key_metrics.update({k: v for k, v in metrics.items()
                              if k in ["average_response_time", "total_requests"]})

        return key_metrics

    def _get_metric_unit(self, metric_name: str) -> str:
        """Get unit for a metric name."""
        unit_mapping = {
            "average_response_time": "seconds",
            "total_requests": "count",
            "successful_requests": "count",
            "failed_requests": "count",
            "throughput": "requests/second",
            "hit_rate": "percentage",
            "cpu_percent": "percentage",
            "memory_percent": "percentage",
        }
        return unit_mapping.get(metric_name, "count")

    def _store_metric_in_cache(self, metric: PerformanceMetric):
        """Store metric in cache for persistence."""
        try:
            cache_key = f"performance_metric:{metric.service}:{metric.name}:{int(metric.timestamp.timestamp())}"
            cache.set(cache_key, metric.to_dict(), 86400)  # 24 hours
        except Exception as e:
            logger.error(f"Failed to store metric in cache: {str(e)}")

    def _calculate_performance_trends(self, recent_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance trends from recent metrics."""
        trends = {}

        # Group metrics by service and name
        metric_groups = {}
        for metric in recent_metrics:
            key = f"{metric['service']}.{metric['name']}"
            if key not in metric_groups:
                metric_groups[key] = []
            metric_groups[key].append(metric)

        # Calculate trends for each metric
        for key, metrics_list in metric_groups.items():
            if len(metrics_list) >= 2:
                # Sort by timestamp
                sorted_metrics = sorted(metrics_list, key=lambda x: x['timestamp'])
                latest = sorted_metrics[-1]['value']
                previous = sorted_metrics[-2]['value']

                if previous != 0:
                    change_percent = ((latest - previous) / previous) * 100
                    trends[key] = {
                        "latest_value": latest,
                        "previous_value": previous,
                        "change_percent": change_percent,
                        "trend": "improving" if change_percent > 0 else "declining" if change_percent < 0 else "stable"
                    }

        return trends

    def _identify_performance_issues(self, recent_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify performance issues from recent metrics."""
        issues = []

        # Check for high response times
        response_time_metrics = [m for m in recent_metrics if "response_time" in m['name']]
        for metric in response_time_metrics[-5:]:  # Last 5 measurements
            if metric['value'] > 5.0:  # 5 seconds threshold
                issues.append({
                    "type": "high_response_time",
                    "service": metric['service'],
                    "value": metric['value'],
                    "threshold": 5.0,
                    "severity": "warning"
                })

        # Check for high error rates
        error_metrics = [m for m in recent_metrics if "error" in m['name'] or "failed" in m['name']]
        for metric in error_metrics[-5:]:
            if metric['value'] > 10:  # 10% error rate threshold
                issues.append({
                    "type": "high_error_rate",
                    "service": metric['service'],
                    "value": metric['value'],
                    "threshold": 10,
                    "severity": "critical"
                })

        return issues

    def _get_service_overview(self) -> Dict[str, Any]:
        """Get overview of all monitored services."""
        return {
            "total_services": len(self.monitored_services),
            "services": list(self.monitored_services.keys()),
            "monitoring_start_time": self.start_time.isoformat(),
            "uptime": (timezone.now() - self.start_time).total_seconds()
        }

    def _generate_performance_recommendations(
        self,
        health_status: SystemHealthStatus,
        trends: Dict[str, Any]
    ) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []

        # Check overall health
        if health_status.overall_health == "unhealthy":
            recommendations.append("System is unhealthy - immediate attention required")

        # Check error rate
        if health_status.error_rate > 5:
            recommendations.append("High error rate detected - investigate failing services")

        # Check service-specific issues
        for service, status in health_status.services_status.items():
            if status == "unhealthy":
                recommendations.append(f"Service {service} is unhealthy - check logs and configuration")

        # Check trends for declining performance
        declining_trends = [k for k, v in trends.items() if v.get("trend") == "declining"]
        if declining_trends:
            recommendations.append(f"Performance declining in: {', '.join(declining_trends)}")

        return recommendations


# Singleton instance
_performance_monitoring_service = None


def get_performance_monitoring_service() -> PerformanceMonitoringService:
    """Get singleton instance of PerformanceMonitoringService."""
    global _performance_monitoring_service
    if _performance_monitoring_service is None:
        _performance_monitoring_service = PerformanceMonitoringService()
    return _performance_monitoring_service
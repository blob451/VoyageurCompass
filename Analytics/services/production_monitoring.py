"""
Production monitoring service for multilingual LLM system.

Provides comprehensive system health monitoring, alerting, and performance tracking
for production environments with automated incident detection and reporting.
"""

import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from threading import Thread, Event
import queue
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from django.conf import settings
from django.core.cache import cache
from django.core.mail import send_mail

from Analytics.services.circuit_breaker import get_all_circuit_breaker_stats
from Analytics.services.feature_flags import get_feature_flags
from Analytics.services.multilingual_metrics import get_multilingual_metrics

logger = logging.getLogger(__name__)


class AlertLevel:
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ProductionAlert:
    """Production alert with context and metadata."""

    def __init__(self, level: str, title: str, message: str, context: Dict[str, Any] = None):
        self.level = level
        self.title = title
        self.message = message
        self.context = context or {}
        self.timestamp = datetime.now()
        self.alert_id = f"{int(time.time())}_{hash(f'{title}_{message}') % 10000}"


class ProductionMonitoringService:
    """
    Comprehensive production monitoring service for multilingual LLM system.
    """

    def __init__(self):
        """Initialize production monitoring service."""
        self.monitoring_enabled = getattr(settings, 'PRODUCTION_MONITORING_ENABLED', True)
        self.check_interval = getattr(settings, 'MONITORING_CHECK_INTERVAL', 60)  # seconds
        self.alert_cooldown = getattr(settings, 'ALERT_COOLDOWN_SECONDS', 300)  # 5 minutes

        # System thresholds
        self.cpu_threshold = getattr(settings, 'CPU_ALERT_THRESHOLD', 80)  # %
        self.memory_threshold = getattr(settings, 'MEMORY_ALERT_THRESHOLD', 85)  # %
        self.disk_threshold = getattr(settings, 'DISK_ALERT_THRESHOLD', 90)  # %
        self.response_time_threshold = getattr(settings, 'RESPONSE_TIME_THRESHOLD', 10)  # seconds
        self.error_rate_threshold = getattr(settings, 'ERROR_RATE_THRESHOLD', 0.1)  # 10%

        # Alert management
        self.alert_queue = queue.Queue()
        self.recent_alerts = {}  # For cooldown management
        self.alert_cache_ttl = 3600  # 1 hour

        # Monitoring thread control
        self.monitoring_thread = None
        self.stop_event = Event()

        # Email settings
        self.email_alerts_enabled = getattr(settings, 'EMAIL_ALERTS_ENABLED', False)
        self.alert_recipients = getattr(settings, 'ALERT_EMAIL_RECIPIENTS', [])

        # Slack/webhook settings
        self.webhook_alerts_enabled = getattr(settings, 'WEBHOOK_ALERTS_ENABLED', False)
        self.webhook_url = getattr(settings, 'ALERT_WEBHOOK_URL', '')

        logger.info(f"Production monitoring service initialized (enabled: {self.monitoring_enabled})")

    def start_monitoring(self) -> None:
        """Start the production monitoring background thread."""
        if not self.monitoring_enabled:
            logger.info("Production monitoring disabled")
            return

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread already running")
            return

        self.stop_event.clear()
        self.monitoring_thread = Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        logger.info("Production monitoring started")

    def stop_monitoring(self) -> None:
        """Stop the production monitoring background thread."""
        if self.monitoring_thread:
            self.stop_event.set()
            self.monitoring_thread.join(timeout=5)
            logger.info("Production monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop that runs checks periodically."""
        while not self.stop_event.wait(self.check_interval):
            try:
                self._run_health_checks()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}", exc_info=True)

    def _run_health_checks(self) -> None:
        """Run comprehensive health checks and generate alerts."""
        checks = [
            self._check_system_resources,
            self._check_multilingual_health,
            self._check_circuit_breakers,
            self._check_feature_flags,
            self._check_performance_metrics,
            self._check_cache_health,
        ]

        for check in checks:
            try:
                check()
            except Exception as e:
                self._generate_alert(
                    AlertLevel.ERROR,
                    f"Health check failed: {check.__name__}",
                    f"Health check error: {str(e)}",
                    {"check_name": check.__name__, "error": str(e)}
                )

    def _check_system_resources(self) -> None:
        """Check system resource usage."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > self.cpu_threshold:
            self._generate_alert(
                AlertLevel.WARNING if cpu_percent < self.cpu_threshold + 10 else AlertLevel.ERROR,
                "High CPU usage detected",
                f"CPU usage: {cpu_percent:.1f}% (threshold: {self.cpu_threshold}%)",
                {"cpu_percent": cpu_percent, "threshold": self.cpu_threshold}
            )

        # Memory usage
        memory = psutil.virtual_memory()
        if memory.percent > self.memory_threshold:
            self._generate_alert(
                AlertLevel.WARNING if memory.percent < self.memory_threshold + 5 else AlertLevel.ERROR,
                "High memory usage detected",
                f"Memory usage: {memory.percent:.1f}% (threshold: {self.memory_threshold}%)",
                {
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "threshold": self.memory_threshold
                }
            )

        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        if disk_percent > self.disk_threshold:
            self._generate_alert(
                AlertLevel.WARNING if disk_percent < self.disk_threshold + 5 else AlertLevel.CRITICAL,
                "High disk usage detected",
                f"Disk usage: {disk_percent:.1f}% (threshold: {self.disk_threshold}%)",
                {
                    "disk_percent": disk_percent,
                    "disk_free_gb": disk.free / (1024**3),
                    "threshold": self.disk_threshold
                }
            )

    def _check_multilingual_health(self) -> None:
        """Check multilingual service health."""
        try:
            metrics_service = get_multilingual_metrics()
            current_metrics = metrics_service.get_real_time_dashboard()

            if "current_rates" in current_metrics:
                rates = current_metrics["current_rates"]

                # Check response time
                avg_response_time = rates.get("avg_response_time", 0)
                if avg_response_time > self.response_time_threshold:
                    self._generate_alert(
                        AlertLevel.WARNING,
                        "High multilingual response time",
                        f"Average response time: {avg_response_time:.2f}s (threshold: {self.response_time_threshold}s)",
                        {"avg_response_time": avg_response_time, "threshold": self.response_time_threshold}
                    )

                # Check error rate
                error_rate = rates.get("error_rate", 0)
                if error_rate > self.error_rate_threshold:
                    self._generate_alert(
                        AlertLevel.ERROR,
                        "High multilingual error rate",
                        f"Error rate: {error_rate:.2%} (threshold: {self.error_rate_threshold:.2%})",
                        {"error_rate": error_rate, "threshold": self.error_rate_threshold}
                    )

                # Check quality scores
                quality_score = rates.get("quality_score", 1.0)
                if quality_score < 0.7:
                    self._generate_alert(
                        AlertLevel.WARNING,
                        "Low multilingual quality score",
                        f"Quality score: {quality_score:.2f} (below 0.7)",
                        {"quality_score": quality_score}
                    )

        except Exception as e:
            self._generate_alert(
                AlertLevel.ERROR,
                "Multilingual metrics check failed",
                f"Unable to retrieve multilingual metrics: {str(e)}",
                {"error": str(e)}
            )

    def _check_circuit_breakers(self) -> None:
        """Check circuit breaker states."""
        try:
            circuit_stats = get_all_circuit_breaker_stats()

            if "circuit_breakers" in circuit_stats:
                for breaker_name, breaker_info in circuit_stats["circuit_breakers"].items():
                    if "languages" in breaker_info:
                        for lang, lang_stats in breaker_info["languages"].items():
                            state = lang_stats.get("state", "unknown")

                            if state == "open":
                                self._generate_alert(
                                    AlertLevel.ERROR,
                                    f"Circuit breaker OPEN for {lang}",
                                    f"Circuit breaker for language '{lang}' is in OPEN state",
                                    {
                                        "language": lang,
                                        "breaker_name": breaker_name,
                                        "state": state,
                                        "failure_count": lang_stats.get("failure_count", 0)
                                    }
                                )
                            elif state == "half_open":
                                self._generate_alert(
                                    AlertLevel.WARNING,
                                    f"Circuit breaker HALF-OPEN for {lang}",
                                    f"Circuit breaker for language '{lang}' is in testing mode",
                                    {
                                        "language": lang,
                                        "breaker_name": breaker_name,
                                        "state": state
                                    }
                                )

        except Exception as e:
            self._generate_alert(
                AlertLevel.WARNING,
                "Circuit breaker check failed",
                f"Unable to check circuit breaker states: {str(e)}",
                {"error": str(e)}
            )

    def _check_feature_flags(self) -> None:
        """Check feature flag states for anomalies."""
        try:
            feature_flags = get_feature_flags()
            flags_status = feature_flags.get_all_flags_status()

            emergency_status = flags_status.get("emergency_status", {})

            # Check for emergency fallback
            if emergency_status.get("emergency_fallback_enabled", False):
                self._generate_alert(
                    AlertLevel.CRITICAL,
                    "Emergency fallback enabled",
                    "Multilingual services are in emergency fallback mode",
                    {"emergency_status": emergency_status}
                )

            # Check for disabled core features
            flags = flags_status.get("flags", {})
            core_flags = [
                "multilingual_llm_enabled",
                "french_generation_enabled",
                "spanish_generation_enabled"
            ]

            for flag_name in core_flags:
                if flag_name in flags and not flags[flag_name].get("enabled", True):
                    self._generate_alert(
                        AlertLevel.WARNING,
                        f"Core feature disabled: {flag_name}",
                        f"Core feature flag '{flag_name}' is disabled",
                        {"flag_name": flag_name, "flag_info": flags[flag_name]}
                    )

        except Exception as e:
            self._generate_alert(
                AlertLevel.WARNING,
                "Feature flags check failed",
                f"Unable to check feature flag states: {str(e)}",
                {"error": str(e)}
            )

    def _check_performance_metrics(self) -> None:
        """Check overall system performance metrics."""
        try:
            # Check request rate anomalies
            metrics_service = get_multilingual_metrics()
            current_metrics = metrics_service.get_real_time_dashboard()

            if "current_rates" in current_metrics:
                rates = current_metrics["current_rates"]
                requests_per_minute = rates.get("requests_per_minute", 0)

                # Alert on extremely low activity (possible service issues)
                if requests_per_minute < 1 and self._is_business_hours():
                    self._generate_alert(
                        AlertLevel.WARNING,
                        "Low request activity detected",
                        f"Request rate: {requests_per_minute} requests/minute during business hours",
                        {"requests_per_minute": requests_per_minute}
                    )

                # Alert on extremely high activity (possible DoS or traffic spike)
                if requests_per_minute > 1000:
                    self._generate_alert(
                        AlertLevel.WARNING,
                        "High request activity detected",
                        f"Request rate: {requests_per_minute} requests/minute (unusually high)",
                        {"requests_per_minute": requests_per_minute}
                    )

        except Exception as e:
            logger.warning(f"Performance metrics check failed: {str(e)}")

    def _check_cache_health(self) -> None:
        """Check cache system health."""
        try:
            # Test cache connectivity
            test_key = "monitoring_cache_test"
            test_value = f"test_{int(time.time())}"

            cache.set(test_key, test_value, 30)
            retrieved_value = cache.get(test_key)

            if retrieved_value != test_value:
                self._generate_alert(
                    AlertLevel.ERROR,
                    "Cache system failure",
                    "Cache read/write test failed",
                    {"test_key": test_key, "expected": test_value, "actual": retrieved_value}
                )
            else:
                cache.delete(test_key)

        except Exception as e:
            self._generate_alert(
                AlertLevel.ERROR,
                "Cache connectivity failure",
                f"Unable to access cache system: {str(e)}",
                {"error": str(e)}
            )

    def _is_business_hours(self) -> bool:
        """Check if current time is during business hours (9 AM - 6 PM)."""
        now = datetime.now()
        return 9 <= now.hour < 18 and now.weekday() < 5  # Monday = 0

    def _generate_alert(self, level: str, title: str, message: str, context: Dict[str, Any] = None) -> None:
        """Generate and process an alert."""
        alert = ProductionAlert(level, title, message, context)

        # Check cooldown to prevent spam
        alert_key = f"{title}_{level}"
        last_alert_time = self.recent_alerts.get(alert_key)

        if last_alert_time and (datetime.now() - last_alert_time).total_seconds() < self.alert_cooldown:
            return

        # Update cooldown
        self.recent_alerts[alert_key] = datetime.now()

        # Log alert
        log_level = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical
        }.get(level, logger.info)

        log_level(f"ALERT [{level.upper()}] {title}: {message}")

        # Store alert for retrieval
        self._store_alert(alert)

        # Send notifications
        if level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
            self._send_alert_notifications(alert)

    def _store_alert(self, alert: ProductionAlert) -> None:
        """Store alert in cache for retrieval."""
        try:
            cache_key = f"monitoring_alert:{alert.alert_id}"
            alert_data = {
                "id": alert.alert_id,
                "level": alert.level,
                "title": alert.title,
                "message": alert.message,
                "context": alert.context,
                "timestamp": alert.timestamp.isoformat()
            }
            cache.set(cache_key, alert_data, self.alert_cache_ttl)

            # Also store in recent alerts list
            recent_alerts_key = "monitoring_recent_alerts"
            recent_alerts = cache.get(recent_alerts_key, [])
            recent_alerts.insert(0, alert_data)

            # Keep only last 100 alerts
            recent_alerts = recent_alerts[:100]
            cache.set(recent_alerts_key, recent_alerts, self.alert_cache_ttl)

        except Exception as e:
            logger.error(f"Failed to store alert: {str(e)}")

    def _send_alert_notifications(self, alert: ProductionAlert) -> None:
        """Send alert notifications via configured channels."""
        # Email notifications
        if self.email_alerts_enabled and self.alert_recipients:
            self._send_email_alert(alert)

        # Webhook notifications
        if self.webhook_alerts_enabled and self.webhook_url:
            self._send_webhook_alert(alert)

    def _send_email_alert(self, alert: ProductionAlert) -> None:
        """Send alert via email."""
        try:
            subject = f"[VoyageurCompass] {alert.level.upper()}: {alert.title}"

            message = f"""
Production Alert - VoyageurCompass Multilingual LLM System

Level: {alert.level.upper()}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Title: {alert.title}

Message:
{alert.message}

Context:
{alert.context}

Alert ID: {alert.alert_id}
            """.strip()

            send_mail(
                subject,
                message,
                settings.DEFAULT_FROM_EMAIL,
                self.alert_recipients,
                fail_silently=False
            )

            logger.info(f"Email alert sent for: {alert.title}")

        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")

    def _send_webhook_alert(self, alert: ProductionAlert) -> None:
        """Send alert via webhook (Slack, etc.)."""
        try:
            import requests

            payload = {
                "alert_id": alert.alert_id,
                "level": alert.level,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "context": alert.context,
                "service": "VoyageurCompass Multilingual LLM"
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()

            logger.info(f"Webhook alert sent for: {alert.title}")

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {str(e)}")

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts for dashboard display."""
        try:
            recent_alerts_key = "monitoring_recent_alerts"
            recent_alerts = cache.get(recent_alerts_key, [])
            return recent_alerts[:limit]
        except Exception as e:
            logger.error(f"Failed to retrieve recent alerts: {str(e)}")
            return []

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring service status."""
        return {
            "monitoring_enabled": self.monitoring_enabled,
            "thread_running": self.monitoring_thread and self.monitoring_thread.is_alive(),
            "check_interval": self.check_interval,
            "last_check": datetime.now().isoformat(),
            "thresholds": {
                "cpu_threshold": self.cpu_threshold,
                "memory_threshold": self.memory_threshold,
                "disk_threshold": self.disk_threshold,
                "response_time_threshold": self.response_time_threshold,
                "error_rate_threshold": self.error_rate_threshold
            },
            "notification_channels": {
                "email_enabled": self.email_alerts_enabled,
                "webhook_enabled": self.webhook_alerts_enabled
            }
        }

    def force_health_check(self) -> Dict[str, Any]:
        """Force an immediate health check and return results."""
        start_time = time.time()

        try:
            self._run_health_checks()

            return {
                "status": "completed",
                "check_duration": round(time.time() - start_time, 2),
                "timestamp": datetime.now().isoformat(),
                "recent_alerts": self.get_recent_alerts(10)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "check_duration": round(time.time() - start_time, 2),
                "timestamp": datetime.now().isoformat()
            }


# Global monitoring service instance
_monitoring_service = None


def get_monitoring_service() -> ProductionMonitoringService:
    """Get global monitoring service instance."""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = ProductionMonitoringService()
    return _monitoring_service


def start_production_monitoring() -> None:
    """Start production monitoring service."""
    service = get_monitoring_service()
    service.start_monitoring()


def stop_production_monitoring() -> None:
    """Stop production monitoring service."""
    service = get_monitoring_service()
    service.stop_monitoring()
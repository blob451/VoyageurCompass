"""
Performance monitoring middleware for VoyageurCompass API.
Tracks response times and performance metrics.
"""

import logging
import time
from datetime import datetime

from django.conf import settings
from django.utils.deprecation import MiddlewareMixin

logger = logging.getLogger(__name__)


class PerformanceMonitoringMiddleware(MiddlewareMixin):
    """
    Middleware to monitor API performance and track response times.
    """

    def __init__(self, get_response):
        """Initialize the middleware."""
        self.get_response = get_response
        super().__init__(get_response)

    def process_request(self, request):
        """Record request start time."""
        request.performance_start_time = time.time()
        request.performance_timestamp = datetime.now()
        return None

    def process_response(self, request, response):
        """Calculate and log response time metrics."""
        if not hasattr(request, "performance_start_time"):
            return response

        # Calculate response time
        response_time = time.time() - request.performance_start_time
        response_time_ms = round(response_time * 1000, 2)

        # Add performance headers
        response["X-Response-Time"] = f"{response_time_ms}ms"
        response["X-Timestamp"] = request.performance_timestamp.isoformat()

        # Log performance metrics
        self._log_performance_metrics(request, response, response_time_ms)

        # Check for performance budget violations
        self._check_performance_budgets(request, response, response_time_ms)

        return response

    def _log_performance_metrics(self, request, response, response_time_ms):
        """Log detailed performance metrics."""
        endpoint = f"{request.method} {request.path}"
        status_code = response.status_code

        # Create performance log entry
        perf_data = {
            "endpoint": endpoint,
            "method": request.method,
            "path": request.path,
            "status_code": status_code,
            "response_time_ms": response_time_ms,
            "timestamp": request.performance_timestamp.isoformat(),
            "user_agent": request.META.get("HTTP_USER_AGENT", "")[:100],
            "remote_addr": self._get_client_ip(request),
        }

        # Log based on performance thresholds
        if response_time_ms > 2000:  # >2s is critical
            logger.error(
                f"CRITICAL_PERFORMANCE: {endpoint} took {response_time_ms}ms",
                extra=perf_data,
            )
        elif response_time_ms > 1000:  # >1s is warning
            logger.warning(
                f"SLOW_PERFORMANCE: {endpoint} took {response_time_ms}ms",
                extra=perf_data,
            )
        elif settings.DEBUG:  # Log all in debug mode
            logger.info(
                f"PERFORMANCE: {endpoint} took {response_time_ms}ms", extra=perf_data
            )

    def _check_performance_budgets(self, request, response, response_time_ms):
        """Check if response violates performance budgets."""
        # Define performance budgets by endpoint type
        budgets = {
            "/api/auth/": 500,  # Auth endpoints: 500ms
            "/api/data/": 1000,  # Data endpoints: 1s
            "/api/analytics/": 2000,  # Analytics endpoints: 2s
            "default": 1500,  # Default budget: 1.5s
        }

        # Find applicable budget
        budget = budgets["default"]
        for pattern, limit in budgets.items():
            if pattern != "default" and request.path.startswith(pattern):
                budget = limit
                break

        # Log budget violations
        if response_time_ms > budget:
            logger.warning(
                f"PERFORMANCE_BUDGET_VIOLATION: {request.method} {request.path} "
                f"took {response_time_ms}ms (budget: {budget}ms)",
                extra={
                    "endpoint": f"{request.method} {request.path}",
                    "response_time_ms": response_time_ms,
                    "budget_ms": budget,
                    "violation_percent": round(
                        ((response_time_ms - budget) / budget) * 100, 1
                    ),
                },
            )

    def _get_client_ip(self, request):
        """Get client IP address."""
        x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        if x_forwarded_for:
            return x_forwarded_for.split(",")[0].strip()
        return request.META.get("REMOTE_ADDR", "")


class RequestLoggingMiddleware(MiddlewareMixin):
    """
    Middleware for logging HTTP requests and responses.
    """

    def __init__(self, get_response):
        self.get_response = get_response
        self.logger = logging.getLogger("django.request")

    def __call__(self, request):
        # Log request
        self.logger.info(
            f"Request: {request.method} {request.path} "
            f"from {self.get_client_ip(request)}"
        )

        # Process request
        response = self.get_response(request)

        # Log response
        self.logger.info(
            f"Response: {response.status_code} for " f"{request.method} {request.path}"
        )

        return response

    def get_client_ip(self, request):
        """Get client IP address from request."""
        x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        if x_forwarded_for:
            ip = x_forwarded_for.split(",")[0]
        else:
            ip = request.META.get("REMOTE_ADDR")
        return ip


class DatabasePerformanceMiddleware(MiddlewareMixin):
    """
    Middleware to monitor database query performance.
    """

    def process_request(self, request):
        """Reset query count at request start."""
        from django.db import connection

        connection.queries_log.clear()
        request.db_query_start_count = len(connection.queries)
        return None

    def process_response(self, request, response):
        """Log database query performance."""
        if not hasattr(request, "db_query_start_count"):
            return response

        from django.db import connection

        # Calculate query metrics
        query_count = len(connection.queries) - request.db_query_start_count
        total_query_time = sum(float(query["time"]) for query in connection.queries)
        total_query_time_ms = round(total_query_time * 1000, 2)

        # Add database performance headers
        response["X-DB-Queries"] = str(query_count)
        response["X-DB-Time"] = f"{total_query_time_ms}ms"

        # Log database performance issues
        if query_count > 20:  # Too many queries
            logger.warning(
                f"HIGH_DB_QUERY_COUNT: {request.method} {request.path} "
                f"executed {query_count} queries in {total_query_time_ms}ms",
                extra={
                    "endpoint": f"{request.method} {request.path}",
                    "query_count": query_count,
                    "total_query_time_ms": total_query_time_ms,
                },
            )
        elif total_query_time_ms > 500:  # Slow queries
            logger.warning(
                f"SLOW_DB_QUERIES: {request.method} {request.path} "
                f"database queries took {total_query_time_ms}ms ({query_count} queries)",
                extra={
                    "endpoint": f"{request.method} {request.path}",
                    "query_count": query_count,
                    "total_query_time_ms": total_query_time_ms,
                },
            )

        return response

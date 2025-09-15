"""Core middleware package."""

import logging
import time
import uuid

from django.utils.deprecation import MiddlewareMixin


class RequestLoggingMiddleware(MiddlewareMixin):
    """Request and response logging middleware with performance metrics and correlation tracking."""

    def __init__(self, get_response=None):
        super().__init__(get_response)
        self.logger = logging.getLogger("VoyageurCompass.requests")

    def process_request(self, request):
        # Honour incoming request ID for distributed tracing
        incoming = request.headers.get("X-Request-Id") or request.headers.get("X-Correlation-Id")
        request.correlation_id = incoming or str(uuid.uuid4())
        request.start_time = time.perf_counter()
        return None

    def process_response(self, request, response):
        if hasattr(request, "start_time"):
            duration = time.perf_counter() - request.start_time
            # Note: userId logging should be reviewed against privacy policy
            # Consider: anonymization, retention policies, or conditional logging
            # based on user consent and regulatory requirements
            user_id = getattr(request.user, "id", None) if hasattr(request, "user") else None

            self.logger.info(
                f'requestId={getattr(request, "correlation_id", "unknown")} '
                f"path={request.path} method={request.method} "
                f"userId={user_id} duration={duration:.3f}s "
                f"status={response.status_code}"
            )
        # Always expose correlation ID to caller
        response["X-Request-Id"] = getattr(request, "correlation_id", "unknown")
        return response

    def process_exception(self, request, exception):
        correlation_id = getattr(request, "correlation_id", "unknown")
        self.logger.error(f"requestId={correlation_id} exception={str(exception)}", exc_info=True)
        return None
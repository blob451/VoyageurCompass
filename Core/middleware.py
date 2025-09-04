import time
import uuid
import logging
from django.http import HttpResponse
from django.utils.cache import patch_vary_headers
from django.utils.deprecation import MiddlewareMixin

class CustomCorsMiddleware:
    """Cross-Origin Resource Sharing middleware with development-focused origin handling."""
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        origin = request.headers.get('Origin')
        
        # Handle preflight requests early to avoid 405s in non-DRF views
        if request.method == 'OPTIONS':
            resp = HttpResponse(status=204)
            
            # Allow specific origins or localhost for development
            if origin and (origin.startswith('http://localhost') or origin.startswith('http://127.0.0.1')):
                resp['Access-Control-Allow-Origin'] = origin
                resp['Access-Control-Allow-Credentials'] = 'true'
            else:
                resp['Access-Control-Allow-Origin'] = 'http://localhost:3000'
                resp['Access-Control-Allow-Credentials'] = 'true'
            
            # Echo requested headers when present, fallback to a sensible default
            requested = request.headers.get('Access-Control-Request-Headers')
            resp['Access-Control-Allow-Headers'] = requested or 'Authorization, Content-Type'
            resp['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            resp['Access-Control-Max-Age'] = '600'
            # Ensure caches vary correctly for CORS preflight
            patch_vary_headers(resp, ['Origin', 'Access-Control-Request-Method', 'Access-Control-Request-Headers'])
            return resp

        response = self.get_response(request)

        # Add CORS headers for non-preflight responses
        if origin and (origin.startswith('http://localhost') or origin.startswith('http://127.0.0.1')):
            response['Access-Control-Allow-Origin'] = origin
            response['Access-Control-Allow-Credentials'] = 'true'
        else:
            response['Access-Control-Allow-Origin'] = 'http://localhost:3000'
            response['Access-Control-Allow-Credentials'] = 'true'
        
        # Expose custom headers to clients
        response['Access-Control-Expose-Headers'] = 'X-Request-Id'
        response['Vary'] = 'Origin'
        return response

class RequestLoggingMiddleware(MiddlewareMixin):
    """Request and response logging middleware with performance metrics and correlation tracking."""
    def __init__(self, get_response=None):
        super().__init__(get_response)
        self.logger = logging.getLogger('VoyageurCompass.requests')

    def process_request(self, request):
        # Honour incoming request ID for distributed tracing
        incoming = request.headers.get('X-Request-Id') or request.headers.get('X-Correlation-Id')
        request.correlation_id = incoming or str(uuid.uuid4())
        request.start_time = time.perf_counter()
        return None

    def process_response(self, request, response):
        if hasattr(request, 'start_time'):
            duration = time.perf_counter() - request.start_time
            # Note: userId logging should be reviewed against privacy policy
            # Consider: anonymization, retention policies, or conditional logging
            # based on user consent and regulatory requirements
            user_id = getattr(request.user, 'id', None) if hasattr(request, 'user') else None
            
            self.logger.info(
                f'requestId={getattr(request, "correlation_id", "unknown")} '
                f'path={request.path} method={request.method} '
                f'userId={user_id} duration={duration:.3f}s '
                f'status={response.status_code}'
            )
        # Always expose correlation ID to caller
        response['X-Request-Id'] = getattr(request, 'correlation_id', 'unknown')
        return response

    def process_exception(self, request, exception):
        correlation_id = getattr(request, 'correlation_id', 'unknown')
        self.logger.error(f'requestId={correlation_id} exception={str(exception)}', exc_info=True)
        return None
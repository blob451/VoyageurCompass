import time
import uuid
import logging
from django.http import HttpResponse
from django.utils.deprecation import MiddlewareMixin

class CustomCorsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Handle preflight requests early to avoid 405s in non-DRF views
        if request.method == 'OPTIONS':
            resp = HttpResponse(status=204)
            resp['Access-Control-Allow-Origin'] = '*'
            # Echo requested headers when present, fallback to a sensible default
            requested = request.headers.get('Access-Control-Request-Headers')
            resp['Access-Control-Allow-Headers'] = requested or 'Authorization, Content-Type'
            resp['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            resp['Access-Control-Max-Age'] = '600'
            return resp

        response = self.get_response(request)

        # Add CORS headers for non-preflight responses
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Headers'] = 'Authorization, Content-Type'
        response['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        return response

class RequestLoggingMiddleware(MiddlewareMixin):
    def __init__(self, get_response=None):
        super().__init__(get_response)
        self.logger = logging.getLogger('VoyageurCompass.requests')

    def process_request(self, request):
        request.correlation_id = str(uuid.uuid4())
        request.start_time = time.perf_counter()
        return None

    def process_response(self, request, response):
        if hasattr(request, 'start_time'):
            duration = time.perf_counter() - request.start_time
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
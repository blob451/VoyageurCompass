"""
Unit tests for Core middleware components with real HTTP operations.
Tests CustomCorsMiddleware and RequestLoggingMiddleware.
"""

import time
import uuid
import tempfile
import logging
from django.test import TestCase, RequestFactory, Client, override_settings
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.urls import path
from django.conf import settings
from io import StringIO

from Core.middleware import CustomCorsMiddleware, RequestLoggingMiddleware


# Real view functions for testing middleware
def test_view(request):
    """Simple test view that returns a basic response."""
    return HttpResponse('Test response')


def test_error_view(request):
    """Test view that raises an exception."""
    raise ValueError("Test exception")


def test_slow_view(request):
    """Test view that simulates slow processing."""
    time.sleep(0.01)  # Small delay for timing tests
    return HttpResponse('Slow response')


# Test URL patterns
test_urlpatterns = [
    path('api/test/', test_view, name='test_view'),
    path('api/error/', test_error_view, name='test_error_view'),
    path('api/slow/', test_slow_view, name='test_slow_view'),
]


class CustomCorsMiddlewareTestCase(TestCase):
    """Test cases for CustomCorsMiddleware with real HTTP operations."""
    
    def setUp(self):
        """Set up test data."""
        self.factory = RequestFactory()
        self.middleware = CustomCorsMiddleware(test_view)
        # Create client for real HTTP testing
        self.client = Client()
    
    def test_preflight_request_with_localhost_origin(self):
        """Test preflight OPTIONS request with localhost origin using real middleware."""
        request = self.factory.options(
            '/api/test/',
            HTTP_ORIGIN='http://localhost:3000',
            HTTP_ACCESS_CONTROL_REQUEST_HEADERS='Authorization, Content-Type'
        )
        
        response = self.middleware(request)
        
        self.assertEqual(response.status_code, 204)
        self.assertEqual(response['Access-Control-Allow-Origin'], 'http://localhost:3000')
        self.assertEqual(response['Access-Control-Allow-Credentials'], 'true')
        self.assertEqual(response['Access-Control-Allow-Headers'], 'Authorization, Content-Type')
        self.assertEqual(response['Access-Control-Allow-Methods'], 'GET, POST, PUT, DELETE, OPTIONS')
        self.assertEqual(response['Access-Control-Max-Age'], '600')
        self.assertIn('Origin', response['Vary'])
        # Verify preflight doesn't execute underlying view (no 'Test response' content)
        self.assertEqual(response.content, b'')
    
    def test_preflight_request_with_127001_origin(self):
        """Test preflight OPTIONS request with 127.0.0.1 origin."""
        request = self.factory.options(
            '/api/test/',
            HTTP_ORIGIN='http://127.0.0.1:3000'
        )
        
        response = self.middleware(request)
        
        self.assertEqual(response.status_code, 204)
        self.assertEqual(response['Access-Control-Allow-Origin'], 'http://127.0.0.1:3000')
        self.assertEqual(response['Access-Control-Allow-Credentials'], 'true')
    
    def test_preflight_request_without_origin(self):
        """Test preflight OPTIONS request without origin header."""
        request = self.factory.options('/api/test/')
        
        response = self.middleware(request)
        
        self.assertEqual(response.status_code, 204)
        self.assertEqual(response['Access-Control-Allow-Origin'], 'http://localhost:3000')
        self.assertEqual(response['Access-Control-Allow-Credentials'], 'true')
    
    def test_preflight_request_with_unknown_origin(self):
        """Test preflight OPTIONS request with unknown origin."""
        request = self.factory.options(
            '/api/test/',
            HTTP_ORIGIN='https://example.com'
        )
        
        response = self.middleware(request)
        
        self.assertEqual(response.status_code, 204)
        self.assertEqual(response['Access-Control-Allow-Origin'], 'http://localhost:3000')
        self.assertEqual(response['Access-Control-Allow-Credentials'], 'true')
    
    def test_preflight_request_without_requested_headers(self):
        """Test preflight OPTIONS request without requested headers."""
        request = self.factory.options(
            '/api/test/',
            HTTP_ORIGIN='http://localhost:3000'
        )
        
        response = self.middleware(request)
        
        self.assertEqual(response['Access-Control-Allow-Headers'], 'Authorization, Content-Type')
    
    def test_normal_request_with_localhost_origin(self):
        """Test normal GET request with localhost origin using real view execution."""
        request = self.factory.get(
            '/api/test/',
            HTTP_ORIGIN='http://localhost:3000'
        )
        
        response = self.middleware(request)
        
        # Verify the underlying view was executed (returns 'Test response')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Test response')
        self.assertEqual(response['Access-Control-Allow-Origin'], 'http://localhost:3000')
        self.assertEqual(response['Access-Control-Allow-Credentials'], 'true')
        self.assertEqual(response['Access-Control-Expose-Headers'], 'X-Request-Id')
        self.assertEqual(response['Vary'], 'Origin')
    
    def test_normal_request_with_127001_origin(self):
        """Test normal GET request with 127.0.0.1 origin."""
        request = self.factory.get(
            '/api/test/',
            HTTP_ORIGIN='http://127.0.0.1:3000'
        )
        
        response = self.middleware(request)
        
        self.assertEqual(response['Access-Control-Allow-Origin'], 'http://127.0.0.1:3000')
        self.assertEqual(response['Access-Control-Allow-Credentials'], 'true')
    
    def test_normal_request_without_origin(self):
        """Test normal GET request without origin header."""
        request = self.factory.get('/api/test/')
        
        response = self.middleware(request)
        
        self.assertEqual(response['Access-Control-Allow-Origin'], 'http://localhost:3000')
        self.assertEqual(response['Access-Control-Allow-Credentials'], 'true')
    
    def test_normal_request_with_unknown_origin(self):
        """Test normal GET request with unknown origin."""
        request = self.factory.get(
            '/api/test/',
            HTTP_ORIGIN='https://example.com'
        )
        
        response = self.middleware(request)
        
        self.assertEqual(response['Access-Control-Allow-Origin'], 'http://localhost:3000')
        self.assertEqual(response['Access-Control-Allow-Credentials'], 'true')
    
    def test_post_request_cors_headers(self):
        """Test POST request includes proper CORS headers with real view execution."""
        request = self.factory.post(
            '/api/test/',
            HTTP_ORIGIN='http://localhost:3000'
        )
        
        response = self.middleware(request)
        
        # Verify the underlying view was executed
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Test response')
        self.assertEqual(response['Access-Control-Allow-Origin'], 'http://localhost:3000')
        self.assertEqual(response['Access-Control-Allow-Credentials'], 'true')
        self.assertEqual(response['Access-Control-Expose-Headers'], 'X-Request-Id')


class RequestLoggingMiddlewareTestCase(TestCase):
    """Test cases for RequestLoggingMiddleware with real logging validation."""
    
    def setUp(self):
        """Set up test data and real logging capture."""
        self.factory = RequestFactory()
        self.middleware = RequestLoggingMiddleware(test_view)
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com'
        )
        
        # Set up real log capture
        self.log_stream = StringIO()
        self.log_handler = logging.StreamHandler(self.log_stream)
        self.log_handler.setLevel(logging.INFO)
        self.logger = logging.getLogger('VoyageurCompass.requests')
        self.logger.addHandler(self.log_handler)
        self.logger.setLevel(logging.INFO)
        
    def tearDown(self):
        """Clean up logging configuration."""
        self.logger.removeHandler(self.log_handler)
        self.log_handler.close()
    
    def test_process_request_logging(self):
        """Test that process_request logs immediately with real stdout/stderr capture."""
        import sys
        from io import StringIO
        
        # Capture real stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            request = self.factory.get('/api/test/')
            result = self.middleware.process_request(request)
            
            self.assertIsNone(result)
            
            # Verify real logging occurred
            stdout_content = stdout_capture.getvalue()
            stderr_content = stderr_capture.getvalue()
            
            self.assertIn('*** MIDDLEWARE: REQUEST GET /api/test/ ***', stdout_content)
            self.assertIn('STDERR: REQUEST GET /api/test/', stderr_content)
            
            # Verify request attributes were set
            self.assertTrue(hasattr(request, 'correlation_id'))
            self.assertTrue(hasattr(request, 'start_time'))
            self.assertIsInstance(request.correlation_id, str)
            self.assertIsInstance(request.start_time, float)
            
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def test_process_request_generates_correlation_id(self):
        """Test that process_request generates correlation ID."""
        request = self.factory.get('/api/test/')
        
        self.middleware.process_request(request)
        
        # Should generate UUID if no incoming ID
        self.assertTrue(len(request.correlation_id) > 0)
        # Should be valid UUID format
        uuid.UUID(request.correlation_id)  # Will raise ValueError if invalid
    
    def test_process_request_honors_incoming_request_id(self):
        """Test that process_request honors incoming X-Request-Id."""
        incoming_id = str(uuid.uuid4())
        request = self.factory.get('/api/test/', HTTP_X_REQUEST_ID=incoming_id)
        
        self.middleware.process_request(request)
        
        self.assertEqual(request.correlation_id, incoming_id)
    
    def test_process_request_honors_incoming_correlation_id(self):
        """Test that process_request honors incoming X-Correlation-Id."""
        incoming_id = str(uuid.uuid4())
        request = self.factory.get('/api/test/', HTTP_X_CORRELATION_ID=incoming_id)
        
        self.middleware.process_request(request)
        
        self.assertEqual(request.correlation_id, incoming_id)
    
    def test_process_request_prefers_request_id_over_correlation_id(self):
        """Test that X-Request-Id takes precedence over X-Correlation-Id."""
        request_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        request = self.factory.get(
            '/api/test/',
            HTTP_X_REQUEST_ID=request_id,
            HTTP_X_CORRELATION_ID=correlation_id
        )
        
        self.middleware.process_request(request)
        
        self.assertEqual(request.correlation_id, request_id)
    
    def test_process_response_with_authenticated_user(self):
        """Test process_response with authenticated user using real logging."""
        request = self.factory.get('/api/test/')
        request.user = self.user
        request.correlation_id = str(uuid.uuid4())
        request.start_time = time.perf_counter()
        
        response = HttpResponse('Test response')
        
        # Clear any existing log content
        self.log_stream.seek(0)
        self.log_stream.truncate(0)
        
        result = self.middleware.process_response(request, response)
        
        # Check real logging output
        log_content = self.log_stream.getvalue()
        self.assertIn(f'requestId={request.correlation_id}', log_content)
        self.assertIn('path=/api/test/', log_content)
        self.assertIn('method=GET', log_content)
        self.assertIn(f'userId={self.user.id}', log_content)
        self.assertIn('duration=', log_content)  # Check duration is present
        self.assertIn('status=200', log_content)
        
        # Check response headers
        self.assertEqual(result['X-Request-Id'], request.correlation_id)
    
    def test_process_response_with_anonymous_user(self):
        """Test process_response with anonymous user using real logging."""
        request = self.factory.get('/api/test/')
        request.correlation_id = str(uuid.uuid4())
        request.start_time = time.perf_counter()
        
        response = HttpResponse('Test response')
        
        # Clear any existing log content
        self.log_stream.seek(0)
        self.log_stream.truncate(0)
        
        result = self.middleware.process_response(request, response)
        
        log_content = self.log_stream.getvalue()
        self.assertIn('userId=None', log_content)
        self.assertIn('duration=', log_content)  # Check duration is present
    
    def test_process_response_without_start_time(self):
        """Test process_response without start_time (no timing) using real logging."""
        request = self.factory.get('/api/test/')
        request.correlation_id = str(uuid.uuid4())
        # No start_time attribute
        
        response = HttpResponse('Test response')
        
        # Clear any existing log content
        self.log_stream.seek(0)
        self.log_stream.truncate(0)
        
        result = self.middleware.process_response(request, response)
        
        # Should not log if no start_time
        log_content = self.log_stream.getvalue()
        self.assertEqual(log_content.strip(), '')  # No logging should occur
        # Should still set correlation ID
        self.assertEqual(result['X-Request-Id'], request.correlation_id)
    
    def test_process_response_without_correlation_id(self):
        """Test process_response without correlation_id."""
        request = self.factory.get('/api/test/')
        # No correlation_id attribute
        
        response = HttpResponse('Test response')
        
        result = self.middleware.process_response(request, response)
        
        self.assertEqual(result['X-Request-Id'], 'unknown')
    
    def test_process_exception_logging(self):
        """Test process_exception logs errors properly using real logging."""
        request = self.factory.get('/api/test/')
        request.correlation_id = str(uuid.uuid4())
        exception = ValueError("Test exception")
        
        # Set up error level logging capture
        error_stream = StringIO()
        error_handler = logging.StreamHandler(error_stream)
        error_handler.setLevel(logging.ERROR)
        self.logger.addHandler(error_handler)
        
        try:
            result = self.middleware.process_exception(request, exception)
            
            self.assertIsNone(result)
            
            # Check real error logging output
            error_content = error_stream.getvalue()
            self.assertIn(f'requestId={request.correlation_id}', error_content)
            self.assertIn('exception=Test exception', error_content)
            # The exc_info=True should include traceback, but format may vary
            # Just verify we have exception information logged
            self.assertTrue(len(error_content) > 0)
            
        finally:
            self.logger.removeHandler(error_handler)
            error_handler.close()
    
    def test_process_exception_without_correlation_id(self):
        """Test process_exception without correlation_id using real logging."""
        request = self.factory.get('/api/test/')
        exception = ValueError("Test exception")
        
        # Set up error level logging capture
        error_stream = StringIO()
        error_handler = logging.StreamHandler(error_stream)
        error_handler.setLevel(logging.ERROR)
        self.logger.addHandler(error_handler)
        
        try:
            result = self.middleware.process_exception(request, exception)
            
            error_content = error_stream.getvalue()
            self.assertIn('requestId=unknown', error_content)
            
        finally:
            self.logger.removeHandler(error_handler)
            error_handler.close()


class MiddlewareIntegrationTestCase(TestCase):
    """Integration tests for middleware components with real chaining."""
    
    def setUp(self):
        """Set up test data and real logging capture."""
        self.factory = RequestFactory()
        self.user = User.objects.create_user(
            username='integrationuser',
            email='integration@example.com'
        )
        
        # Set up real logging capture for integration tests
        self.log_stream = StringIO()
        self.log_handler = logging.StreamHandler(self.log_stream)
        self.log_handler.setLevel(logging.INFO)
        self.logger = logging.getLogger('VoyageurCompass.requests')
        self.logger.addHandler(self.log_handler)
        self.logger.setLevel(logging.INFO)
        
    def tearDown(self):
        """Clean up logging configuration."""
        self.logger.removeHandler(self.log_handler)
        self.log_handler.close()
    
    def test_middleware_chain_integration(self):
        """Test CORS and RequestLogging middleware working together with real operations."""
        # Create the real middleware chain (in reverse order as Django does)
        def cors_and_logging_chain(request):
            # First middleware: RequestLogging
            logging_middleware = RequestLoggingMiddleware(test_view)
            
            # Second middleware: CORS (wraps logging)
            cors_middleware = CustomCorsMiddleware(logging_middleware)
            
            return cors_middleware(request)
        
        request = self.factory.get(
            '/api/test/',
            HTTP_ORIGIN='http://localhost:3000'
        )
        request.user = self.user
        
        # Clear any existing log content
        self.log_stream.seek(0)
        self.log_stream.truncate(0)
        
        # Capture stdout/stderr for real request processing
        import sys
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            response = cors_and_logging_chain(request)
            
            # Verify real middleware chain execution
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, b'Test response')  # Real view executed
            
            # Should have both CORS and logging attributes
            self.assertEqual(response['Access-Control-Allow-Origin'], 'http://localhost:3000')
            self.assertEqual(response['Access-Control-Allow-Credentials'], 'true')
            self.assertIn('X-Request-Id', response)
            self.assertEqual(response['Vary'], 'Origin')
            
            # Verify real logging occurred
            log_content = self.log_stream.getvalue()
            self.assertIn('path=/api/test/', log_content)
            self.assertIn('method=GET', log_content)
            self.assertIn(f'userId={self.user.id}', log_content)
            
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def test_preflight_request_skips_logging_chain(self):
        """Test that preflight requests are handled by CORS middleware only."""
        # Create middleware chain with real components
        logging_middleware = RequestLoggingMiddleware(test_view)
        cors_middleware = CustomCorsMiddleware(logging_middleware)
        
        request = self.factory.options(
            '/api/test/',
            HTTP_ORIGIN='http://localhost:3000'
        )
        
        # Clear any existing log content
        self.log_stream.seek(0)
        self.log_stream.truncate(0)
        
        response = cors_middleware(request)
        
        # CORS middleware should handle preflight directly
        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.content, b'')  # No view content
        
        # Should not reach the logging middleware (no logging should occur)
        log_content = self.log_stream.getvalue()
        self.assertEqual(log_content.strip(), '')  # No logging for preflight
    
    def test_timing_accuracy_in_middleware_chain(self):
        """Test that timing measurements are accurate through middleware chain with real operations."""
        logging_middleware = RequestLoggingMiddleware(test_slow_view)
        cors_middleware = CustomCorsMiddleware(logging_middleware)
        
        request = self.factory.get('/api/test/')
        request.user = self.user
        
        # Clear any existing log content
        self.log_stream.seek(0)
        self.log_stream.truncate(0)
        
        # Capture stdout/stderr for real request processing
        import sys
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            response = cors_middleware(request)
            
            # Verify real view executed
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, b'Slow response')
            
            # Check that real timing was logged
            log_content = self.log_stream.getvalue()
            self.assertIn('duration=', log_content)
            self.assertIn('path=/api/test/', log_content)
            
            # Verify actual duration was measured (should be > 0.01 seconds due to sleep)
            import re
            duration_match = re.search(r'duration=(\d+\.\d+)s', log_content)
            self.assertIsNotNone(duration_match)
            duration = float(duration_match.group(1))
            self.assertGreater(duration, 0.005)  # Should have some measurable duration
            
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

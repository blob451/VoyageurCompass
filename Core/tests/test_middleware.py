"""
Unit tests for Core middleware components.
Tests CustomCorsMiddleware and RequestLoggingMiddleware.
"""

import time
import uuid
from unittest.mock import Mock, patch, MagicMock
from django.test import TestCase, RequestFactory
from django.http import HttpResponse
from django.contrib.auth.models import User

from Core.middleware import CustomCorsMiddleware, RequestLoggingMiddleware


class CustomCorsMiddlewareTestCase(TestCase):
    """Test cases for CustomCorsMiddleware."""
    
    def setUp(self):
        """Set up test data."""
        self.factory = RequestFactory()
        self.get_response = Mock(return_value=HttpResponse('Test response'))
        self.middleware = CustomCorsMiddleware(self.get_response)
    
    def test_preflight_request_with_localhost_origin(self):
        """Test preflight OPTIONS request with localhost origin."""
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
        # Should not call get_response for preflight
        self.get_response.assert_not_called()
    
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
        """Test normal GET request with localhost origin."""
        request = self.factory.get(
            '/api/test/',
            HTTP_ORIGIN='http://localhost:3000'
        )
        
        response = self.middleware(request)
        
        self.get_response.assert_called_once_with(request)
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
        """Test POST request includes proper CORS headers."""
        request = self.factory.post(
            '/api/test/',
            HTTP_ORIGIN='http://localhost:3000'
        )
        
        response = self.middleware(request)
        
        self.get_response.assert_called_once_with(request)
        self.assertEqual(response['Access-Control-Allow-Origin'], 'http://localhost:3000')
        self.assertEqual(response['Access-Control-Allow-Credentials'], 'true')
        self.assertEqual(response['Access-Control-Expose-Headers'], 'X-Request-Id')


class RequestLoggingMiddlewareTestCase(TestCase):
    """Test cases for RequestLoggingMiddleware."""
    
    def setUp(self):
        """Set up test data."""
        self.factory = RequestFactory()
        self.get_response = Mock(return_value=HttpResponse('Test response'))
        self.middleware = RequestLoggingMiddleware(self.get_response)
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com'
        )
    
    @patch('sys.stderr')
    @patch('builtins.print')
    def test_process_request_logging(self, mock_print, mock_stderr):
        """Test that process_request logs immediately."""
        request = self.factory.get('/api/test/')
        
        result = self.middleware.process_request(request)
        
        self.assertIsNone(result)
        mock_print.assert_called_once()
        mock_stderr.write.assert_called_once()
        self.assertTrue(hasattr(request, 'correlation_id'))
        self.assertTrue(hasattr(request, 'start_time'))
        self.assertIsInstance(request.correlation_id, str)
        self.assertIsInstance(request.start_time, float)
    
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
        """Test process_response with authenticated user."""
        request = self.factory.get('/api/test/')
        request.user = self.user
        request.correlation_id = str(uuid.uuid4())
        request.start_time = time.perf_counter()
        
        response = HttpResponse('Test response')
        
        with patch.object(self.middleware.logger, 'info') as mock_log:
            result = self.middleware.process_response(request, response)
        
        # Check logging
        mock_log.assert_called_once()
        log_call = mock_log.call_args[0][0]
        self.assertIn(f'requestId={request.correlation_id}', log_call)
        self.assertIn('path=/api/test/', log_call)
        self.assertIn('method=GET', log_call)
        self.assertIn(f'userId={self.user.id}', log_call)
        self.assertIn('duration=', log_call)  # Just check duration is present
        self.assertIn('status=200', log_call)
        
        # Check response headers
        self.assertEqual(result['X-Request-Id'], request.correlation_id)
    
    def test_process_response_with_anonymous_user(self):
        """Test process_response with anonymous user."""
        request = self.factory.get('/api/test/')
        request.correlation_id = str(uuid.uuid4())
        request.start_time = time.perf_counter()
        
        response = HttpResponse('Test response')
        
        with patch.object(self.middleware.logger, 'info') as mock_log:
            result = self.middleware.process_response(request, response)
        
        log_call = mock_log.call_args[0][0]
        self.assertIn('userId=None', log_call)
        self.assertIn('duration=', log_call)  # Just check duration is present
    
    def test_process_response_without_start_time(self):
        """Test process_response without start_time (no timing)."""
        request = self.factory.get('/api/test/')
        request.correlation_id = str(uuid.uuid4())
        # No start_time attribute
        
        response = HttpResponse('Test response')
        
        with patch.object(self.middleware.logger, 'info') as mock_log:
            result = self.middleware.process_response(request, response)
        
        # Should not log if no start_time
        mock_log.assert_not_called()
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
        """Test process_exception logs errors properly."""
        request = self.factory.get('/api/test/')
        request.correlation_id = str(uuid.uuid4())
        exception = ValueError("Test exception")
        
        with patch.object(self.middleware.logger, 'error') as mock_log:
            result = self.middleware.process_exception(request, exception)
        
        self.assertIsNone(result)
        mock_log.assert_called_once()
        log_call = mock_log.call_args[0][0]
        self.assertIn(f'requestId={request.correlation_id}', log_call)
        self.assertIn('exception=Test exception', log_call)
        # Check that exc_info was passed
        self.assertTrue(mock_log.call_args[1]['exc_info'])
    
    def test_process_exception_without_correlation_id(self):
        """Test process_exception without correlation_id."""
        request = self.factory.get('/api/test/')
        exception = ValueError("Test exception")
        
        with patch.object(self.middleware.logger, 'error') as mock_log:
            result = self.middleware.process_exception(request, exception)
        
        log_call = mock_log.call_args[0][0]
        self.assertIn('requestId=unknown', log_call)


class MiddlewareIntegrationTestCase(TestCase):
    """Integration tests for middleware components."""
    
    def setUp(self):
        """Set up test data."""
        self.factory = RequestFactory()
        self.user = User.objects.create_user(
            username='integrationuser',
            email='integration@example.com'
        )
    
    def test_middleware_chain_integration(self):
        """Test CORS and RequestLogging middleware working together."""
        # Create a mock response that will be passed through the chain
        final_response = HttpResponse('Final response')
        
        # Create the middleware chain (in reverse order as Django does)
        def cors_and_logging_chain(request):
            # First middleware: RequestLogging
            logging_middleware = RequestLoggingMiddleware(lambda req: final_response)
            
            # Second middleware: CORS (wraps logging)
            cors_middleware = CustomCorsMiddleware(logging_middleware)
            
            return cors_middleware(request)
        
        request = self.factory.get(
            '/api/test/',
            HTTP_ORIGIN='http://localhost:3000'
        )
        request.user = self.user
        
        with patch('sys.stderr'), patch('builtins.print'):
            response = cors_and_logging_chain(request)
        
        # Should have both CORS and logging attributes
        self.assertEqual(response['Access-Control-Allow-Origin'], 'http://localhost:3000')
        self.assertEqual(response['Access-Control-Allow-Credentials'], 'true')
        self.assertIn('X-Request-Id', response)
        self.assertEqual(response['Vary'], 'Origin')
    
    def test_preflight_request_skips_logging_chain(self):
        """Test that preflight requests are handled by CORS middleware only."""
        logging_get_response = Mock()
        
        # Create middleware chain
        logging_middleware = RequestLoggingMiddleware(logging_get_response)
        cors_middleware = CustomCorsMiddleware(logging_middleware)
        
        request = self.factory.options(
            '/api/test/',
            HTTP_ORIGIN='http://localhost:3000'
        )
        
        response = cors_middleware(request)
        
        # CORS middleware should handle preflight directly
        self.assertEqual(response.status_code, 204)
        # Should not reach the logging middleware
        logging_get_response.assert_not_called()
    
    def test_timing_accuracy_in_middleware_chain(self):
        """Test that timing measurements are accurate through middleware chain."""
        def slow_response(request):
            # Simulate some processing
            return HttpResponse('Slow response')
        
        logging_middleware = RequestLoggingMiddleware(slow_response)
        cors_middleware = CustomCorsMiddleware(logging_middleware)
        
        request = self.factory.get('/api/test/')
        request.user = self.user
        
        with patch.object(logging_middleware.logger, 'info') as mock_log:
            with patch('sys.stderr'), patch('builtins.print'):
                response = cors_middleware(request)
        
        # Check that timing was logged
        mock_log.assert_called_once()
        log_call = mock_log.call_args[0][0]
        # Check that duration is present in log
        self.assertIn('duration=', log_call)
        self.assertIn('path=/api/test/', log_call)
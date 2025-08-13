"""
Tests for Core middleware.
"""

from unittest.mock import Mock

import pytest
from django.http import HttpRequest, HttpResponse

from Core.middleware.cors import CustomCorsMiddleware
from Core.middleware.performance import (
    PerformanceMonitoringMiddleware,
    RequestLoggingMiddleware,
)


class TestCustomCorsMiddleware:
    """Test cases for CustomCorsMiddleware."""

    def test_cors_headers_added(self):
        """Test that CORS headers are added to response."""
        get_response = Mock(return_value=HttpResponse())
        middleware = CustomCorsMiddleware(get_response)
        request = Mock(spec=HttpRequest)
        request.method = "GET"

        response = middleware(request)

        assert response["Access-Control-Allow-Origin"] == "*"
        assert "GET, POST, PUT, PATCH, DELETE, OPTIONS" in response["Access-Control-Allow-Methods"]
        assert "Content-Type, Authorization" in response["Access-Control-Allow-Headers"]
        assert response["Access-Control-Max-Age"] == "3600"

    def test_options_request_handling(self):
        """Test that OPTIONS requests are handled for preflight."""
        get_response = Mock(return_value=HttpResponse())
        middleware = CustomCorsMiddleware(get_response)
        request = Mock(spec=HttpRequest)
        request.method = "OPTIONS"

        response = middleware.process_request(request)

        # Should return a response for OPTIONS requests
        assert response is not None
        assert response["Access-Control-Allow-Origin"] == "*"

    def test_non_options_request(self):
        """Test that non-OPTIONS requests return None."""
        get_response = Mock(return_value=HttpResponse())
        middleware = CustomCorsMiddleware(get_response)
        request = Mock(spec=HttpRequest)
        request.method = "GET"

        response = middleware.process_request(request)

        # Should return None for non-OPTIONS requests
        assert response is None


class TestPerformanceMonitoringMiddleware:
    """Test cases for PerformanceMonitoringMiddleware."""

    def test_performance_tracking(self):
        """Test that performance tracking works."""
        get_response = Mock(return_value=HttpResponse())
        middleware = PerformanceMonitoringMiddleware(get_response)
        request = Mock(spec=HttpRequest)
        request.path = "/test/"
        request.method = "GET"
        request.META = {"HTTP_USER_AGENT": "test-agent", "REMOTE_ADDR": "127.0.0.1"}

        response = middleware(request)

        # Should return a response
        assert response is not None
        get_response.assert_called_once_with(request)


class TestRequestLoggingMiddleware:
    """Test cases for RequestLoggingMiddleware."""

    def test_request_logging(self):
        """Test that requests are logged."""
        get_response = Mock(return_value=HttpResponse())
        middleware = RequestLoggingMiddleware(get_response)
        request = Mock(spec=HttpRequest)
        request.path = "/api/test/"
        request.method = "POST"
        request.META = {"HTTP_X_FORWARDED_FOR": "10.0.0.1", "REMOTE_ADDR": "127.0.0.1"}

        response = middleware(request)

        # Should return a response
        assert response is not None
        get_response.assert_called_once_with(request)
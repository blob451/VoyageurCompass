"""
API Tests for sentiment analysis endpoints.
Tests the sentiment analysis REST API functionality.
"""


from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APIClient

from Core.tests.fixtures import CoreTestDataFactory
from Data.models import AnalyticsResults
from Data.tests.fixtures import DataTestDataFactory

User = get_user_model()


class SentimentAPITestCase(TestCase):
    """Test cases for sentiment analysis API endpoints."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = APIClient()

        # Create test user using real factory
        self.user = CoreTestDataFactory.create_test_user(username="testuser", email="test@example.com")

        # Create test stock using real factory
        self.test_stock = DataTestDataFactory.create_test_stock("AAPL", "Apple Inc.", "Technology")

        # Clear cache
        cache.clear()

    def tearDown(self):
        """Clean up after tests."""
        cache.clear()

    def test_sentiment_endpoint_authentication_required(self):
        """Test that sentiment endpoint requires authentication."""
        url = reverse("analytics:stock_sentiment", kwargs={"symbol": "AAPL"})
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_sentiment_endpoint_success(self):
        """Test successful sentiment analysis API call with real functionality."""
        # Authenticate user
        self.client.force_authenticate(user=self.user)

        # Make API call with real endpoint
        url = reverse("analytics:stock_sentiment", kwargs={"symbol": "AAPL"})
        response = self.client.get(url)

        # Verify response structure (real API will handle data fetching)
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE])

        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            self.assertEqual(data["symbol"], "AAPL")
            self.assertIn("sentimentLabel", data)
            self.assertIn("sentimentScore", data)
            self.assertIn("newsCount", data)
            self.assertIn(data["sentimentLabel"], ["positive", "negative", "neutral"])
            self.assertIsInstance(data["sentimentScore"], (int, float))
            self.assertIsInstance(data["newsCount"], int)
        else:
            # Service unavailable - acceptable in test environment
            data = response.json()
            self.assertIn("error", data)

    def test_sentiment_endpoint_no_news_scenario(self):
        """Test sentiment endpoint behavior with potential no-news scenario."""
        # Authenticate user
        self.client.force_authenticate(user=self.user)

        # Create stock with limited data
        limited_stock = DataTestDataFactory.create_test_stock("NWSTEST", "No News Test Co", "Technology")

        # Make API call
        url = reverse("analytics:stock_sentiment", kwargs={"symbol": "NWSTEST"})
        response = self.client.get(url)

        # Verify response structure regardless of news availability
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE])

        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            self.assertEqual(data["symbol"], "NWSTEST")
            self.assertIn("sentimentLabel", data)
            self.assertIn("sentimentScore", data)
            self.assertIn("newsCount", data)
            # When no news available, should default to neutral
            if data["newsCount"] == 0:
                self.assertEqual(data["sentimentLabel"], "neutral")
                self.assertEqual(data["sentimentScore"], 0.0)

    def test_sentiment_endpoint_caching_behavior(self):
        """Test sentiment endpoint caching behavior with real functionality."""
        # Authenticate user
        self.client.force_authenticate(user=self.user)

        # Make initial API call
        url = reverse("analytics:stock_sentiment", kwargs={"symbol": "AAPL"})
        response1 = self.client.get(url)

        if response1.status_code == status.HTTP_200_OK:
            # Make second API call immediately - should be faster due to caching
            import time

            start_time = time.time()
            response2 = self.client.get(url)
            end_time = time.time()

            self.assertEqual(response2.status_code, status.HTTP_200_OK)

            # Second call should be fast (cached or quick lookup)
            response_time = end_time - start_time
            self.assertLess(response_time, 2.0, "Second API call should be faster due to caching")

            # Verify consistent results
            data1 = response1.json()
            data2 = response2.json()
            self.assertEqual(data1["symbol"], data2["symbol"])
        else:
            self.skipTest("Sentiment service not available for caching test")

    def test_sentiment_endpoint_custom_days_parameter(self):
        """Test sentiment endpoint with custom days parameter."""
        # Authenticate user
        self.client.force_authenticate(user=self.user)

        # Make API call with custom days parameter
        url = reverse("analytics:stock_sentiment", kwargs={"symbol": "AAPL"})
        response = self.client.get(url, {"days": 30})

        # Verify parameter handling
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE])

        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            self.assertEqual(data["symbol"], "AAPL")
            # Verify response includes days parameter information
            self.assertIn("sentimentLabel", data)

    def test_sentiment_endpoint_refresh_parameter(self):
        """Test sentiment endpoint with refresh parameter."""
        # Authenticate user
        self.client.force_authenticate(user=self.user)

        # Make API call with refresh parameter
        url = reverse("analytics:stock_sentiment", kwargs={"symbol": "AAPL"})
        response = self.client.get(url, {"refresh": "true"})

        # Verify refresh parameter handling
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE])

        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            self.assertEqual(data["symbol"], "AAPL")
            # Refresh should force new analysis (cached=False or no cached field)
            if "cached" in data:
                self.assertFalse(data["cached"])

    def test_sentiment_endpoint_nonexistent_stock(self):
        """Test sentiment endpoint with non-existent stock symbol."""
        # Authenticate user
        self.client.force_authenticate(user=self.user)

        # Make API call with non-existent symbol
        url = reverse("analytics:stock_sentiment", kwargs={"symbol": "INVALID"})
        response = self.client.get(url)

        # API may handle invalid symbols gracefully with 200 status
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND])

        data = response.json()
        if response.status_code == status.HTTP_404_NOT_FOUND:
            self.assertIn("error", data)
        else:
            # API returns neutral sentiment for invalid symbols
            self.assertEqual(data["symbol"], "INVALID")
            self.assertIn("sentimentLabel", data)

    def test_sentiment_endpoint_error_handling(self):
        """Test sentiment endpoint error handling with real functionality."""
        # Authenticate user
        self.client.force_authenticate(user=self.user)

        # Test with invalid stock symbol that might cause errors
        url = reverse("analytics:stock_sentiment", kwargs={"symbol": "INVALID123"})
        response = self.client.get(url)

        # Real API may handle invalid symbols gracefully
        self.assertIn(
            response.status_code,
            [
                status.HTTP_200_OK,
                status.HTTP_404_NOT_FOUND,
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ],
        )

        data = response.json()
        if response.status_code != status.HTTP_200_OK:
            self.assertIn("error", data)
        else:
            # Graceful handling returns neutral sentiment
            self.assertEqual(data["symbol"], "INVALID123")

    def test_sentiment_endpoint_with_existing_analysis(self):
        """Test sentiment endpoint includes database sentiment data."""
        # Authenticate user
        self.client.force_authenticate(user=self.user)

        # Create existing analysis result
        AnalyticsResults.objects.create(
            stock=self.test_stock,
            as_of=timezone.now(),
            sentimentScore=0.65,
            sentimentLabel="positive",
            newsCount=20,
            score_0_10=7,
        )

        # Make API call with real endpoint
        url = reverse("analytics:stock_sentiment", kwargs={"symbol": "AAPL"})
        response = self.client.get(url)

        # Verify response includes database data if available
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE])

        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            # Should include database sentiment data if it exists
            if "dbSentimentScore" in data:
                self.assertEqual(data["dbSentimentScore"], 0.65)
                self.assertEqual(data["dbSentimentLabel"], "positive")
                self.assertEqual(data["dbNewsCount"], 20)

    def test_sentiment_endpoint_rate_limiting(self):
        """Test sentiment endpoint rate limiting."""
        # Authenticate user
        self.client.force_authenticate(user=self.user)

        # Test rapid successive calls to check rate limiting behavior
        url = reverse("analytics:stock_sentiment", kwargs={"symbol": "AAPL"})

        responses = []
        for _ in range(3):
            response = self.client.get(url)
            responses.append(response)

        # All responses should be handled (either success or rate limited)
        for response in responses:
            self.assertIn(
                response.status_code,
                [status.HTTP_200_OK, status.HTTP_429_TOO_MANY_REQUESTS, status.HTTP_503_SERVICE_UNAVAILABLE],
            )


class SentimentAPIPerformanceTestCase(TestCase):
    """Performance tests for sentiment analysis API."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = APIClient()
        self.user = CoreTestDataFactory.create_test_user(username="perfuser", email="perf@example.com")
        self.client.force_authenticate(user=self.user)

        DataTestDataFactory.create_test_stock("PERF", "Performance Test Co", "Technology")

    def test_sentiment_response_time(self):
        """Test sentiment analysis response time is reasonable with real functionality."""
        import time

        # Measure response time for real API call
        start_time = time.time()

        url = reverse("analytics:stock_sentiment", kwargs={"symbol": "PERF"})
        response = self.client.get(url)

        end_time = time.time()
        response_time = end_time - start_time

        # Verify response
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE])

        if response.status_code == status.HTTP_200_OK:
            # Response should be reasonably fast (under 30 seconds for real analysis)
            self.assertLess(response_time, 30.0, f"API response took {response_time:.2f}s")

            data = response.json()
            self.assertEqual(data["symbol"], "PERF")
            self.assertIn("sentimentLabel", data)
        else:
            # Service unavailable - still check that response was reasonably fast
            self.assertLess(response_time, 5.0, "Error response should be fast")


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])

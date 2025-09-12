"""
Integration tests for Analytics API endpoints.
Tests the full API flow including authentication and data validation.
"""

from datetime import date, timedelta
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from Data.models import Stock, StockPrice

User = get_user_model()


class TestAnalyticsAPI(APITestCase):
    """Test suite for Analytics API endpoints."""

    def setUp(self):
        """Set up test data before each test."""
        # Create test user
        self.user = User.objects.create_user(username="testuser", email="test@example.com", password="testpass123")

        # Create test stock
        self.stock = Stock.objects.create(
            symbol="AAPL",
            short_name="Apple",
            long_name="Apple Inc.",
            sector="Technology",
            industry="Consumer Electronics",
            market_cap=3000000000000,
        )

        # Create historical price data
        base_date = date.today()
        for i in range(30):
            StockPrice.objects.create(
                stock=self.stock,
                date=base_date - timedelta(days=i),
                open=Decimal("150.00") + Decimal(str(i % 5)),
                high=Decimal("155.00") + Decimal(str(i % 5)),
                low=Decimal("148.00") + Decimal(str(i % 5)),
                close=Decimal("152.00") + Decimal(str(i % 5)),
                volume=50000000 + (i * 1000000),
            )

    def test_analysis_endpoint_authenticated(self):
        """Test analysis endpoint with authenticated user."""
        # Authenticate the user
        self.client.force_authenticate(user=self.user)

        # Make request to analysis endpoint
        url = reverse("analytics:stock-analysis", kwargs={"symbol": "AAPL"})
        response = self.client.get(url)

        # Should return 200 OK
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Check response structure
        data = response.json()
        self.assertIn("symbol", data)
        self.assertIn("name", data)
        self.assertIn("analysis", data)
        self.assertIn("technical_indicators", data["analysis"])
        self.assertIn("price_history", data["analysis"])

        # Verify stock data
        self.assertEqual(data["symbol"], "AAPL")
        self.assertEqual(data["name"], "Apple Inc.")

    def test_analysis_endpoint_nonexistent_stock(self):
        """Test analysis endpoint with non-existent stock."""
        self.client.force_authenticate(user=self.user)

        url = reverse("analytics:stock-analysis", kwargs={"symbol": "INVALID"})
        response = self.client.get(url)

        # Should return 404 Not Found
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        data = response.json()
        self.assertIn("error", data)

    def test_batch_analysis_endpoint(self):
        """Test batch analysis for multiple stocks."""
        # Create additional test stocks
        stock2 = Stock.objects.create(symbol="GOOGL", short_name="Alphabet")
        stock3 = Stock.objects.create(symbol="MSFT", short_name="Microsoft")

        # Add some price data for the new stocks
        for stock in [stock2, stock3]:
            for i in range(5):
                StockPrice.objects.create(
                    stock=stock,
                    date=date.today() - timedelta(days=i),
                    open=Decimal("100.00"),
                    high=Decimal("102.00"),
                    low=Decimal("99.00"),
                    close=Decimal("100.00") + Decimal(str(i)),
                )

        self.client.force_authenticate(user=self.user)

        # Request batch analysis
        url = reverse("analytics:batch-analysis")
        response = self.client.post(url, {"symbols": ["AAPL", "GOOGL", "MSFT"]}, format="json")

        # Should return 200 OK
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        data = response.json()
        self.assertIn("results", data)
        self.assertEqual(len(data["results"]), 3)

        # Verify each stock result
        symbols = [result["symbol"] for result in data["results"]]
        self.assertIn("AAPL", symbols)
        self.assertIn("GOOGL", symbols)
        self.assertIn("MSFT", symbols)

    def test_technical_indicators_endpoint(self):
        """Test technical indicators calculation endpoint."""
        self.client.force_authenticate(user=self.user)

        url = reverse("analytics:technical-indicators", kwargs={"symbol": "AAPL"})
        response = self.client.get(url, {"indicators": "sma,ema,rsi,macd"})

        # Should return 200 OK
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        data = response.json()
        self.assertIn("symbol", data)
        self.assertIn("indicators", data)

        indicators = data["indicators"]
        self.assertIn("sma", indicators)
        self.assertIn("ema", indicators)
        self.assertIn("rsi", indicators)
        self.assertIn("macd", indicators)

    def test_historical_data_endpoint(self):
        """Test historical data retrieval with date range."""
        self.client.force_authenticate(user=self.user)

        # Request last 7 days of data
        start_date = (date.today() - timedelta(days=7)).isoformat()
        end_date = date.today().isoformat()

        url = reverse("analytics:historical-data", kwargs={"symbol": "AAPL"})
        response = self.client.get(url, {"start_date": start_date, "end_date": end_date})

        # Should return 200 OK
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        data = response.json()
        self.assertIn("symbol", data)
        self.assertIn("price_data", data)

        # Should have approximately 7 days of data
        self.assertGreater(len(data["price_data"]), 0)
        self.assertLessEqual(len(data["price_data"]), 8)

        # Verify data structure
        if data["price_data"]:
            first_entry = data["price_data"][0]
            self.assertIn("date", first_entry)
            self.assertIn("open", first_entry)
            self.assertIn("high", first_entry)
            self.assertIn("low", first_entry)
            self.assertIn("close", first_entry)
            self.assertIn("volume", first_entry)

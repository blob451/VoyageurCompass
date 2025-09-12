"""
Integration tests for VoyageurCompass API endpoints.
These tests verify that different components work together correctly.
"""

import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth.models import User
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient, APITestCase

from Data.models import Portfolio, PortfolioHolding, Stock, StockPrice


class FullWorkflowIntegrationTest(APITestCase):
    """
    Integration tests that simulate complete user workflows.
    """

    def setUp(self):
        """Set up test data for integration tests."""
        self.client = APIClient()

        # Create test user
        self.user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpass123",
            "first_name": "Test",
            "last_name": "User",
        }

        # Create test stocks
        self.create_test_stocks()

    def create_test_stocks(self):
        """Create test stocks with price history."""
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        self.stocks = {}

        for i, symbol in enumerate(symbols):
            stock = Stock.objects.create(
                symbol=symbol,
                short_name=f"{symbol} Inc.",
                long_name=f"{symbol} Corporation",
                currency="USD",
                exchange="NASDAQ",
                sector="Technology",
                market_cap=(i + 1) * 1000000000000,
                is_active=True,
            )
            self.stocks[symbol] = stock

            # Create price history
            base_price = Decimal("100.00") + (i * Decimal("50.00"))
            for j in range(30):
                price_date = date.today() - timedelta(days=j)
                StockPrice.objects.create(
                    stock=stock,
                    date=price_date,
                    open=base_price + Decimal(str(j % 10)),
                    high=base_price + Decimal(str((j % 10) + 5)),
                    low=base_price - Decimal(str(j % 5)),
                    close=base_price + Decimal(str(j % 8)),
                    volume=1000000 + (j * 50000),
                )

    def test_complete_user_registration_and_portfolio_creation(self):
        """Test complete workflow: register -> login -> create portfolio -> add holdings."""

        # Step 1: Register new user
        register_url = reverse("core:auth-register")
        register_data = {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "newpass123",
            "password2": "newpass123",
            "first_name": "New",
            "last_name": "User",
        }
        register_response = self.client.post(register_url, register_data, format="json")

        self.assertEqual(register_response.status_code, status.HTTP_201_CREATED)
        self.assertIn("access", register_response.data)
        self.assertIn("refresh", register_response.data)

        access_token = register_response.data["access"]

        # Step 2: Authenticate with the new token
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {access_token}")

        # Step 3: Create a portfolio
        portfolio_url = reverse("data:portfolio-list")
        portfolio_data = {
            "name": "My First Portfolio",
            "description": "Integration test portfolio",
            "initial_value": 10000.00,
            "risk_tolerance": "moderate",
        }
        portfolio_response = self.client.post(portfolio_url, portfolio_data, format="json")

        self.assertEqual(portfolio_response.status_code, status.HTTP_201_CREATED)
        portfolio_id = portfolio_response.data["id"]

        # Step 4: Add holdings to the portfolio
        holdings_data = [
            {
                "stock_symbol": "AAPL",
                "quantity": 10,
                "average_price": 150.00,
                "purchase_date": date.today().isoformat(),
            },
            {"stock_symbol": "MSFT", "quantity": 5, "average_price": 200.00, "purchase_date": date.today().isoformat()},
        ]

        for holding_data in holdings_data:
            add_holding_url = reverse("data:portfolio-add-holding", args=[portfolio_id])
            holding_response = self.client.post(add_holding_url, holding_data, format="json")
            self.assertEqual(holding_response.status_code, status.HTTP_201_CREATED)

        # Step 5: Verify portfolio details
        portfolio_detail_url = reverse("data:portfolio-detail", args=[portfolio_id])
        detail_response = self.client.get(portfolio_detail_url)

        self.assertEqual(detail_response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(detail_response.data["holdings"]), 2)

        # Step 6: Get portfolio performance
        performance_url = reverse("data:portfolio-performance", args=[portfolio_id])
        performance_response = self.client.get(performance_url)

        self.assertEqual(performance_response.status_code, status.HTTP_200_OK)
        self.assertIn("total_holdings", performance_response.data)
        self.assertIn("total_cost", performance_response.data)
        self.assertIn("total_value", performance_response.data)

        # Step 7: Get portfolio allocation
        allocation_url = reverse("data:portfolio-allocation", args=[portfolio_id])
        allocation_response = self.client.get(allocation_url)

        self.assertEqual(allocation_response.status_code, status.HTTP_200_OK)
        self.assertIn("by_stock", allocation_response.data)
        self.assertIn("by_sector", allocation_response.data)

    @patch("Data.services.yahoo_finance.yahoo_finance_service.get_stock_data")
    def test_stock_data_synchronization_workflow(self, mock_get_stock_data):
        """Test workflow for syncing stock data from Yahoo Finance."""

        # Mock Yahoo Finance response
        mock_get_stock_data.return_value = {
            "symbol": "AAPL",
            "prices": [155.00, 154.50, 156.20],
            "volumes": [50000000, 48000000, 52000000],
            "dates": [
                date.today().isoformat(),
                (date.today() - timedelta(days=1)).isoformat(),
                (date.today() - timedelta(days=2)).isoformat(),
            ],
        }

        # Step 1: Register and authenticate user
        User.objects.create_user(**self.user_data)
        login_url = reverse("core:auth-login")
        login_response = self.client.post(login_url, {"username": "testuser", "password": "testpass123"})

        self.assertEqual(login_response.status_code, status.HTTP_200_OK)
        access_token = login_response.data["access"]
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {access_token}")

        # Step 2: List available stocks
        stocks_url = reverse("data:stock-list")
        stocks_response = self.client.get(stocks_url)

        self.assertEqual(stocks_response.status_code, status.HTTP_200_OK)
        self.assertGreater(len(stocks_response.data["results"]), 0)

        # Step 3: Get stock details
        aapl_stock = self.stocks["AAPL"]
        stock_detail_url = reverse("data:stock-detail", args=[aapl_stock.id])
        stock_response = self.client.get(stock_detail_url)

        self.assertEqual(stock_response.status_code, status.HTTP_200_OK)
        self.assertEqual(stock_response.data["symbol"], "AAPL")

        # Step 4: Sync stock data
        sync_url = reverse("data:stock-sync", args=[aapl_stock.id])
        sync_response = self.client.post(sync_url, {"period": "1mo"})

        self.assertEqual(sync_response.status_code, status.HTTP_200_OK)
        self.assertIn("message", sync_response.data)
        mock_get_stock_data.assert_called_once()

        # Step 5: Get updated price history
        prices_url = reverse("data:stock-prices", args=[aapl_stock.id])
        prices_response = self.client.get(prices_url, {"days": 7})

        self.assertEqual(prices_response.status_code, status.HTTP_200_OK)
        self.assertIsInstance(prices_response.data, list)

        # Step 6: Get technical indicators (Analytics integration)
        indicators_url = reverse("analytics:technical-indicators", args=["AAPL"])
        indicators_response = self.client.get(indicators_url)

        self.assertEqual(indicators_response.status_code, status.HTTP_200_OK)
        self.assertIn("indicators", indicators_response.data)

    def test_market_data_integration(self):
        """Test integration between market data endpoints."""

        # Step 1: Get market overview
        market_overview_url = reverse("data:market-overview")
        overview_response = self.client.get(market_overview_url)

        self.assertEqual(overview_response.status_code, status.HTTP_200_OK)
        self.assertIn("market_status", overview_response.data)

        # Step 2: Get sector performance
        sector_url = reverse("data:sector-performance")
        sector_response = self.client.get(sector_url)

        self.assertEqual(sector_response.status_code, status.HTTP_200_OK)
        self.assertIn("sectors", sector_response.data)

        # Step 3: Compare stocks
        compare_url = reverse("data:compare-stocks")
        compare_data = {"symbols": ["AAPL", "MSFT"], "metrics": ["price", "market_cap"]}
        compare_response = self.client.post(compare_url, compare_data, format="json")

        self.assertEqual(compare_response.status_code, status.HTTP_200_OK)
        self.assertIn("comparison", compare_response.data)
        self.assertEqual(len(compare_response.data["comparison"]), 2)

    def test_analytics_integration_workflow(self):
        """Test analytics functionality with real data."""

        # Step 1: Get stock analysis
        analysis_url = reverse("analytics:stock-analysis", args=["AAPL"])
        analysis_response = self.client.get(analysis_url)

        self.assertEqual(analysis_response.status_code, status.HTTP_200_OK)
        self.assertIn("technical_indicators", analysis_response.data)

        # Step 2: Get market sentiment
        sentiment_url = reverse("analytics:market-sentiment")
        sentiment_response = self.client.get(sentiment_url)

        self.assertEqual(sentiment_response.status_code, status.HTTP_200_OK)
        self.assertIn("sentiment_score", sentiment_response.data)

        # Step 3: Create portfolio for analysis
        User.objects.create_user(**self.user_data)
        login_url = reverse("core:auth-login")
        login_response = self.client.post(login_url, {"username": "testuser", "password": "testpass123"})

        access_token = login_response.data["access"]
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {access_token}")

        # Create portfolio
        portfolio = Portfolio.objects.create(
            user_id=login_response.data["user"]["id"],
            name="Analytics Test Portfolio",
            initial_value=Decimal("10000.00"),
        )

        # Add holdings
        PortfolioHolding.objects.create(
            portfolio=portfolio,
            stock=self.stocks["AAPL"],
            quantity=Decimal("10"),
            average_price=Decimal("150.00"),
            current_price=Decimal("155.00"),
            purchase_date=date.today(),
        )

        # Step 4: Get portfolio analytics
        portfolio_analysis_url = reverse("analytics:portfolio-analysis", args=[portfolio.id])
        portfolio_response = self.client.get(portfolio_analysis_url)

        self.assertEqual(portfolio_response.status_code, status.HTTP_200_OK)
        self.assertIn("diversification", portfolio_response.data)

        # Step 5: Get risk assessment
        risk_url = reverse("analytics:risk-assessment", args=[portfolio.id])
        risk_response = self.client.get(risk_url)

        self.assertEqual(risk_response.status_code, status.HTTP_200_OK)
        self.assertIn("risk_score", risk_response.data)

    def test_error_handling_across_services(self):
        """Test error handling integration across different services."""

        # Step 1: Test authentication errors
        protected_url = reverse("data:portfolio-list")
        response = self.client.get(protected_url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

        # Step 2: Test invalid stock symbol
        invalid_stock_url = reverse("analytics:stock-analysis", args=["INVALID"])
        response = self.client.get(invalid_stock_url)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        # Step 3: Test invalid portfolio access
        User.objects.create_user(**self.user_data)
        other_user = User.objects.create_user(username="otheruser", password="otherpass")

        # Create portfolio for other user
        other_portfolio = Portfolio.objects.create(user=other_user, name="Other Portfolio")

        # Login as first user
        login_url = reverse("core:auth-login")
        login_response = self.client.post(login_url, {"username": "testuser", "password": "testpass123"})

        access_token = login_response.data["access"]
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {access_token}")

        # Try to access other user's portfolio
        portfolio_url = reverse("data:portfolio-detail", args=[other_portfolio.id])
        response = self.client.get(portfolio_url)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_data_consistency_across_operations(self):
        """Test that data remains consistent across multiple operations."""

        # Create user and authenticate
        User.objects.create_user(**self.user_data)
        login_url = reverse("core:auth-login")
        login_response = self.client.post(login_url, {"username": "testuser", "password": "testpass123"})

        access_token = login_response.data["access"]
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {access_token}")

        # Create portfolio
        portfolio_url = reverse("data:portfolio-list")
        portfolio_data = {"name": "Consistency Test Portfolio", "initial_value": 5000.00}
        portfolio_response = self.client.post(portfolio_url, portfolio_data, format="json")
        portfolio_id = portfolio_response.data["id"]

        # Add holding
        add_holding_url = reverse("data:portfolio-add-holding", args=[portfolio_id])
        holding_data = {
            "stock_symbol": "AAPL",
            "quantity": 10,
            "average_price": 150.00,
            "purchase_date": date.today().isoformat(),
        }
        self.client.post(add_holding_url, holding_data, format="json")

        # Check portfolio value is updated
        portfolio_detail_url = reverse("data:portfolio-detail", args=[portfolio_id])
        detail_response = self.client.get(portfolio_detail_url)

        self.assertEqual(len(detail_response.data["holdings"]), 1)

        # Update holding
        update_holding_url = reverse("data:portfolio-update-holding", args=[portfolio_id])
        update_data = {"symbol": "AAPL", "quantity": 20, "average_price": 145.00}
        self.client.post(update_holding_url, update_data, format="json")

        # Verify update
        detail_response = self.client.get(portfolio_detail_url)
        holding = detail_response.data["holdings"][0]
        self.assertEqual(float(holding["quantity"]), 20.0)

        # Remove holding
        remove_holding_url = reverse("data:portfolio-remove-holding", args=[portfolio_id])
        remove_data = {"symbol": "AAPL"}
        self.client.post(remove_holding_url, remove_data, format="json")

        # Verify removal
        detail_response = self.client.get(portfolio_detail_url)
        active_holdings = [h for h in detail_response.data["holdings"] if h["is_active"]]
        self.assertEqual(len(active_holdings), 0)

    @patch("Data.services.yahoo_finance.yahoo_finance_service.get_multiple_stocks")
    def test_bulk_operations_integration(self, mock_get_multiple):
        """Test bulk operations across the system."""

        mock_get_multiple.return_value = {
            "AAPL": {"symbol": "AAPL", "prices": [155.00]},
            "MSFT": {"symbol": "MSFT", "prices": [200.00]},
            "GOOGL": {"symbol": "GOOGL", "prices": [120.00]},
        }

        # Authenticate user
        User.objects.create_user(**self.user_data)
        login_url = reverse("core:auth-login")
        login_response = self.client.post(login_url, {"username": "testuser", "password": "testpass123"})

        access_token = login_response.data["access"]
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {access_token}")

        # Test bulk sync
        sync_watchlist_url = reverse("data:sync-watchlist")
        sync_data = {"symbols": ["AAPL", "MSFT", "GOOGL"], "period": "1mo"}
        sync_response = self.client.post(sync_watchlist_url, sync_data, format="json")

        self.assertEqual(sync_response.status_code, status.HTTP_200_OK)
        self.assertIn("success_count", sync_response.data)

        # Test bulk price update
        bulk_update_url = reverse("data:bulk-price-update")
        update_response = self.client.post(bulk_update_url)

        self.assertEqual(update_response.status_code, status.HTTP_200_OK)
        self.assertIn("updated", update_response.data)


class PerformanceIntegrationTest(APITestCase):
    """Integration tests focused on performance and scalability."""

    @classmethod
    def setUpTestData(cls):
        """Set up test data once for the entire test class."""
        # Create 100 stocks for performance testing
        for i in range(100):
            symbol = f"TEST{i:03d}"
            Stock.objects.create(
                symbol=symbol,
                short_name=f"Test Stock {i}",
                sector=["Technology", "Healthcare", "Finance", "Energy"][i % 4],
                market_cap=1000000000 * (i + 1),
                is_active=True,
            )

    def setUp(self):
        """Set up performance test data."""
        self.client = APIClient()
        self.user = User.objects.create_user(username="perfuser", password="perfpass123")
        self.client.force_authenticate(user=self.user)

    def test_large_portfolio_performance(self):
        """Test performance with large portfolio."""
        # Create portfolio with many holdings
        portfolio = Portfolio.objects.create(user=self.user, name="Large Portfolio", initial_value=Decimal("100000.00"))

        # Add 50 holdings
        stocks = Stock.objects.all()[:50]
        for i, stock in enumerate(stocks):
            PortfolioHolding.objects.create(
                portfolio=portfolio,
                stock=stock,
                quantity=Decimal("10"),
                average_price=Decimal("100.00") + i,
                current_price=Decimal("105.00") + i,
                purchase_date=date.today(),
            )

        # Test portfolio endpoints performance
        portfolio_detail_url = reverse("data:portfolio-detail", args=[portfolio.id])
        response = self.client.get(portfolio_detail_url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["holdings"]), 50)

        # Test performance analytics
        performance_url = reverse("data:portfolio-performance", args=[portfolio.id])
        perf_response = self.client.get(performance_url)

        self.assertEqual(perf_response.status_code, status.HTTP_200_OK)
        self.assertEqual(perf_response.data["total_holdings"], 50)

    def test_stock_list_pagination(self):
        """Test stock list pagination with large dataset."""
        stocks_url = reverse("data:stock-list")

        # Test first page
        response = self.client.get(stocks_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("results", response.data)
        self.assertIn("count", response.data)

        # Test pagination
        if response.data.get("next"):
            next_response = self.client.get(response.data["next"])
            self.assertEqual(next_response.status_code, status.HTTP_200_OK)

    def test_search_performance(self):
        """Test search functionality performance."""
        stocks_url = reverse("data:stock-list")

        # Test search
        search_response = self.client.get(stocks_url, {"search": "Technology"})
        self.assertEqual(search_response.status_code, status.HTTP_200_OK)

        # All results should be in Technology sector
        for stock in search_response.data["results"]:
            if "sector" in stock:
                self.assertIn("Technology", stock.get("sector", ""))

    def test_concurrent_operations(self):
        """Test handling of concurrent operations."""
        import threading

        portfolio = Portfolio.objects.create(user=self.user, name="Concurrent Test Portfolio")

        results = []

        def add_holding(stock_symbol):
            """Add holding to portfolio."""
            try:
                # Create a separate client instance for each thread to avoid race conditions
                thread_client = APIClient()
                thread_client.force_authenticate(user=self.user)

                add_url = reverse("data:portfolio-add-holding", args=[portfolio.id])
                response = thread_client.post(
                    add_url,
                    {
                        "stock_symbol": stock_symbol,
                        "quantity": 1,
                        "average_price": 100.00,
                        "purchase_date": date.today().isoformat(),
                    },
                    format="json",
                )
                results.append(response.status_code)
            except Exception as e:
                results.append(str(e))

        # Create multiple threads to add holdings concurrently
        threads = []
        stock_symbols = [f"TEST{i:03d}" for i in range(5)]

        for symbol in stock_symbols:
            thread = threading.Thread(target=add_holding, args=(symbol,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check that all concurrent operations succeeded
        success_count = sum(1 for result in results if result == 201)
        total_operations = len(stock_symbols)

        # For concurrency safety validation, all operations should succeed
        # since we're adding different stock symbols to the same portfolio
        self.assertEqual(
            success_count,
            total_operations,
            f"Expected all {total_operations} operations to succeed, but only {success_count} succeeded. "
            f"Results: {results}",
        )

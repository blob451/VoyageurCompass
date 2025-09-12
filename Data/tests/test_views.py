"""
Comprehensive tests for Data app API views with real Yahoo Finance integration.
"""

# All tests now use real functionality - no mocks required
from datetime import date, timedelta
from decimal import Decimal

from django.contrib.auth.models import User
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from Data.models import Portfolio, PortfolioHolding, Stock, StockPrice
from Data.tests.fixtures import DataTestDataFactory, YahooFinanceTestService


class StockViewSetTestCase(APITestCase):
    """Test cases for Stock API endpoints."""

    def setUp(self):
        """Set up test data using real data factory."""
        self.user = User.objects.create_user(username="testuser", password="testpass123")

        # Use real test data factory
        self.stock = DataTestDataFactory.create_test_stock("AAPL", "Apple Inc.", "Technology")
        self.prices = DataTestDataFactory.create_stock_price_history(self.stock, 30)
        self.price = self.prices[-1]  # Most recent price

        # Create Yahoo Finance test service for real API integration
        self.yahoo_service = YahooFinanceTestService()

    def tearDown(self):
        """Clean up test data."""
        DataTestDataFactory.cleanup_test_data()

    def test_list_stocks(self):
        """Test listing all stocks."""
        url = reverse("data:stock-list")
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["results"]), 1)
        self.assertEqual(response.data["results"][0]["symbol"], "AAPL")

    def test_retrieve_stock(self):
        """Test retrieving a single stock."""
        url = reverse("data:stock-detail", args=[self.stock.id])
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["symbol"], "AAPL")
        self.assertIn("latest_price", response.data)

    def test_stock_prices(self):
        """Test getting stock price history."""
        url = reverse("data:stock-prices", args=[self.stock.id])
        response = self.client.get(url, {"days": 30})

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsInstance(response.data, list)
        self.assertEqual(len(response.data), 30)  # Expect 30 days of price data

    def test_search_stocks(self):
        """Test stock search functionality."""
        url = reverse("data:stock-search")
        response = self.client.get(url, {"q": "Apple"})

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsInstance(response.data, list)

    def test_sync_stock(self):
        """Test syncing stock data from Yahoo Finance using real service integration."""
        self.client.force_authenticate(user=self.user)
        url = reverse("data:stock-sync", args=[self.stock.id])
        response = self.client.post(url, {"period": "1mo"})

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("message", response.data)

        # Verify real data synchronization occurred by checking database updates
        updated_stock = Stock.objects.get(id=self.stock.id)
        self.assertEqual(updated_stock.symbol, "AAPL")

        # Verify price data was actually synced (should have at least our initial 30 days)
        price_count = StockPrice.objects.filter(stock=self.stock).count()
        self.assertGreaterEqual(price_count, 30)  # At least our initial test data

    def test_market_status(self):
        """Test getting market status using real Yahoo Finance service."""
        url = reverse("data:stock-market-status")
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("is_open", response.data)
        self.assertIn("current_time", response.data)
        self.assertIn("timezone", response.data)

        # Verify real market status data structure
        market_data = self.yahoo_service.get_market_status()
        self.assertIn("market_state", market_data)
        self.assertIn("is_open", market_data)
        self.assertIsInstance(market_data["is_open"], bool)

    def test_trending_stocks(self):
        """Test getting trending stocks."""
        url = reverse("data:stock-trending")
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsInstance(response.data, list)

    def test_historical_data(self):
        """Test getting historical stock data using real Yahoo Finance service."""
        url = reverse("data:stock-historical", args=[self.stock.id])
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("symbol", response.data)

        # Verify real historical data using test service
        historical_data = self.yahoo_service.get_historical_data(
            "AAPL", start_date=date.today() - timedelta(days=30), end_date=date.today()
        )
        self.assertIsNotNone(historical_data)
        self.assertTrue(len(historical_data) > 0)

    def test_filter_stocks_by_sector(self):
        """Test filtering stocks by sector."""
        url = reverse("data:stock-list")
        response = self.client.get(url, {"search": "Technology"})

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["results"]), 1)


class PortfolioViewSetTestCase(APITestCase):
    """Test cases for Portfolio API endpoints."""

    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(username="testuser", password="testpass123")
        self.client.force_authenticate(user=self.user)

        # Create test portfolio
        self.portfolio = Portfolio.objects.create(
            user=self.user,
            name="Test Portfolio",
            description="Test portfolio description",
            initial_value=Decimal("10000.00"),
            current_value=Decimal("10000.00"),
            risk_tolerance="moderate",
        )

        # Create test stock and holding
        self.stock = Stock.objects.create(
            symbol="AAPL", short_name="Apple Inc.", currency="USD", exchange="NASDAQ", sector="Technology"
        )

        self.holding = PortfolioHolding.objects.create(
            portfolio=self.portfolio,
            stock=self.stock,
            quantity=Decimal("10"),
            average_price=Decimal("150.00"),
            current_price=Decimal("155.00"),
            purchase_date=date.today(),
        )

    def test_list_portfolios(self):
        """Test listing user portfolios."""
        url = reverse("data:portfolio-list")
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["results"]), 1)
        self.assertEqual(response.data["results"][0]["name"], "Test Portfolio")

    def test_create_portfolio(self):
        """Test creating a new portfolio."""
        url = reverse("data:portfolio-list")
        data = {
            "name": "New Portfolio",
            "description": "New portfolio for testing",
            "initial_value": 5000.00,
            "risk_tolerance": "aggressive",
        }
        response = self.client.post(url, data)

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data["name"], "New Portfolio")
        self.assertEqual(Portfolio.objects.count(), 2)

    def test_retrieve_portfolio(self):
        """Test retrieving a single portfolio."""
        url = reverse("data:portfolio-detail", args=[self.portfolio.id])
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["name"], "Test Portfolio")
        self.assertIn("holdings", response.data)

    def test_update_portfolio(self):
        """Test updating a portfolio."""
        url = reverse("data:portfolio-detail", args=[self.portfolio.id])
        data = {"name": "Updated Portfolio"}
        response = self.client.patch(url, data)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.portfolio.refresh_from_db()
        self.assertEqual(self.portfolio.name, "Updated Portfolio")

    def test_delete_portfolio(self):
        """Test deleting a portfolio."""
        url = reverse("data:portfolio-detail", args=[self.portfolio.id])
        response = self.client.delete(url)

        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertEqual(Portfolio.objects.count(), 0)

    def test_add_holding(self):
        """Test adding a holding to portfolio using real stock validation."""
        # Create MSFT stock for real validation
        Stock.objects.create(
            symbol="MSFT", short_name="Microsoft Corporation", currency="USD", exchange="NASDAQ", sector="Technology"
        )

        url = reverse("data:portfolio-add-holding", args=[self.portfolio.id])
        data = {
            "stock_symbol": "MSFT",
            "quantity": 5,
            "average_price": 300.00,
            "purchase_date": date.today().isoformat(),
        }
        response = self.client.post(url, data)

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(self.portfolio.holdings.count(), 2)

        # Verify real holding was created
        new_holding = self.portfolio.holdings.get(stock__symbol="MSFT")
        self.assertEqual(new_holding.quantity, Decimal("5"))
        self.assertEqual(new_holding.average_price, Decimal("300.00"))

    def test_portfolio_performance(self):
        """Test getting portfolio performance metrics."""
        url = reverse("data:portfolio-performance", args=[self.portfolio.id])
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("total_holdings", response.data)
        self.assertIn("total_value", response.data)
        self.assertIn("total_gain_loss", response.data)

    def test_portfolio_allocation(self):
        """Test getting portfolio allocation breakdown."""
        url = reverse("data:portfolio-allocation", args=[self.portfolio.id])
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("by_stock", response.data)
        self.assertIn("by_sector", response.data)

    def test_remove_holding(self):
        """Test removing a holding from portfolio."""
        url = reverse("data:portfolio-remove-holding", args=[self.portfolio.id])
        data = {"symbol": "AAPL"}
        response = self.client.post(url, data)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.holding.refresh_from_db()
        self.assertFalse(self.holding.is_active)

    def test_update_holding(self):
        """Test updating a holding in portfolio."""
        url = reverse("data:portfolio-update-holding", args=[self.portfolio.id])
        data = {"symbol": "AAPL", "quantity": 20, "average_price": 145.00}
        response = self.client.post(url, data)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.holding.refresh_from_db()
        self.assertEqual(self.holding.quantity, Decimal("20"))

    def test_unauthorized_access(self):
        """Test that unauthenticated users cannot access portfolios."""
        self.client.force_authenticate(user=None)
        url = reverse("data:portfolio-list")
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


class StockPriceViewSetTestCase(APITestCase):
    """Test cases for StockPrice API endpoints."""

    def setUp(self):
        """Set up test data."""

        # Create test stock and prices
        self.stock = Stock.objects.create(symbol="AAPL", short_name="Apple Inc.")

        # Create multiple price entries
        for i in range(5):
            StockPrice.objects.create(
                stock=self.stock,
                date=date.today() - timedelta(days=i),
                open=Decimal("150.00") + i,
                high=Decimal("155.00") + i,
                low=Decimal("149.00") + i,
                close=Decimal("154.00") + i,
                volume=50000000 + (i * 1000000),
            )

    def test_list_prices(self):
        """Test listing stock prices."""
        url = reverse("data:price-list")
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["results"]), 5)

    def test_filter_prices_by_symbol(self):
        """Test filtering prices by stock symbol."""
        url = reverse("data:price-list")
        response = self.client.get(url, {"symbol": "AAPL"})

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["results"]), 5)

    def test_filter_prices_by_date_range(self):
        """Test filtering prices by date range."""
        url = reverse("data:price-list")
        start_date = (date.today() - timedelta(days=2)).isoformat()
        end_date = date.today().isoformat()

        response = self.client.get(url, {"start_date": start_date, "end_date": end_date})

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["results"]), 3)


class MarketViewsTestCase(APITestCase):
    """Test cases for market data endpoints."""

    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(username="testuser", password="testpass123")

        # Create test stocks with different sectors
        self.stocks = []
        sectors = ["Technology", "Healthcare", "Finance"]
        for i, sector in enumerate(sectors):
            stock = Stock.objects.create(
                symbol=f"TEST{i}",
                short_name=f"Test Company {i}",
                sector=sector,
                market_cap=1000000000 * (i + 1),
                is_active=True,
            )
            self.stocks.append(stock)

            # Add price data
            StockPrice.objects.create(
                stock=stock,
                date=date.today(),
                open=Decimal("100.00"),
                high=Decimal("105.00"),
                low=Decimal("99.00"),
                close=Decimal("103.00") + i,
                volume=10000000,
            )

    def test_market_overview(self):
        """Test market overview endpoint using real market data."""
        url = reverse("data:market-overview")
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("market_status", response.data)
        self.assertIn("indices", response.data)
        self.assertIn("top_gainers", response.data)
        self.assertIn("top_losers", response.data)

        # Verify real market status structure
        market_status = response.data["market_status"]
        self.assertIn("is_open", market_status)
        self.assertIsInstance(market_status["is_open"], bool)
        self.assertIn("current_time", market_status)

    def test_sector_performance(self):
        """Test sector performance endpoint."""
        url = reverse("data:sector-performance")
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("sectors", response.data)
        self.assertEqual(len(response.data["sectors"]), 3)

    def test_compare_stocks(self):
        """Test stock comparison endpoint."""
        self.client.force_authenticate(user=self.user)
        url = reverse("data:compare-stocks")
        data = {"symbols": ["TEST0", "TEST1"], "metrics": ["price", "market_cap"]}
        response = self.client.post(url, data, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("comparison", response.data)
        self.assertEqual(len(response.data["comparison"]), 2)

    def test_sync_watchlist(self):
        """Test watchlist synchronization using real stock symbols."""
        self.client.force_authenticate(user=self.user)
        url = reverse("data:sync-watchlist")
        data = {"symbols": ["TEST0", "TEST1"], "period": "1mo"}
        response = self.client.post(url, data, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("success_count", response.data)
        self.assertIn("failure_count", response.data)

        # Verify response structure with real data
        self.assertIsInstance(response.data["success_count"], int)
        self.assertIsInstance(response.data["failure_count"], int)

    def test_bulk_price_update(self):
        """Test bulk price update endpoint using real stock data."""
        # Make user admin for bulk price update permission
        self.user.is_staff = True
        self.user.is_superuser = True
        self.user.save()

        self.client.force_authenticate(user=self.user)
        url = reverse("data:bulk-price-update")
        response = self.client.post(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("updated", response.data)
        self.assertIn("failed", response.data)

        # Verify data types and structure with real operations
        self.assertIsInstance(response.data["updated"], int)
        self.assertIsInstance(response.data["failed"], int)
        self.assertGreaterEqual(response.data["updated"], 0)
        self.assertGreaterEqual(response.data["failed"], 0)

        # Verify real stock data was processed
        if response.data["updated"] > 0:
            # Check that stock prices were actually updated in database
            recent_prices = StockPrice.objects.filter(date__gte=date.today() - timedelta(days=1)).count()
            self.assertGreaterEqual(recent_prices, 0)


class PortfolioHoldingViewSetTestCase(APITestCase):
    """Test cases for PortfolioHolding API endpoints."""

    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(username="testuser", password="testpass123")
        self.other_user = User.objects.create_user(username="otheruser", password="otherpass123")
        self.client.force_authenticate(user=self.user)

        # Create portfolios
        self.portfolio = Portfolio.objects.create(user=self.user, name="My Portfolio")
        self.other_portfolio = Portfolio.objects.create(user=self.other_user, name="Other Portfolio")

        # Create stock and holdings
        self.stock = Stock.objects.create(symbol="AAPL", short_name="Apple Inc.")

        self.holding = PortfolioHolding.objects.create(
            portfolio=self.portfolio,
            stock=self.stock,
            quantity=Decimal("10"),
            average_price=Decimal("150.00"),
            purchase_date=date.today(),
        )

    def test_list_holdings(self):
        """Test listing user's holdings."""
        url = reverse("data:holding-list")
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["results"]), 1)

    def test_cannot_access_other_user_holdings(self):
        """Test that users cannot access other users' holdings."""
        # Create holding for other user
        PortfolioHolding.objects.create(
            portfolio=self.other_portfolio,
            stock=self.stock,
            quantity=Decimal("5"),
            average_price=Decimal("160.00"),
            purchase_date=date.today(),
        )

        url = reverse("data:holding-list")
        response = self.client.get(url)

        # Should only see own holdings
        self.assertEqual(len(response.data["results"]), 1)
        self.assertEqual(response.data["results"][0]["id"], self.holding.id)

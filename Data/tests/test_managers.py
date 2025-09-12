"""
Unit tests for custom database managers and querysets with data source filtering.
"""

from django.db import models
from django.test import TestCase

from Data.models import DataSourceChoices, Stock


class TestStockQuerySet(TestCase):
    """Test custom StockQuerySet filtering methods."""

    @classmethod
    def setUpTestData(cls):
        """Create test stocks with various data source and activity states."""
        cls.stock_yahoo = Stock.objects.create(
            symbol="AAPL", short_name="Apple Inc.", data_source=DataSourceChoices.YAHOO, is_active=True
        )

        cls.stock_mock = Stock.objects.create(
            symbol="MOCK1", short_name="Mock Stock 1", data_source=DataSourceChoices.MOCK, is_active=True
        )

        cls.stock_mock_inactive = Stock.objects.create(
            symbol="MOCK2", short_name="Mock Stock 2", data_source=DataSourceChoices.MOCK, is_active=False
        )

        cls.stock_yahoo_inactive = Stock.objects.create(
            symbol="MSFT", short_name="Microsoft", data_source=DataSourceChoices.YAHOO, is_active=False
        )

        cls.stock_null = Stock.objects.create(
            symbol="NULL1",
            short_name="Null Source Stock",
            data_source="",  # Empty string as Django typically doesn't allow NULL for CharField
            is_active=True,
        )

        cls.stock_other = Stock.objects.create(
            symbol="OTHER", short_name="Other Source Stock", data_source="other", is_active=True
        )

    def test_real_method_excludes_mock(self):
        """Test that real() excludes mock data."""
        real_stocks = Stock.objects.all().real()

        # Should include yahoo, null, and other sources
        self.assertIn(self.stock_yahoo, real_stocks)
        self.assertIn(self.stock_yahoo_inactive, real_stocks)
        self.assertIn(self.stock_null, real_stocks)
        self.assertIn(self.stock_other, real_stocks)

        # Should exclude mock sources
        self.assertNotIn(self.stock_mock, real_stocks)
        self.assertNotIn(self.stock_mock_inactive, real_stocks)

    def test_mock_method_filters_mock_only(self):
        """Test that mock() returns only mock data."""
        mock_stocks = Stock.objects.all().mock()

        # Should include only mock sources
        self.assertIn(self.stock_mock, mock_stocks)
        self.assertIn(self.stock_mock_inactive, mock_stocks)

        # Should exclude all non-mock sources
        self.assertNotIn(self.stock_yahoo, mock_stocks)
        self.assertNotIn(self.stock_yahoo_inactive, mock_stocks)
        self.assertNotIn(self.stock_null, mock_stocks)
        self.assertNotIn(self.stock_other, mock_stocks)

    def test_active_method_filters_active_only(self):
        """Test that active() returns only active stocks."""
        active_stocks = Stock.objects.all().active()

        # Should include active stocks regardless of data source
        self.assertIn(self.stock_yahoo, active_stocks)
        self.assertIn(self.stock_mock, active_stocks)
        self.assertIn(self.stock_null, active_stocks)
        self.assertIn(self.stock_other, active_stocks)

        # Should exclude inactive stocks
        self.assertNotIn(self.stock_mock_inactive, active_stocks)
        self.assertNotIn(self.stock_yahoo_inactive, active_stocks)

    def test_method_chaining_active_real(self):
        """Test chaining active() and real() methods."""
        active_real_stocks = Stock.objects.all().active().real()

        # Should include active non-mock stocks
        self.assertIn(self.stock_yahoo, active_real_stocks)
        self.assertIn(self.stock_null, active_real_stocks)
        self.assertIn(self.stock_other, active_real_stocks)

        # Should exclude mock stocks (even if active)
        self.assertNotIn(self.stock_mock, active_real_stocks)

        # Should exclude inactive stocks (even if real)
        self.assertNotIn(self.stock_yahoo_inactive, active_real_stocks)
        self.assertNotIn(self.stock_mock_inactive, active_real_stocks)

    def test_method_chaining_real_active(self):
        """Test chaining real() and active() methods (reverse order)."""
        real_active_stocks = Stock.objects.all().real().active()

        # Result should be the same regardless of order
        active_real_stocks = Stock.objects.all().active().real()

        self.assertQuerySetEqual(
            real_active_stocks.order_by("symbol"), active_real_stocks.order_by("symbol"), transform=lambda x: x
        )

    def test_method_chaining_mock_active(self):
        """Test chaining mock() and active() methods."""
        mock_active_stocks = Stock.objects.all().mock().active()

        # Should include only active mock stocks
        self.assertIn(self.stock_mock, mock_active_stocks)

        # Should exclude inactive mock stocks
        self.assertNotIn(self.stock_mock_inactive, mock_active_stocks)

        # Should exclude all non-mock stocks
        self.assertNotIn(self.stock_yahoo, mock_active_stocks)
        self.assertNotIn(self.stock_null, mock_active_stocks)


class TestStockManager(TestCase):
    """Test the StockManager custom manager."""

    @classmethod
    def setUpTestData(cls):
        """Create test stocks."""
        cls.stock_yahoo_active = Stock.objects.create(
            symbol="GOOGL", short_name="Google", data_source=DataSourceChoices.YAHOO, is_active=True
        )

        cls.stock_mock_active = Stock.objects.create(
            symbol="TEST1", short_name="Test Stock", data_source=DataSourceChoices.MOCK, is_active=True
        )

        cls.stock_yahoo_inactive = Stock.objects.create(
            symbol="AMZN", short_name="Amazon", data_source=DataSourceChoices.YAHOO, is_active=False
        )

    def test_manager_real_data_method(self):
        """Test StockManager.real_data() method."""
        real_stocks = Stock.objects.real_data()

        self.assertIn(self.stock_yahoo_active, real_stocks)
        self.assertIn(self.stock_yahoo_inactive, real_stocks)
        self.assertNotIn(self.stock_mock_active, real_stocks)

    def test_manager_mock_data_method(self):
        """Test StockManager.mock_data() method."""
        mock_stocks = Stock.objects.mock_data()

        self.assertIn(self.stock_mock_active, mock_stocks)
        self.assertNotIn(self.stock_yahoo_active, mock_stocks)
        self.assertNotIn(self.stock_yahoo_inactive, mock_stocks)

    def test_manager_active_real_stocks_method(self):
        """Test StockManager.active_real_stocks() method."""
        active_real = Stock.objects.active_real_stocks()

        self.assertIn(self.stock_yahoo_active, active_real)
        self.assertNotIn(self.stock_mock_active, active_real)
        self.assertNotIn(self.stock_yahoo_inactive, active_real)

    def test_manager_returns_queryset(self):
        """Test that manager methods return QuerySet instances."""
        # All methods should return QuerySet instances
        self.assertIsInstance(Stock.objects.real_data(), models.QuerySet)
        self.assertIsInstance(Stock.objects.mock_data(), models.QuerySet)
        self.assertIsInstance(Stock.objects.active_real_stocks(), models.QuerySet)

    def test_manager_chaining_with_filters(self):
        """Test that manager methods can be chained with standard filters."""
        # Should be able to chain with filter()
        result = Stock.objects.real_data().filter(symbol="GOOGL")
        self.assertEqual(result.count(), 1)
        self.assertEqual(result.first(), self.stock_yahoo_active)

        # Should be able to chain with exclude()
        result = Stock.objects.real_data().exclude(is_active=False)
        self.assertIn(self.stock_yahoo_active, result)
        self.assertNotIn(self.stock_yahoo_inactive, result)


class TestRealDataManager(TestCase):
    """Test the RealDataManager that excludes mock data by default."""

    @classmethod
    def setUpTestData(cls):
        """Create test stocks."""
        cls.stock_real = Stock.objects.create(
            symbol="TSLA", short_name="Tesla", data_source=DataSourceChoices.YAHOO, is_active=True
        )

        cls.stock_mock = Stock.objects.create(
            symbol="FAKE1", short_name="Fake Stock", data_source=DataSourceChoices.MOCK, is_active=True
        )

        cls.stock_real_inactive = Stock.objects.create(
            symbol="NFLX", short_name="Netflix", data_source=DataSourceChoices.YAHOO, is_active=False
        )

    def test_real_data_manager_default_excludes_mock(self):
        """Test that RealDataManager excludes mock data by default."""
        # Using real_data manager should exclude mock by default
        real_stocks = Stock.real_data.all()

        self.assertIn(self.stock_real, real_stocks)
        self.assertIn(self.stock_real_inactive, real_stocks)
        self.assertNotIn(self.stock_mock, real_stocks)

    def test_real_data_manager_has_queryset_methods(self):
        """Test that RealDataManager exposes StockQuerySet methods."""
        # Should have access to active() method
        active_real = Stock.real_data.active()
        self.assertIn(self.stock_real, active_real)
        self.assertNotIn(self.stock_real_inactive, active_real)
        self.assertNotIn(self.stock_mock, active_real)

        # Mock() method on real_data manager returns empty because
        # real_data.get_queryset() already excludes mock, then mock() filters for mock
        # This results in an empty queryset (correct behavior)
        mock_stocks = Stock.real_data.mock()
        self.assertEqual(mock_stocks.count(), 0)

        # Should have access to real() method (redundant but should work)
        real_stocks = Stock.real_data.real()
        self.assertIn(self.stock_real, real_stocks)
        self.assertNotIn(self.stock_mock, real_stocks)

    def test_real_data_manager_chaining(self):
        """Test chaining methods on RealDataManager."""
        # Chain active() with default real filtering
        result = Stock.real_data.active()
        self.assertEqual(result.count(), 1)
        self.assertEqual(result.first(), self.stock_real)

        # Chain with standard Django filters
        result = Stock.real_data.filter(symbol="TSLA")
        self.assertEqual(result.count(), 1)
        self.assertEqual(result.first(), self.stock_real)


class TestEdgeCases(TestCase):
    """Test edge cases and special scenarios."""

    def test_empty_queryset_methods(self):
        """Test methods on empty querysets."""
        # Clear all stocks
        Stock.objects.all().delete()

        # Methods should return empty querysets without errors
        self.assertEqual(Stock.objects.real_data().count(), 0)
        self.assertEqual(Stock.objects.mock_data().count(), 0)
        self.assertEqual(Stock.objects.active_real_stocks().count(), 0)
        self.assertEqual(Stock.real_data.all().count(), 0)
        self.assertEqual(Stock.real_data.active().count(), 0)

    def test_null_data_source_handling(self):
        """Test handling of null/empty data_source values."""
        # Create stock with empty data_source
        stock_empty = Stock.objects.create(symbol="EMPTY", short_name="Empty Source", data_source="", is_active=True)

        # Empty string should not match 'mock'
        real_stocks = Stock.objects.real_data()
        mock_stocks = Stock.objects.mock_data()

        self.assertIn(stock_empty, real_stocks)
        self.assertNotIn(stock_empty, mock_stocks)

    def test_case_sensitivity(self):
        """Test that data_source comparison is case-sensitive."""
        # Create stock with uppercase 'MOCK'
        stock_upper = Stock.objects.create(
            symbol="UPPER", short_name="Upper Mock", data_source="MOCK", is_active=True  # Uppercase
        )

        # Should not be treated as mock (case-sensitive)
        mock_stocks = Stock.objects.mock_data()
        real_stocks = Stock.objects.real_data()

        self.assertNotIn(stock_upper, mock_stocks)
        self.assertIn(stock_upper, real_stocks)

    def test_multiple_chaining(self):
        """Test complex chaining scenarios."""
        # Create diverse test data
        Stock.objects.all().delete()

        Stock.objects.create(symbol="A1", data_source=DataSourceChoices.YAHOO, is_active=True, sector="Tech")
        Stock.objects.create(symbol="A2", data_source=DataSourceChoices.MOCK, is_active=True, sector="Tech")
        Stock.objects.create(symbol="A3", data_source=DataSourceChoices.YAHOO, is_active=False, sector="Tech")

        # Complex chain: real -> active -> filter by sector
        result = Stock.objects.all().real().active().filter(sector="Tech")
        self.assertEqual(result.count(), 1)
        self.assertEqual(result.first().symbol, "A1")

"""
Comprehensive tests for Data app models.
"""

from datetime import date, timedelta
from decimal import Decimal

from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.db import IntegrityError
from django.utils import timezone

import pytest

from Data.models import (
    DataIndustry,
    DataSector,
    Portfolio,
    PortfolioHolding,
    Stock,
    StockPrice,
)


@pytest.mark.django_db
class TestStockModel:
    """Test cases for Stock model."""

    def test_create_stock(self):
        """Test creating a stock instance."""
        stock = Stock.objects.create(
            symbol="AAPL",
            short_name="Apple Inc.",
            long_name="Apple Inc.",
            currency="USD",
            exchange="NASDAQ",
            sector="Technology",
            industry="Consumer Electronics",
            country="USA",
            website="https://www.apple.com",
            description="Technology company",
            market_cap=3000000000000,
            shares_outstanding=15500000000,
            is_active=True,
        )

        assert stock.symbol == "AAPL"
        assert stock.short_name == "Apple Inc."
        assert stock.sector == "Technology"
        assert stock.is_active
        assert str(stock) == "AAPL - Apple Inc."

    def test_stock_unique_symbol(self):
        """Test that stock symbols must be unique."""
        Stock.objects.create(symbol="AAPL", short_name="Apple")

        with pytest.raises(IntegrityError):
            Stock.objects.create(symbol="AAPL", short_name="Another Apple")

    def test_get_latest_price(self):
        """Test getting the latest price for a stock."""
        stock = Stock.objects.create(symbol="AAPL", short_name="Apple")

        # Create multiple prices
        older_price = StockPrice.objects.create(
            stock=stock,
            date=date.today() - timedelta(days=2),
            open=Decimal("150.00"),
            high=Decimal("155.00"),
            low=Decimal("149.00"),
            close=Decimal("154.00"),
            volume=50000000,
        )

        latest_price = StockPrice.objects.create(
            stock=stock,
            date=date.today(),
            open=Decimal("155.00"),
            high=Decimal("160.00"),
            low=Decimal("154.00"),
            close=Decimal("159.00"),
            volume=60000000,
        )

        result = stock.get_latest_price()
        assert result == latest_price
        assert result.close == Decimal("159.00")

    def test_get_price_history(self):
        """Test getting price history for a stock."""
        stock = Stock.objects.create(symbol="AAPL", short_name="Apple")

        # Create prices for different dates
        for i in range(10):
            StockPrice.objects.create(
                stock=stock,
                date=date.today() - timedelta(days=i),
                open=Decimal("150.00"),
                high=Decimal("155.00"),
                low=Decimal("149.00"),
                close=Decimal("154.00") + i,
                volume=50000000,
            )

        # Get last 7 days (includes today + 7 previous days = 8 total)
        history = stock.get_price_history(days=7)
        assert history.count() == 8

        # Get last 30 days (should return all 10)
        history = stock.get_price_history(days=30)
        assert history.count() == 10

    def test_needs_sync_property(self):
        """Test the needs_sync property."""
        stock = Stock.objects.create(symbol="AAPL", short_name="Apple")

        # No last_sync, should need sync
        assert stock.needs_sync is True

        # Recent sync, should not need sync
        stock.last_sync = timezone.now()
        stock.save()
        assert stock.needs_sync is False

        # Old sync (2 hours ago), should need sync
        stock.last_sync = timezone.now() - timedelta(hours=2)
        stock.save()
        assert stock.needs_sync is True


@pytest.mark.django_db
class TestStockPriceModel:
    """Test cases for StockPrice model."""

    def test_create_stock_price(self):
        """Test creating a stock price instance."""
        stock = Stock.objects.create(symbol="AAPL", short_name="Apple")
        price = StockPrice.objects.create(
            stock=stock,
            date=date.today(),
            open=Decimal("150.00"),
            high=Decimal("155.00"),
            low=Decimal("149.00"),
            close=Decimal("154.00"),
            adjusted_close=Decimal("153.50"),
            volume=50000000,
        )

        assert price.stock == stock
        assert price.open == Decimal("150.00")
        assert price.close == Decimal("154.00")
        assert str(price) == f"AAPL - {date.today()}: $154.00"

    def test_unique_stock_date_constraint(self):
        """Test that stock-date combination must be unique."""
        stock = Stock.objects.create(symbol="AAPL", short_name="Apple")
        StockPrice.objects.create(
            stock=stock,
            date=date.today(),
            open=Decimal("150.00"),
            high=Decimal("155.00"),
            low=Decimal("149.00"),
            close=Decimal("154.00"),
            volume=50000000,
        )

        # Try to create another price for the same stock and date
        with pytest.raises(IntegrityError):
            StockPrice.objects.create(
                stock=stock,
                date=date.today(),
                open=Decimal("151.00"),
                high=Decimal("156.00"),
                low=Decimal("150.00"),
                close=Decimal("155.00"),
                volume=60000000,
            )

    def test_daily_change_property(self):
        """Test the daily_change property."""
        stock = Stock.objects.create(symbol="AAPL", short_name="Apple")
        price = StockPrice.objects.create(
            stock=stock,
            date=date.today(),
            open=Decimal("150.00"),
            high=Decimal("155.00"),
            low=Decimal("149.00"),
            close=Decimal("154.00"),
            volume=50000000,
        )

        assert price.daily_change == Decimal("4.00")

    def test_daily_change_percent_property(self):
        """Test the daily_change_percent property."""
        stock = Stock.objects.create(symbol="AAPL", short_name="Apple")
        price = StockPrice.objects.create(
            stock=stock,
            date=date.today(),
            open=Decimal("150.00"),
            high=Decimal("155.00"),
            low=Decimal("149.00"),
            close=Decimal("154.00"),
            volume=50000000,
        )

        expected_percent = (Decimal("4.00") / Decimal("150.00")) * Decimal("100")
        assert abs(price.daily_change_percent - expected_percent) < Decimal("0.01")

    def test_daily_range_property(self):
        """Test the daily_range property."""
        stock = Stock.objects.create(symbol="AAPL", short_name="Apple")
        price = StockPrice.objects.create(
            stock=stock,
            date=date.today(),
            open=Decimal("150.00"),
            high=Decimal("155.00"),
            low=Decimal("149.00"),
            close=Decimal("154.00"),
            volume=50000000,
        )

        assert price.daily_range == "149.00 - 155.00"

    def test_is_gain_property(self):
        """Test the is_gain property."""
        stock = Stock.objects.create(symbol="AAPL", short_name="Apple")

        # Gain day
        gain_price = StockPrice.objects.create(
            stock=stock,
            date=date.today(),
            open=Decimal("150.00"),
            high=Decimal("155.00"),
            low=Decimal("149.00"),
            close=Decimal("154.00"),
            volume=50000000,
        )
        assert gain_price.is_gain is True

        # Loss day
        loss_price = StockPrice.objects.create(
            stock=stock,
            date=date.today() - timedelta(days=1),
            open=Decimal("154.00"),
            high=Decimal("155.00"),
            low=Decimal("149.00"),
            close=Decimal("150.00"),
            volume=50000000,
        )
        assert loss_price.is_gain is False


@pytest.mark.django_db
class TestPortfolioModel:
    """Test cases for Portfolio model."""

    def test_create_portfolio(self):
        """Test creating a portfolio instance."""
        user = User.objects.create_user(username="testuser", password="testpass")
        portfolio = Portfolio.objects.create(
            user=user,
            name="My Portfolio",
            description="Test portfolio",
            initial_value=Decimal("10000.00"),
            current_value=Decimal("11000.00"),
            risk_tolerance="moderate",
        )

        assert portfolio.user == user
        assert portfolio.name == "My Portfolio"
        assert portfolio.initial_value == Decimal("10000.00")
        assert portfolio.risk_tolerance == "moderate"
        assert str(portfolio) == "My Portfolio"

    def test_calculate_returns(self):
        """Test calculating portfolio returns."""
        user = User.objects.create_user(username="testuser", password="testpass")
        portfolio = Portfolio.objects.create(
            user=user,
            name="My Portfolio",
            initial_value=Decimal("10000.00"),
            current_value=Decimal("12000.00"),
        )

        returns = portfolio.calculate_returns()
        assert returns == Decimal("20.00")  # 20% return

    def test_update_value(self):
        """Test updating portfolio value based on holdings."""
        user = User.objects.create_user(username="testuser", password="testpass")
        portfolio = Portfolio.objects.create(
            user=user, name="My Portfolio", initial_value=Decimal("10000.00")
        )

        # Create holdings
        stock1 = Stock.objects.create(symbol="AAPL", short_name="Apple")
        stock2 = Stock.objects.create(symbol="MSFT", short_name="Microsoft")

        PortfolioHolding.objects.create(
            portfolio=portfolio,
            stock=stock1,
            quantity=Decimal("10"),
            average_price=Decimal("150.00"),
            current_price=Decimal("160.00"),
            purchase_date=date.today(),
        )

        PortfolioHolding.objects.create(
            portfolio=portfolio,
            stock=stock2,
            quantity=Decimal("5"),
            average_price=Decimal("300.00"),
            current_price=Decimal("320.00"),
            purchase_date=date.today(),
        )

        portfolio.update_value()
        portfolio.refresh_from_db()

        # 10 * 160 + 5 * 320 = 1600 + 1600 = 3200
        assert portfolio.current_value == Decimal("3200.00")


@pytest.mark.django_db
class TestPortfolioHoldingModel:
    """Test cases for PortfolioHolding model."""

    def test_create_holding(self):
        """Test creating a portfolio holding."""
        user = User.objects.create_user(username="testuser", password="testpass")
        portfolio = Portfolio.objects.create(user=user, name="My Portfolio")
        stock = Stock.objects.create(symbol="AAPL", short_name="Apple")

        holding = PortfolioHolding.objects.create(
            portfolio=portfolio,
            stock=stock,
            quantity=Decimal("10"),
            average_price=Decimal("150.00"),
            current_price=Decimal("160.00"),
            purchase_date=date.today(),
        )

        assert holding.portfolio == portfolio
        assert holding.stock == stock
        assert holding.quantity == Decimal("10")
        assert str(holding) == "My Portfolio - AAPL: 10 shares"

    def test_automatic_calculations_on_save(self):
        """Test that derived fields are calculated automatically."""
        user = User.objects.create_user(username="testuser", password="testpass")
        portfolio = Portfolio.objects.create(user=user, name="My Portfolio")
        stock = Stock.objects.create(symbol="AAPL", short_name="Apple")

        holding = PortfolioHolding.objects.create(
            portfolio=portfolio,
            stock=stock,
            quantity=Decimal("10"),
            average_price=Decimal("150.00"),
            current_price=Decimal("160.00"),
            purchase_date=date.today(),
        )

        # Check calculated fields
        assert holding.cost_basis == Decimal("1500.00")  # 10 * 150
        assert holding.current_value == Decimal("1600.00")  # 10 * 160
        assert holding.unrealized_gain_loss == Decimal("100.00")  # 1600 - 1500
        assert abs(holding.unrealized_gain_loss_percent - Decimal("6.67")) < Decimal(
            "0.01"
        )

    def test_unique_portfolio_stock_constraint(self):
        """Test that portfolio-stock combination must be unique."""
        user = User.objects.create_user(username="testuser", password="testpass")
        portfolio = Portfolio.objects.create(user=user, name="My Portfolio")
        stock = Stock.objects.create(symbol="AAPL", short_name="Apple")

        PortfolioHolding.objects.create(
            portfolio=portfolio,
            stock=stock,
            quantity=Decimal("10"),
            average_price=Decimal("150.00"),
            purchase_date=date.today(),
        )

        # Try to create another holding for the same stock in the same portfolio
        with pytest.raises(IntegrityError):
            PortfolioHolding.objects.create(
                portfolio=portfolio,
                stock=stock,
                quantity=Decimal("5"),
                average_price=Decimal("155.00"),
                purchase_date=date.today(),
            )


@pytest.mark.django_db
class TestDataSectorModel:
    """Test cases for DataSector model."""

    def test_create_data_sector(self):
        """Test creating a data sector instance."""
        sector = DataSector.objects.create(
            sectorKey="technology",
            sectorName="Technology",
            isActive=True,
        )

        assert sector.sectorKey == "technology"
        assert sector.sectorName == "Technology"
        assert sector.isActive is True
        assert str(sector) == "Technology"

    def test_sector_unique_key(self):
        """Test that sector keys must be unique."""
        DataSector.objects.create(sectorKey="technology", sectorName="Technology")

        with pytest.raises(IntegrityError):
            DataSector.objects.create(sectorKey="technology", sectorName="Tech")

    def test_sector_str_representation(self):
        """Test the string representation of DataSector."""
        sector = DataSector.objects.create(
            sectorKey="healthcare", sectorName="Healthcare Services"
        )
        assert str(sector) == "Healthcare Services"

    def test_sector_meta_ordering(self):
        """Test that sectors are ordered by name."""
        DataSector.objects.create(sectorKey="tech", sectorName="Technology")
        DataSector.objects.create(sectorKey="health", sectorName="Healthcare")

        sectors = list(DataSector.objects.all())
        assert sectors[0].sectorName == "Healthcare"
        assert sectors[1].sectorName == "Technology"


@pytest.mark.django_db
class TestDataIndustryModel:
    """Test cases for DataIndustry model."""

    def test_create_data_industry(self):
        """Test creating a data industry instance."""
        sector = DataSector.objects.create(
            sectorKey="technology", sectorName="Technology"
        )
        industry = DataIndustry.objects.create(
            sector=sector,
            industryKey="software_infrastructure",
            industryName="Software - Infrastructure",
            isActive=True,
        )

        assert industry.sector == sector
        assert industry.industryKey == "software_infrastructure"
        assert industry.industryName == "Software - Infrastructure"
        assert industry.isActive is True
        assert str(industry) == "Software - Infrastructure (Technology)"

    def test_industry_unique_key(self):
        """Test that industry keys must be unique."""
        sector = DataSector.objects.create(
            sectorKey="technology", sectorName="Technology"
        )
        DataIndustry.objects.create(
            sector=sector,
            industryKey="software_infrastructure",
            industryName="Software - Infrastructure",
        )

        with pytest.raises(IntegrityError):
            DataIndustry.objects.create(
                sector=sector,
                industryKey="software_infrastructure",
                industryName="Software Infrastructure",
            )

    def test_industry_str_representation(self):
        """Test the string representation of DataIndustry."""
        sector = DataSector.objects.create(
            sectorKey="healthcare", sectorName="Healthcare"
        )
        industry = DataIndustry.objects.create(
            sector=sector,
            industryKey="biotech",
            industryName="Biotechnology",
        )
        assert str(industry) == "Biotechnology (Healthcare)"

    def test_industry_foreign_key_relationship(self):
        """Test the foreign key relationship with DataSector."""
        sector = DataSector.objects.create(
            sectorKey="technology", sectorName="Technology"
        )
        industry = DataIndustry.objects.create(
            sector=sector,
            industryKey="software",
            industryName="Software Services",
        )

        # Test the relationship
        assert industry.sector == sector
        assert industry in sector.industries.all()

        # Test cascade deletion
        sector_id = sector.id
        sector.delete()
        with pytest.raises(DataIndustry.DoesNotExist):
            DataIndustry.objects.get(id=industry.id)

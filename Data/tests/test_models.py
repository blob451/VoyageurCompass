"""
Comprehensive tests for Data app models.
"""

from django.test import TestCase
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.db import IntegrityError
from datetime import date, timedelta
from decimal import Decimal
from django.utils import timezone

from Data.models import Stock, StockPrice, Portfolio, PortfolioHolding


class TestStockModel(TestCase):
    """Test cases for Stock model."""
    
    def test_create_stock(self):
        """Test creating a stock instance."""
        stock = Stock.objects.create(
            symbol='AAPL',
            short_name='Apple Inc.',
            long_name='Apple Inc.',
            currency='USD',
            exchange='NASDAQ',
            sector='Technology',
            industry='Consumer Electronics',
            country='USA',
            website='https://www.apple.com',
            description='Technology company',
            market_cap=3000000000000,
            shares_outstanding=15500000000,
            is_active=True
        )
        
        self.assertEqual(stock.symbol, 'AAPL')
        self.assertEqual(stock.short_name, 'Apple Inc.')
        self.assertEqual(stock.sector, 'Technology')
        self.assertTrue(stock.is_active)
        self.assertEqual(str(stock), 'AAPL - Apple Inc.')
    
    def test_stock_unique_symbol(self):
        """Test that stock symbols must be unique."""
        Stock.objects.get_or_create(symbol='AAPL_UNIQUE_TEST', defaults={'short_name': 'Apple'})
        
        with self.assertRaises(IntegrityError):
            Stock.objects.create(symbol='AAPL_UNIQUE_TEST', short_name='Another Apple')
    
    def test_get_latest_price(self):
        """Test getting the latest price for a stock."""
        stock, _ = Stock.objects.get_or_create(symbol='AAPL_PRICE', defaults={'short_name': 'Apple Price Test'})
        
        # Create multiple prices
        older_price = StockPrice.objects.create(
            stock=stock,
            date=date.today() - timedelta(days=2),
            open=Decimal('150.00'),
            high=Decimal('155.00'),
            low=Decimal('149.00'),
            close=Decimal('154.00'),
            volume=50000000
        )
        
        latest_price = StockPrice.objects.create(
            stock=stock,
            date=date.today(),
            open=Decimal('155.00'),
            high=Decimal('160.00'),
            low=Decimal('154.00'),
            close=Decimal('159.00'),
            volume=60000000
        )
        
        result = stock.get_latest_price()
        self.assertEqual(result, latest_price)
        self.assertEqual(result.close, Decimal('159.00'))
    
    def test_get_price_history(self):
        """Test getting price history for a stock."""
        stock, _ = Stock.objects.get_or_create(symbol='AAPL_HIST', defaults={'short_name': 'Apple History Test'})
        
        # Create prices for different dates (start from 1 day ago to avoid today's date issues)
        for i in range(1, 11):  # Creates dates from 1-10 days ago
            StockPrice.objects.create(
                stock=stock,
                date=date.today() - timedelta(days=i),
                open=Decimal('150.00'),
                high=Decimal('155.00'),
                low=Decimal('149.00'),
                close=Decimal('154.00') + i,
                volume=50000000
            )
        
        # Get last 7 days (test data creates 1-10 days ago, so expecting 7 in range: days 1-7)
        history = stock.get_price_history(days=7)
        self.assertEqual(history.count(), 7)
        
        # Get last 30 days (should return all 10)
        history = stock.get_price_history(days=30)
        self.assertEqual(history.count(), 10)
    
    def test_needs_sync_property(self):
        """Test the needs_sync property."""
        stock, _ = Stock.objects.get_or_create(symbol='AAPL_SYNC', defaults={'short_name': 'Apple Sync Test'})
        
        # No last_sync, should need sync
        self.assertTrue(stock.needs_sync)
        
        # Recent sync, should not need sync
        stock.last_sync = timezone.now()
        stock.save()
        self.assertFalse(stock.needs_sync)
        
        # Old sync (2 hours ago), should need sync
        stock.last_sync = timezone.now() - timedelta(hours=2)
        stock.save()
        self.assertTrue(stock.needs_sync)


class TestStockPriceModel(TestCase):
    """Test cases for StockPrice model."""
    
    def test_create_stock_price(self):
        """Test creating a stock price instance."""
        stock, _ = Stock.objects.get_or_create(symbol='AAPL_PRICE_CREATE', defaults={'short_name': 'Apple Price Create Test'})
        price = StockPrice.objects.create(
            stock=stock,
            date=date.today(),
            open=Decimal('150.00'),
            high=Decimal('155.00'),
            low=Decimal('149.00'),
            close=Decimal('154.00'),
            adjusted_close=Decimal('153.50'),
            volume=50000000
        )
        
        self.assertEqual(price.stock, stock)
        self.assertEqual(price.open, Decimal('150.00'))
        self.assertEqual(price.close, Decimal('154.00'))
        self.assertEqual(str(price), f'AAPL_PRICE_CREATE - {date.today()}: $154.00')
    
    def test_unique_stock_date_constraint(self):
        """Test that stock-date combination must be unique."""
        stock, _ = Stock.objects.get_or_create(symbol='AAPL_DATE_UNIQUE', defaults={'short_name': 'Apple Date Unique Test'})
        test_date = date.today() - timedelta(days=100)  # Use a unique date to avoid conflicts
        StockPrice.objects.create(
            stock=stock,
            date=test_date,
            open=Decimal('150.00'),
            high=Decimal('155.00'),
            low=Decimal('149.00'),
            close=Decimal('154.00'),
            volume=50000000
        )
        
        # Try to create another price for the same stock and date
        with self.assertRaises(IntegrityError):
            StockPrice.objects.create(
                stock=stock,
                date=test_date,
                open=Decimal('151.00'),
                high=Decimal('156.00'),
                low=Decimal('150.00'),
                close=Decimal('155.00'),
                volume=60000000
            )
    
    def test_daily_change_property(self):
        """Test the daily_change property."""
        stock, _ = Stock.objects.get_or_create(symbol='AAPL_CHANGE', defaults={'short_name': 'Apple Change Test'})
        price = StockPrice.objects.create(
            stock=stock,
            date=date.today(),
            open=Decimal('150.00'),
            high=Decimal('155.00'),
            low=Decimal('149.00'),
            close=Decimal('154.00'),
            volume=50000000
        )
        
        self.assertEqual(price.daily_change, Decimal('4.00'))
    
    def test_daily_change_percent_property(self):
        """Test the daily_change_percent property."""
        stock, _ = Stock.objects.get_or_create(symbol='AAPL_CHANGE_PCT', defaults={'short_name': 'Apple Change Percent Test'})
        price = StockPrice.objects.create(
            stock=stock,
            date=date.today(),
            open=Decimal('150.00'),
            high=Decimal('155.00'),
            low=Decimal('149.00'),
            close=Decimal('154.00'),
            volume=50000000
        )
        
        expected_percent = (Decimal('4.00') / Decimal('150.00')) * Decimal('100')
        self.assertLess(abs(price.daily_change_percent - expected_percent), Decimal('0.01'))
    
    def test_daily_range_property(self):
        """Test the daily_range property."""
        stock, _ = Stock.objects.get_or_create(symbol='AAPL_RANGE', defaults={'short_name': 'Apple Range Test'})
        price = StockPrice.objects.create(
            stock=stock,
            date=date.today(),
            open=Decimal('150.00'),
            high=Decimal('155.00'),
            low=Decimal('149.00'),
            close=Decimal('154.00'),
            volume=50000000
        )
        
        self.assertEqual(price.daily_range, '149.00 - 155.00')
    
    def test_is_gain_property(self):
        """Test the is_gain property."""
        stock, _ = Stock.objects.get_or_create(symbol='AAPL_GAIN', defaults={'short_name': 'Apple Gain Test'})
        
        # Gain day
        gain_price = StockPrice.objects.create(
            stock=stock,
            date=date.today(),
            open=Decimal('150.00'),
            high=Decimal('155.00'),
            low=Decimal('149.00'),
            close=Decimal('154.00'),
            volume=50000000
        )
        self.assertTrue(gain_price.is_gain)
        
        # Loss day
        loss_price = StockPrice.objects.create(
            stock=stock,
            date=date.today() - timedelta(days=1),
            open=Decimal('154.00'),
            high=Decimal('155.00'),
            low=Decimal('149.00'),
            close=Decimal('150.00'),
            volume=50000000
        )
        self.assertFalse(loss_price.is_gain)


class TestPortfolioModel(TestCase):
    """Test cases for Portfolio model."""
    
    def test_create_portfolio(self):
        """Test creating a portfolio instance."""
        user = User.objects.create_user(username='testuser', password='testpass')
        portfolio = Portfolio.objects.create(
            user=user,
            name='My Portfolio',
            description='Test portfolio',
            initial_value=Decimal('10000.00'),
            current_value=Decimal('11000.00'),
            risk_tolerance='moderate'
        )
        
        self.assertEqual(portfolio.user, user)
        self.assertEqual(portfolio.name, 'My Portfolio')
        self.assertEqual(portfolio.initial_value, Decimal('10000.00'))
        self.assertEqual(portfolio.risk_tolerance, 'moderate')
        self.assertEqual(str(portfolio), 'My Portfolio')
    
    def test_calculate_returns(self):
        """Test calculating portfolio returns."""
        user = User.objects.create_user(username='testuser', password='testpass')
        portfolio = Portfolio.objects.create(
            user=user,
            name='My Portfolio',
            initial_value=Decimal('10000.00'),
            current_value=Decimal('12000.00')
        )
        
        returns = portfolio.calculate_returns()
        self.assertEqual(returns, Decimal('20.00'))  # 20% return
    
    def test_update_value(self):
        """Test updating portfolio value based on holdings."""
        user = User.objects.create_user(username='testuser', password='testpass')
        portfolio = Portfolio.objects.create(
            user=user,
            name='My Portfolio',
            initial_value=Decimal('10000.00')
        )
        
        # Create holdings
        stock1, _ = Stock.objects.get_or_create(symbol='AAPL_CALC', defaults={'short_name': 'Apple Calc Test'})
        stock2, _ = Stock.objects.get_or_create(symbol='MSFT_CALC', defaults={'short_name': 'Microsoft Calc Test'})
        
        PortfolioHolding.objects.create(
            portfolio=portfolio,
            stock=stock1,
            quantity=Decimal('10'),
            average_price=Decimal('150.00'),
            current_price=Decimal('160.00'),
            purchase_date=date.today()
        )
        
        PortfolioHolding.objects.create(
            portfolio=portfolio,
            stock=stock2,
            quantity=Decimal('5'),
            average_price=Decimal('300.00'),
            current_price=Decimal('320.00'),
            purchase_date=date.today()
        )
        
        portfolio.update_value()
        portfolio.refresh_from_db()
        
        # 10 * 160 + 5 * 320 = 1600 + 1600 = 3200
        self.assertEqual(portfolio.current_value, Decimal('3200.00'))


class TestPortfolioHoldingModel(TestCase):
    """Test cases for PortfolioHolding model."""
    
    def test_create_holding(self):
        """Test creating a portfolio holding."""
        user = User.objects.create_user(username='testuser', password='testpass')
        portfolio = Portfolio.objects.create(
            user=user,
            name='My Portfolio'
        )
        stock, _ = Stock.objects.get_or_create(symbol='AAPL_HOLDING', defaults={'short_name': 'Apple Holding Test'})
        
        holding = PortfolioHolding.objects.create(
            portfolio=portfolio,
            stock=stock,
            quantity=Decimal('10'),
            average_price=Decimal('150.00'),
            current_price=Decimal('160.00'),
            purchase_date=date.today()
        )
        
        self.assertEqual(holding.portfolio, portfolio)
        self.assertEqual(holding.stock, stock)
        self.assertEqual(holding.quantity, Decimal('10'))
        self.assertEqual(str(holding), 'My Portfolio - AAPL_HOLDING: 10 shares')
    
    def test_automatic_calculations_on_save(self):
        """Test that derived fields are calculated automatically."""
        user = User.objects.create_user(username='testuser', password='testpass')
        portfolio = Portfolio.objects.create(
            user=user,
            name='My Portfolio'
        )
        stock, _ = Stock.objects.get_or_create(symbol='AAPL_HOLDING', defaults={'short_name': 'Apple Holding Test'})
        
        holding = PortfolioHolding.objects.create(
            portfolio=portfolio,
            stock=stock,
            quantity=Decimal('10'),
            average_price=Decimal('150.00'),
            current_price=Decimal('160.00'),
            purchase_date=date.today()
        )
        
        # Check calculated fields
        self.assertEqual(holding.cost_basis, Decimal('1500.00'))  # 10 * 150
        self.assertEqual(holding.current_value, Decimal('1600.00'))  # 10 * 160
        self.assertEqual(holding.unrealized_gain_loss, Decimal('100.00'))  # 1600 - 1500
        self.assertLess(abs(holding.unrealized_gain_loss_percent - Decimal('6.67')), Decimal('0.01'))
    
    def test_unique_portfolio_stock_constraint(self):
        """Test that portfolio-stock combination must be unique."""
        user = User.objects.create_user(username='testuser_portfolio_unique', password='testpass')
        portfolio = Portfolio.objects.create(
            user=user,
            name='My Portfolio Unique Test'
        )
        stock, _ = Stock.objects.get_or_create(symbol='AAPL_HOLDING_UNIQUE', defaults={'short_name': 'Apple Holding Unique Test'})
        
        PortfolioHolding.objects.create(
            portfolio=portfolio,
            stock=stock,
            quantity=Decimal('10'),
            average_price=Decimal('150.00'),
            purchase_date=date.today()
        )
        
        # Try to create another holding for the same stock in the same portfolio
        with self.assertRaises(IntegrityError):
            PortfolioHolding.objects.create(
                portfolio=portfolio,
                stock=stock,
                quantity=Decimal('5'),
                average_price=Decimal('155.00'),
                purchase_date=date.today()
            )

"""
Unit tests for Data models.
Tests the Stock and StockPrice models to ensure they work correctly.
"""

import pytest
from datetime import date
from decimal import Decimal
from django.contrib.auth import get_user_model
from Data.models import Stock, StockPrice

User = get_user_model()


@pytest.mark.django_db
class TestStockModel:
    """Test suite for the Stock model."""
    
    def test_create_stock(self):
        """Test creating a stock instance."""
        stock = Stock.objects.create(
            symbol='AAPL',
            short_name='Apple',
            long_name='Apple Inc.',
            currency='USD',
            exchange='NASDAQ',
            sector='Technology',
            industry='Consumer Electronics',
            country='USA',
            market_cap=3000000000000
        )
        
        assert stock.symbol == 'AAPL'
        assert stock.short_name == 'Apple'
        assert stock.long_name == 'Apple Inc.'
        assert stock.sector == 'Technology'
        assert stock.industry == 'Consumer Electronics'
        assert stock.market_cap == 3000000000000
        assert stock.is_active == True  # Default value
    
    def test_stock_unique_symbol(self):
        """Test that stock symbols must be unique."""
        Stock.objects.create(
            symbol='MSFT',
            short_name='Microsoft'
        )
        
        with pytest.raises(Exception):  # Should raise IntegrityError
            Stock.objects.create(
                symbol='MSFT',
                short_name='Microsoft Duplicate'
            )
    
    def test_stock_optional_fields(self):
        """Test that optional fields default to empty strings."""
        stock = Stock.objects.create(
            symbol='GOOGL'
        )
        
        # CharField fields default to '' not None in Django
        assert stock.short_name == ''
        assert stock.long_name == ''
        assert stock.sector == ''
        assert stock.industry == ''


@pytest.mark.django_db
class TestStockPriceModel:
    """Test suite for the StockPrice model."""
    
    def test_create_stock_price(self):
        """Test creating a stock price instance."""
        stock = Stock.objects.create(
            symbol='TSLA',
            short_name='Tesla'
        )
        
        stock_price = StockPrice.objects.create(
            stock=stock,
            date=date(2025, 8, 8),
            open=Decimal('250.50'),
            high=Decimal('255.75'),
            low=Decimal('248.30'),
            close=Decimal('253.20'),
            volume=15000000
        )
        
        assert stock_price.stock == stock
        assert stock_price.date == date(2025, 8, 8)
        assert stock_price.open == Decimal('250.50')
        assert stock_price.high == Decimal('255.75')
        assert stock_price.low == Decimal('248.30')
        assert stock_price.close == Decimal('253.20')
        assert stock_price.volume == 15000000
    
    def test_stock_price_unique_constraint(self):
        """Test that stock-date combination must be unique."""
        stock = Stock.objects.create(
            symbol='AMZN',
            short_name='Amazon'
        )
        
        test_date = date(2025, 8, 7)
        
        StockPrice.objects.create(
            stock=stock,
            date=test_date,
            open=Decimal('180.00'),
            high=Decimal('182.00'),
            low=Decimal('179.00'),
            close=Decimal('182.00')
        )
        
        # Attempt to create duplicate should fail
        with pytest.raises(Exception):  # Should raise IntegrityError
            StockPrice.objects.create(
                stock=stock,
                date=test_date,
                open=Decimal('181.00'),
                high=Decimal('183.00'),
                low=Decimal('180.00'),
                close=Decimal('183.00')
            )
    
    def test_stock_price_ordering(self):
        """Test that stock prices are ordered by date descending."""
        stock = Stock.objects.create(
            symbol='NFLX',
            short_name='Netflix'
        )
        
        # Create prices in non-chronological order
        price1 = StockPrice.objects.create(
            stock=stock,
            date=date(2025, 8, 6),
            open=Decimal('450.00'),
            high=Decimal('455.00'),
            low=Decimal('448.00'),
            close=Decimal('450.00')
        )
        price2 = StockPrice.objects.create(
            stock=stock,
            date=date(2025, 8, 8),
            open=Decimal('452.00'),
            high=Decimal('457.00'),
            low=Decimal('451.00'),
            close=Decimal('455.00')
        )
        price3 = StockPrice.objects.create(
            stock=stock,
            date=date(2025, 8, 7),
            open=Decimal('451.00'),
            high=Decimal('454.00'),
            low=Decimal('449.00'),
            close=Decimal('452.00')
        )
        
        prices = StockPrice.objects.filter(stock=stock).order_by('-date')
        
        # Should be ordered by date descending (most recent first)
        assert prices[0].date == date(2025, 8, 8)
        assert prices[1].date == date(2025, 8, 7)
        assert prices[2].date == date(2025, 8, 6)
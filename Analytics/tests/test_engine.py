"""
Unit tests for Analytics engine service.
Uses mocking to test service logic in isolation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import date, timedelta
from decimal import Decimal
from Analytics.services.engine import AnalyticsEngine


class TestAnalyticsEngine:
    """Test suite for the AnalyticsEngine service."""
    
    @pytest.fixture
    def engine(self):
        """Create an AnalyticsEngine instance for testing."""
        return AnalyticsEngine()
    
    @patch('Analytics.services.engine.StockPrice')
    def test_calculate_moving_average(self, mock_stock_price, engine):
        """Test moving average calculation."""
        # Mock the database query
        mock_prices = [
            Mock(close_price=Decimal('100.00'), date=date(2025, 8, 8)),
            Mock(close_price=Decimal('102.00'), date=date(2025, 8, 7)),
            Mock(close_price=Decimal('98.00'), date=date(2025, 8, 6)),
            Mock(close_price=Decimal('101.00'), date=date(2025, 8, 5)),
            Mock(close_price=Decimal('99.00'), date=date(2025, 8, 4)),
        ]
        
        mock_stock_price.objects.filter.return_value.order_by.return_value[:5] = mock_prices
        
        # Calculate 5-day moving average
        result = engine.calculate_moving_average('AAPL', days=5)
        
        # Expected average: (100 + 102 + 98 + 101 + 99) / 5 = 100
        assert result == Decimal('100.00')
        
        # Verify the query was made correctly
        mock_stock_price.objects.filter.assert_called_once()
    
    @patch('Analytics.services.engine.StockPrice')
    def test_calculate_rsi(self, mock_stock_price, engine):
        """Test RSI (Relative Strength Index) calculation."""
        # Create mock price data with gains and losses
        mock_prices = []
        base_price = 100
        
        for i in range(15):
            if i % 2 == 0:
                price = Decimal(str(base_price + i))
            else:
                price = Decimal(str(base_price - 1))
            
            mock_prices.append(
                Mock(close_price=price, date=date(2025, 8, 8) - timedelta(days=i))
            )
        
        mock_stock_price.objects.filter.return_value.order_by.return_value[:15] = mock_prices
        
        # Calculate RSI
        result = engine.calculate_rsi('AAPL', period=14)
        
        # RSI should be between 0 and 100
        assert 0 <= result <= 100
        assert isinstance(result, (int, float, Decimal))
    
    @patch('Analytics.services.engine.requests')
    @patch('Analytics.services.engine.Stock')
    def test_fetch_external_data(self, mock_stock, mock_requests, engine):
        """Test fetching data from external API."""
        # Mock the stock lookup
        mock_stock_obj = Mock(symbol='AAPL', name='Apple Inc.')
        mock_stock.objects.get.return_value = mock_stock_obj
        
        # Mock the API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'symbol': 'AAPL',
            'price': 175.50,
            'change': 2.35,
            'changePercent': 1.36,
            'volume': 52000000,
            'marketCap': 2750000000000
        }
        mock_requests.get.return_value = mock_response
        
        # Fetch external data
        result = engine.fetch_external_data('AAPL')
        
        # Verify the result
        assert result['symbol'] == 'AAPL'
        assert result['price'] == 175.50
        assert result['change'] == 2.35
        assert result['changePercent'] == 1.36
        
        # Verify API was called
        mock_requests.get.assert_called_once()
    
    @patch('Analytics.services.engine.StockPrice')
    def test_calculate_volatility(self, mock_stock_price, engine):
        """Test volatility calculation."""
        # Create mock price data
        mock_prices = [
            Mock(close_price=Decimal('100.00')),
            Mock(close_price=Decimal('102.00')),
            Mock(close_price=Decimal('98.00')),
            Mock(close_price=Decimal('103.00')),
            Mock(close_price=Decimal('97.00')),
            Mock(close_price=Decimal('101.00')),
            Mock(close_price=Decimal('99.00')),
            Mock(close_price=Decimal('104.00')),
            Mock(close_price=Decimal('96.00')),
            Mock(close_price=Decimal('102.00')),
        ]
        
        mock_stock_price.objects.filter.return_value.order_by.return_value[:30] = mock_prices
        
        # Calculate volatility
        result = engine.calculate_volatility('AAPL', days=30)
        
        # Volatility should be positive
        assert result > 0
        assert isinstance(result, (int, float, Decimal))
    
    def test_analyze_stock_comprehensive(self, engine):
        """Test comprehensive stock analysis with all mocked components."""
        with patch.object(engine, 'calculate_moving_average') as mock_ma, \
             patch.object(engine, 'calculate_rsi') as mock_rsi, \
             patch.object(engine, 'calculate_volatility') as mock_vol, \
             patch.object(engine, 'fetch_external_data') as mock_external:
            
            # Set up mock return values
            mock_ma.return_value = Decimal('150.00')
            mock_rsi.return_value = 55.5
            mock_vol.return_value = 0.25
            mock_external.return_value = {
                'price': 152.00,
                'change': 2.00,
                'volume': 50000000
            }
            
            # Perform comprehensive analysis
            result = engine.analyze_stock('AAPL')
            
            # Verify all components were called
            mock_ma.assert_called_once_with('AAPL', days=20)
            mock_rsi.assert_called_once_with('AAPL', period=14)
            mock_vol.assert_called_once_with('AAPL', days=30)
            mock_external.assert_called_once_with('AAPL')
            
            # Verify result structure
            assert 'moving_average' in result
            assert 'rsi' in result
            assert 'volatility' in result
            assert 'current_data' in result
            assert result['moving_average'] == Decimal('150.00')
            assert result['rsi'] == 55.5
            assert result['volatility'] == 0.25
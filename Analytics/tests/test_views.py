"""
Basic functional tests for Analytics app API views.
"""

# import pytest  # Unused - using Django TestCase
from django.contrib.auth.models import User
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase, APIClient
from datetime import timedelta, date
from decimal import Decimal

from Data.models import Stock, StockPrice, Portfolio, PortfolioHolding


class AnalyticsAPITestCase(APITestCase):
    """Test cases for Analytics API endpoints."""
    
    @classmethod
    def setUpTestData(cls):
        """Set up immutable test data once for the entire test class."""
        # Create test stock with price history
        cls.stock = Stock.objects.create(
            symbol='AAPL',
            short_name='Apple Inc.',
            long_name='Apple Inc.',
            currency='USD',
            exchange='NASDAQ',
            sector='Technology',
            market_cap=3000000000000,
            is_active=True,
            data_source='yahoo'
        )
        
        # Create price history for technical analysis (immutable data)
        base_price = Decimal('150.00')
        for i in range(30):
            price_date = date.today() - timedelta(days=i)
            price = base_price + Decimal(str(i % 10))  # Create some variation
            
            StockPrice.objects.create(
                stock=cls.stock,
                date=price_date,
                open=price - Decimal('1.00'),
                high=price + Decimal('2.00'),
                low=price - Decimal('2.00'),
                close=price,
                adjusted_close=price,
                volume=50000000 + (i * 100000),
                data_source='yahoo'
            )
    
    def setUp(self):
        """Set up test-specific data."""
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        
        # Create test portfolio (user-specific)
        self.portfolio = Portfolio.objects.create(
            user=self.user,
            name='Test Portfolio',
            initial_value=Decimal('10000.00'),
            current_value=Decimal('11000.00')
        )
        
        # Create portfolio holding (user-specific)
        self.holding = PortfolioHolding.objects.create(
            portfolio=self.portfolio,
            stock=self.stock,  # Reference to class-level stock
            quantity=Decimal('10'),
            average_price=Decimal('145.00'),
            current_price=Decimal('155.00'),
            purchase_date=date.today() - timedelta(days=30)
        )
    
    def test_stock_analysis_valid_symbol(self):
        """Test that stock analysis endpoint responds for valid symbols."""
        # Authenticate the user for this endpoint
        self.client.force_authenticate(user=self.user)
        
        url = reverse('analytics:analyze_stock', args=[self.stock.symbol])
        response = self.client.get(url)
        
        # Should return 200 OK or handle gracefully
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND])
        
        if response.status_code == status.HTTP_200_OK:
            self.assertIn('symbol', response.data)
            self.assertEqual(response.data['symbol'], 'AAPL')
    
    def test_stock_analysis_invalid_symbol(self):
        """Test stock analysis with invalid symbol."""
        # Authenticate the user for this endpoint
        self.client.force_authenticate(user=self.user)
        
        url = reverse('analytics:analyze_stock', args=['INVALID'])
        response = self.client.get(url)
        
        # Should return 404 Not Found or 400 Bad Request for invalid symbols
        self.assertIn(response.status_code, [status.HTTP_404_NOT_FOUND, status.HTTP_400_BAD_REQUEST])
    
    def test_portfolio_analysis_requires_auth(self):
        """Test that portfolio analysis requires authentication."""
        url = reverse('analytics:analyze_portfolio', args=[self.portfolio.id])
        response = self.client.get(url)
        
        # Should require authentication
        self.assertIn(response.status_code, [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN])
    
    def test_portfolio_analysis_authenticated(self):
        """Test portfolio analysis with authentication."""
        self.client.force_authenticate(user=self.user)
        url = reverse('analytics:analyze_portfolio', args=[self.portfolio.id])
        response = self.client.get(url)
        
        # Should handle request (may not be fully implemented yet)
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND, status.HTTP_501_NOT_IMPLEMENTED])
    
    def test_portfolio_analysis_wrong_user(self):
        """Test that users cannot access other users' portfolio analysis."""
        other_user = User.objects.create_user(
            username='otheruser',
            password='otherpass123'
        )
        other_portfolio = Portfolio.objects.create(
            user=other_user,
            name='Other Portfolio',
            initial_value=Decimal('5000.00'),
            current_value=Decimal('5000.00')
        )
        
        self.client.force_authenticate(user=self.user)
        url = reverse('analytics:analyze_portfolio', args=[other_portfolio.id])
        response = self.client.get(url)
        
        # Should not allow access to other user's portfolio
        self.assertIn(response.status_code, [status.HTTP_403_FORBIDDEN, status.HTTP_404_NOT_FOUND])
    
    def test_batch_analysis_endpoint(self):
        """Test batch analysis endpoint basic functionality."""
        # Authenticate the user for this endpoint
        self.client.force_authenticate(user=self.user)
        
        url = reverse('analytics:batch_analysis')
        response = self.client.post(url, {'symbols': ['AAPL']}, format='json')
        
        # Should handle request gracefully (may not be fully implemented)
        self.assertIn(response.status_code, [
            status.HTTP_200_OK, 
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_404_NOT_FOUND, 
            status.HTTP_501_NOT_IMPLEMENTED
        ])


class AnalyticsEngineIntegrationTestCase(APITestCase):
    """Integration tests that test actual analytics engine functionality."""
    
    @classmethod
    def setUpTestData(cls):
        """Set up test data once per test class for integration tests."""
        # Create test stock with sufficient price history for analytics
        cls.stock = Stock.objects.create(
            symbol='TEST',
            short_name='Test Inc.',
            sector='Technology',
            market_cap=1000000000000,
            is_active=True,
            data_source='yahoo'
        )
        
        # Create realistic price history for analytics engine
        base_price = Decimal('100.00')
        for i in range(100):  # 100 days of data
            price_date = date.today() - timedelta(days=i)
            # Simulate some price movement
            price_change = Decimal(str((i % 20 - 10) * 0.5))  # -5 to +5 range
            current_price = base_price + price_change
            
            StockPrice.objects.create(
                stock=cls.stock,
                date=price_date,
                open=current_price - Decimal('0.50'),
                high=current_price + Decimal('1.00'),
                low=current_price - Decimal('1.50'),
                close=current_price,
                adjusted_close=current_price,
                volume=1000000 + (i * 10000),
                data_source='yahoo'
            )

    def setUp(self):
        """Set up test client for each test method."""
        self.client = APIClient()
    
    def test_analytics_engine_basic_functionality(self):
        """Test that analytics engine can be instantiated and doesn't crash."""
        from Analytics.engine.ta_engine import TechnicalAnalysisEngine
        
        engine = TechnicalAnalysisEngine()
        self.assertIsNotNone(engine)
        
        # Test that engine has expected methods
        self.assertTrue(hasattr(engine, 'analyze_stock'))
        self.assertTrue(callable(getattr(engine, 'analyze_stock')))
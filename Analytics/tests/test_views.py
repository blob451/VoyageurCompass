"""
Comprehensive tests for Analytics app API views.
"""

import pytest
from django.contrib.auth.models import User
from django.urls import reverse
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase, APIClient
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta, date
from decimal import Decimal

from Data.models import Stock, StockPrice, Portfolio, PortfolioHolding
from Analytics.services.engine import analytics_engine


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
            is_active=True
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
                volume=50000000 + (i * 100000)
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
    
    def test_stock_analysis_unauthenticated(self):
        """Test that stock analysis is accessible without authentication."""
        url = reverse('analytics:stock-analysis', args=[self.stock.symbol])
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('symbol', response.data)
        self.assertIn('technical_indicators', response.data)
        self.assertEqual(response.data['symbol'], 'AAPL')
    
    @patch('Analytics.services.engine.analytics_engine.analyze_stock')
    def test_stock_analysis_with_mocked_engine(self, mock_analyze):
        """Test stock analysis with mocked analytics engine."""
        mock_analyze.return_value = {
            'symbol': 'AAPL',
            'current_price': 155.00,
            'technical_indicators': {
                'sma_20': 152.00,
                'sma_50': 150.00,
                'rsi': 65.5,
                'macd': 2.5
            },
            'signals': ['BUY'],
            'analysis_date': timezone.now().isoformat()
        }
        
        url = reverse('analytics:stock-analysis', args=[self.stock.symbol])
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['symbol'], 'AAPL')
        self.assertIn('technical_indicators', response.data)
        mock_analyze.assert_called_once_with('AAPL')
    
    def test_stock_analysis_invalid_symbol(self):
        """Test stock analysis with invalid symbol."""
        url = reverse('analytics:stock-analysis', args=['INVALID'])
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn('error', response.data)
    
    def test_portfolio_analysis_requires_auth(self):
        """Test that portfolio analysis requires authentication."""
        url = reverse('analytics:portfolio-analysis', args=[self.portfolio.id])
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
    
    def test_portfolio_analysis_authenticated(self):
        """Test portfolio analysis with authentication."""
        self.client.force_authenticate(user=self.user)
        url = reverse('analytics:portfolio-analysis', args=[self.portfolio.id])
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('portfolio_id', response.data)
        self.assertIn('diversification', response.data)
        self.assertIn('risk_metrics', response.data)
    
    def test_portfolio_analysis_wrong_user(self):
        """Test that users cannot access other users' portfolio analysis."""
        other_user = User.objects.create_user(
            username='otheruser',
            password='otherpass123'
        )
        other_portfolio = Portfolio.objects.create(
            user=other_user,
            name='Other Portfolio',
            initial_value=Decimal('5000.00')
        )
        
        self.client.force_authenticate(user=self.user)
        url = reverse('analytics:portfolio-analysis', args=[other_portfolio.id])
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
    
    @patch('Analytics.services.engine.analytics_engine.analyze_portfolio')
    def test_portfolio_analysis_with_mocked_engine(self, mock_analyze):
        """Test portfolio analysis with mocked analytics engine."""
        mock_analyze.return_value = {
            'portfolio_id': self.portfolio.id,
            'total_value': 11000.00,
            'diversification': {
                'by_sector': {'Technology': 100.0},
                'by_asset_class': {'Stocks': 100.0}
            },
            'risk_metrics': {
                'volatility': 0.25,
                'beta': 1.2,
                'sharpe_ratio': 1.5
            },
            'performance': {
                'total_return': 10.0,
                'annualized_return': 12.0
            },
            'recommendations': ['Consider diversifying into other sectors']
        }
        
        self.client.force_authenticate(user=self.user)
        url = reverse('analytics:portfolio-analysis', args=[self.portfolio.id])
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['portfolio_id'], self.portfolio.id)
        self.assertIn('diversification', response.data)
        self.assertIn('risk_metrics', response.data)
        mock_analyze.assert_called_once()
    
    def test_market_sentiment(self):
        """Test market sentiment endpoint."""
        url = reverse('analytics:market-sentiment')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('sentiment_score', response.data)
        self.assertIn('market_indicators', response.data)
    
    @patch('Analytics.services.engine.analytics_engine.get_market_sentiment')
    def test_market_sentiment_with_mocked_engine(self, mock_sentiment):
        """Test market sentiment with mocked analytics engine."""
        mock_sentiment.return_value = {
            'sentiment_score': 0.65,  # Positive sentiment
            'sentiment_label': 'BULLISH',
            'confidence': 0.8,
            'market_indicators': {
                'vix': 18.5,
                'put_call_ratio': 0.85,
                'advance_decline': 1.2
            },
            'sector_sentiment': {
                'Technology': 0.7,
                'Healthcare': 0.6,
                'Finance': 0.4
            },
            'analysis_timestamp': timezone.now().isoformat()
        }
        
        url = reverse('analytics:market-sentiment')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['sentiment_score'], 0.65)
        self.assertEqual(response.data['sentiment_label'], 'BULLISH')
        mock_sentiment.assert_called_once()
    
    def test_technical_indicators(self):
        """Test technical indicators endpoint."""
        url = reverse('analytics:technical-indicators', args=[self.stock.symbol])
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('symbol', response.data)
        self.assertIn('indicators', response.data)
        self.assertEqual(response.data['symbol'], 'AAPL')
    
    @patch('Analytics.services.engine.analytics_engine.calculate_technical_indicators')
    def test_technical_indicators_with_mocked_engine(self, mock_indicators):
        """Test technical indicators with mocked analytics engine."""
        mock_indicators.return_value = {
            'symbol': 'AAPL',
            'indicators': {
                'sma_20': 152.50,
                'sma_50': 148.75,
                'ema_12': 154.20,
                'ema_26': 151.30,
                'rsi': 68.5,
                'macd': {
                    'macd_line': 2.90,
                    'signal_line': 1.85,
                    'histogram': 1.05
                },
                'bollinger_bands': {
                    'upper': 158.00,
                    'middle': 152.00,
                    'lower': 146.00
                },
                'stochastic': {
                    'k': 75.2,
                    'd': 72.8
                }
            },
            'signals': {
                'trend': 'UPTREND',
                'momentum': 'STRONG',
                'volatility': 'NORMAL'
            },
            'calculation_date': timezone.now().isoformat()
        }
        
        url = reverse('analytics:technical-indicators', args=[self.stock.symbol])
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['symbol'], 'AAPL')
        self.assertIn('indicators', response.data)
        self.assertIn('sma_20', response.data['indicators'])
        mock_indicators.assert_called_once_with('AAPL')
    
    def test_stock_recommendations(self):
        """Test stock recommendations endpoint."""
        url = reverse('analytics:stock-recommendations', args=[self.stock.symbol])
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('symbol', response.data)
        self.assertIn('recommendations', response.data)
        self.assertEqual(response.data['symbol'], 'AAPL')
    
    @patch('Analytics.services.engine.analytics_engine.generate_recommendations')
    def test_stock_recommendations_with_mocked_engine(self, mock_recommendations):
        """Test stock recommendations with mocked analytics engine."""
        mock_recommendations.return_value = {
            'symbol': 'AAPL',
            'overall_rating': 'BUY',
            'confidence': 0.85,
            'target_price': 165.00,
            'recommendations': [
                {
                    'type': 'TECHNICAL',
                    'signal': 'BUY',
                    'reason': 'RSI indicates oversold condition',
                    'confidence': 0.8
                },
                {
                    'type': 'MOMENTUM',
                    'signal': 'HOLD',
                    'reason': 'MACD showing mixed signals',
                    'confidence': 0.6
                }
            ],
            'risk_factors': [
                'High volatility expected',
                'Market sentiment uncertainty'
            ],
            'time_horizon': '3-6 months',
            'generated_at': timezone.now().isoformat()
        }
        
        url = reverse('analytics:stock-recommendations', args=[self.stock.symbol])
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['symbol'], 'AAPL')
        self.assertEqual(response.data['overall_rating'], 'BUY')
        self.assertIn('recommendations', response.data)
        mock_recommendations.assert_called_once_with('AAPL')
    
    def test_portfolio_optimization(self):
        """Test portfolio optimization endpoint."""
        self.client.force_authenticate(user=self.user)
        url = reverse('analytics:portfolio-optimization', args=[self.portfolio.id])
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('portfolio_id', response.data)
        self.assertIn('current_allocation', response.data)
        self.assertIn('suggested_allocation', response.data)
    
    @patch('Analytics.services.engine.analytics_engine.optimize_portfolio')
    def test_portfolio_optimization_with_mocked_engine(self, mock_optimize):
        """Test portfolio optimization with mocked analytics engine."""
        mock_optimize.return_value = {
            'portfolio_id': self.portfolio.id,
            'current_allocation': {
                'AAPL': {'weight': 1.0, 'value': 11000.00}
            },
            'suggested_allocation': {
                'AAPL': {'weight': 0.6, 'value': 6600.00},
                'MSFT': {'weight': 0.2, 'value': 2200.00},
                'GOOGL': {'weight': 0.2, 'value': 2200.00}
            },
            'optimization_metrics': {
                'expected_return': 0.12,
                'volatility': 0.18,
                'sharpe_ratio': 0.67
            },
            'rebalancing_suggestions': [
                {
                    'action': 'REDUCE',
                    'symbol': 'AAPL',
                    'from_weight': 1.0,
                    'to_weight': 0.6,
                    'reason': 'Over-concentration risk'
                },
                {
                    'action': 'ADD',
                    'symbol': 'MSFT',
                    'to_weight': 0.2,
                    'reason': 'Diversification benefit'
                }
            ],
            'optimization_date': timezone.now().isoformat()
        }
        
        self.client.force_authenticate(user=self.user)
        url = reverse('analytics:portfolio-optimization', args=[self.portfolio.id])
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['portfolio_id'], self.portfolio.id)
        self.assertIn('current_allocation', response.data)
        self.assertIn('suggested_allocation', response.data)
        mock_optimize.assert_called_once()
    
    def test_risk_assessment(self):
        """Test risk assessment endpoint."""
        self.client.force_authenticate(user=self.user)
        url = reverse('analytics:risk-assessment', args=[self.portfolio.id])
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('portfolio_id', response.data)
        self.assertIn('risk_score', response.data)
        self.assertIn('risk_factors', response.data)
    
    @patch('Analytics.services.engine.analytics_engine.assess_portfolio_risk')
    def test_risk_assessment_with_mocked_engine(self, mock_assess):
        """Test risk assessment with mocked analytics engine."""
        mock_assess.return_value = {
            'portfolio_id': self.portfolio.id,
            'risk_score': 7.5,  # Scale 1-10
            'risk_level': 'MODERATE_HIGH',
            'risk_factors': [
                {
                    'factor': 'Concentration Risk',
                    'score': 9.0,
                    'description': 'Portfolio heavily concentrated in single stock',
                    'impact': 'HIGH'
                },
                {
                    'factor': 'Sector Concentration',
                    'score': 8.0,
                    'description': 'All holdings in Technology sector',
                    'impact': 'HIGH'
                }
            ],
            'metrics': {
                'volatility': 0.28,
                'beta': 1.35,
                'var_95': -0.15,
                'max_drawdown': -0.22
            },
            'recommendations': [
                'Diversify across multiple sectors',
                'Consider adding defensive stocks',
                'Reduce position size in individual holdings'
            ],
            'assessment_date': timezone.now().isoformat()
        }
        
        self.client.force_authenticate(user=self.user)
        url = reverse('analytics:risk-assessment', args=[self.portfolio.id])
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['portfolio_id'], self.portfolio.id)
        self.assertEqual(response.data['risk_score'], 7.5)
        self.assertEqual(response.data['risk_level'], 'MODERATE_HIGH')
        mock_assess.assert_called_once()
    
    def test_performance_analytics(self):
        """Test performance analytics endpoint."""
        self.client.force_authenticate(user=self.user)
        url = reverse('analytics:performance-analytics', args=[self.portfolio.id])
        response = self.client.get(url, {'period': '1Y'})
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('portfolio_id', response.data)
        self.assertIn('performance_metrics', response.data)
    
    @patch('Analytics.services.engine.analytics_engine.calculate_performance_metrics')
    def test_performance_analytics_with_mocked_engine(self, mock_performance):
        """Test performance analytics with mocked analytics engine."""
        mock_performance.return_value = {
            'portfolio_id': self.portfolio.id,
            'period': '1Y',
            'performance_metrics': {
                'total_return': 0.15,
                'annualized_return': 0.15,
                'volatility': 0.22,
                'sharpe_ratio': 0.68,
                'alpha': 0.03,
                'beta': 1.25,
                'max_drawdown': -0.18,
                'calmar_ratio': 0.83
            },
            'benchmark_comparison': {
                'benchmark': 'S&P 500',
                'portfolio_return': 0.15,
                'benchmark_return': 0.12,
                'excess_return': 0.03,
                'tracking_error': 0.08
            },
            'monthly_returns': [
                {'month': '2024-01', 'return': 0.05},
                {'month': '2024-02', 'return': -0.02},
                # ... more monthly data
            ],
            'calculation_date': timezone.now().isoformat()
        }
        
        self.client.force_authenticate(user=self.user)
        url = reverse('analytics:performance-analytics', args=[self.portfolio.id])
        response = self.client.get(url, {'period': '1Y'})
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['portfolio_id'], self.portfolio.id)
        self.assertIn('performance_metrics', response.data)
        self.assertIn('benchmark_comparison', response.data)
        mock_performance.assert_called_once()


class AnalyticsEngineIntegrationTestCase(APITestCase):
    """Integration tests that test actual analytics engine functionality."""
    
    @classmethod
    def setUpTestData(cls):
        """Set up test data once per test class for integration tests."""
        # Create multiple stocks with price history for realistic testing
        cls.stocks = []
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        
        for symbol in symbols:
            stock = Stock.objects.create(
                symbol=symbol,
                short_name=f'{symbol} Inc.',
                sector='Technology',
                market_cap=1000000000000,
                is_active=True
            )
            cls.stocks.append(stock)
            
            # Create realistic price history
            base_price = Decimal('100.00')
            for i in range(100):  # 100 days of data
                price_date = date.today() - timedelta(days=i)
                # Simulate some price movement
                price_change = Decimal(str((i % 20 - 10) * 0.5))  # -5 to +5 range
                current_price = base_price + price_change
                
                StockPrice.objects.create(
                    stock=stock,
                    date=price_date,
                    open=current_price - Decimal('0.50'),
                    high=current_price + Decimal('1.00'),
                    low=current_price - Decimal('1.50'),
                    close=current_price,
                    volume=1000000 + (i * 10000)
                )

    def setUp(self):
        """Set up test client and authentication for each test method."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.client.force_authenticate(user=self.user)
    
    def test_technical_indicators_integration(self):
        """Test that technical indicators are calculated correctly."""
        url = reverse('analytics:technical-indicators', args=['AAPL'])
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('indicators', response.data)
        
        indicators = response.data['indicators']
        
        # Check that key indicators are present and are numeric
        for indicator in ['sma_20', 'sma_50', 'rsi']:
            self.assertIn(indicator, indicators)
            self.assertIsInstance(indicators[indicator], (int, float))
            
        # RSI should be between 0 and 100
        if 'rsi' in indicators:
            self.assertGreaterEqual(indicators['rsi'], 0)
            self.assertLessEqual(indicators['rsi'], 100)
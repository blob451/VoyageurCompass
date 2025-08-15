"""
API Tests for sentiment analysis endpoints.
Tests the sentiment analysis REST API functionality.
"""

import json
from unittest.mock import patch, Mock
from django.test import TestCase
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.core.cache import cache
from rest_framework.test import APIClient
from rest_framework import status

from Data.models import Stock, AnalyticsResults
from Analytics.services.sentiment_analyzer import get_sentiment_analyzer

User = get_user_model()


class SentimentAPITestCase(TestCase):
    """Test cases for sentiment analysis API endpoints."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = APIClient()
        
        # Create test user
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        
        # Create test stock
        self.test_stock = Stock.objects.create(
            symbol='AAPL',
            company_name='Apple Inc.',
            sector='Technology',
            industry='Consumer Electronics'
        )
        
        # Clear cache
        cache.clear()
    
    def tearDown(self):
        """Clean up after tests."""
        cache.clear()
    
    def test_sentiment_endpoint_authentication_required(self):
        """Test that sentiment endpoint requires authentication."""
        url = reverse('analytics:stock_sentiment', kwargs={'symbol': 'AAPL'})
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
    
    @patch('Analytics.sentiment_views.yahoo_finance_service')
    @patch('Analytics.sentiment_views.get_sentiment_analyzer')
    def test_sentiment_endpoint_success(self, mock_get_analyzer, mock_yahoo):
        """Test successful sentiment analysis API call."""
        # Authenticate user
        self.client.force_authenticate(user=self.user)
        
        # Mock news data
        mock_yahoo.fetchNewsForStock.return_value = [
            {
                'title': 'Apple reports record quarterly revenue',
                'summary': 'Strong iPhone sales drive exceptional growth',
                'publishedDate': '2024-01-15T10:30:00Z',
                'source': 'Reuters'
            },
            {
                'title': 'Apple stock rises on positive outlook',
                'summary': 'Analysts upgrade price targets',
                'publishedDate': '2024-01-15T09:15:00Z', 
                'source': 'Bloomberg'
            }
        ]
        
        mock_yahoo.preprocessNewsText.side_effect = [
            "Apple reports record quarterly revenue. Strong iPhone sales drive exceptional growth",
            "Apple stock rises on positive outlook. Analysts upgrade price targets"
        ]
        
        # Mock sentiment analyzer
        mock_analyzer = Mock()
        mock_analyzer.generateCacheKey.return_value = "sentiment:stock:AAPL:90d"
        mock_analyzer.getCachedSentiment.return_value = None  # Cache miss
        mock_analyzer.setCachedSentiment.return_value = True
        
        mock_analyzer.analyzeSentimentBatch.return_value = [
            {
                'sentimentScore': 0.75,
                'sentimentLabel': 'positive',
                'sentimentConfidence': 0.89
            },
            {
                'sentimentScore': 0.68,
                'sentimentLabel': 'positive', 
                'sentimentConfidence': 0.82
            }
        ]
        
        mock_analyzer.aggregateSentiment.return_value = {
            'sentimentScore': 0.715,
            'sentimentLabel': 'positive',
            'sentimentConfidence': 0.855,
            'distribution': {'positive': 2, 'negative': 0, 'neutral': 0},
            'sourceBreakdown': {
                'Reuters': {'count': 1, 'avg_score': 0.75},
                'Bloomberg': {'count': 1, 'avg_score': 0.68}
            }
        }
        
        mock_get_analyzer.return_value = mock_analyzer
        
        # Make API call
        url = reverse('analytics:stock_sentiment', kwargs={'symbol': 'AAPL'})
        response = self.client.get(url)
        
        # Verify response
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        data = response.json()
        self.assertEqual(data['symbol'], 'AAPL')
        self.assertEqual(data['sentimentLabel'], 'positive')
        self.assertAlmostEqual(data['sentimentScore'], 0.715, places=3)
        self.assertEqual(data['newsCount'], 2)
        self.assertIn('distribution', data)
        self.assertIn('sources', data)
        self.assertEqual(len(data['articles']), 2)
    
    @patch('Analytics.sentiment_views.yahoo_finance_service')
    @patch('Analytics.sentiment_views.get_sentiment_analyzer')
    def test_sentiment_endpoint_no_news(self, mock_get_analyzer, mock_yahoo):
        """Test sentiment endpoint when no news is available."""
        # Authenticate user
        self.client.force_authenticate(user=self.user)
        
        # Mock no news found
        mock_yahoo.fetchNewsForStock.return_value = []
        
        mock_analyzer = Mock()
        mock_analyzer.generateCacheKey.return_value = "sentiment:stock:AAPL:90d"
        mock_analyzer.getCachedSentiment.return_value = None
        mock_get_analyzer.return_value = mock_analyzer
        
        # Make API call
        url = reverse('analytics:stock_sentiment', kwargs={'symbol': 'AAPL'})
        response = self.client.get(url)
        
        # Verify response
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        data = response.json()
        self.assertEqual(data['symbol'], 'AAPL')
        self.assertEqual(data['sentimentLabel'], 'neutral')
        self.assertEqual(data['sentimentScore'], 0.0)
        self.assertEqual(data['newsCount'], 0)
        self.assertIn('message', data)
    
    @patch('Analytics.sentiment_views.yahoo_finance_service')
    @patch('Analytics.sentiment_views.get_sentiment_analyzer')
    def test_sentiment_endpoint_cached_result(self, mock_get_analyzer, mock_yahoo):
        """Test sentiment endpoint returning cached result."""
        # Authenticate user
        self.client.force_authenticate(user=self.user)
        
        # Mock cached result
        cached_result = {
            'symbol': 'AAPL',
            'sentimentScore': 0.42,
            'sentimentLabel': 'positive',
            'confidence': 0.78,
            'newsCount': 15,
            'cached': True
        }
        
        mock_analyzer = Mock()
        mock_analyzer.generateCacheKey.return_value = "sentiment:stock:AAPL:90d"
        mock_analyzer.getCachedSentiment.return_value = cached_result
        mock_get_analyzer.return_value = mock_analyzer
        
        # Make API call
        url = reverse('analytics:stock_sentiment', kwargs={'symbol': 'AAPL'})
        response = self.client.get(url)
        
        # Verify cached response returned
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data['cached'], True)
        self.assertEqual(data['sentimentScore'], 0.42)
        
        # Verify yahoo service not called due to cache hit
        mock_yahoo.fetchNewsForStock.assert_not_called()
    
    def test_sentiment_endpoint_custom_days_parameter(self):
        """Test sentiment endpoint with custom days parameter."""
        # Authenticate user
        self.client.force_authenticate(user=self.user)
        
        with patch('Analytics.sentiment_views.yahoo_finance_service') as mock_yahoo, \
             patch('Analytics.sentiment_views.get_sentiment_analyzer') as mock_get_analyzer:
            
            mock_yahoo.fetchNewsForStock.return_value = []
            mock_analyzer = Mock()
            mock_analyzer.generateCacheKey.return_value = "sentiment:stock:AAPL:30d"
            mock_analyzer.getCachedSentiment.return_value = None
            mock_get_analyzer.return_value = mock_analyzer
            
            # Make API call with custom days
            url = reverse('analytics:stock_sentiment', kwargs={'symbol': 'AAPL'})
            response = self.client.get(url, {'days': 30})
            
            self.assertEqual(response.status_code, status.HTTP_200_OK)
            
            # Verify days parameter passed correctly
            mock_yahoo.fetchNewsForStock.assert_called_with('AAPL', days=30, max_items=50)
    
    def test_sentiment_endpoint_refresh_parameter(self):
        """Test sentiment endpoint with refresh parameter."""
        # Authenticate user
        self.client.force_authenticate(user=self.user)
        
        with patch('Analytics.sentiment_views.yahoo_finance_service') as mock_yahoo, \
             patch('Analytics.sentiment_views.get_sentiment_analyzer') as mock_get_analyzer:
            
            mock_yahoo.fetchNewsForStock.return_value = []
            mock_analyzer = Mock()
            mock_analyzer.generateCacheKey.return_value = "sentiment:stock:AAPL:90d"
            mock_analyzer.getCachedSentiment.return_value = {'cached': 'data'}  # Cached data exists
            mock_get_analyzer.return_value = mock_analyzer
            
            # Make API call with refresh=true (should bypass cache)
            url = reverse('analytics:stock_sentiment', kwargs={'symbol': 'AAPL'})
            response = self.client.get(url, {'refresh': 'true'})
            
            self.assertEqual(response.status_code, status.HTTP_200_OK)
            
            # Verify cache was bypassed
            mock_analyzer.getCachedSentiment.assert_not_called()
    
    def test_sentiment_endpoint_nonexistent_stock(self):
        """Test sentiment endpoint with non-existent stock symbol."""
        # Authenticate user
        self.client.force_authenticate(user=self.user)
        
        # Make API call with non-existent symbol
        url = reverse('analytics:stock_sentiment', kwargs={'symbol': 'INVALID'})
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        
        data = response.json()
        self.assertIn('error', data)
        self.assertIn('not found', data['error'].lower())
    
    @patch('Analytics.sentiment_views.get_sentiment_analyzer')
    def test_sentiment_endpoint_error_handling(self, mock_get_analyzer):
        """Test sentiment endpoint error handling."""
        # Authenticate user
        self.client.force_authenticate(user=self.user)
        
        # Mock analyzer to raise exception
        mock_get_analyzer.side_effect = Exception("Model loading failed")
        
        # Make API call
        url = reverse('analytics:stock_sentiment', kwargs={'symbol': 'AAPL'})
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        data = response.json()
        self.assertIn('error', data)
    
    def test_sentiment_endpoint_with_existing_analysis(self):
        """Test sentiment endpoint includes database sentiment data."""
        # Authenticate user
        self.client.force_authenticate(user=self.user)
        
        # Create existing analysis result
        AnalyticsResults.objects.create(
            stock=self.test_stock,
            as_of=timezone.now(),
            sentimentScore=0.65,
            sentimentLabel='positive',
            newsCount=20,
            score_0_10=7
        )
        
        with patch('Analytics.sentiment_views.yahoo_finance_service') as mock_yahoo, \
             patch('Analytics.sentiment_views.get_sentiment_analyzer') as mock_get_analyzer:
            
            mock_yahoo.fetchNewsForStock.return_value = []
            mock_analyzer = Mock()
            mock_analyzer.generateCacheKey.return_value = "sentiment:stock:AAPL:90d"
            mock_analyzer.getCachedSentiment.return_value = None
            mock_get_analyzer.return_value = mock_analyzer
            
            # Make API call
            url = reverse('analytics:stock_sentiment', kwargs={'symbol': 'AAPL'})
            response = self.client.get(url)
            
            self.assertEqual(response.status_code, status.HTTP_200_OK)
            
            data = response.json()
            # Should include database sentiment data
            self.assertIn('dbSentimentScore', data)
            self.assertEqual(data['dbSentimentScore'], 0.65)
            self.assertEqual(data['dbSentimentLabel'], 'positive')
            self.assertEqual(data['dbNewsCount'], 20)
    
    def test_sentiment_endpoint_rate_limiting(self):
        """Test sentiment endpoint rate limiting."""
        # Authenticate user
        self.client.force_authenticate(user=self.user)
        
        # This test would need actual rate limiting configuration
        # For now, just verify the throttle class is applied
        from Analytics.sentiment_views import stock_sentiment
        
        # Check that throttle classes are configured
        throttle_classes = getattr(stock_sentiment, 'throttle_classes', [])
        self.assertTrue(len(throttle_classes) > 0)


class SentimentAPIPerformanceTestCase(TestCase):
    """Performance tests for sentiment analysis API."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='perfuser',
            email='perf@example.com', 
            password='perfpass123'
        )
        self.client.force_authenticate(user=self.user)
        
        Stock.objects.create(
            symbol='PERF',
            company_name='Performance Test Co',
            sector='Technology'
        )
    
    @patch('Analytics.sentiment_views.yahoo_finance_service')
    @patch('Analytics.sentiment_views.get_sentiment_analyzer')
    def test_sentiment_response_time(self, mock_get_analyzer, mock_yahoo):
        """Test sentiment analysis response time is reasonable."""
        import time
        
        # Mock fast responses
        mock_yahoo.fetchNewsForStock.return_value = [
            {'title': f'News {i}', 'summary': f'Summary {i}', 'publishedDate': '2024-01-15T10:30:00Z', 'source': 'Test'}
            for i in range(10)
        ]
        mock_yahoo.preprocessNewsText.return_value = "Test news content"
        
        mock_analyzer = Mock()
        mock_analyzer.generateCacheKey.return_value = "sentiment:stock:PERF:90d"
        mock_analyzer.getCachedSentiment.return_value = None
        mock_analyzer.setCachedSentiment.return_value = True
        mock_analyzer.analyzeSentimentBatch.return_value = [
            {'sentimentScore': 0.5, 'sentimentLabel': 'neutral', 'sentimentConfidence': 0.7}
            for _ in range(10)
        ]
        mock_analyzer.aggregateSentiment.return_value = {
            'sentimentScore': 0.5,
            'sentimentLabel': 'neutral',
            'sentimentConfidence': 0.7,
            'distribution': {'positive': 3, 'negative': 2, 'neutral': 5}
        }
        mock_get_analyzer.return_value = mock_analyzer
        
        # Measure response time
        start_time = time.time()
        
        url = reverse('analytics:stock_sentiment', kwargs={'symbol': 'PERF'})
        response = self.client.get(url)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Response should be reasonably fast (under 5 seconds for mocked data)
        self.assertLess(response_time, 5.0)
        self.assertEqual(response.status_code, status.HTTP_200_OK)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
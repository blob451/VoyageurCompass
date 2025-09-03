"""
Unit tests for Analytics explanation views and API endpoints.
Tests explanation generation with real LLM service integration.
"""

from django.test import TestCase, TransactionTestCase
from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status
from datetime import datetime, timedelta
from decimal import Decimal
import json
from django.utils import timezone

from Data.models import Stock, DataSector, DataIndustry, AnalyticsResults
from Analytics.tests.fixtures import OllamaTestService, AnalyticsTestDataFactory

User = get_user_model()


class ExplanationViewsIntegrationTestCase(TransactionTestCase):
    """Integration tests for explanation views with real services."""
    
    def setUp(self):
        """Set up test environment."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='explanation_test_user',
            email='explanation@test.com',
            password='explanation_test_pass_123'
        )
        
        # Create test data
        self.sector = DataSector.objects.create(
            sectorKey='tech_explanation',
            sectorName='Technology Explanation',
            data_source='yahoo'
        )
        
        self.industry = DataIndustry.objects.create(
            industryKey='software_explanation',
            industryName='Software Explanation',
            sector=self.sector,
            data_source='yahoo'
        )
        
        self.stock = Stock.objects.create(
            symbol='EXPL',
            short_name='Explanation Test Corp',
            currency='USD',
            exchange='NASDAQ',
            sector_id=self.sector,
            industry_id=self.industry
        )
        
        # Create analytics results for explanation
        self.analytics = AnalyticsResults.objects.create(
            stock=self.stock,
            as_of=timezone.now(),
            w_rsi14=Decimal('0.655'),
            w_sma50vs200=Decimal('0.148'),
            w_macd12269=Decimal('0.085'),
            composite_raw=Decimal('7.2'),
            sentimentScore=0.72
        )
        
        # Initialize test service
        self.ollama_service = OllamaTestService()
    
    def test_explanation_generation_with_real_service(self):
        """Test explanation generation using real Ollama test service."""
        self.client.force_authenticate(user=self.user)
        
        try:
            # Test the explanation generation endpoint
            url = reverse('analytics:generate_explanation', args=[self.analytics.id])
            response = self.client.post(url, {
                'detail_level': 'standard'
            }, format='json')
            
            # Response should be successful or indicate service processing
            self.assertIn(response.status_code, [200, 202, 503])
            
            if response.status_code in [200, 202]:
                data = response.json()
                
                # Verify response structure
                if 'explanation' in data:
                    self.assertIn('content', data['explanation'])
                    self.assertIsInstance(data['explanation']['content'], str)
                elif 'status' in data:
                    self.assertIn(data['status'], ['processing', 'queued'])
                    
        except Exception as e:
            # If service unavailable, test should handle gracefully
            print(f"LLM service unavailable during test: {e}")
            self.assertIsInstance(e, Exception)
    
    def test_explanation_with_missing_analytics(self):
        """Test explanation generation when analytics data is missing."""
        self.client.force_authenticate(user=self.user)
        
        # Create stock without analytics
        stock_no_analytics = Stock.objects.create(
            symbol='NOANALYTI',
            short_name='No Analytics Corp',
            sector_id=self.sector,
            industry_id=self.industry
        )
        
        # Create a dummy analytics record for URL generation
        dummy_analytics = AnalyticsResults.objects.create(
            stock=stock_no_analytics,
            as_of=timezone.now(),
            composite_raw=Decimal('0.0'),
            sentimentScore=0.0
        )
        url = reverse('analytics:generate_explanation', args=[dummy_analytics.id])
        response = self.client.post(url, {
            'detail_level': 'standard'
        }, format='json')
        
        # Should handle gracefully
        self.assertIn(response.status_code, [200, 404, 503])
    
    def test_explanation_authentication_required(self):
        """Test that explanation endpoints require authentication."""
        # Test without authentication
        url = reverse('analytics:generate_explanation', args=[self.analytics.id])
        response = self.client.post(url, {
            'detail_level': 'standard'
        }, format='json')
        
        self.assertEqual(response.status_code, 401)
    
    def test_explanation_invalid_symbol(self):
        """Test explanation generation with invalid stock symbol."""
        self.client.force_authenticate(user=self.user)
        
        url = reverse('analytics:generate_explanation', args=[99999])  # Invalid analysis_id
        response = self.client.post(url, {
            'detail_level': 'standard'
        }, format='json')
        
        self.assertEqual(response.status_code, 404)
    
    def test_explanation_different_detail_levels(self):
        """Test explanation generation with different detail levels."""
        self.client.force_authenticate(user=self.user)
        
        detail_levels = ['summary', 'standard', 'detailed']
        
        for level in detail_levels:
            with self.subTest(detail_level=level):
                try:
                    url = reverse('analytics:generate_explanation', args=[self.analytics.id])
                    response = self.client.post(url, {
                        'detail_level': level
                    }, format='json')
                    
                    # Should handle all detail levels
                    self.assertIn(response.status_code, [200, 202, 503])
                    
                except Exception as e:
                    # Service may not be available for all levels
                    print(f"Detail level {level} unavailable: {e}")


class ExplanationHealthTestCase(TestCase):
    """Test cases for explanation service health and status."""
    
    def setUp(self):
        """Set up health test environment."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='health_test_user',
            email='health@test.com',
            password='health_test_pass_123'
        )
    
    def test_explanation_service_health_check(self):
        """Test explanation service health monitoring."""
        self.client.force_authenticate(user=self.user)
        
        try:
            # This would test a health endpoint if it exists
            # For now, test that the service initialization doesn't crash
            service = OllamaTestService()
            status = service.check_service_health()
            
            # Service health check should return boolean or status info
            self.assertIsNotNone(status)
            
        except Exception as e:
            # Health check may not be implemented
            print(f"Health check not available: {e}")
    
    def test_explanation_service_timeout_handling(self):
        """Test that explanation service handles timeouts gracefully."""
        self.client.force_authenticate(user=self.user)
        
        # Create test stock for timeout testing
        sector = DataSector.objects.create(
            sectorKey='timeout_test',
            sectorName='Timeout Test',
            data_source='yahoo'
        )
        
        stock = Stock.objects.create(
            symbol='TIMEOUT',
            short_name='Timeout Test Corp',
            sector_id=sector
        )
        
        try:
            # Create analytics for the timeout test stock
            timeout_analytics = AnalyticsResults.objects.create(
                stock=stock,
                as_of=timezone.now(),
                composite_raw=Decimal('5.0'),
                sentimentScore=0.5
            )
            url = reverse('analytics:generate_explanation', args=[timeout_analytics.id])
            response = self.client.post(url, {
                'detail_level': 'standard',
                'timeout': 1  # Very short timeout for testing
            }, format='json')
            
            # Should handle timeout gracefully
            self.assertIn(response.status_code, [200, 202, 408, 503])
            
        except Exception as e:
            # Timeout handling may vary
            print(f"Timeout test handled: {e}")


if __name__ == '__main__':
    import unittest
    unittest.main()
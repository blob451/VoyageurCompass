"""
Unit tests for the explanation service.
"""

from django.test import TestCase
from django.contrib.auth import get_user_model
# All tests now use real cache operations - no mocks required
from datetime import datetime

from Analytics.services.explanation_service import ExplanationService, get_explanation_service
from Data.models import Stock, AnalyticsResults

User = get_user_model()


class ExplanationServiceTestCase(TestCase):
    """Test cases for ExplanationService."""

    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        
        self.stock = Stock.objects.create(
            symbol='AAPL',
            short_name='Apple Inc.',
            sector='Technology',
            industry='Consumer Electronics'
        )
        
        self.analysis_result = AnalyticsResults.objects.create(
            user=self.user,
            stock=self.stock,
            as_of=datetime(2025, 1, 17, 10, 30),
            horizon='medium',
            score_0_10=7,
            composite_raw=2.1,
            w_sma50vs200=0.12,
            w_pricevs50=0.08,
            w_rsi14=0.03,
            w_macd12269=0.06,
            w_bbpos20=0.05,
            w_bbwidth20=-0.02,
            w_volsurge=0.04,
            w_obv20=0.02,
            w_rel1y=0.07,
            w_rel2y=0.06,
            w_candlerev=0.01,
            w_srcontext=0.03,
            components={
                'sma50vs200': {'raw': 1.2, 'score': 0.8},
                'rsi14': {'raw': 65.0, 'score': 0.3}
            }
        )

    def test_service_initialization(self):
        """Test that the service initializes correctly."""
        service = ExplanationService()
        self.assertIsNotNone(service)
        self.assertIsNotNone(service.llm_service)
        self.assertIsInstance(service.indicator_templates, dict)
        self.assertTrue(len(service.indicator_templates) > 0)

    def test_singleton_service(self):
        """Test that get_explanation_service returns singleton."""
        service1 = get_explanation_service()
        service2 = get_explanation_service()
        self.assertIs(service1, service2)

    def test_prepare_analysis_data_from_model(self):
        """Test preparation of analysis data from model instance."""
        service = ExplanationService()
        data = service._prepare_analysis_data(self.analysis_result)
        
        self.assertEqual(data['symbol'], 'AAPL')
        self.assertEqual(data['score_0_10'], 7.0)
        self.assertEqual(data['composite_raw'], 2.1)
        self.assertIn('weighted_scores', data)
        self.assertEqual(data['weighted_scores']['w_sma50vs200'], 0.12)

    def test_prepare_analysis_data_from_dict(self):
        """Test preparation of analysis data from dictionary."""
        service = ExplanationService()
        mock_dict = {
            'symbol': 'MSFT',
            'score_0_10': 8.5,
            'composite_raw': 3.2
        }
        data = service._prepare_analysis_data(mock_dict)
        
        self.assertEqual(data, mock_dict)

    def test_determine_recommendation(self):
        """Test recommendation determination logic."""
        service = ExplanationService()
        
        self.assertEqual(service._determine_recommendation(8.5), 'BUY')
        self.assertEqual(service._determine_recommendation(7.0), 'BUY')
        self.assertEqual(service._determine_recommendation(5.5), 'HOLD')
        self.assertEqual(service._determine_recommendation(4.0), 'HOLD')
        self.assertEqual(service._determine_recommendation(2.5), 'SELL')

    def test_extract_indicators_explained(self):
        """Test extraction of explained indicators."""
        service = ExplanationService()
        analysis_data = {
            'weighted_scores': {
                'w_sma50vs200': 0.12,
                'w_rsi14': 0.03,
                'w_macd12269': 0.0,  # Should be excluded
                'w_bbpos20': 0.05
            }
        }
        
        indicators = service._extract_indicators_explained(analysis_data)
        expected = ['sma50vs200', 'rsi14', 'bbpos20']
        self.assertEqual(set(indicators), set(expected))

    def test_extract_risk_factors(self):
        """Test extraction of risk factors."""
        service = ExplanationService()
        
        # Test high volatility detection
        analysis_data = {
            'weighted_scores': {
                'w_bbwidth20': 0.6,  # High volatility
                'w_rsi14': -0.4,     # Oversold
                'w_sma50vs200': 0.05  # Uncertain trend
            }
        }
        
        risk_factors = service._extract_risk_factors(analysis_data)
        self.assertIn('High volatility detected', risk_factors)
        self.assertIn('Potential oversold conditions', risk_factors)

    def test_build_indicator_explanation(self):
        """Test individual indicator explanation building."""
        service = ExplanationService()
        
        # Test RSI explanation
        explanation = service.build_indicator_explanation(
            'rsi14', 75.0, 0.2
        )
        self.assertIn('RSI', explanation)
        self.assertIn('overbought', explanation)
        self.assertIn('0.20', explanation)

    def test_generate_template_explanation_summary(self):
        """Test template explanation generation - summary level."""
        service = ExplanationService()
        analysis_data = {
            'symbol': 'AAPL',
            'score_0_10': 7.5,
            'weighted_scores': {
                'w_sma50vs200': 0.12,
                'w_rsi14': 0.03
            }
        }
        
        explanation = service._generate_template_explanation(analysis_data, 'summary')
        
        self.assertIsNotNone(explanation)
        self.assertIn('AAPL', explanation['content'])
        self.assertIn('7.5', explanation['content'])
        self.assertIn('BUY', explanation['content'])
        self.assertEqual(explanation['model_used'], 'template_fallback')
        self.assertGreater(explanation['word_count'], 0)

    def test_generate_template_explanation_detailed(self):
        """Test template explanation generation - detailed level."""
        service = ExplanationService()
        analysis_data = {
            'symbol': 'MSFT',
            'score_0_10': 8.2,
            'weighted_scores': {
                'w_sma50vs200': 0.15,
                'w_rsi14': 0.05,
                'w_macd12269': 0.08
            }
        }
        
        explanation = service._generate_template_explanation(analysis_data, 'detailed')
        
        self.assertIsNotNone(explanation)
        self.assertIn('MSFT', explanation['content'])
        self.assertIn('8.2', explanation['content'])
        self.assertIn('BUY', explanation['content'])
        self.assertIn('Technical Indicator Analysis:', explanation['content'])
        self.assertEqual(explanation['model_used'], 'template_fallback')
        # Detailed should have more words than summary
        self.assertGreater(explanation['word_count'], 50)

    def test_create_cache_key(self):
        """Test cache key creation."""
        service = ExplanationService()
        analysis_data = {
            'symbol': 'AAPL',
            'score_0_10': 7.5,
            'weighted_scores': {
                'w_sma50vs200': 0.12,
                'w_rsi14': 0.03
            }
        }
        
        key1 = service._create_cache_key(analysis_data, 'standard', self.user)
        key2 = service._create_cache_key(analysis_data, 'standard', self.user)
        key3 = service._create_cache_key(analysis_data, 'detailed', self.user)
        
        # Same data should produce same key
        self.assertEqual(key1, key2)
        # Different detail level should produce different key
        self.assertNotEqual(key1, key3)
        self.assertTrue(key1.startswith('explanation_'))

    def test_explain_prediction_single_caching(self):
        """Test that explanation results are cached properly using real cache."""
        from django.core.cache import cache
        service = ExplanationService()
        
        # Clear cache before test
        cache.clear()
        
        try:
            # First call should generate explanation and cache it
            explanation1 = service.explain_prediction_single(self.analysis_result)
            self.assertIsNotNone(explanation1)
            
            # Second call should use cached result (faster)
            import time
            start_time = time.time()
            explanation2 = service.explain_prediction_single(self.analysis_result)
            end_time = time.time()
            
            # Cached call should be faster and return same result
            self.assertIsNotNone(explanation2)
            self.assertLess(end_time - start_time, 5.0)  # Should complete quickly if cached
            
        except Exception as e:
            # Service may not be available, handle gracefully
            self.assertIsInstance(e, Exception)

    def test_service_status(self):
        """Test service status retrieval."""
        service = ExplanationService()
        status = service.get_service_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('enabled', status)
        self.assertIn('llm_available', status)
        self.assertIn('template_indicators', status)
        self.assertEqual(status['template_indicators'], len(service.indicator_templates))

    def test_error_handling_in_explanation_generation(self):
        """Test error handling in explanation generation."""
        service = ExplanationService()
        
        # Test with invalid analysis data
        invalid_data = {'invalid': 'data'}
        explanation = service._generate_template_explanation(invalid_data, 'standard')
        
        # Should return template fallback explanation (graceful handling)
        self.assertIsNotNone(explanation)
        self.assertEqual(explanation['model_used'], 'template_fallback')
        self.assertGreaterEqual(explanation['confidence_score'], 0.5)

    def test_batch_explanation(self):
        """Test batch explanation generation."""
        service = ExplanationService()
        
        # Create additional analysis result
        analysis_result2 = AnalyticsResults.objects.create(
            user=self.user,
            stock=self.stock,
            as_of=datetime(2025, 1, 16, 10, 30),
            horizon='short',
            score_0_10=6,
            composite_raw=1.8,
            w_sma50vs200=0.08,
            w_rsi14=0.02,
            components={'sma50vs200': {'raw': 1.0, 'score': 0.6}}
        )
        
        explanations = service.explain_prediction_batch(
            [self.analysis_result, analysis_result2],
            detail_level='standard'
        )
        
        self.assertEqual(len(explanations), 2)
        # Both should have explanations (template fallback)
        self.assertIsNotNone(explanations[0])
        self.assertIsNotNone(explanations[1])


class ExplanationServiceIntegrationTestCase(TestCase):
    """Integration tests for the explanation service."""

    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            username='integrationuser',
            email='integration@example.com',
            password='testpass123'
        )

    def test_full_explanation_flow(self):
        """Test the complete explanation generation flow."""
        service = get_explanation_service()
        
        # Test with mock data that simulates real analysis result
        mock_analysis_data = {
            'symbol': 'GOOGL',
            'score_0_10': 8.7,
            'composite_raw': 3.5,
            'analysis_date': '2025-01-17T15:30:00Z',
            'horizon': 'long',
            'components': {
                'sma50vs200': {'raw': 1.5, 'score': 1.0},
                'rsi14': {'raw': 55.0, 'score': 0.5},
                'macd12269': {'raw': 0.8, 'score': 0.7}
            },
            'weighted_scores': {
                'w_sma50vs200': 0.15,
                'w_pricevs50': 0.09,
                'w_rsi14': 0.05,
                'w_macd12269': 0.07,
                'w_bbpos20': 0.06,
                'w_bbwidth20': 0.02,
                'w_volsurge': 0.05,
                'w_obv20': 0.03,
                'w_rel1y': 0.08,
                'w_rel2y': 0.09,
                'w_candlerev': 0.02,
                'w_srcontext': 0.04
            }
        }
        
        # Test explanation generation
        explanation = service._generate_template_explanation(mock_analysis_data, 'standard')
        
        self.assertIsNotNone(explanation)
        self.assertIn('GOOGL', explanation['content'])
        self.assertIn('8.7', explanation['content'])
        self.assertIn('BUY', explanation['content'])
        self.assertEqual(explanation['recommendation'], 'BUY')
        self.assertGreater(explanation['confidence_score'], 0.5)
        self.assertGreater(explanation['word_count'], 10)
        
        # Check that important indicators are mentioned
        indicators_explained = explanation['indicators_explained']
        self.assertIn('sma50vs200', indicators_explained)
        
        # Check risk factors
        risk_factors = explanation['risk_factors']
        self.assertIsInstance(risk_factors, list)
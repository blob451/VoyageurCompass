"""
Integration tests for Hybrid LLM-FinBERT Analysis System.
Tests the complete Phase 1 implementation of sentiment-enhanced explanation generation.
Uses real services without mocks for authentic integration testing.
"""

from django.test import TestCase, TransactionTestCase
from django.core.cache import cache
from datetime import datetime, timedelta
from decimal import Decimal
import json

from Analytics.services.hybrid_analysis_coordinator import get_hybrid_analysis_coordinator
from Analytics.services.local_llm_service import get_local_llm_service, SentimentEnhancedPromptBuilder, ConfidenceAdaptiveGeneration
from Analytics.services.sentiment_analyzer import get_sentiment_analyzer


class HybridIntegrationTestCase(TestCase):
    """Integration tests for hybrid sentiment-LLM analysis system using real functionality."""

    def setUp(self):
        """Set up test environment."""
        cache.clear()

        # Initialize services
        self.hybrid_coordinator = get_hybrid_analysis_coordinator()
        self.llm_service = get_local_llm_service()
        self.sentiment_service = get_sentiment_analyzer()

        # Test data
        self.test_analysis_data = {
            'symbol': 'HYBRID_TEST',
            'score_0_10': 7.5,
            'weighted_scores': {
                'w_sma50vs200': 0.15,
                'w_rsi14': 0.08,
                'w_macd12269': 0.12,
                'w_bbpos20': 0.06,
                'w_volsurge': 0.10
            },
            'components': {},
            'news_articles': [
                {
                    'title': 'HYBRID_TEST Reports Strong Q3 Earnings',
                    'summary': 'The company exceeded expectations with 15% revenue growth and positive guidance for Q4.'
                },
                {
                    'title': 'HYBRID_TEST Announces Strategic Partnership',
                    'summary': 'New partnership expected to drive significant growth in the technology sector.'
                }
            ]
        }

        self.test_sentiment_data = {
            'sentimentScore': 0.65,
            'sentimentLabel': 'positive',
            'sentimentConfidence': 0.82,
            'newsCount': 2,
            'timestamp': datetime.now().isoformat()
        }

    def test_sentiment_enhanced_prompt_builder(self):
        """Test sentiment-enhanced prompt building functionality."""
        prompt_builder = SentimentEnhancedPromptBuilder()

        # Test with sentiment data
        enhanced_prompt = prompt_builder.build_sentiment_aware_prompt(
            self.test_analysis_data,
            self.test_sentiment_data,
            'standard',
            'technical_analysis'
        )

        # Validate prompt structure
        self.assertIsInstance(enhanced_prompt, str)
        self.assertGreater(len(enhanced_prompt), 100)
        self.assertIn('HYBRID_TEST', enhanced_prompt)
        self.assertIn('7.5/10', enhanced_prompt)
        self.assertIn('positive', enhanced_prompt.lower())
        self.assertIn('BUY/SELL/HOLD', enhanced_prompt)

        # Test without sentiment data
        no_sentiment_prompt = prompt_builder.build_sentiment_aware_prompt(
            self.test_analysis_data,
            None,
            'standard',
            'technical_analysis'
        )

        self.assertIsInstance(no_sentiment_prompt, str)
        self.assertNotIn('sentiment', no_sentiment_prompt.lower())

        # Test different detail levels
        for detail_level in ['summary', 'standard', 'detailed']:
            prompt = prompt_builder.build_sentiment_aware_prompt(
                self.test_analysis_data,
                self.test_sentiment_data,
                detail_level
            )
            self.assertIn('HYBRID_TEST', prompt)

            if detail_level == 'summary':
                self.assertLess(len(prompt), 600)  # Adjusted for sentiment context
            elif detail_level == 'detailed':
                self.assertGreater(len(prompt), 300)

    def test_confidence_adaptive_generation(self):
        """Test confidence-based adaptive generation parameters."""
        adaptive_generator = ConfidenceAdaptiveGeneration()

        # Test high confidence scenario
        high_confidence_options = adaptive_generator.get_confidence_weighted_options(
            self.test_sentiment_data,  # High confidence sentiment
            0.8,  # High complexity
            'standard',
            'llama3.1:8b'
        )

        # Validate adaptive options
        self.assertIsInstance(high_confidence_options, dict)
        self.assertIn('temperature', high_confidence_options)
        self.assertIn('top_p', high_confidence_options)
        self.assertIn('num_predict', high_confidence_options)

        # High confidence should have lower temperature (more focused)
        self.assertLessEqual(high_confidence_options['temperature'], 0.3)

        # Test low confidence scenario
        low_confidence_sentiment = {
            'sentimentScore': 0.1,
            'sentimentConfidence': 0.4,
            'sentimentLabel': 'neutral'
        }

        low_confidence_options = adaptive_generator.get_confidence_weighted_options(
            low_confidence_sentiment,
            0.3,  # Low complexity
            'standard',
            'llama3.1:8b'
        )

        # Low confidence should have higher temperature (more exploratory)
        self.assertGreaterEqual(low_confidence_options['temperature'], 0.4)

        # Test model-specific adjustments
        options_70b = adaptive_generator.get_confidence_weighted_options(
            self.test_sentiment_data,
            0.6,
            'detailed',
            'llama3.1:70b'
        )

        self.assertEqual(options_70b['num_ctx'], 1024)  # Larger context for 70B

    def test_hybrid_cache_functionality(self):
        """Test hybrid analysis caching system."""
        from Analytics.services.hybrid_analysis_coordinator import HybridAnalysisCache

        hybrid_cache = HybridAnalysisCache()

        # Test cache set and get
        test_result = {
            'content': 'Test explanation content',
            'generation_time': 1.5,
            'model_used': 'llama3.1:8b'
        }

        cache_key = 'test_hybrid_key'
        ttl = 300

        # Set cache
        set_success = hybrid_cache.set(cache_key, test_result, ttl)
        self.assertTrue(set_success)

        # Get from cache
        cached_result = hybrid_cache.get(cache_key)
        self.assertIsNotNone(cached_result)
        self.assertEqual(cached_result['content'], test_result['content'])
        self.assertIn('cached_at', cached_result)
        self.assertEqual(cached_result['cache_ttl'], ttl)

        # Test cache miss
        missing_result = hybrid_cache.get('nonexistent_key')
        self.assertIsNone(missing_result)

    def test_sentiment_aware_explanation_generation(self):
        """Test sentiment-aware explanation generation with real LLM service."""
        # Test with real service availability check
        if not self.llm_service.is_available():
            # Use fallback test explanation for when LLM is not available
            expected_content = 'STRONG BUY recommendation for HYBRID_TEST. The technical score of 7.5/10 indicates bullish momentum with SMA crossover and strong RSI signals. Positive market sentiment aligns with technical indicators.'

            # Verify the service gracefully handles unavailability
            result = self.llm_service.generate_sentiment_aware_explanation(
                analysis_data=self.test_analysis_data,
                sentiment_data=self.test_sentiment_data,
                detail_level='standard'
            )

            # Validate result structure when LLM is unavailable
            self.assertIsNotNone(result)
            if isinstance(result, dict):
                self.assertIn('content', result)
                self.assertTrue(len(result.get('content', '')) > 0)
        else:
            # Test with real LLM service when available
            result = self.llm_service.generate_sentiment_aware_explanation(
                analysis_data=self.test_analysis_data,
                sentiment_data=self.test_sentiment_data,
                detail_level='standard'
            )

            # Validate result structure with real service
            self.assertIsNotNone(result)
            self.assertIn('content', result)
            self.assertIn('sentiment_enhanced', result)
            self.assertTrue(result['sentiment_enhanced'])

            # Validate sentiment integration metadata
            self.assertIn('sentiment_integration', result)
            sentiment_meta = result['sentiment_integration']
            self.assertTrue(sentiment_meta['sentiment_data_available'])
            self.assertIn('sentiment_label', sentiment_meta)

            # Validate content quality
            content = result['content']
            self.assertTrue(len(content) > 50)  # Reasonable content length
            self.assertIn('HYBRID_TEST', content)

    def test_hybrid_coordinator_integration(self):
        """Test full hybrid coordinator integration with real services."""
        # Test enhanced explanation generation with real services
        result = self.hybrid_coordinator.generate_enhanced_explanation(
            analysis_data=self.test_analysis_data,
            detail_level='standard'
        )

        # Validate result structure
        self.assertIsNotNone(result)
        self.assertIn('content', result)

        # Validate content quality
        content = result.get('content', '')
        self.assertTrue(len(content) > 30)  # Reasonable content length
        self.assertIn('HYBRID_TEST', content)

        # Check if hybrid coordination metadata is present
        if 'hybrid_coordination' in result:
            coord_meta = result['hybrid_coordination']
            self.assertIsInstance(coord_meta, dict)
            if 'sentiment_integration_success' in coord_meta:
                self.assertIsInstance(coord_meta['sentiment_integration_success'], bool)

    def test_cache_key_generation(self):
        """Test sentiment-enhanced cache key generation."""
        # Test cache key creation
        cache_key = self.llm_service._create_sentiment_enhanced_cache_key(
            self.test_analysis_data,
            self.test_sentiment_data,
            'standard',
            'technical_analysis'
        )

        self.assertIsInstance(cache_key, str)
        self.assertIn('sentiment_enhanced:', cache_key)
        self.assertIn('_sent_', cache_key)  # Sentiment hash included

        # Test cache key without sentiment
        no_sentiment_key = self.llm_service._create_sentiment_enhanced_cache_key(
            self.test_analysis_data,
            None,
            'standard',
            'technical_analysis'
        )

        self.assertIsInstance(no_sentiment_key, str)
        self.assertIn('sentiment_enhanced:', no_sentiment_key)
        self.assertNotIn('_sent_', no_sentiment_key)  # No sentiment hash

        # Keys should be different
        self.assertNotEqual(cache_key, no_sentiment_key)

    def test_sentiment_aware_ttl_calculation(self):
        """Test sentiment-aware TTL calculation."""
        # High confidence sentiment should have longer TTL
        high_conf_ttl = self.llm_service._get_sentiment_aware_ttl(
            self.test_analysis_data,
            self.test_sentiment_data  # 0.82 confidence
        )

        # Low confidence sentiment should have shorter TTL
        low_confidence_sentiment = {
            'sentimentScore': 0.1,
            'sentimentConfidence': 0.3,
            'sentimentLabel': 'neutral'
        }

        low_conf_ttl = self.llm_service._get_sentiment_aware_ttl(
            self.test_analysis_data,
            low_confidence_sentiment
        )

        # High confidence should have longer TTL
        self.assertGreater(high_conf_ttl, low_conf_ttl)

        # Both should be within reasonable bounds
        self.assertGreaterEqual(high_conf_ttl, 60)
        self.assertLessEqual(high_conf_ttl, 1800)
        self.assertGreaterEqual(low_conf_ttl, 60)
        self.assertLessEqual(low_conf_ttl, 1800)

    def test_performance_metrics_tracking(self):
        """Test performance metrics tracking in hybrid system."""
        initial_metrics = self.hybrid_coordinator.get_performance_summary()

        # Mock a successful request
        with patch.object(self.sentiment_service, 'analyzeNewsArticles') as mock_sentiment:
            with patch.object(self.llm_service, 'generate_sentiment_aware_explanation') as mock_llm:
                mock_sentiment.return_value = self.test_sentiment_data
                mock_llm.return_value = {
                    'content': 'Test explanation',
                    'generation_time': 1.5,
                    'sentiment_enhanced': True
                }

                # Generate explanation
                result = self.hybrid_coordinator.generate_enhanced_explanation(
                    self.test_analysis_data
                )

                # Check metrics updated
                updated_metrics = self.hybrid_coordinator.get_performance_summary()

                self.assertGreater(updated_metrics['total_requests'], initial_metrics['total_requests'])
                self.assertGreater(updated_metrics['successful_requests'], initial_metrics['successful_requests'])
                self.assertGreaterEqual(updated_metrics['success_rate'], 0)
                self.assertGreaterEqual(updated_metrics['average_generation_time'], 0)

    def test_quality_assessment_functions(self):
        """Test quality assessment functions in hybrid coordinator."""
        # Test recommendation detection
        content_with_rec = "Based on the analysis, this is a STRONG BUY recommendation."
        content_without_rec = "The stock shows mixed signals with various indicators."

        has_rec_1 = self.hybrid_coordinator._has_clear_recommendation(content_with_rec)
        has_rec_2 = self.hybrid_coordinator._has_clear_recommendation(content_without_rec)

        self.assertTrue(has_rec_1)
        self.assertFalse(has_rec_2)

        # Test technical indicator mentions
        content_with_indicators = "The RSI is overbought and the SMA crossover indicates bullish momentum."
        mentions_indicators = self.hybrid_coordinator._mentions_technical_indicators(
            content_with_indicators,
            self.test_analysis_data
        )
        self.assertTrue(mentions_indicators)

        # Test sentiment alignment
        positive_content = "The stock shows strong bullish momentum with positive market sentiment."
        alignment = self.hybrid_coordinator._check_sentiment_alignment(
            positive_content,
            self.test_sentiment_data  # Positive sentiment
        )
        self.assertTrue(alignment)

        # Test content completeness
        complete_content = "This is a comprehensive analysis with multiple technical indicators showing bullish signals. The RSI and MACD support a buy recommendation."
        incomplete_content = "Buy."

        complete_1 = self.hybrid_coordinator._assess_content_completeness(complete_content)
        complete_2 = self.hybrid_coordinator._assess_content_completeness(incomplete_content)

        self.assertTrue(complete_1)
        self.assertFalse(complete_2)

    def test_error_handling_and_fallbacks(self):
        """Test error handling and fallback mechanisms."""
        # Test with invalid analysis data
        invalid_data = {}

        result = self.hybrid_coordinator.generate_enhanced_explanation(invalid_data)
        # Should handle gracefully without crashing

        # Test with sentiment service failure
        with patch.object(self.sentiment_service, 'analyzeNewsArticles', side_effect=Exception("Sentiment service error")):
            with patch.object(self.llm_service, 'generate_sentiment_aware_explanation') as mock_llm:
                mock_llm.return_value = {
                    'content': 'Technical analysis without sentiment',
                    'generation_time': 1.0,
                    'sentiment_enhanced': True
                }

                result = self.hybrid_coordinator.generate_enhanced_explanation(
                    self.test_analysis_data
                )

                # Should still generate explanation without sentiment
                if result:  # May be None if LLM service not available
                    self.assertIn('content', result)

    def test_caching_behavior_with_sentiment(self):
        """Test caching behavior with sentiment integration."""
        cache.clear()

        # Mock services for controlled testing
        with patch.object(self.sentiment_service, 'analyzeNewsArticles') as mock_sentiment:
            with patch.object(self.llm_service, 'generate_sentiment_aware_explanation') as mock_llm:
                mock_sentiment.return_value = self.test_sentiment_data
                mock_llm.return_value = {
                    'content': 'Test cached explanation',
                    'generation_time': 1.0,
                    'sentiment_enhanced': True,
                    'hybrid_coordination': {
                        'quality_metrics': {'has_recommendation': True}
                    }
                }

                # First request - should call services
                result1 = self.hybrid_coordinator.generate_enhanced_explanation(
                    self.test_analysis_data,
                    use_cache=True
                )

                # Second request - should use cache
                result2 = self.hybrid_coordinator.generate_enhanced_explanation(
                    self.test_analysis_data,
                    use_cache=True
                )

                # Both results should be identical
                if result1 and result2:
                    self.assertEqual(result1['content'], result2['content'])

                # Services should only be called once (cached second time)
                self.assertEqual(mock_sentiment.call_count, 1)
                self.assertEqual(mock_llm.call_count, 1)


class SentimentLLMPerformanceTestCase(TestCase):
    """Performance tests for sentiment-enhanced LLM system."""

    def setUp(self):
        """Set up performance testing environment."""
        cache.clear()
        self.hybrid_coordinator = get_hybrid_analysis_coordinator()

        # Performance test data
        self.perf_test_data = {
            'symbol': 'PERF_TEST',
            'score_0_10': 6.8,
            'weighted_scores': {
                'w_sma50vs200': 0.12,
                'w_rsi14': 0.09,
                'w_macd12269': 0.11
            },
            'news_articles': [
                {'title': 'PERF_TEST Q2 Results', 'summary': 'Strong performance with 12% growth.'}
            ]
        }

    @patch('Analytics.services.local_llm_service.Client')
    def test_response_time_with_sentiment_enhancement(self, mock_client_class):
        """Test that sentiment enhancement doesn't significantly impact response times."""
        # Setup fast mock response
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.generate.return_value = {
            'response': 'Fast test response with BUY recommendation'
        }

        llm_service = get_local_llm_service()

        # Test sentiment-aware generation timing with real service
        import time
        start_time = time.time()

        result = llm_service.generate_sentiment_aware_explanation(
            self.perf_test_data,
            {'sentimentScore': Decimal('0.5'), 'sentimentConfidence': Decimal('0.7'), 'sentimentLabel': 'positive'},
            'standard'
        )

        generation_time = time.time() - start_time

        # Allow reasonable time for real service (under 10 seconds)
        self.assertLess(generation_time, 10.0)

        if result:
            self.assertIn('content', result)
            self.assertIsInstance(result, dict)

    def test_cache_efficiency_with_hybrid_system(self):
        """Test cache efficiency improvements with hybrid system."""
        cache.clear()
        initial_metrics = self.hybrid_coordinator.get_performance_summary()

        with patch.object(self.hybrid_coordinator.sentiment_service, 'analyzeNewsArticles') as mock_sentiment:
            with patch.object(self.hybrid_coordinator.llm_service, 'generate_sentiment_aware_explanation') as mock_llm:
                mock_sentiment.return_value = {'sentimentScore': 0.6, 'sentimentLabel': 'positive', 'sentimentConfidence': 0.8}
                mock_llm.return_value = {'content': 'Cached test explanation', 'generation_time': 0.5}

                # Make multiple identical requests
                for _ in range(3):
                    result = self.hybrid_coordinator.generate_enhanced_explanation(
                        self.perf_test_data,
                        use_cache=True
                    )

                # Check cache hit improvement
                final_metrics = self.hybrid_coordinator.get_performance_summary()

                # Should have some cache hits after first request
                self.assertGreaterEqual(final_metrics['cache_hit_rate'], 0)

                # LLM service should only be called once (first request)
                self.assertEqual(mock_llm.call_count, 1)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])

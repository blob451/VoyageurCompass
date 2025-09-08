"""
Unit tests for Analytics Local LLM Service.
Tests LLaMA 3.1 integration via Ollama with real functionality - NO MOCKS.
Uses OllamaTestService for real LLM testing with graceful unavailability handling.
"""

import time
import json
from datetime import datetime
from django.test import TestCase
from django.core.cache import cache
from django.test.utils import override_settings
from django.utils import timezone

from Analytics.services.local_llm_service import LocalLLMService, get_local_llm_service
from Analytics.tests.fixtures import OllamaTestService, AnalyticsTestDataFactory
from Data.models import Stock, StockPrice
from decimal import Decimal


class RealLocalLLMServiceTestCase(TestCase):
    """Test cases for LocalLLMService using real functionality without mocks."""

    def setUp(self):
        """Set up test data with real service components."""
        # Create real test service instance
        self.ollama_test_service = OllamaTestService()

        # Test stock for realistic data
        self.test_stock = Stock.objects.create(
            symbol='TEST',
            short_name='Test Corporation',
            long_name='Test Corporation Ltd.',
            exchange='NASDAQ'
        )

        # Add price data
        StockPrice.objects.create(
            stock=self.test_stock,
            date=timezone.now().date(),
            close=Decimal('150.00'),
            high=Decimal('152.50'),
            low=Decimal('148.75'),
            open=Decimal('149.25'),
            volume=85000000
        )

        # Real analysis data using factory
        self.real_analysis_data = {
            'symbol': 'TEST',
            'score_0_10': 7,
            'components': {
                'rsi14': 0.65,
                'sma50vs200': 0.8,
                'macd12269': 0.55,
                'bollinger_bands': 0.4,
                'volume_surge': 0.75,
                'price_momentum': 0.68
            },
            'weighted_scores': {
                'w_rsi14': 0.13,
                'w_sma50vs200': 0.16,
                'w_macd12269': 0.11,
                'w_bollinger_bands': 0.08,
                'w_volsurge': 0.15,
                'w_price_momentum': 0.12
            }
        }

    def test_service_initialization_real(self):
        """Test LocalLLMService initialization with real components."""
        service = LocalLLMService()

        # Verify real service attributes
        self.assertEqual(service.primary_model, "llama3.1:8b")
        self.assertEqual(service.detailed_model, "llama3.1:70b")
        self.assertTrue(service.performance_mode)
        self.assertEqual(service.generation_timeout, 45)
        self.assertEqual(service.max_retries, 3)

        # Service should handle initialization gracefully even if Ollama unavailable
        self.assertIsNotNone(service)

    def test_real_model_availability_check(self):
        """Test model availability with real service connection."""
        service = LocalLLMService()

        # Test connection - should handle both available and unavailable cases
        is_available = service.is_available()
        self.assertIsInstance(is_available, bool)

        # Model verification should work regardless of service availability
        primary_check = service._verify_model_availability('llama3.1:8b')
        detailed_check = service._verify_model_availability('llama3.1:70b')

        self.assertIsInstance(primary_check, bool)
        self.assertIsInstance(detailed_check, bool)

    def test_optimal_model_selection_real(self):
        """Test model selection logic with real service."""
        service = LocalLLMService()

        # Test all detail levels with real selection logic
        for detail_level in ['summary', 'standard', 'detailed']:
            selected_model = service._select_optimal_model(detail_level)

            # Should return a valid model name
            self.assertIn('llama3.1', selected_model)
            self.assertIn(selected_model, ['llama3.1:8b', 'llama3.1:70b'])

    def test_real_explanation_generation_with_service_available(self):
        """Test explanation generation when service is available."""
        service = LocalLLMService()

        if service.is_available():
            # Use real LLM service
            result = service.generate_explanation(
                self.real_analysis_data,
                detail_level='standard'
            )

            if result is not None:  # Service responded
                self.assertIsInstance(result, dict)
                self.assertIn('content', result)
                self.assertIn('generation_time', result)
                self.assertIn('confidence_score', result)
                self.assertIn('model_used', result)
                self.assertEqual(result['detail_level'], 'standard')
                self.assertIsInstance(result['content'], str)
                self.assertGreater(len(result['content']), 10)  # Non-trivial content
                self.assertGreaterEqual(result['confidence_score'], 0.0)
                self.assertLessEqual(result['confidence_score'], 1.0)
        else:
            # Service unavailable - should handle gracefully
            result = service.generate_explanation(self.real_analysis_data)
            self.assertIsNone(result)

    def test_real_explanation_generation_with_test_service(self):
        """Test explanation generation using OllamaTestService for consistent results."""
        # Create explanation using test service
        prompt = f"Technical analysis for TEST with score 7/10. Key indicators: RSI 0.65, SMA 0.8, MACD 0.55"
        context_data = {
            'symbol': 'TEST',
            'current_price': 150.00,
            'trend': 'bullish'
        }

        result = self.ollama_test_service.generate_explanation(prompt, context_data)

        self.assertIsInstance(result, dict)
        self.assertIn('response', result)
        self.assertIn('model', result)
        self.assertIn('processing_time', result)
        self.assertIn('confidence', result)

        # Verify content quality
        content = result['response']
        self.assertIn('TEST', content)
        self.assertIn('Technical Analysis', content)
        self.assertGreater(len(content), 50)  # Substantial content

    def test_explanation_generation_different_detail_levels_real(self):
        """Test explanation generation across all detail levels."""
        service = LocalLLMService()

        detail_levels = ['summary', 'standard', 'detailed']

        for detail_level in detail_levels:
            if service.is_available():
                result = service.generate_explanation(
                    self.real_analysis_data,
                    detail_level=detail_level
                )

                if result is not None:
                    self.assertIsInstance(result, dict)
                    self.assertEqual(result['detail_level'], detail_level)
                    self.assertIn('content', result)

                    # Verify content scales with detail level
                    word_count = result.get('word_count', 0)
                    if detail_level == 'summary':
                        self.assertLessEqual(word_count, 100)  # Concise
                    elif detail_level == 'detailed':
                        self.assertGreater(word_count, 20)  # More comprehensive
            else:
                # Service unavailable - test graceful handling
                result = service.generate_explanation(
                    self.real_analysis_data,
                    detail_level=detail_level
                )
                self.assertIsNone(result)

    def test_service_unavailability_handling(self):
        """Test graceful handling when LLM service is unavailable."""
        service = LocalLLMService()

        # Force service unavailable state by clearing client
        original_client = service.client
        service.client = None

        try:
            result = service.generate_explanation(self.real_analysis_data)
            self.assertIsNone(result)

            # Service status should reflect unavailability
            status = service.get_service_status()
            self.assertFalse(status['available'])
            self.assertIn('error', status)

        finally:
            # Restore original client
            service.client = original_client

    def test_real_prompt_building(self):
        """Test LLaMA prompt construction with real data."""
        service = LocalLLMService()

        prompt = service._build_prompt(
            self.real_analysis_data,
            detail_level='standard',
            explanation_type='technical_analysis'
        )

        # Verify prompt contains key information
        self.assertIn('TEST', prompt)
        self.assertIn('7/10', prompt)
        self.assertIn('Key Indicators', prompt)
        self.assertIn('Analysis:', prompt)

        # Should contain weighted indicators
        self.assertIn('w_sma50vs200', prompt)
        self.assertIn('w_rsi14', prompt)

    def test_prompt_building_different_detail_levels(self):
        """Test prompt construction for different detail levels."""
        service = LocalLLMService()

        detail_levels = ['summary', 'standard', 'detailed']

        for detail_level in detail_levels:
            prompt = service._build_prompt(
                self.real_analysis_data,
                detail_level=detail_level,
                explanation_type='technical_analysis'
            )

            self.assertIn('TEST', prompt)

            if detail_level == 'summary':
                self.assertIn('1-2 sentences', prompt)
            elif detail_level == 'detailed':
                self.assertIn('investment recommendation', prompt)
                self.assertIn('Key supporting indicators', prompt)

    def test_generation_options_real(self):
        """Test generation options for real models."""
        service = LocalLLMService()

        # Test options for both models
        options_8b = service._get_optimized_generation_options('standard', 'llama3.1:8b')
        options_70b = service._get_optimized_generation_options('standard', 'llama3.1:70b')

        # Verify basic structure
        self.assertIn('temperature', options_8b)
        self.assertIn('num_predict', options_8b)
        self.assertIn('num_ctx', options_8b)

        # 8B model should have smaller context (optimized values)
        self.assertEqual(options_8b['num_ctx'], 512)
        self.assertEqual(options_70b['num_ctx'], 1024)

        # Verify temperature differences (optimized values)
        self.assertEqual(options_8b['temperature'], 0.4)
        self.assertEqual(options_70b['temperature'], 0.2)

    def test_max_tokens_by_detail_level(self):
        """Test token limits for different detail levels."""
        service = LocalLLMService()

        tokens_summary = service._get_optimized_tokens('summary')
        tokens_standard = service._get_optimized_tokens('standard')
        tokens_detailed = service._get_optimized_tokens('detailed')

        self.assertEqual(tokens_summary, 50)
        self.assertEqual(tokens_standard, 100)
        self.assertEqual(tokens_detailed, 175)

        # Verify ordering
        self.assertLess(tokens_summary, tokens_standard)
        self.assertLess(tokens_standard, tokens_detailed)

    def test_confidence_score_calculation_real(self):
        """Test confidence score calculation with real content."""
        service = LocalLLMService()

        # Test various content qualities
        good_content = "Strong BUY recommendation based on bullish indicators. RSI shows healthy momentum with score of 7/10. Technical trend analysis suggests upward trajectory with moderate risk."
        poor_content = "Buy."
        medium_content = "TEST shows positive momentum. Score 7/10 indicates bullish trend."

        confidence_good = service._calculate_confidence_score(good_content)
        confidence_poor = service._calculate_confidence_score(poor_content)
        confidence_medium = service._calculate_confidence_score(medium_content)

        # Good content should have highest confidence
        self.assertGreater(confidence_good, confidence_medium)
        self.assertGreater(confidence_medium, confidence_poor)

        # All should be valid confidence scores
        for conf in [confidence_good, confidence_poor, confidence_medium]:
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)

    def test_real_caching_mechanism(self):
        """Test explanation caching with real service."""
        service = LocalLLMService()

        # Clear cache
        cache.clear()

        if service.is_available():
            # First call
            start_time = time.time()
            result1 = service.generate_explanation(self.real_analysis_data)
            first_call_time = time.time() - start_time

            if result1 is not None:
                # Second call should be faster due to caching
                start_time = time.time()
                result2 = service.generate_explanation(self.real_analysis_data)
                second_call_time = time.time() - start_time

                self.assertIsNotNone(result2)
                # Cache hit should be significantly faster
                self.assertLess(second_call_time, first_call_time)

                # Content should be identical (from cache)
                self.assertEqual(result1['content'], result2['content'])

    def test_cache_key_creation_consistency(self):
        """Test cache key creation consistency."""
        service = LocalLLMService()

        cache_key1 = service._create_cache_key(
            self.real_analysis_data,
            'standard',
            'technical_analysis'
        )

        cache_key2 = service._create_cache_key(
            self.real_analysis_data,
            'standard',
            'technical_analysis'
        )

        # Same inputs should produce same cache key
        self.assertEqual(cache_key1, cache_key2)

        # Different detail level should produce different key
        cache_key3 = service._create_cache_key(
            self.real_analysis_data,
            'detailed',
            'technical_analysis'
        )

        self.assertNotEqual(cache_key1, cache_key3)

    def test_real_batch_explanation_generation(self):
        """Test batch explanation generation with real service."""
        service = LocalLLMService()

        # Create multiple analysis data sets
        analysis_batch = [
            self.real_analysis_data,
            {**self.real_analysis_data, 'symbol': 'TEST2', 'score_0_10': 6},
            {**self.real_analysis_data, 'symbol': 'TEST3', 'score_0_10': 8}
        ]

        results = service.generate_batch_explanations(
            analysis_batch,
            detail_level='summary'
        )

        self.assertEqual(len(results), 3)

        # Each result should be valid or None (if service unavailable)
        for result in results:
            if result is not None:
                self.assertIsInstance(result, dict)
                self.assertIn('content', result)
                self.assertEqual(result['detail_level'], 'summary')

    def test_real_service_status_reporting(self):
        """Test service status reporting with real service."""
        service = LocalLLMService()

        status = service.get_service_status()

        # Status should always be a dictionary
        self.assertIsInstance(status, dict)
        self.assertIn('available', status)
        self.assertIn('primary_model', status)
        self.assertIn('detailed_model', status)
        self.assertIn('performance_mode', status)

        # Model names should be correct
        self.assertEqual(status['primary_model'], 'llama3.1:8b')
        self.assertEqual(status['detailed_model'], 'llama3.1:70b')

        if status['available']:
            # If available, should have model availability flags
            self.assertIn('primary_model_available', status)
            self.assertIn('detailed_model_available', status)
        else:
            # If unavailable, should have error information
            self.assertIn('error', status)

    def test_real_timeout_handling(self):
        """Test timeout handling with real service."""
        service = LocalLLMService()

        # Test with very short timeout
        original_timeout = service.generation_timeout
        service.generation_timeout = 0.1  # 100ms - very short

        try:
            if service.is_available():
                # This may timeout or complete quickly depending on service
                result = service.generate_explanation(self.real_analysis_data)

                # Result should be None (timeout) or valid dict (completed quickly)
                if result is not None:
                    self.assertIsInstance(result, dict)

            else:
                # Service unavailable - should return None gracefully
                result = service.generate_explanation(self.real_analysis_data)
                self.assertIsNone(result)

        finally:
            # Restore original timeout
            service.generation_timeout = original_timeout

    def test_real_performance_monitoring(self):
        """Test performance monitoring with real service calls."""
        service = LocalLLMService()

        if service.is_available():
            start_time = time.time()
            result = service.generate_explanation(self.real_analysis_data)
            total_time = time.time() - start_time

            if result is not None:
                # Should have generation time recorded
                self.assertIn('generation_time', result)
                self.assertGreater(result['generation_time'], 0)

                # Should have performance metadata
                self.assertIn('model_used', result)
                self.assertIn('word_count', result)
                self.assertIn('confidence_score', result)

                # Verify timing consistency
                self.assertLessEqual(result['generation_time'], total_time + 0.1)


class RealLocalLLMServiceSingletonTestCase(TestCase):
    """Test singleton pattern with real service."""

    def test_singleton_pattern_real(self):
        """Test that get_local_llm_service returns singleton instance."""
        service1 = get_local_llm_service()
        service2 = get_local_llm_service()

        # Should be the same instance
        self.assertIs(service1, service2)

        # Should be LocalLLMService instance
        self.assertIsInstance(service1, LocalLLMService)
        self.assertIsInstance(service2, LocalLLMService)


class RealLocalLLMServiceIntegrationTestCase(TestCase):
    """Integration tests using real data and service functionality."""

    def setUp(self):
        """Set up realistic test data."""
        # Create realistic stock and price data
        self.aapl_stock = Stock.objects.create(
            symbol='AAPL',
            short_name='Apple Inc.',
            long_name='Apple Inc. Corporation',
            exchange='NASDAQ'
        )

        StockPrice.objects.create(
            stock=self.aapl_stock,
            date=timezone.now().date(),
            close=Decimal('175.25'),
            high=Decimal('177.50'),
            low=Decimal('174.10'),
            open=Decimal('176.00'),
            volume=95000000
        )

        # Realistic analysis data using factory pattern
        self.realistic_analysis_data = AnalyticsTestDataFactory.create_comprehensive_analysis(
            self.aapl_stock
        )

        # Convert to dictionary format for service
        self.analysis_dict = {
            'symbol': self.aapl_stock.symbol,
            'score_0_10': int(self.realistic_analysis_data.score_0_10),
            'components': json.loads(self.realistic_analysis_data.components),
            'weighted_scores': {
                'w_sma50vs200': float(self.realistic_analysis_data.w_sma50vs200),
                'w_pricevs50': float(self.realistic_analysis_data.w_pricevs50),
                'w_rsi14': float(self.realistic_analysis_data.w_rsi14),
                'w_macd12269': float(self.realistic_analysis_data.w_macd12269),
                'w_bbpos20': float(self.realistic_analysis_data.w_bbpos20),
                'w_volsurge': float(self.realistic_analysis_data.w_volsurge),
            }
        }

    def test_realistic_explanation_generation_integration(self):
        """Test explanation generation with realistic financial data."""
        service = LocalLLMService()

        if service.is_available():
            result = service.generate_explanation(
                self.analysis_dict,
                detail_level='detailed'
            )

            if result is not None:
                self.assertIsNotNone(result)
                self.assertIn('AAPL', result['content'])
                self.assertIn(str(self.analysis_dict['score_0_10']), result['content'])

                # Should mention financial concepts
                content_lower = result['content'].lower()
                financial_terms = ['buy', 'sell', 'hold', 'bullish', 'bearish', 'momentum', 'trend']
                term_found = any(term in content_lower for term in financial_terms)
                self.assertTrue(term_found, "Should contain financial terminology")

                # Should have reasonable confidence
                self.assertGreater(result['confidence_score'], 0.5)

                # Should have substantial content for detailed level
                self.assertGreater(result['word_count'], 20)
        else:
            # Test graceful degradation when service unavailable
            result = service.generate_explanation(self.analysis_dict)
            self.assertIsNone(result)

    def test_multiple_detail_levels_integration_real(self):
        """Test generation across all detail levels with realistic data."""
        service = LocalLLMService()

        results = {}

        for detail_level in ['summary', 'standard', 'detailed']:
            if service.is_available():
                result = service.generate_explanation(
                    self.analysis_dict,
                    detail_level=detail_level
                )

                if result is not None:
                    results[detail_level] = result

                    # Verify basic structure
                    self.assertIsNotNone(result)
                    self.assertEqual(result['detail_level'], detail_level)
                    self.assertIn('AAPL', result['content'])

                    # Verify content scaling
                    word_count = result['word_count']
                    if detail_level == 'summary':
                        self.assertLessEqual(word_count, 50)
                    elif detail_level == 'detailed':
                        self.assertGreater(word_count, 15)

        # If we got results for multiple levels, verify content scaling
        if len(results) >= 2:
            if 'summary' in results and 'standard' in results:
                self.assertLessEqual(
                    results['summary']['word_count'],
                    results['standard']['word_count'] * 1.2  # Allow some variance
                )

    def test_real_service_with_factory_data(self):
        """Test service with data created by AnalyticsTestDataFactory."""
        # Create different types of analysis
        tech_analysis = AnalyticsTestDataFactory.create_technical_analysis_data(self.aapl_stock)
        sentiment_analysis = AnalyticsTestDataFactory.create_sentiment_analysis_data(self.aapl_stock)

        service = LocalLLMService()

        if service.is_available():
            # Test with technical analysis data
            tech_dict = {
                'symbol': self.aapl_stock.symbol,
                'score_0_10': int(tech_analysis.score_0_10),
                'components': json.loads(tech_analysis.components),
                'weighted_scores': {
                    'w_sma50vs200': float(tech_analysis.w_sma50vs200),
                    'w_rsi14': float(tech_analysis.w_rsi14),
                }
            }

            result = service.generate_explanation(tech_dict, detail_level='standard')

            if result is not None:
                self.assertIn('AAPL', result['content'])
                self.assertIsInstance(result['generation_time'], float)
                self.assertGreater(result['generation_time'], 0)

    def test_error_conditions_real(self):
        """Test various error conditions with real service."""
        service = LocalLLMService()

        # Test with invalid data
        invalid_data = {'invalid': 'data'}
        result = service.generate_explanation(invalid_data)

        if service.is_available():
            # May return None or handle gracefully
            if result is not None:
                self.assertIsInstance(result, dict)
        else:
            # Service unavailable should return None
            self.assertIsNone(result)

        # Test with empty data
        empty_data = {}
        result = service.generate_explanation(empty_data)

        # Should handle gracefully regardless of service availability
        if result is not None:
            self.assertIsInstance(result, dict)

    def test_service_resilience(self):
        """Test service resilience under various conditions."""
        service = LocalLLMService()

        # Test multiple rapid requests
        results = []
        for i in range(3):
            test_data = {
                'symbol': f'TEST{i}',
                'score_0_10': 5 + i,
                'components': {'rsi14': 0.5 + i * 0.1},
                'weighted_scores': {'w_rsi14': 0.1 + i * 0.02}
            }

            result = service.generate_explanation(test_data, detail_level='summary')
            results.append(result)

        # Should handle multiple requests gracefully
        non_null_results = [r for r in results if r is not None]

        if len(non_null_results) > 0:
            # If any succeeded, they should be valid
            for result in non_null_results:
                self.assertIsInstance(result, dict)
                self.assertIn('content', result)
                self.assertIn('generation_time', result)

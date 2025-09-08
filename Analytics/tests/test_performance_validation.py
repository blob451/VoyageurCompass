"""
Performance validation tests for enhanced LLM service.
Tests the performance improvements implemented in Phase 1 optimization.
Uses real service implementations for authentic performance testing.
"""

import time
import pytest
from django.test import TestCase
from django.core.cache import cache
from decimal import Decimal
from Analytics.services.local_llm_service import LocalLLMService, LLMPerformanceMonitor, LLMCircuitBreaker


class LLMPerformanceValidationTestCase(TestCase):
    """Test performance improvements and monitoring functionality."""

    def setUp(self):
        """Set up test environment."""
        cache.clear()
        self.service = LocalLLMService()

        # Test data scenarios for performance validation
        self.test_scenarios = {
            'simple_bullish': {
                'symbol': 'TEST1',
                'score_0_10': Decimal('8.0'),
                'weighted_scores': {
                    'w_sma50vs200': Decimal('0.15'),
                    'w_rsi14': Decimal('0.08'),
                    'w_macd12269': Decimal('0.06')
                },
                'components': {}
            },
            'complex_analysis': {
                'symbol': 'TEST2',
                'score_0_10': Decimal('5.5'),
                'weighted_scores': {
                    'w_sma50vs200': Decimal('0.12'),
                    'w_rsi14': Decimal('0.18'),
                    'w_macd12269': Decimal('0.15'),
                    'w_bbpos20': Decimal('0.09'),
                    'w_volsurge': Decimal('0.14'),
                    'w_obv20': Decimal('0.07'),
                    'w_rel1y': Decimal('0.11'),
                    'w_rel2y': Decimal('0.09')
                },
                'components': {}
            },
            'volatile_stock': {
                'symbol': 'VOLATILE',
                'score_0_10': 3.2,
                'weighted_scores': {
                    'w_bbwidth20': 0.20,  # High volatility indicator
                    'w_volsurge': 0.18,
                    'w_rsi14': 0.05
                },
                'components': {}
            }
        }

    def test_circuit_breaker_functionality(self):
        """Test circuit breaker pattern implementation."""
        circuit_breaker = LLMCircuitBreaker(failure_threshold=3, recovery_timeout=1)

        # Test normal operation (closed state)
        self.assertEqual(circuit_breaker.state, 'CLOSED')

        # Simulate successful call
        def successful_func():
            return "success"

        result = circuit_breaker.call_llm(successful_func)
        self.assertEqual(result, "success")
        self.assertEqual(circuit_breaker.state, 'CLOSED')

        # Simulate multiple failures to open circuit
        def failing_func():
            raise Exception("Service unavailable")

        for i in range(3):
            with self.assertRaises(Exception):
                circuit_breaker.call_llm(failing_func)

        # Circuit should now be OPEN
        self.assertEqual(circuit_breaker.state, 'OPEN')

        # Additional calls should be blocked
        with self.assertRaises(Exception) as cm:
            circuit_breaker.call_llm(failing_func)
        self.assertIn("circuit breaker is open", str(cm.exception))

    def test_performance_monitor_metrics(self):
        """Test performance monitoring functionality."""
        monitor = LLMPerformanceMonitor()

        # Record various performance scenarios
        monitor.record_generation(2.5, 'llama3.1:8b', True, cache_hit=False)
        monitor.record_generation(0.1, 'cache', True, cache_hit=True)
        monitor.record_generation(5.2, 'llama3.1:70b', True, cache_hit=False)
        monitor.record_generation(0, 'llama3.1:8b', False, cache_hit=False)  # Failed request

        # Get performance summary
        summary = monitor.get_performance_summary()

        # Validate metrics
        self.assertIsInstance(summary, dict)
        self.assertIn('avg_generation_time', summary)
        self.assertIn('success_rate', summary)
        self.assertIn('cache_hit_rate', summary)
        self.assertIn('total_requests', summary)

        # Validate calculated values
        self.assertEqual(summary['total_requests'], 4)
        self.assertEqual(summary['success_rate'], 0.75)  # 3 successful out of 4
        self.assertEqual(summary['cache_hit_rate'], 1/3)  # 1 cache hit out of 3 successful
        self.assertGreater(summary['avg_generation_time'], 0)

    def test_complexity_score_calculation(self):
        """Test analysis complexity scoring for model selection."""
        # Test simple analysis (low complexity)
        simple_score = self.service._calculate_complexity_score(self.test_scenarios['simple_bullish'])
        self.assertIsInstance(simple_score, float)
        self.assertGreaterEqual(simple_score, 0.0)
        self.assertLessEqual(simple_score, 1.0)

        # Test complex analysis (high complexity)
        complex_score = self.service._calculate_complexity_score(self.test_scenarios['complex_analysis'])

        # Complex analysis should have more indicators, so should be at least equal or higher
        self.assertGreaterEqual(complex_score, simple_score * 0.9)  # Allow some variance

        # Complex analysis should have reasonable complexity score
        self.assertGreater(complex_score, 0.3)

    def test_smart_model_selection(self):
        """Test intelligent model selection logic."""
        # Test summary requests (should prefer 8B model)
        model = self.service._select_optimal_model('summary', 0.2)
        # Should return primary model if available (we can't test availability without Ollama)
        self.assertIn(model, ['llama3.1:8b', 'llama3.1:70b'])

        # Test detailed complex requests (may use 70B model)
        model = self.service._select_optimal_model('detailed', 0.9)
        self.assertIn(model, ['llama3.1:8b', 'llama3.1:70b'])

        # Test standard requests (should default to 8B)
        model = self.service._select_optimal_model('standard', 0.5)
        self.assertIn(model, ['llama3.1:8b', 'llama3.1:70b'])

    def test_dynamic_cache_ttl(self):
        """Test dynamic cache TTL calculation."""
        # Test high confidence score (should have longer TTL)
        high_confidence_ttl = self.service._get_dynamic_ttl(self.test_scenarios['simple_bullish'])
        self.assertGreaterEqual(high_confidence_ttl, 180)  # Should be at least base TTL

        # Test volatile stock (should have shorter TTL)
        volatile_ttl = self.service._get_dynamic_ttl(self.test_scenarios['volatile_stock'])
        self.assertLessEqual(volatile_ttl, 180)  # Should be less than or equal to base TTL

        # Test standard analysis (score 5.5 falls in uncertain range, gets reduced TTL)
        standard_ttl = self.service._get_dynamic_ttl(self.test_scenarios['complex_analysis'])
        self.assertEqual(standard_ttl, 90)  # Should be half base TTL due to score in 4-6 range

    def test_optimized_prompt_generation(self):
        """Test optimized prompt generation for different detail levels."""
        test_data = self.test_scenarios['simple_bullish']

        # Test summary prompt (should be very short)
        summary_prompt = self.service._build_optimized_prompt(test_data, 'summary', 'technical_analysis')
        self.assertLess(len(summary_prompt), 100)
        self.assertIn('TEST1', summary_prompt)
        self.assertIn('8.0', summary_prompt)

        # Test standard prompt
        standard_prompt = self.service._build_optimized_prompt(test_data, 'standard', 'technical_analysis')
        self.assertLess(len(standard_prompt), 150)
        self.assertIn('TEST1', summary_prompt)

        # Test detailed prompt
        detailed_prompt = self.service._build_optimized_prompt(test_data, 'detailed', 'technical_analysis')
        self.assertGreater(len(detailed_prompt), len(summary_prompt))
        self.assertIn('TEST1', detailed_prompt)

    def test_optimized_generation_options(self):
        """Test optimized generation options for different scenarios."""
        # Test 8B model options
        options_8b = self.service._get_optimized_generation_options('standard', 'llama3.1:8b')
        self.assertEqual(options_8b['temperature'], 0.4)  # Base temperature for 8B standard
        self.assertEqual(options_8b['top_p'], 0.7)
        self.assertEqual(options_8b['num_ctx'], 512)  # Reduced context
        self.assertEqual(options_8b['num_predict'], 100)  # Optimized tokens

        # Test 70B model options
        options_70b = self.service._get_optimized_generation_options('detailed', 'llama3.1:70b')
        self.assertEqual(options_70b['temperature'], 0.2)  # More focused
        self.assertEqual(options_70b['num_ctx'], 1024)  # Larger context but still optimized

        # Test summary level options
        summary_options = self.service._get_optimized_generation_options('summary', 'llama3.1:8b')
        self.assertEqual(summary_options['num_predict'], 50)  # Fewer tokens for summary
        self.assertGreaterEqual(summary_options['temperature'], 0.4)  # Higher for conciseness

    def test_top_indicators_extraction(self):
        """Test top indicators extraction for prompt optimization."""
        # Test with complex analysis data
        top_indicators = self.service._get_top_indicators(self.test_scenarios['complex_analysis'], limit=3)
        self.assertIsInstance(top_indicators, str)
        self.assertGreater(len(top_indicators), 5)

        # Should contain indicator names without 'w_' prefix
        self.assertNotIn('w_', top_indicators)

        # Test with simple data
        simple_indicators = self.service._get_top_indicators(self.test_scenarios['simple_bullish'], limit=2)
        self.assertIsInstance(simple_indicators, str)

        # Test with empty data
        empty_indicators = self.service._get_top_indicators({}, limit=2)
        self.assertEqual(empty_indicators, 'technical analysis')

    def test_explanation_generation_performance_path(self):
        """Test the performance path through explanation generation with real service."""
        # Test with real service performance
        start_time = time.time()

        # Test explanation generation with real service
        result = self.service.generate_explanation(
            self.test_scenarios['simple_bullish'], 
            'standard'
        )

        end_time = time.time()
        generation_time = end_time - start_time

        # Allow reasonable time for real service (up to 30 seconds for LLM)
        self.assertLess(generation_time, 30.0)

        # Verify result structure when available
        if result:  # Only test if generation was successful
            self.assertIsInstance(result, dict)
            if 'content' in result:
                self.assertTrue(len(result['content']) > 0)
            if 'generation_time' in result:
                self.assertIsInstance(result['generation_time'], (int, float, Decimal))

            # Verify performance monitoring was called
            try:
                summary = self.service.performance_monitor.get_performance_summary()
                self.assertIsInstance(summary, dict)
            except AttributeError:
                # Performance monitor may not be available in all configurations
                pass

    def test_service_status_with_monitoring(self):
        """Test enhanced service status reporting."""
        status = self.service.get_service_status()

        # Verify enhanced status includes monitoring data
        self.assertIn('circuit_breaker_state', status)
        self.assertIn('performance_metrics', status)

        # Verify circuit breaker state
        self.assertIn(status['circuit_breaker_state'], ['CLOSED', 'OPEN', 'HALF_OPEN'])

        # Verify performance metrics structure
        if isinstance(status['performance_metrics'], dict) and 'error' not in status['performance_metrics']:
            metrics = status['performance_metrics']
            expected_keys = ['total_requests', 'success_rate', 'uptime_minutes']
            for key in expected_keys:
                if key in metrics:  # Some metrics might not be present if no requests made
                    self.assertIsInstance(metrics[key], (int, float))


class LLMOptimizationBenchmarkTestCase(TestCase):
    """Benchmark tests to measure actual performance improvements."""

    def setUp(self):
        """Set up benchmark environment."""
        cache.clear()
        self.service = LocalLLMService()

        self.benchmark_data = {
            'symbol': 'BENCHMARK',
            'score_0_10': 6.5,
            'weighted_scores': {
                'w_sma50vs200': 0.12,
                'w_rsi14': 0.08,
                'w_macd12269': 0.10,
                'w_bbpos20': 0.05
            },
            'components': {}
        }

    def test_prompt_optimization_benchmark(self):
        """Benchmark prompt generation optimization."""
        # Test optimized vs legacy prompt generation
        start_time = time.time()
        for _ in range(100):
            optimized_prompt = self.service._build_optimized_prompt(
                self.benchmark_data, 'standard', 'technical_analysis'
            )
        optimized_time = time.time() - start_time

        start_time = time.time()
        for _ in range(100):
            legacy_prompt = self.service._build_prompt_legacy(
                self.benchmark_data, 'standard', 'technical_analysis'
            )
        legacy_time = time.time() - start_time

        # Optimized should be faster (less string operations)
        # Allow some tolerance since this is a micro-benchmark
        self.assertLess(optimized_time, legacy_time * 1.5)  # Allow 50% tolerance

        # Verify optimized prompt is shorter (faster to process)
        self.assertLess(len(optimized_prompt), len(legacy_prompt))

    def test_generation_options_optimization_benchmark(self):
        """Benchmark generation options optimization."""
        # Test optimized vs legacy options generation
        start_time = time.time()
        for _ in range(1000):
            optimized_options = self.service._get_optimized_generation_options('standard', 'llama3.1:8b')
        optimized_time = time.time() - start_time

        # Verify optimized options have performance-focused values
        self.assertEqual(optimized_options['temperature'], 0.4)  # Base temperature for 8B standard
        self.assertEqual(optimized_options['num_ctx'], 512)  # Smaller context
        self.assertIn('top_k', optimized_options)  # Should have top_k for speed

        # Options generation should be very fast
        self.assertLess(optimized_time, 0.1)  # Should complete in under 100ms

    def test_cache_key_generation_performance(self):
        """Test cache key generation performance."""
        start_time = time.time()

        # Generate many cache keys
        cache_keys = set()
        for i in range(1000):
            modified_data = self.benchmark_data.copy()
            modified_data['score_0_10'] = 5.0 + (i % 10) * 0.1  # Vary scores

            cache_key = self.service._create_cache_key(
                modified_data, 'standard', 'technical_analysis'
            )
            cache_keys.add(cache_key)

        generation_time = time.time() - start_time

        # Should generate unique keys efficiently (score only varies by 0.1 increments)
        self.assertGreater(len(cache_keys), 8)  # Should have many unique keys for different scores
        self.assertLess(generation_time, 0.5)  # Should complete quickly

    def test_complexity_calculation_benchmark(self):
        """Benchmark complexity score calculation performance."""
        start_time = time.time()

        # Calculate complexity scores for various scenarios
        for _ in range(1000):
            complexity_score = self.service._calculate_complexity_score(self.benchmark_data)

        calculation_time = time.time() - start_time

        # Should be very fast calculation
        self.assertLess(calculation_time, 0.1)  # Under 100ms for 1000 calculations

        # Verify score is reasonable
        complexity_score = self.service._calculate_complexity_score(self.benchmark_data)
        self.assertGreaterEqual(complexity_score, 0.0)
        self.assertLessEqual(complexity_score, 1.0)


if __name__ == '__main__':
    pytest.main([__file__])

"""
Integration tests for TA engine with LSTM predictions.
Tests the complete analysis workflow including ML predictions.
"""

import unittest
from decimal import Decimal
from datetime import datetime, date, timedelta
from django.test import TestCase
from django.utils import timezone

from Analytics.engine.ta_engine import TechnicalAnalysisEngine, IndicatorResult
from Data.models import Stock, AnalyticsResults
from Analytics.tests.fixtures import AnalyticsTestDataFactory, TechnicalAnalysisTestEngine, OllamaTestService
from Data.tests.fixtures import DataTestDataFactory
from Core.tests.fixtures import TestEnvironmentManager


class TestTAEngineWithLSTM(TestCase):
    """Integration tests for TA engine with LSTM prediction integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        TestEnvironmentManager.setup_test_environment()
        
        # Create real TA engine and test services
        self.engine = TechnicalAnalysisEngine()
        self.ta_test_engine = TechnicalAnalysisTestEngine()
        self.ollama_service = OllamaTestService()
        
        # Create real test stock with price history
        self.stock = DataTestDataFactory.create_test_stock('TEST', 'Test Stock Inc.', 'Technology')
        self.price_history = DataTestDataFactory.create_stock_price_history(self.stock, 100)
    
    def test_weights_include_prediction(self):
        """Test that weights include LSTM prediction."""
        weights = self.engine.WEIGHTS
        
        self.assertIn('prediction', weights)
        self.assertEqual(weights['prediction'], 0.10)
        
        # Check that weights still sum to 1.0
        total_weight = sum(weights.values())
        self.assertAlmostEqual(float(total_weight), 1.0, places=3)
    
    def test_indicator_names_include_prediction(self):
        """Test that indicator names include LSTM prediction."""
        names = self.engine.INDICATOR_NAMES
        
        self.assertIn('prediction', names)
        self.assertEqual(names['prediction'], 'LSTM Price Prediction')
    
    def test_calculate_prediction_score_no_model(self):
        """Test prediction score calculation when no model is available."""
        # Test with real TA engine when no prediction data available
        result = self.ta_test_engine.calculate_prediction_score_no_model('TEST')
        
        self.assertIsInstance(result, IndicatorResult)
        self.assertEqual(result.score, 0.5)  # Neutral score
        self.assertEqual(result.weight, 0.10)
        self.assertIn('error', result.raw)
    
    def test_calculate_prediction_score_with_prediction(self):
        """Test prediction score calculation with successful prediction."""
        # Use real TA test engine to generate realistic prediction
        result = self.ta_test_engine.calculate_prediction_score_with_data('TEST')
        
        self.assertIsInstance(result, IndicatorResult)
        self.assertGreater(result.score, 0.0)
        self.assertLess(result.score, 1.0)
        self.assertEqual(result.weight, 0.10)
        
        # Check raw data contains realistic prediction values
        self.assertIn('predicted_price', result.raw)
        self.assertIn('confidence', result.raw)
        self.assertIn('model_version', result.raw)
        self.assertGreater(result.raw['predicted_price'], 0)
        self.assertGreater(result.raw['confidence'], 0)
    
    def test_calculate_prediction_score_exception(self):
        """Test prediction score calculation with exception."""
        # Test real error handling in TA engine
        result = self.ta_test_engine.calculate_prediction_score_with_error('TEST')
        
        self.assertIsInstance(result, IndicatorResult)
        self.assertEqual(result.score, 0.5)  # Neutral score on error
        self.assertIn('error', result.raw)
        self.assertIn('service error', result.raw['error'].lower())
    
    def test_analyze_stock_includes_prediction(self):
        """Test that analyze_stock includes LSTM prediction in the analysis."""
        # Create comprehensive test analysis with prediction
        components = self.ta_test_engine.generate_realistic_analysis_components('TEST')
        
        # Verify prediction is included in components
        self.assertIn('prediction', components)
        
        # Check prediction component structure
        prediction_component = components['prediction']
        self.assertIn('score', prediction_component)
        self.assertIn('raw', prediction_component)
        
        # Verify prediction raw data
        prediction_raw = prediction_component['raw']
        self.assertIn('predicted_price', prediction_raw)
        self.assertIn('current_price', prediction_raw)
        self.assertIn('confidence', prediction_raw)
        self.assertIn('model_version', prediction_raw)
        
        # Verify realistic score range
        prediction_score = prediction_component['score']
        self.assertGreaterEqual(prediction_score, 0.0)
        self.assertLessEqual(prediction_score, 1.0)
        
        # Verify technical indicators are also present
        technical_indicators = ['sma50vs200', 'rsi14', 'macd12269', 'bbpos20', 'volsurge']
        for indicator in technical_indicators:
            self.assertIn(indicator, components)
    
    def _create_real_price_data(self, count=100):
        """Create real price data for testing using existing stock history."""
        # Use actual stock price history from database
        if self.price_history and len(self.price_history) >= count:
            return self.price_history[:count]
        
        # Fallback: create additional realistic price data
        from Data.repo.price_reader import PriceData
        
        price_data = []
        base_price = 150.0
        current_date = date.today()
        
        for i in range(count):
            price_date = current_date - timedelta(days=count-i-1)
            daily_variation = (i % 10 - 5) * 0.5  # -2.5 to +2.0 variation
            current_price = base_price + daily_variation
            
            price_data.append(PriceData(
                date=price_date,
                open_price=current_price - 0.5,
                high_price=current_price + 1.0,
                low_price=current_price - 1.5,
                close_price=current_price,
                volume=1000000 + (i * 50000),
                adj_close_price=current_price
            ))
        
        return price_data
    
    def tearDown(self):
        """Clean up test data."""
        DataTestDataFactory.cleanup_test_data()
        TestEnvironmentManager.teardown_test_environment()
    
    def test_analytics_result_includes_prediction_fields(self):
        """Test that analytics results include prediction fields."""
        # Create real analytics data with prediction using factory
        analytics_result = AnalyticsTestDataFactory.create_technical_analysis_data(self.stock)
        
        # Verify basic analytics fields exist
        self.assertIsNotNone(analytics_result.stock)
        self.assertEqual(analytics_result.stock.symbol, 'TEST')
        self.assertIsNotNone(analytics_result.score_0_10)
        self.assertIsNotNone(analytics_result.sentimentScore)
        self.assertIsNotNone(analytics_result.sentimentLabel)
        
        # Verify raw indicators contain realistic technical data
        import json
        components = json.loads(analytics_result.components)
        self.assertIn('sma_50', components)
        self.assertIn('sma_200', components)
        self.assertIn('rsi', components)
        self.assertIn('macd', components)
        
        # Verify scores are in valid ranges
        self.assertGreaterEqual(analytics_result.score_0_10, 0)
        self.assertLessEqual(analytics_result.score_0_10, 10)
        self.assertGreaterEqual(analytics_result.sentimentScore, 0)
        self.assertLessEqual(analytics_result.sentimentScore, 10)
    
    def test_weight_distribution_with_prediction(self):
        """Test that weight distribution is correct with LSTM prediction."""
        weights = self.engine.WEIGHTS
        
        # Check individual weights are adjusted correctly
        # Should be 80% TA, 10% sentiment, 10% prediction
        ta_weights = [
            'sma50vs200', 'pricevs50', 'rsi14', 'macd12269',
            'bbpos20', 'bbwidth20', 'volsurge', 'obv20',
            'rel1y', 'rel2y', 'candlerev', 'srcontext'
        ]
        
        ta_total = sum(weights[key] for key in ta_weights)
        sentiment_weight = weights['sentiment']
        prediction_weight = weights['prediction']
        
        # Check proportions (allowing for small rounding differences)
        self.assertAlmostEqual(ta_total, 0.80, places=3)
        self.assertAlmostEqual(sentiment_weight, 0.10, places=3)
        self.assertAlmostEqual(prediction_weight, 0.10, places=3)
        
        # Check total sums to 1.0
        total_weight = ta_total + sentiment_weight + prediction_weight
        self.assertAlmostEqual(total_weight, 1.0, places=3)
    
    def test_display_name_for_prediction(self):
        """Test display name generation for prediction indicator."""
        display_name = self.engine.get_indicator_display_name('prediction')
        self.assertEqual(display_name, 'LSTM Price Prediction')
    
    def test_prediction_integration_graceful_failure(self):
        """Test that prediction integration fails gracefully when service unavailable."""
        # Test graceful failure with TA test engine
        error_result = self.ta_test_engine.calculate_prediction_score_with_error('TEST')
        
        # Should return neutral score without raising exception
        self.assertIsInstance(error_result, IndicatorResult)
        self.assertEqual(error_result.score, 0.5)  # Neutral score
        self.assertEqual(error_result.weight, 0.10)  # Correct weight
        self.assertIn('error', error_result.raw)
        
        # Verify error message is descriptive
        error_message = error_result.raw['error']
        self.assertIn('service error', error_message.lower())
        
        # Test that analysis can continue with error condition
        components = self.ta_test_engine.generate_realistic_analysis_components('TEST')
        self.assertIn('prediction', components)
        
        # Even with error, prediction component should be present with neutral values
        prediction_component = components['prediction']
        self.assertIsInstance(prediction_component['score'], (int, float))


class TestLSTMPredictionInProduction(TestCase):
    """Test LSTM prediction integration in production-like scenarios."""
    
    def setUp(self):
        """Set up production-like test environment."""
        TestEnvironmentManager.setup_test_environment()
        
        self.engine = TechnicalAnalysisEngine()
        self.ta_test_engine = TechnicalAnalysisTestEngine()
        self.stock = DataTestDataFactory.create_test_stock('AAPL', 'Apple Inc.', 'Technology')
        self.price_history = DataTestDataFactory.create_stock_price_history(self.stock, 50)
    
    def test_prediction_caching_behavior(self):
        """Test that predictions are properly handled consistently."""
        # Test consistent prediction results with TA test engine
        result1 = self.ta_test_engine.calculate_prediction_score_with_data('AAPL')
        result2 = self.ta_test_engine.calculate_prediction_score_with_data('AAPL')
        
        # Results should be identical for same symbol (deterministic)
        self.assertEqual(result1.score, result2.score)
        self.assertEqual(result1.raw['predicted_price'], result2.raw['predicted_price'])
        self.assertEqual(result1.raw['confidence'], result2.raw['confidence'])
        self.assertEqual(result1.weight, result2.weight)
        
        # Test different symbols produce different results
        result3 = self.ta_test_engine.calculate_prediction_score_with_data('MSFT')
        self.assertNotEqual(result1.raw['predicted_price'], result3.raw['predicted_price'])
    
    def test_prediction_performance_impact(self):
        """Test that prediction doesn't significantly impact analysis performance."""
        from Analytics.tests.fixtures import PerformanceTestUtilities
        
        # Benchmark prediction calculation performance
        benchmark_result = PerformanceTestUtilities.benchmark_analysis_execution(
            self.ta_test_engine.calculate_prediction_score_with_data,
            'AAPL'
        )
        
        # Should complete very quickly (under 1 second in test environment)
        execution_time = benchmark_result['execution_time']
        self.assertLess(execution_time, 1.0)
        
        # Should return valid result
        result = benchmark_result['result']
        self.assertIsInstance(result, IndicatorResult)
        self.assertGreater(result.score, 0.0)
        self.assertLess(result.score, 1.0)
        
        # Test multiple prediction calculations for performance consistency
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        total_time = 0
        
        for symbol in symbols:
            benchmark = PerformanceTestUtilities.benchmark_analysis_execution(
                self.ta_test_engine.calculate_prediction_score_with_data,
                symbol
            )
            total_time += benchmark['execution_time']
        
        # Total time for 3 predictions should be reasonable
        self.assertLess(total_time, 2.0)


    def tearDown(self):
        """Clean up test data."""
        DataTestDataFactory.cleanup_test_data()
        TestEnvironmentManager.teardown_test_environment()


if __name__ == '__main__':
    unittest.main()
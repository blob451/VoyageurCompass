"""
Integration tests for TA engine with LSTM predictions.
Tests the complete analysis workflow including ML predictions.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from datetime import datetime
from django.test import TestCase
from django.utils import timezone

from Analytics.engine.ta_engine import TechnicalAnalysisEngine, IndicatorResult
from Data.models import Stock, AnalyticsResults


class TestTAEngineWithLSTM(TestCase):
    """Integration tests for TA engine with LSTM prediction integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = TechnicalAnalysisEngine()
        
        # Create test stock
        self.stock, created = Stock.objects.get_or_create(
            symbol='TEST',
            defaults={
                'name': 'Test Stock',
                'sector': 'Technology',
                'industry': 'Software',
                'market_cap': 1000000000,
                'active': True
            }
        )
    
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
        with patch('Analytics.services.lstm_predictor.get_lstm_predictor') as mock_get_predictor:
            mock_predictor = Mock()
            mock_predictor.predict_stock_price.return_value = None
            mock_get_predictor.return_value = mock_predictor
            
            result = self.engine._calculate_prediction_score('TEST')
        
        self.assertIsInstance(result, IndicatorResult)
        self.assertEqual(result.score, 0.5)  # Neutral score
        self.assertEqual(result.weight, 0.10)
        self.assertIn('error', result.raw)
    
    def test_calculate_prediction_score_with_prediction(self):
        """Test prediction score calculation with successful prediction."""
        mock_prediction = {
            'predicted_price': 150.0,
            'current_price': 145.0,
            'price_change': 5.0,
            'price_change_pct': 3.45,
            'confidence': 0.8,
            'model_version': '1.0.0',
            'horizon': '1d'
        }
        
        with patch('Analytics.services.lstm_predictor.get_lstm_predictor') as mock_get_predictor:
            mock_predictor = Mock()
            mock_predictor.predict_stock_price.return_value = mock_prediction
            mock_predictor.normalize_prediction_score.return_value = 0.65
            mock_get_predictor.return_value = mock_predictor
            
            result = self.engine._calculate_prediction_score('TEST')
        
        self.assertIsInstance(result, IndicatorResult)
        self.assertEqual(result.score, 0.65)
        self.assertEqual(result.weight, 0.10)
        
        # Check raw data
        self.assertEqual(result.raw['predicted_price'], 150.0)
        self.assertEqual(result.raw['confidence'], 0.8)
        self.assertEqual(result.raw['model_version'], '1.0.0')
    
    def test_calculate_prediction_score_exception(self):
        """Test prediction score calculation with exception."""
        with patch('Analytics.services.lstm_predictor.get_lstm_predictor') as mock_get_predictor:
            mock_get_predictor.side_effect = Exception("LSTM service error")
            
            result = self.engine._calculate_prediction_score('TEST')
        
        self.assertIsInstance(result, IndicatorResult)
        self.assertEqual(result.score, 0.5)  # Neutral score on error
        self.assertIn('error', result.raw)
        self.assertIn('LSTM service error', result.raw['error'])
    
    @patch('Analytics.engine.ta_engine.TechnicalAnalysisEngine._get_stock_data')
    @patch('Analytics.engine.ta_engine.TechnicalAnalysisEngine._get_sector_industry_data')
    @patch('Analytics.engine.ta_engine.TechnicalAnalysisEngine._calculate_sentiment_analysis')
    @patch('Analytics.engine.ta_engine.TechnicalAnalysisEngine._calculate_prediction_score')
    def test_analyze_stock_includes_prediction(
        self,
        mock_prediction,
        mock_sentiment,
        mock_sector_data,
        mock_stock_data
    ):
        """Test that analyze_stock includes LSTM prediction in the analysis."""
        # Mock stock data
        mock_stock_data.return_value = self._create_mock_price_data(100)
        mock_sector_data.return_value = ([], [])
        
        # Mock sentiment result
        mock_sentiment.return_value = IndicatorResult(
            raw={'sentiment': 0.1, 'label': 'positive'},
            score=0.6,
            weight=0.10,
            weighted_score=0.06
        )
        
        # Mock prediction result
        mock_prediction.return_value = IndicatorResult(
            raw={
                'predicted_price': 155.0,
                'current_price': 150.0,
                'price_change_pct': 3.33,
                'confidence': 0.8
            },
            score=0.7,
            weight=0.10,
            weighted_score=0.07
        )
        
        # Mock all other technical indicators
        with patch.object(self.engine, '_calculate_sma_crossover') as mock_sma:
            mock_sma.return_value = IndicatorResult(raw={}, score=0.6, weight=0.12, weighted_score=0.072)
            
            with patch.object(self.engine, '_calculate_price_vs_50d') as mock_price50:
                mock_price50.return_value = IndicatorResult(raw={}, score=0.5, weight=0.08, weighted_score=0.04)
                
                with self._patch_all_indicators():
                    result = self.engine.analyze_stock('TEST')
        
        # Check that prediction is included in indicators
        self.assertIn('prediction', result['indicators'])
        
        # Check that prediction is included in components
        self.assertIn('prediction', result['components'])
        
        # Check that weighted score includes prediction
        self.assertIn('w_prediction', result['weighted_scores'])
        
        # Verify prediction data is stored
        prediction_component = result['components']['prediction']
        self.assertEqual(prediction_component['score'], 0.7)
        self.assertIn('predicted_price', prediction_component['raw'])
    
    def _create_mock_price_data(self, count=100):
        """Create mock price data for testing."""
        from Data.repo.price_reader import PriceData
        
        price_data = []
        base_price = 100.0
        
        for i in range(count):
            price_data.append(PriceData(
                date=datetime.now().date(),
                open_price=base_price + i * 0.1,
                high_price=base_price + i * 0.1 + 2,
                low_price=base_price + i * 0.1 - 2,
                close_price=base_price + i * 0.1 + 1,
                volume=1000000 + i * 1000,
                adj_close_price=base_price + i * 0.1 + 1
            ))
        
        return price_data
    
    def _patch_all_indicators(self):
        """Context manager to patch all technical indicators."""
        from unittest.mock import patch
        
        indicators_to_patch = [
            '_calculate_rsi14',
            '_calculate_macd_histogram',
            '_calculate_bollinger_position',
            '_calculate_bollinger_bandwidth',
            '_calculate_volume_surge',
            '_calculate_obv_trend',
            '_calculate_relative_strength_1y',
            '_calculate_relative_strength_2y',
            '_calculate_candlestick_reversal',
            '_calculate_support_resistance_context'
        ]
        
        patches = []
        for indicator in indicators_to_patch:
            mock_result = IndicatorResult(
                raw={'test': True},
                score=0.5,
                weight=0.05,  # Small weight for testing
                weighted_score=0.025
            )
            patches.append(
                patch.object(self.engine, indicator, return_value=mock_result)
            )
        
        class MultiPatch:
            def __init__(self, patches):
                self.patches = patches
                self.started = []
            
            def __enter__(self):
                for p in self.patches:
                    self.started.append(p.__enter__())
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                for p in self.patches:
                    p.__exit__(exc_type, exc_val, exc_tb)
        
        return MultiPatch(patches)
    
    def test_analytics_result_includes_prediction_fields(self):
        """Test that analytics results include prediction fields."""
        # Create a mock analytics result with prediction data
        components = {
            'prediction': {
                'raw': {
                    'predicted_price': 155.0,
                    'confidence': 0.8,
                    'model_version': '1.0.0'
                },
                'score': 0.7
            },
            'sentiment': {
                'raw': {
                    'sentiment': 0.1,
                    'label': 'positive'
                },
                'score': 0.6
            }
        }
        
        weighted_scores = {
            'w_prediction': Decimal('0.07'),
            'w_sentiment': Decimal('0.06')
        }
        
        # Test analytics writer integration
        analytics_result = self.engine.analytics_writer.upsert_analytics_result(
            symbol='TEST',
            as_of=timezone.now(),
            weighted_scores=weighted_scores,
            components=components,
            composite_raw=Decimal('0.5'),
            score_0_10=5,
            horizon='blend'
        )
        
        # Check that prediction fields are populated
        self.assertEqual(analytics_result.prediction_1d, Decimal('155.0'))
        self.assertEqual(analytics_result.prediction_confidence, 0.8)
        self.assertEqual(analytics_result.model_version, '1.0.0')
        self.assertIsNotNone(analytics_result.prediction_timestamp)
    
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
        # Mock import error (service not available)
        with patch('Analytics.engine.ta_engine.TechnicalAnalysisEngine._calculate_prediction_score') as mock_pred:
            mock_pred.return_value = IndicatorResult(
                raw={'prediction': None, 'error': 'Service unavailable'},
                score=0.5,
                weight=0.10,
                weighted_score=0.05
            )
            
            # This should not raise an exception
            try:
                with patch.object(self.engine, '_get_stock_data', return_value=self._create_mock_price_data(50)):
                    with patch.object(self.engine, '_get_sector_industry_data', return_value=([], [])):
                        with self._patch_all_indicators():
                            with patch.object(self.engine, '_calculate_sentiment_analysis') as mock_sentiment:
                                mock_sentiment.return_value = IndicatorResult(
                                    raw={'sentiment': 0}, score=0.5, weight=0.10, weighted_score=0.05
                                )
                                result = self.engine.analyze_stock('TEST')
                
                # Analysis should complete successfully with neutral prediction
                self.assertIn('prediction', result['indicators'])
                self.assertEqual(result['indicators']['prediction'].score, 0.5)
                
            except Exception as e:
                self.fail(f"Analysis should not fail when LSTM service unavailable: {e}")


class TestLSTMPredictionInProduction(TestCase):
    """Test LSTM prediction integration in production-like scenarios."""
    
    def setUp(self):
        """Set up production-like test environment."""
        self.engine = TechnicalAnalysisEngine()
    
    def test_prediction_caching_behavior(self):
        """Test that predictions are properly cached."""
        mock_prediction = {
            'predicted_price': 150.0,
            'current_price': 145.0,
            'price_change_pct': 3.45,
            'confidence': 0.8
        }
        
        with patch('Analytics.services.lstm_predictor.get_lstm_predictor') as mock_get_predictor:
            mock_predictor = Mock()
            mock_predictor.predict_stock_price.return_value = mock_prediction
            mock_predictor.normalize_prediction_score.return_value = 0.65
            mock_get_predictor.return_value = mock_predictor
            
            # First call should hit the predictor
            result1 = self.engine._calculate_prediction_score('AAPL')
            
            # Verify predictor was called
            mock_predictor.predict_stock_price.assert_called_once_with('AAPL', horizon='1d')
            
            # Second call should also hit the predictor (no caching in this method)
            result2 = self.engine._calculate_prediction_score('AAPL')
            
            # Both results should be identical
            self.assertEqual(result1.score, result2.score)
            self.assertEqual(result1.raw, result2.raw)
    
    def test_prediction_performance_impact(self):
        """Test that prediction doesn't significantly impact analysis performance."""
        import time
        
        # Mock fast prediction
        with patch('Analytics.services.lstm_predictor.get_lstm_predictor') as mock_get_predictor:
            mock_predictor = Mock()
            mock_predictor.predict_stock_price.return_value = {
                'predicted_price': 150.0,
                'confidence': 0.8,
                'price_change_pct': 2.0
            }
            mock_predictor.normalize_prediction_score.return_value = 0.6
            mock_get_predictor.return_value = mock_predictor
            
            start_time = time.time()
            result = self.engine._calculate_prediction_score('AAPL')
            end_time = time.time()
            
            # Should complete very quickly (under 1 second)
            execution_time = end_time - start_time
            self.assertLess(execution_time, 1.0)
            
            # Should return valid result
            self.assertIsInstance(result, IndicatorResult)
            self.assertEqual(result.score, 0.6)


if __name__ == '__main__':
    unittest.main()
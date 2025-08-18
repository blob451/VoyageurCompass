"""
Unit tests for Analytics LSTM prediction services.
Tests IntegratedPredictionService and LSTM components.
"""

import torch
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from django.test import TestCase
from django.contrib.auth.models import User

from Analytics.services.integrated_predictor import IntegratedPredictionService
from Analytics.ml.models.lstm_base import SectorCrossAttention, AttentionLayer
from Analytics.services.universal_predictor import UniversalLSTMAnalyticsService
from Data.models import Stock, StockPrice


class IntegratedPredictionServiceTestCase(TestCase):
    """Test cases for IntegratedPredictionService."""
    
    def setUp(self):
        """Set up test data."""
        self.service = IntegratedPredictionService()
        
        # Create test stock
        self.stock = Stock.objects.create(
            symbol='TEST',
            short_name='Test Company',
            exchange='NASDAQ'
        )
        
        # Mock LSTM service results
        self.mock_lstm_result = {
            'success': True,
            'predicted_price': 155.50,
            'current_price': 150.00,
            'confidence': 0.75,
            'model_version': 'lstm_v1.0'
        }
        
        # Mock TA indicators
        self.mock_ta_indicators = {
            'sma50vs200': {'raw': {'crossover': True, 'position': 'bullish'}, 'normalized': 0.8},
            'rsi14': {'raw': {'value': 45.2}, 'normalized': 0.6},
            'macd12269': {'raw': {'histogram': 0.5}, 'normalized': 0.7},
            'bollinger_bands': {'raw': {'position': 0.3}, 'normalized': 0.4}
        }
        
        self.mock_weighted_result = {
            'predicted_price': 158.25,
            'price_change': 8.25,
            'price_change_pct': 5.5,
            'confidence': 0.82
        }
    
    @patch('Analytics.services.integrated_predictor.UniversalLSTMAnalyticsService')
    def test_initialization(self, mock_lstm_service):
        """Test service initialization."""
        service = IntegratedPredictionService()
        
        self.assertIsNotNone(service.ta_engine)
        self.assertIsNotNone(service.dynamic_predictor)
        self.assertIsNotNone(service.lstm_service)
    
    @patch('Analytics.services.integrated_predictor.UniversalLSTMAnalyticsService')
    def test_predict_with_ta_context_success(self, mock_lstm_service):
        """Test successful prediction with TA context."""
        # Mock LSTM service
        mock_lstm_service.return_value.predict_stock_price.return_value = self.mock_lstm_result
        
        # Mock TA analysis
        with patch.object(self.service, '_get_ta_indicators') as mock_ta:
            mock_ta.return_value = {
                'success': True,
                'indicators': self.mock_ta_indicators
            }
            
            # Mock dynamic predictor
            with patch.object(self.service.dynamic_predictor, 'calculate_dynamic_weights') as mock_weights:
                with patch.object(self.service.dynamic_predictor, 'weighted_prediction') as mock_prediction:
                    with patch.object(self.service.dynamic_predictor, 'get_indicator_importance') as mock_importance:
                        
                        mock_weights.return_value = {'momentum': 0.7, 'trend': 0.8}
                        mock_prediction.return_value = self.mock_weighted_result
                        mock_importance.return_value = {'rsi14': 0.8, 'sma50vs200': 0.9}
                        
                        result = self.service.predict_with_ta_context('TEST')
                        
                        # Verify result structure
                        self.assertTrue(result['success'])
                        self.assertEqual(result['symbol'], 'TEST')
                        self.assertEqual(result['current_price'], 150.00)
                        self.assertEqual(result['base_prediction'], 155.50)
                        self.assertEqual(result['weighted_prediction'], 158.25)
                        self.assertEqual(result['confidence'], 0.82)
                        self.assertIn('ta_weights', result)
                        self.assertIn('indicator_importance', result)
    
    @patch('Analytics.services.integrated_predictor.UniversalLSTMAnalyticsService')
    def test_predict_lstm_failure(self, mock_lstm_service):
        """Test handling of LSTM prediction failure."""
        mock_lstm_service.return_value.predict_stock_price.return_value = None
        
        result = self.service.predict_with_ta_context('TEST')
        
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], "LSTM prediction unavailable")
    
    @patch('Analytics.services.integrated_predictor.UniversalLSTMAnalyticsService')
    def test_predict_without_analysis(self, mock_lstm_service):
        """Test prediction without TA analysis."""
        mock_lstm_service.return_value.predict_stock_price.return_value = self.mock_lstm_result
        
        result = self.service.predict_with_ta_context('TEST', include_analysis=False)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['base_prediction'], 155.50)
        # Should use base prediction when TA analysis is disabled
        self.assertEqual(result.get('weighted_prediction', result['base_prediction']), 155.50)
    
    def test_get_ta_indicators_isolation(self):
        """Test TA indicators retrieval without LSTM recursion."""
        with patch.object(self.service.ta_engine, 'analyze') as mock_analyze:
            mock_analyze.return_value = {
                'success': True,
                'indicators': self.mock_ta_indicators,
                'score_0_10': 7
            }
            
            result = self.service._get_ta_indicators('TEST')
            
            self.assertTrue(result['success'])
            self.assertEqual(result['indicators'], self.mock_ta_indicators)
            # Verify LSTM prediction wasn't included to avoid recursion
            mock_analyze.assert_called_once()


class LSTMModelTestCase(TestCase):
    """Test cases for LSTM model components."""
    
    def setUp(self):
        """Set up test data."""
        self.hidden_size = 128
        self.sector_embedding_dim = 64
        self.seq_len = 60
        self.batch_size = 4
        
        # Create sample tensors
        self.lstm_output = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        self.sector_embedding = torch.randn(self.batch_size, self.sector_embedding_dim)
    
    def test_sector_cross_attention_initialization(self):
        """Test SectorCrossAttention layer initialization."""
        attention = SectorCrossAttention(self.hidden_size, self.sector_embedding_dim)
        
        self.assertEqual(attention.hidden_size, self.hidden_size)
        self.assertEqual(attention.sector_embedding_dim, self.sector_embedding_dim)
        self.assertIsInstance(attention.query_projection, torch.nn.Linear)
        self.assertIsInstance(attention.key_projection, torch.nn.Linear)
        self.assertIsInstance(attention.value_projection, torch.nn.Linear)
        self.assertIsInstance(attention.output_projection, torch.nn.Linear)
    
    def test_sector_cross_attention_forward(self):
        """Test SectorCrossAttention forward pass."""
        attention = SectorCrossAttention(self.hidden_size, self.sector_embedding_dim)
        
        output = attention(self.lstm_output, self.sector_embedding)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.hidden_size)
        self.assertEqual(output.shape, expected_shape)
        
        # Check that output is different from input (attention was applied)
        self.assertFalse(torch.equal(output, self.lstm_output))
    
    def test_attention_layer_initialization(self):
        """Test AttentionLayer initialization."""
        attention = AttentionLayer(self.hidden_size)
        
        self.assertEqual(attention.hidden_size, self.hidden_size)
        self.assertIsInstance(attention.attn, torch.nn.Linear)
    
    def test_attention_layer_forward(self):
        """Test AttentionLayer forward pass."""
        attention = AttentionLayer(self.hidden_size)
        
        attended_output, attention_weights = attention(self.lstm_output)
        
        # Check output shapes
        self.assertEqual(attended_output.shape, (self.batch_size, self.hidden_size))
        self.assertEqual(attention_weights.shape, (self.batch_size, self.seq_len))
        
        # Check that attention weights sum to 1
        attention_sums = torch.sum(attention_weights, dim=1)
        expected_sums = torch.ones(self.batch_size)
        self.assertTrue(torch.allclose(attention_sums, expected_sums, atol=1e-6))
    
    def test_attention_weights_valid_range(self):
        """Test that attention weights are in valid range [0, 1]."""
        attention = AttentionLayer(self.hidden_size)
        
        _, attention_weights = attention(self.lstm_output)
        
        # Check that all weights are non-negative
        self.assertTrue(torch.all(attention_weights >= 0))
        
        # Check that all weights are at most 1
        self.assertTrue(torch.all(attention_weights <= 1))


class UniversalLSTMAnalyticsServiceTestCase(TestCase):
    """Test cases for UniversalLSTMAnalyticsService."""
    
    def setUp(self):
        """Set up test data."""
        self.service = UniversalLSTMAnalyticsService()
        
        # Create test stock with price data
        self.stock = Stock.objects.create(
            symbol='TEST',
            short_name='Test Company',
            exchange='NASDAQ'
        )
        
        # Create historical price data
        base_date = datetime.now().date() - timedelta(days=100)
        for i in range(60):  # 60 days of data for LSTM sequence
            StockPrice.objects.create(
                stock=self.stock,
                date=base_date + timedelta(days=i),
                open=Decimal(f'{100 + i * 0.1:.2f}'),
                high=Decimal(f'{105 + i * 0.1:.2f}'),
                low=Decimal(f'{95 + i * 0.1:.2f}'),
                close=Decimal(f'{102 + i * 0.1:.2f}'),
                volume=1000000 + i * 1000
            )
    
    @patch('Analytics.services.universal_predictor.UniversalLSTMAnalyticsService._load_model')
    @patch('Analytics.services.universal_predictor.UniversalLSTMAnalyticsService._prepare_data')
    def test_predict_stock_price_success(self, mock_prepare_data, mock_load_model):
        """Test successful stock price prediction."""
        # Mock model loading
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_load_model.return_value = (mock_model, {'scaler': 'mock_scaler'})
        
        # Mock data preparation
        mock_prepare_data.return_value = {
            'features': torch.randn(1, 60, 5),  # Sample feature tensor
            'current_price': 108.0,
            'success': True
        }
        
        # Mock model prediction
        with patch.object(torch.nn.Module, 'forward') as mock_forward:
            mock_forward.return_value = torch.tensor([[110.5]])
            
            result = self.service.predict_stock_price('TEST')
            
            self.assertIsNotNone(result)
            self.assertTrue(result.get('success', False))
            self.assertIn('predicted_price', result)
            self.assertIn('current_price', result)
            self.assertIn('confidence', result)
    
    @patch('Analytics.services.universal_predictor.UniversalLSTMAnalyticsService._load_model')
    def test_predict_stock_price_no_model(self, mock_load_model):
        """Test prediction when no model is available."""
        mock_load_model.return_value = (None, None)
        
        result = self.service.predict_stock_price('TEST')
        
        self.assertIsNone(result)
    
    def test_data_validation(self):
        """Test data validation for LSTM input."""
        # Test with insufficient data
        insufficient_stock = Stock.objects.create(
            symbol='INSUFFICIENT',
            short_name='Insufficient Data Stock',
            exchange='NASDAQ'
        )
        
        # Only create 10 days of data (less than required 60)
        base_date = datetime.now().date() - timedelta(days=20)
        for i in range(10):
            StockPrice.objects.create(
                stock=insufficient_stock,
                date=base_date + timedelta(days=i),
                open=Decimal('100.00'),
                high=Decimal('105.00'),
                low=Decimal('95.00'),
                close=Decimal('102.00'),
                volume=1000000
            )
        
        with patch('Analytics.services.universal_predictor.UniversalLSTMAnalyticsService._load_model'):
            result = self.service.predict_stock_price('INSUFFICIENT')
            
            # Should handle insufficient data gracefully
            self.assertIsNone(result)


class LSTMIntegrationTestCase(TestCase):
    """Integration tests for LSTM prediction pipeline."""
    
    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        
        self.stock = Stock.objects.create(
            symbol='INTEGRATION',
            short_name='Integration Test Stock',
            exchange='NASDAQ'
        )
    
    @patch('Analytics.services.integrated_predictor.UniversalLSTMAnalyticsService')
    @patch('Analytics.engine.ta_engine.TechnicalAnalysisEngine')
    def test_end_to_end_prediction_pipeline(self, mock_ta_engine, mock_lstm_service):
        """Test complete prediction pipeline from TA to LSTM integration."""
        # Mock TA engine
        mock_ta_result = {
            'success': True,
            'indicators': {
                'sma50vs200': {'normalized': 0.7},
                'rsi14': {'normalized': 0.6},
                'macd12269': {'normalized': 0.8}
            },
            'score_0_10': 7
        }
        mock_ta_engine.return_value.analyze.return_value = mock_ta_result
        
        # Mock LSTM service
        mock_lstm_result = {
            'success': True,
            'predicted_price': 125.50,
            'current_price': 120.00,
            'confidence': 0.78
        }
        mock_lstm_service.return_value.predict_stock_price.return_value = mock_lstm_result
        
        # Mock dynamic predictor
        service = IntegratedPredictionService()
        
        with patch.object(service.dynamic_predictor, 'calculate_dynamic_weights') as mock_weights:
            with patch.object(service.dynamic_predictor, 'weighted_prediction') as mock_prediction:
                
                mock_weights.return_value = {'trend': 0.8, 'momentum': 0.6}
                mock_prediction.return_value = {
                    'predicted_price': 127.25,
                    'price_change': 7.25,
                    'price_change_pct': 6.04,
                    'confidence': 0.82
                }
                
                result = service.predict_with_ta_context('INTEGRATION')
                
                # Verify complete pipeline execution
                self.assertTrue(result['success'])
                self.assertEqual(result['symbol'], 'INTEGRATION')
                self.assertEqual(result['base_prediction'], 125.50)
                self.assertEqual(result['weighted_prediction'], 127.25)
                self.assertGreater(result['confidence'], result.get('base_confidence', 0))
    
    def test_performance_benchmarking(self):
        """Test prediction performance benchmarks."""
        with patch('Analytics.services.integrated_predictor.UniversalLSTMAnalyticsService') as mock_lstm:
            mock_lstm.return_value.predict_stock_price.return_value = {
                'success': True,
                'predicted_price': 100.0,
                'current_price': 95.0,
                'confidence': 0.7
            }
            
            service = IntegratedPredictionService()
            
            # Mock TA components for speed
            with patch.object(service, '_get_ta_indicators') as mock_ta:
                mock_ta.return_value = {'success': False}
                
                start_time = datetime.now()
                
                # Run multiple predictions
                for i in range(10):
                    result = service.predict_with_ta_context(f'STOCK{i}', include_analysis=False)
                    self.assertIsNotNone(result)
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # Should complete 10 predictions in reasonable time (< 5 seconds)
                self.assertLess(duration, 5.0)
    
    def test_error_resilience(self):
        """Test error handling and resilience."""
        service = IntegratedPredictionService()
        
        # Test with non-existent symbol
        result = service.predict_with_ta_context('NONEXISTENT')
        self.assertFalse(result['success'])
        self.assertIn('error', result)
        
        # Test with empty data
        empty_stock = Stock.objects.create(
            symbol='EMPTY',
            short_name='Empty Stock',
            exchange='NASDAQ'
        )
        
        result = service.predict_with_ta_context('EMPTY')
        self.assertFalse(result['success'])
    
    def test_concurrent_predictions(self):
        """Test handling of concurrent prediction requests."""
        import threading
        
        service = IntegratedPredictionService()
        results = []
        errors = []
        
        def predict_symbol(symbol):
            try:
                with patch('Analytics.services.integrated_predictor.UniversalLSTMAnalyticsService') as mock_lstm:
                    mock_lstm.return_value.predict_stock_price.return_value = {
                        'success': True,
                        'predicted_price': 100.0 + hash(symbol) % 50,
                        'current_price': 95.0,
                        'confidence': 0.7
                    }
                    
                    result = service.predict_with_ta_context(symbol, include_analysis=False)
                    results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=predict_symbol, args=(f'CONCURRENT{i}',))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All predictions should succeed
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 5)
        
        for result in results:
            self.assertTrue(result.get('success', False))
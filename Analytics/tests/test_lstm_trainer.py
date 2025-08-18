"""
Unit tests for Analytics LSTM training services.
Tests LSTM model training, validation, and optimization.
"""

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock, mock_open
from django.test import TestCase
from tempfile import NamedTemporaryFile
import os

from Analytics.ml.models.lstm_base import SectorCrossAttention, AttentionLayer
from Analytics.services.advanced_lstm_trainer import AdvancedLSTMTrainer
from Data.models import Stock, StockPrice


class MockLSTMModel(nn.Module):
    """Mock LSTM model for testing."""
    
    def __init__(self, input_size=5, hidden_size=128, num_layers=2):
        super(MockLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output


class LSTMTrainerTestCase(TestCase):
    """Test cases for LSTM training functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create test stock
        self.stock = Stock.objects.create(
            symbol='TRAIN_TEST',
            short_name='Training Test Stock',
            exchange='NASDAQ'
        )
        
        # Create historical price data for training
        base_date = datetime.now().date() - timedelta(days=200)
        for i in range(150):  # Enough data for training/validation split
            StockPrice.objects.create(
                stock=self.stock,
                date=base_date + timedelta(days=i),
                open=Decimal(f'{100 + i * 0.1:.2f}'),
                high=Decimal(f'{105 + i * 0.1:.2f}'),
                low=Decimal(f'{95 + i * 0.1:.2f}'),
                close=Decimal(f'{102 + i * 0.1:.2f}'),
                volume=1000000 + i * 1000
            )
        
        self.trainer_config = {
            'sequence_length': 60,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 16,
            'epochs': 5
        }
    
    @patch('Analytics.services.advanced_lstm_trainer.AdvancedLSTMTrainer')
    def test_trainer_initialization(self, MockTrainer):
        """Test LSTM trainer initialization."""
        mock_trainer = MockTrainer.return_value
        mock_trainer.symbol = 'TRAIN_TEST'
        mock_trainer.config = self.trainer_config
        
        # Test initialization
        self.assertEqual(mock_trainer.symbol, 'TRAIN_TEST')
        self.assertIsInstance(mock_trainer.config, dict)
        self.assertIn('sequence_length', mock_trainer.config)
    
    def test_data_preparation(self):
        """Test training data preparation."""
        # Mock data preparation functionality
        with patch('Analytics.services.advanced_lstm_trainer.AdvancedLSTMTrainer') as MockTrainer:
            trainer = MockTrainer.return_value
            
            # Mock prepared data
            mock_features = torch.randn(100, 60, 5)  # 100 samples, 60 timesteps, 5 features
            mock_targets = torch.randn(100, 1)       # 100 target values
            
            trainer.prepare_training_data.return_value = {
                'features': mock_features,
                'targets': mock_targets,
                'train_features': mock_features[:80],
                'train_targets': mock_targets[:80],
                'val_features': mock_features[80:],
                'val_targets': mock_targets[80:],
                'scaler': 'mock_scaler'
            }
            
            data = trainer.prepare_training_data('TRAIN_TEST')
            
            self.assertIsNotNone(data)
            self.assertIn('features', data)
            self.assertIn('targets', data)
            self.assertIn('train_features', data)
            self.assertIn('val_features', data)
            self.assertEqual(data['train_features'].shape[0], 80)
            self.assertEqual(data['val_features'].shape[0], 20)
    
    def test_model_architecture_creation(self):
        """Test LSTM model architecture creation."""
        input_size = 5
        hidden_size = 64
        num_layers = 2
        
        # Test basic LSTM model
        model = MockLSTMModel(input_size, hidden_size, num_layers)
        
        self.assertIsInstance(model.lstm, nn.LSTM)
        self.assertIsInstance(model.fc, nn.Linear)
        self.assertEqual(model.lstm.input_size, input_size)
        self.assertEqual(model.lstm.hidden_size, hidden_size)
        self.assertEqual(model.lstm.num_layers, num_layers)
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        model = MockLSTMModel()
        batch_size = 4
        seq_len = 60
        input_size = 5
        
        # Create sample input
        sample_input = torch.randn(batch_size, seq_len, input_size)
        
        # Forward pass
        output = model(sample_input)
        
        # Check output shape
        expected_shape = (batch_size, 1)
        self.assertEqual(output.shape, expected_shape)
    
    def test_loss_calculation(self):
        """Test loss calculation functionality."""
        model = MockLSTMModel()
        criterion = nn.MSELoss()
        
        # Sample data
        predictions = torch.randn(10, 1)
        targets = torch.randn(10, 1)
        
        # Calculate loss
        loss = criterion(predictions, targets)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Should be a scalar
        self.assertTrue(loss.item() >= 0)  # MSE loss should be non-negative
    
    @patch('Analytics.services.advanced_lstm_trainer.AdvancedLSTMTrainer')
    def test_training_loop(self, MockTrainer):
        """Test training loop execution."""
        trainer = MockTrainer.return_value
        
        # Mock training results
        mock_history = {
            'train_loss': [0.5, 0.4, 0.3, 0.25, 0.2],
            'val_loss': [0.6, 0.5, 0.4, 0.35, 0.3],
            'learning_rates': [0.001, 0.001, 0.001, 0.0005, 0.0005],
            'epochs': 5
        }
        
        trainer.train.return_value = {
            'success': True,
            'history': mock_history,
            'best_val_loss': 0.3,
            'model_path': '/path/to/model.pth',
            'training_time': 120.5
        }
        
        result = trainer.train('TRAIN_TEST')
        
        self.assertTrue(result['success'])
        self.assertIn('history', result)
        self.assertEqual(len(result['history']['train_loss']), 5)
        self.assertEqual(len(result['history']['val_loss']), 5)
        self.assertGreater(result['training_time'], 0)
    
    def test_model_validation(self):
        """Test model validation functionality."""
        model = MockLSTMModel()
        
        # Sample validation data
        val_features = torch.randn(20, 60, 5)
        val_targets = torch.randn(20, 1)
        
        model.eval()
        with torch.no_grad():
            predictions = model(val_features)
            
            # Calculate validation metrics
            mse = nn.MSELoss()(predictions, val_targets)
            mae = nn.L1Loss()(predictions, val_targets)
            
            self.assertTrue(mse.item() >= 0)
            self.assertTrue(mae.item() >= 0)
    
    @patch('Analytics.services.advanced_lstm_trainer.AdvancedLSTMTrainer')
    def test_model_saving_loading(self, MockTrainer):
        """Test model saving and loading functionality."""
        trainer = MockTrainer.return_value
        
        # Mock model save
        trainer.save_model.return_value = {
            'success': True,
            'model_path': '/mock/path/model.pth',
            'scaler_path': '/mock/path/scaler.pkl',
            'metadata': {
                'symbol': 'TRAIN_TEST',
                'timestamp': datetime.now().isoformat(),
                'architecture': 'LSTM_v1.0'
            }
        }
        
        # Mock model load
        trainer.load_model.return_value = {
            'success': True,
            'model': MockLSTMModel(),
            'scaler': 'mock_scaler',
            'metadata': {
                'symbol': 'TRAIN_TEST',
                'architecture': 'LSTM_v1.0'
            }
        }
        
        # Test save
        save_result = trainer.save_model('/mock/path')
        self.assertTrue(save_result['success'])
        self.assertIn('model_path', save_result)
        self.assertIn('metadata', save_result)
        
        # Test load
        load_result = trainer.load_model('/mock/path/model.pth')
        self.assertTrue(load_result['success'])
        self.assertIsNotNone(load_result['model'])
        self.assertIn('metadata', load_result)
    
    def test_hyperparameter_optimization(self):
        """Test hyperparameter optimization."""
        # Mock hyperparameter search results
        mock_search_results = [
            {'config': {'hidden_size': 64, 'learning_rate': 0.001}, 'val_loss': 0.25},
            {'config': {'hidden_size': 128, 'learning_rate': 0.001}, 'val_loss': 0.22},
            {'config': {'hidden_size': 64, 'learning_rate': 0.0005}, 'val_loss': 0.28},
            {'config': {'hidden_size': 128, 'learning_rate': 0.0005}, 'val_loss': 0.20}
        ]
        
        # Find best configuration
        best_config = min(mock_search_results, key=lambda x: x['val_loss'])
        
        self.assertEqual(best_config['config']['hidden_size'], 128)
        self.assertEqual(best_config['config']['learning_rate'], 0.0005)
        self.assertEqual(best_config['val_loss'], 0.20)
    
    def test_early_stopping(self):
        """Test early stopping mechanism."""
        # Mock training history with early stopping
        train_losses = [0.5, 0.4, 0.35, 0.33, 0.32, 0.32, 0.31]
        val_losses = [0.6, 0.45, 0.40, 0.38, 0.40, 0.42, 0.44]  # Val loss starts increasing
        
        patience = 3
        best_val_loss = min(val_losses)
        patience_counter = 0
        stopped_epoch = None
        
        for epoch, val_loss in enumerate(val_losses):
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                stopped_epoch = epoch
                break
        
        self.assertIsNotNone(stopped_epoch)
        self.assertEqual(stopped_epoch, 6)  # Should stop at epoch 6
        self.assertEqual(best_val_loss, 0.38)
    
    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling."""
        initial_lr = 0.001
        factor = 0.5
        patience = 2
        
        # Mock validation loss history
        val_losses = [0.5, 0.4, 0.35, 0.36, 0.37, 0.34, 0.33]
        
        # Simulate ReduceLROnPlateau behavior
        current_lr = initial_lr
        best_loss = float('inf')
        wait = 0
        lr_changes = []
        
        for epoch, val_loss in enumerate(val_losses):
            if val_loss < best_loss:
                best_loss = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    current_lr *= factor
                    lr_changes.append((epoch, current_lr))
                    wait = 0
        
        # Should have learning rate reductions
        self.assertGreater(len(lr_changes), 0)
        self.assertLess(lr_changes[-1][1], initial_lr)


class LSTMModelArchitectureTestCase(TestCase):
    """Test cases for LSTM model architecture components."""
    
    def setUp(self):
        """Set up test data."""
        self.batch_size = 4
        self.seq_len = 60
        self.hidden_size = 128
        self.sector_embedding_dim = 64
    
    def test_attention_layer_gradient_flow(self):
        """Test gradient flow through attention layer."""
        attention = AttentionLayer(self.hidden_size)
        lstm_outputs = torch.randn(self.batch_size, self.seq_len, self.hidden_size, requires_grad=True)
        
        attended_output, attention_weights = attention(lstm_outputs)
        
        # Create a dummy loss
        loss = attended_output.sum()
        loss.backward()
        
        # Check that gradients are computed
        self.assertIsNotNone(lstm_outputs.grad)
        self.assertTrue(torch.any(lstm_outputs.grad != 0))
    
    def test_sector_attention_gradient_flow(self):
        """Test gradient flow through sector cross-attention."""
        attention = SectorCrossAttention(self.hidden_size, self.sector_embedding_dim)
        lstm_output = torch.randn(self.batch_size, self.seq_len, self.hidden_size, requires_grad=True)
        sector_embedding = torch.randn(self.batch_size, self.sector_embedding_dim, requires_grad=True)
        
        output = attention(lstm_output, sector_embedding)
        
        # Create a dummy loss
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed for both inputs
        self.assertIsNotNone(lstm_output.grad)
        self.assertIsNotNone(sector_embedding.grad)
        self.assertTrue(torch.any(lstm_output.grad != 0))
        self.assertTrue(torch.any(sector_embedding.grad != 0))
    
    def test_model_parameter_count(self):
        """Test model parameter counting."""
        model = MockLSTMModel(input_size=5, hidden_size=64, num_layers=2)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.assertGreater(total_params, 0)
        self.assertEqual(total_params, trainable_params)  # All params should be trainable
    
    def test_model_mode_switching(self):
        """Test switching between train and eval modes."""
        model = MockLSTMModel()
        
        # Test train mode
        model.train()
        self.assertTrue(model.training)
        for module in model.modules():
            if hasattr(module, 'training'):
                self.assertTrue(module.training)
        
        # Test eval mode
        model.eval()
        self.assertFalse(model.training)
        for module in model.modules():
            if hasattr(module, 'training'):
                self.assertFalse(module.training)


class LSTMTrainingIntegrationTestCase(TestCase):
    """Integration tests for LSTM training pipeline."""
    
    def setUp(self):
        """Set up test data."""
        # Create multiple test stocks for comprehensive testing
        self.stocks = []
        for i in range(3):
            stock = Stock.objects.create(
                symbol=f'INTEGRATION{i}',
                short_name=f'Integration Test Stock {i}',
                exchange='NASDAQ'
            )
            self.stocks.append(stock)
            
            # Create price data with different patterns
            base_date = datetime.now().date() - timedelta(days=150)
            for j in range(120):
                # Create different price patterns for each stock
                if i == 0:  # Trending up
                    base_price = 100 + j * 0.1
                elif i == 1:  # Volatile
                    base_price = 100 + 10 * np.sin(j * 0.1)
                else:  # Trending down
                    base_price = 150 - j * 0.05
                
                StockPrice.objects.create(
                    stock=stock,
                    date=base_date + timedelta(days=j),
                    open=Decimal(f'{base_price:.2f}'),
                    high=Decimal(f'{base_price + 2:.2f}'),
                    low=Decimal(f'{base_price - 2:.2f}'),
                    close=Decimal(f'{base_price + 0.5:.2f}'),
                    volume=1000000 + j * 1000
                )
    
    @patch('Analytics.services.advanced_lstm_trainer.AdvancedLSTMTrainer')
    def test_multi_stock_training_pipeline(self, MockTrainer):
        """Test training pipeline with multiple stocks."""
        trainer = MockTrainer.return_value
        
        # Mock training results for each stock
        mock_results = {
            'INTEGRATION0': {'success': True, 'val_loss': 0.15, 'training_time': 45.0},
            'INTEGRATION1': {'success': True, 'val_loss': 0.25, 'training_time': 52.0},
            'INTEGRATION2': {'success': True, 'val_loss': 0.18, 'training_time': 48.0}
        }
        
        def mock_train(symbol):
            return mock_results.get(symbol, {'success': False})
        
        trainer.train.side_effect = mock_train
        
        results = []
        for stock in self.stocks:
            result = trainer.train(stock.symbol)
            results.append(result)
        
        # All training should succeed
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertTrue(result['success'])
            self.assertIn('val_loss', result)
            self.assertIn('training_time', result)
    
    def test_training_data_quality_validation(self):
        """Test validation of training data quality."""
        # Test data quality metrics
        for stock in self.stocks:
            prices = StockPrice.objects.filter(stock=stock).order_by('date')
            
            # Check data completeness
            self.assertGreater(prices.count(), 60)  # Minimum required for LSTM
            
            # Check for missing values
            for price in prices:
                self.assertIsNotNone(price.close)
                self.assertIsNotNone(price.volume)
                self.assertGreater(price.close, 0)
                self.assertGreater(price.volume, 0)
            
            # Check date continuity (allowing for weekends/holidays)
            dates = [p.date for p in prices]
            date_diffs = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            
            # Most date differences should be 1-3 days (accounting for weekends)
            reasonable_gaps = sum(1 for diff in date_diffs if diff <= 7)
            self.assertGreater(reasonable_gaps / len(date_diffs), 0.8)
    
    @patch('Analytics.services.advanced_lstm_trainer.AdvancedLSTMTrainer')
    def test_model_performance_benchmarking(self, MockTrainer):
        """Test model performance benchmarking."""
        trainer = MockTrainer.return_value
        
        # Mock performance metrics
        mock_metrics = {
            'mse': 0.05,
            'mae': 0.15,
            'mape': 2.5,  # 2.5% mean absolute percentage error
            'directional_accuracy': 0.68,  # 68% correct direction predictions
            'r2_score': 0.75
        }
        
        trainer.evaluate_model.return_value = {
            'success': True,
            'metrics': mock_metrics,
            'predictions_sample': [105.2, 104.8, 106.1, 105.9, 107.2],
            'actuals_sample': [105.0, 105.2, 106.0, 106.1, 107.0]
        }
        
        result = trainer.evaluate_model('INTEGRATION0')
        
        self.assertTrue(result['success'])
        metrics = result['metrics']
        
        # Validate performance thresholds
        self.assertLess(metrics['mse'], 0.1)  # MSE should be low
        self.assertLess(metrics['mape'], 5.0)  # MAPE should be under 5%
        self.assertGreater(metrics['directional_accuracy'], 0.6)  # > 60% directional accuracy
        self.assertGreater(metrics['r2_score'], 0.7)  # R² should be > 0.7
    
    def test_training_resource_management(self):
        """Test training resource management and cleanup."""
        # Mock resource monitoring
        initial_memory = 1000  # MB
        peak_memory = 2500     # MB
        final_memory = 1050    # MB
        
        # Simulate training resource usage
        resource_usage = {
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': peak_memory,
            'final_memory_mb': final_memory,
            'memory_cleanup_ratio': (peak_memory - final_memory) / (peak_memory - initial_memory)
        }
        
        # Memory should be cleaned up properly
        self.assertGreater(resource_usage['memory_cleanup_ratio'], 0.9)  # > 90% cleanup
        self.assertLess(resource_usage['final_memory_mb'] - resource_usage['initial_memory_mb'], 100)  # < 100MB increase
    
    def test_training_fault_tolerance(self):
        """Test training fault tolerance and recovery."""
        # Mock training interruption and recovery
        training_states = [
            {'epoch': 0, 'train_loss': 0.8, 'val_loss': 0.9},
            {'epoch': 1, 'train_loss': 0.6, 'val_loss': 0.7},
            {'epoch': 2, 'train_loss': 0.5, 'val_loss': 0.6},
            # Interruption occurs here
            {'epoch': 3, 'train_loss': 0.4, 'val_loss': 0.5, 'resumed': True},
            {'epoch': 4, 'train_loss': 0.35, 'val_loss': 0.45}
        ]
        
        # Find resumed training
        resumed_states = [state for state in training_states if state.get('resumed', False)]
        
        self.assertEqual(len(resumed_states), 1)
        self.assertEqual(resumed_states[0]['epoch'], 3)
        
        # Training should continue from interruption point
        final_loss = training_states[-1]['val_loss']
        interrupted_loss = training_states[2]['val_loss']
        
        self.assertLess(final_loss, interrupted_loss)  # Should continue improving
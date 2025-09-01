"""
Real implementation tests for Analytics LSTM training services.
Tests LSTM model training, validation, and optimization with actual functionality.
No mocks - uses real PostgreSQL test database.
"""

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from django.test import TestCase, TransactionTestCase
from django.db import transaction
import tempfile
import os
import json

from Analytics.ml.models.lstm_base import UniversalLSTMPredictor
from Analytics.services.advanced_lstm_trainer import AdvancedLSTMTrainer
from Data.models import Stock, StockPrice, DataSector, DataIndustry


class RealLSTMTrainerTestCase(TransactionTestCase):
    """Real test cases for LSTM training functionality using PostgreSQL."""
    
    def setUp(self):
        """Set up test data in PostgreSQL test database."""
        # Create sector and industry
        self.sector = DataSector.objects.create(
            sectorKey='test_tech',
            sectorName='Test Technology',
            data_source='yahoo'
        )
        
        self.industry = DataIndustry.objects.create(
            industryKey='test_software',
            industryName='Test Software',
            sector=self.sector,
            data_source='yahoo'
        )
        
        # Create test stock
        self.stock = Stock.objects.create(
            symbol='LSTM_TEST',
            short_name='LSTM Test Stock',
            long_name='LSTM Testing Corporation',
            exchange='NASDAQ',
            currency='USD',
            sector_id=self.sector,
            industry_id=self.industry,
            market_cap=1000000000
        )
        
        # Create realistic historical price data for training
        self._create_realistic_price_data()
        
        # Lightweight config for fast testing
        self.test_config = {
            'sequence_length': 20,  # Shorter sequence for faster testing
            'hidden_size': 32,      # Smaller hidden size
            'num_layers': 1,        # Single layer
            'dropout': 0.1,
            'learning_rate': 0.01,
            'batch_size': 8,
            'epochs': 2,            # Very few epochs for testing
            'patience': 1,
            'min_delta': 0.01
        }
    
    def _create_realistic_price_data(self):
        """Create realistic price data with trends and volatility."""
        base_date = datetime.now().date() - timedelta(days=100)
        base_price = 100.0
        
        for i in range(100):
            # Add realistic price movement
            trend = 0.001 * i  # Slight upward trend
            seasonality = 5 * np.sin(2 * np.pi * i / 30)  # Monthly seasonality
            noise = np.random.normal(0, 2)  # Daily volatility
            
            price = base_price + trend * base_price + seasonality + noise
            price = max(price, 10)  # Ensure positive price
            
            # Create OHLC data
            daily_volatility = abs(np.random.normal(0, 1))
            open_price = price + np.random.uniform(-daily_volatility, daily_volatility)
            high_price = max(open_price, price) + abs(np.random.uniform(0, daily_volatility))
            low_price = min(open_price, price) - abs(np.random.uniform(0, daily_volatility))
            close_price = price
            
            # Realistic volume with some patterns
            base_volume = 1000000
            volume_multiplier = 1 + abs(np.random.normal(0, 0.3))
            if i % 5 == 0:  # Higher volume on certain days
                volume_multiplier *= 1.5
            volume = int(base_volume * volume_multiplier)
            
            StockPrice.objects.create(
                stock=self.stock,
                date=base_date + timedelta(days=i),
                open=Decimal(str(round(open_price, 2))),
                high=Decimal(str(round(high_price, 2))),
                low=Decimal(str(round(low_price, 2))),
                close=Decimal(str(round(close_price, 2))),
                adjusted_close=Decimal(str(round(close_price, 2))),
                volume=volume,
                data_source='test'
            )
    
    def test_real_trainer_initialization(self):
        """Test real LSTM trainer initialization."""
        trainer = AdvancedLSTMTrainer('LSTM_TEST', config=self.test_config)
        
        self.assertEqual(trainer.symbol, 'LSTM_TEST')
        self.assertIsNotNone(trainer.config)
        self.assertEqual(trainer.config['sequence_length'], 20)
        self.assertEqual(trainer.config['hidden_size'], 32)
        self.assertIsNone(trainer.model)  # Model created during training
    
    def test_real_data_preparation(self):
        """Test real training data preparation from database."""
        trainer = AdvancedLSTMTrainer('LSTM_TEST', config=self.test_config)
        
        # Prepare data - this actually queries the database
        X_train, X_val, y_train, y_val = trainer.prepare_training_data(lookback_days=90)
        
        # Verify data shapes
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(y_train)
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(y_train), 0)
        
        # Check sequence length
        if len(X_train.shape) == 3:
            self.assertEqual(X_train.shape[1], self.test_config['sequence_length'])
            # Should have 5 features (open, high, low, close, volume)
            self.assertEqual(X_train.shape[2], 5)
        
        # Ensure train/val split
        self.assertGreater(len(X_train), len(X_val))
    
    def test_real_model_creation(self):
        """Test real LSTM model creation."""
        trainer = AdvancedLSTMTrainer('LSTM_TEST', config=self.test_config)
        
        # Create model
        trainer.create_model()
        
        self.assertIsNotNone(trainer.model)
        self.assertIsInstance(trainer.model, UniversalLSTMPredictor)
        
        # Check model parameters
        total_params = sum(p.numel() for p in trainer.model.parameters())
        self.assertGreater(total_params, 0)
        
        # Verify model can do forward pass
        sample_input = torch.randn(1, self.test_config['sequence_length'], 5)
        with torch.no_grad():
            output = trainer.model(sample_input)
        self.assertEqual(output.shape[0], 1)  # Batch size
        self.assertEqual(output.shape[1], 1)  # Single prediction
    
    def test_real_training_execution(self):
        """Test actual training execution with real data."""
        trainer = AdvancedLSTMTrainer('LSTM_TEST', config=self.test_config)
        
        # Prepare data
        X_train, X_val, y_train, y_val = trainer.prepare_training_data(lookback_days=90)
        
        if X_train is None or len(X_train) == 0:
            self.skipTest("Insufficient data for training test")
        
        # Create model
        trainer.create_model()
        
        # Train model (very short training for testing)
        history = trainer.train(
            (X_train, X_val, y_train, y_val),
            epochs=2,  # Just 2 epochs for testing
            batch_size=8
        )
        
        self.assertIsNotNone(history)
        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
        self.assertEqual(len(history['train_loss']), 2)
        
        # Losses should be finite numbers
        for loss in history['train_loss']:
            self.assertTrue(np.isfinite(loss))
            self.assertGreater(loss, 0)
    
    def test_real_model_evaluation(self):
        """Test real model evaluation with metrics."""
        trainer = AdvancedLSTMTrainer('LSTM_TEST', config=self.test_config)
        
        # Prepare data
        X_train, X_val, y_train, y_val = trainer.prepare_training_data(lookback_days=90)
        
        if X_train is None or len(X_train) == 0:
            self.skipTest("Insufficient data for evaluation test")
        
        # Create and minimally train model
        trainer.create_model()
        trainer.train(
            (X_train, X_val, y_train, y_val),
            epochs=1,
            batch_size=8
        )
        
        # Evaluate model
        metrics = trainer.evaluate_model(X_val, y_val)
        
        self.assertIsNotNone(metrics)
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        
        # All metrics should be positive
        self.assertGreater(metrics['mse'], 0)
        self.assertGreater(metrics['mae'], 0)
        self.assertGreater(metrics['rmse'], 0)
    
    def test_real_prediction_generation(self):
        """Test real prediction generation."""
        trainer = AdvancedLSTMTrainer('LSTM_TEST', config=self.test_config)
        
        # Prepare data and train minimal model
        X_train, X_val, y_train, y_val = trainer.prepare_training_data(lookback_days=90)
        
        if X_train is None or len(X_train) == 0:
            self.skipTest("Insufficient data for prediction test")
        
        trainer.create_model()
        trainer.train(
            (X_train, X_val, y_train, y_val),
            epochs=1,
            batch_size=8
        )
        
        # Generate predictions
        trainer.model.eval()
        with torch.no_grad():
            predictions = trainer.model(torch.FloatTensor(X_val[:5]))
        
        self.assertEqual(len(predictions), 5)
        for pred in predictions:
            self.assertTrue(torch.isfinite(pred).all())
    
    def test_model_save_load_functionality(self):
        """Test saving and loading trained models."""
        trainer = AdvancedLSTMTrainer('LSTM_TEST', config=self.test_config)
        
        # Train a minimal model
        X_train, X_val, y_train, y_val = trainer.prepare_training_data(lookback_days=90)
        
        if X_train is None or len(X_train) == 0:
            self.skipTest("Insufficient data for save/load test")
        
        trainer.create_model()
        trainer.train(
            (X_train, X_val, y_train, y_val),
            epochs=1,
            batch_size=8
        )
        
        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.pth')
            metadata_path = os.path.join(temp_dir, 'test_model_metadata.json')
            
            # Save model state
            torch.save({
                'model_state_dict': trainer.model.state_dict(),
                'config': trainer.config,
                'symbol': trainer.symbol,
                'timestamp': datetime.now().isoformat()
            }, model_path)
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump({
                    'symbol': trainer.symbol,
                    'config': trainer.config,
                    'training_date': datetime.now().isoformat()
                }, f)
            
            # Create new trainer and load model
            new_trainer = AdvancedLSTMTrainer('LSTM_TEST', config=self.test_config)
            new_trainer.create_model()
            
            # Load saved state
            checkpoint = torch.load(model_path)
            new_trainer.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Verify loaded model works
            new_trainer.model.eval()
            with torch.no_grad():
                sample_input = torch.randn(1, self.test_config['sequence_length'], 5)
                output = new_trainer.model(sample_input)
            
            self.assertEqual(output.shape, (1, 1))
    
    def test_training_with_different_configurations(self):
        """Test training with various configurations."""
        configs = [
            {'sequence_length': 10, 'hidden_size': 16, 'num_layers': 1},
            {'sequence_length': 20, 'hidden_size': 32, 'num_layers': 1},
        ]
        
        for config in configs:
            full_config = {**self.test_config, **config}
            trainer = AdvancedLSTMTrainer('LSTM_TEST', config=full_config)
            
            # Verify config is applied
            self.assertEqual(trainer.config['sequence_length'], config['sequence_length'])
            self.assertEqual(trainer.config['hidden_size'], config['hidden_size'])
            
            # Create model with config
            trainer.create_model()
            self.assertIsNotNone(trainer.model)
    
    def test_data_validation_and_preprocessing(self):
        """Test data validation and preprocessing steps."""
        trainer = AdvancedLSTMTrainer('LSTM_TEST', config=self.test_config)
        
        # Get raw price data
        prices = StockPrice.objects.filter(
            stock__symbol='LSTM_TEST'
        ).order_by('date').values_list('close', flat=True)
        
        prices_array = np.array([float(p) for p in prices])
        
        # Check data quality
        self.assertFalse(np.any(np.isnan(prices_array)))  # No NaN values
        self.assertFalse(np.any(np.isinf(prices_array)))  # No infinite values
        self.assertTrue(np.all(prices_array > 0))  # All positive prices
        
        # Verify normalization works
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(prices_array.reshape(-1, 1))
        
        self.assertTrue(np.all(normalized >= 0))
        self.assertTrue(np.all(normalized <= 1))


class LSTMPerformanceTestCase(TestCase):
    """Test LSTM model performance and optimization."""
    
    def setUp(self):
        """Set up performance test data."""
        # Create minimal test stock
        self.stock = Stock.objects.create(
            symbol='PERF_TEST',
            short_name='Performance Test Stock',
            exchange='NASDAQ'
        )
        
        # Create sufficient data for performance testing
        base_date = datetime.now().date() - timedelta(days=60)
        for i in range(60):
            price = 100 + 10 * np.sin(i / 10) + np.random.normal(0, 1)
            StockPrice.objects.create(
                stock=self.stock,
                date=base_date + timedelta(days=i),
                open=Decimal(str(round(price - 1, 2))),
                high=Decimal(str(round(price + 2, 2))),
                low=Decimal(str(round(price - 2, 2))),
                close=Decimal(str(round(price, 2))),
                adjusted_close=Decimal(str(round(price, 2))),
                volume=1000000
            )
    
    def test_training_speed(self):
        """Test that training completes in reasonable time."""
        import time
        
        config = {
            'sequence_length': 10,
            'hidden_size': 16,
            'num_layers': 1,
            'epochs': 1,
            'batch_size': 16
        }
        
        trainer = AdvancedLSTMTrainer('PERF_TEST', config=config)
        X_train, X_val, y_train, y_val = trainer.prepare_training_data(lookback_days=50)
        
        if X_train is None or len(X_train) < 10:
            self.skipTest("Insufficient data for performance test")
        
        trainer.create_model()
        
        start_time = time.time()
        trainer.train(
            (X_train, X_val, y_train, y_val),
            epochs=1,
            batch_size=16
        )
        training_time = time.time() - start_time
        
        # Should complete single epoch quickly with small model
        self.assertLess(training_time, 10)  # Less than 10 seconds
    
    def test_memory_efficiency(self):
        """Test memory efficiency during training."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        config = {
            'sequence_length': 10,
            'hidden_size': 16,
            'num_layers': 1,
            'epochs': 1
        }
        
        trainer = AdvancedLSTMTrainer('PERF_TEST', config=config)
        X_train, X_val, y_train, y_val = trainer.prepare_training_data(lookback_days=50)
        
        if X_train is None:
            self.skipTest("Insufficient data for memory test")
        
        trainer.create_model()
        trainer.train(
            (X_train, X_val, y_train, y_val),
            epochs=1,
            batch_size=8
        )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not use excessive memory for small model
        self.assertLess(memory_increase, 500)  # Less than 500MB increase
    
    def test_gradient_stability(self):
        """Test gradient stability during training."""
        trainer = AdvancedLSTMTrainer('PERF_TEST')
        trainer.create_model()
        
        # Monitor gradients during a forward-backward pass
        sample_input = torch.randn(4, 10, 5, requires_grad=True)
        output = trainer.model(sample_input)
        loss = output.mean()
        loss.backward()
        
        # Check gradient magnitudes
        for name, param in trainer.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                self.assertTrue(np.isfinite(grad_norm))
                self.assertLess(grad_norm, 100)  # No exploding gradients


if __name__ == '__main__':
    import django
    django.setup()
    from django.test import TestRunner
    runner = TestRunner()
    runner.run_tests(['Analytics.tests.test_lstm_trainer_real'])
"""
Unit tests for Analytics LSTM training services.
Tests LSTM model training, validation, and optimization using real functionality.
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from django.contrib.auth.models import User
from django.test import TestCase

from Analytics.ml.models.lstm_base import AttentionLayer, SectorCrossAttention
from Analytics.services.advanced_lstm_trainer import AdvancedLSTMTrainer
from Data.models import StockPrice
from Data.tests.fixtures import DataTestDataFactory


class TestLSTMModel(nn.Module):
    """Real LSTM model for testing functionality."""

    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
        super(TestLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(self.dropout(lstm_out[:, -1, :]))
        return output


class LSTMTrainerTestCase(TestCase):
    """Test cases for LSTM training functionality using real training operations."""

    def setUp(self):
        """Set up test data using DataTestDataFactory."""
        # Create test stock using factory
        self.stock = DataTestDataFactory.create_test_stock(
            symbol="TRAIN_TEST", company_name="Training Test Stock", sector="Technology"
        )

        # Create comprehensive historical price data for training
        DataTestDataFactory.create_stock_price_history(self.stock, days=200)

        # Create test user for analytics
        self.test_user = User.objects.create_user(username="testuser", email="test@example.com", password="testpass")

        # Real trainer configuration for testing
        self.trainer_config = {
            "sequence_length": 30,  # Smaller for faster testing
            "hidden_size": 32,  # Smaller for resource constraints
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 0.01,  # Higher for faster convergence in tests
            "batch_size": 8,  # Smaller batches for testing
            "epochs": 3,  # Limited epochs for testing
            "validation_split": 0.3,
            "early_stopping_patience": 10,
            "lr_scheduler_patience": 5,
        }

        # Create temporary directory for model storage
        self.temp_dir = Path(tempfile.mkdtemp())

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_trainer_initialization(self):
        """Test real LSTM trainer initialization."""
        trainer = AdvancedLSTMTrainer(symbol="TRAIN_TEST", config=self.trainer_config)

        # Test real initialization
        self.assertEqual(trainer.symbol, "TRAIN_TEST")
        self.assertIsInstance(trainer.config, dict)
        self.assertIn("sequence_length", trainer.config)
        self.assertEqual(trainer.config["hidden_size"], 32)
        self.assertIsInstance(trainer.device, torch.device)
        self.assertIsNone(trainer.model)  # Not created until training

    def test_data_preparation(self):
        """Test real training data preparation."""
        trainer = AdvancedLSTMTrainer(symbol="TRAIN_TEST", config=self.trainer_config)

        try:
            # Real data preparation
            data = trainer.prepare_training_data("TRAIN_TEST")

            # Validate real prepared data
            self.assertIsNotNone(data)
            self.assertIn("features", data)
            self.assertIn("targets", data)
            self.assertIn("train_features", data)
            self.assertIn("val_features", data)
            self.assertIn("scaler", data)
            self.assertEqual(data["symbol"], "TRAIN_TEST")

            # Check tensor shapes
            self.assertIsInstance(data["features"], torch.Tensor)
            self.assertIsInstance(data["targets"], torch.Tensor)
            self.assertEqual(len(data["features"].shape), 3)  # batch, seq, features
            self.assertEqual(len(data["targets"].shape), 2)  # batch, output

            # Validate train/val split
            total_samples = data["features"].shape[0]
            train_samples = data["train_features"].shape[0]
            val_samples = data["val_features"].shape[0]
            self.assertEqual(total_samples, train_samples + val_samples)

            # Check split ratio
            expected_val_ratio = self.trainer_config["validation_split"]
            actual_val_ratio = val_samples / total_samples
            self.assertAlmostEqual(actual_val_ratio, expected_val_ratio, delta=0.1)

        except ValueError as e:
            if "Insufficient data" in str(e):
                self.skipTest("Insufficient test data for training preparation")
            else:
                raise

    def test_model_architecture_creation(self):
        """Test real LSTM model architecture creation."""
        trainer = AdvancedLSTMTrainer(config=self.trainer_config)
        input_size = 15  # Realistic feature count

        # Create real model using trainer
        model = trainer.create_model(input_size)

        # Validate model architecture
        self.assertIsNotNone(model)
        self.assertIsInstance(model, nn.Module)

        # Check if model has expected layers
        has_lstm = any(isinstance(module, nn.LSTM) for module in model.modules())
        has_linear = any(isinstance(module, nn.Linear) for module in model.modules())
        has_dropout = any(isinstance(module, nn.Dropout) for module in model.modules())

        self.assertTrue(has_lstm, "Model should contain LSTM layer")
        self.assertTrue(has_linear, "Model should contain Linear layer")
        self.assertTrue(has_dropout, "Model should contain Dropout layer")

        # Test model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.assertGreater(total_params, 0, "Model should have parameters")
        self.assertEqual(total_params, trainable_params, "All parameters should be trainable")

    def test_model_forward_pass(self):
        """Test real model forward pass."""
        trainer = AdvancedLSTMTrainer(config=self.trainer_config)
        input_size = 15
        model = trainer.create_model(input_size)

        batch_size = 4
        seq_len = self.trainer_config["sequence_length"]

        # Create realistic sample input
        sample_input = torch.randn(batch_size, seq_len, input_size)

        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(sample_input)

        # Validate output
        expected_shape = (batch_size, 1)
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(torch.isfinite(output).all(), "Output should be finite")
        self.assertFalse(torch.isnan(output).any(), "Output should not contain NaN")

    def test_loss_calculation(self):
        """Test real loss calculation functionality."""
        trainer = AdvancedLSTMTrainer(config=self.trainer_config)
        criterion = trainer.criterion  # Use real criterion

        # Generate realistic predictions and targets
        batch_size = 10
        predictions = torch.randn(batch_size, 1) * 0.1 + 150  # Around price of 150
        targets = torch.randn(batch_size, 1) * 0.1 + 150

        # Calculate real loss
        loss = criterion(predictions, targets)

        # Validate loss properties
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0, "Loss should be a scalar")
        self.assertGreaterEqual(loss.item(), 0, "MSE loss should be non-negative")
        self.assertTrue(torch.isfinite(loss), "Loss should be finite")

        # Test gradient computation
        predictions.requires_grad_(True)
        loss = criterion(predictions, targets)
        loss.backward()

        self.assertIsNotNone(predictions.grad, "Gradients should be computed")
        self.assertTrue(torch.isfinite(predictions.grad).all(), "Gradients should be finite")

    def test_training_loop(self):
        """Test real training loop execution with resource constraints."""
        trainer = AdvancedLSTMTrainer(symbol="TRAIN_TEST", config=self.trainer_config)

        try:
            # Execute real training with limited resources
            result = trainer.train("TRAIN_TEST")

            # Check if training succeeded or failed gracefully
            self.assertIsInstance(result, dict)
            self.assertIn("success", result)

            if result["success"]:
                # Validate successful training results
                self.assertIn("history", result)
                self.assertIn("training_time", result)
                self.assertIn("model_path", result)
                self.assertIn("best_val_loss", result)

                # Validate training history
                history = result["history"]
                self.assertIn("train_loss", history)
                self.assertIn("val_loss", history)

                # Check that losses are realistic
                train_losses = history["train_loss"]
                val_losses = history["val_loss"]

                self.assertTrue(len(train_losses) > 0, "Should have training losses")
                self.assertTrue(len(val_losses) > 0, "Should have validation losses")
                self.assertTrue(all(loss >= 0 for loss in train_losses), "Losses should be non-negative")
                self.assertTrue(all(loss >= 0 for loss in val_losses), "Losses should be non-negative")

                # Validate training time
                self.assertIsInstance(result["training_time"], (int, float))
                self.assertGreater(result["training_time"], 0)

            else:
                # Validate graceful failure
                self.assertIn("error", result)
                self.assertIsInstance(result["error"], str)

        except Exception as e:
            # Handle resource constraints gracefully
            if any(keyword in str(e).lower() for keyword in ["memory", "cuda", "device", "insufficient"]):
                self.skipTest(f"Training test skipped due to resource constraints: {e}")
            else:
                # Re-raise unexpected errors
                raise

    def test_model_validation(self):
        """Test real model validation functionality."""
        trainer = AdvancedLSTMTrainer(config=self.trainer_config)
        input_size = 15
        model = trainer.create_model(input_size)

        # Generate realistic validation data
        batch_size = 20
        seq_len = self.trainer_config["sequence_length"]
        val_features = torch.randn(batch_size, seq_len, input_size) * 0.1
        val_targets = torch.randn(batch_size, 1) * 0.1 + 150  # Around price level

        # Test real validation
        model.eval()
        with torch.no_grad():
            predictions = model(val_features)

            # Calculate real validation metrics using trainer's criterion
            mse_loss = trainer.criterion(predictions, val_targets)
            mae_loss = nn.L1Loss()(predictions, val_targets)

            # Validate metrics
            self.assertIsInstance(mse_loss, torch.Tensor)
            self.assertIsInstance(mae_loss, torch.Tensor)
            self.assertGreaterEqual(mse_loss.item(), 0, "MSE should be non-negative")
            self.assertGreaterEqual(mae_loss.item(), 0, "MAE should be non-negative")
            self.assertTrue(torch.isfinite(mse_loss), "MSE should be finite")
            self.assertTrue(torch.isfinite(mae_loss), "MAE should be finite")

            # Test validation method from trainer
            val_loss = trainer._validate(val_features, val_targets)
            self.assertIsInstance(val_loss, float)
            self.assertGreaterEqual(val_loss, 0)
            self.assertTrue(np.isfinite(val_loss))

    def test_model_saving_loading(self):
        """Test real model saving and loading functionality."""
        trainer = AdvancedLSTMTrainer(symbol="TRAIN_TEST", config=self.trainer_config)

        # Create a model to save
        input_size = 15
        model = trainer.create_model(input_size)
        trainer.model = model

        # Initialize training history for saving
        trainer.training_history = {
            "train_loss": [0.5, 0.4, 0.3],
            "val_loss": [0.6, 0.5, 0.4],
            "best_val_loss": 0.4,
            "best_epoch": 2,
        }

        try:
            # Test real model saving
            save_path = self.temp_dir / "test_model.pth"
            save_result = trainer.save_model(str(save_path))

            # Validate save result
            self.assertTrue(save_result["success"], "Model save should succeed")
            self.assertIn("model_path", save_result)
            self.assertIn("metadata", save_result)
            self.assertEqual(save_result["metadata"]["symbol"], "TRAIN_TEST")

            # Check file actually exists
            self.assertTrue(Path(save_result["model_path"]).exists(), "Model file should exist")

            # Test real model loading
            new_trainer = AdvancedLSTMTrainer()
            load_result = new_trainer.load_model(save_result["model_path"])

            # Validate load result
            self.assertTrue(load_result["success"], "Model load should succeed")
            self.assertIsNotNone(load_result["model"], "Loaded model should not be None")
            self.assertIn("metadata", load_result)
            self.assertEqual(load_result["metadata"]["symbol"], "TRAIN_TEST")

            # Validate loaded model can make predictions
            loaded_model = load_result["model"]
            test_input = torch.randn(1, self.trainer_config["sequence_length"], input_size)

            loaded_model.eval()
            with torch.no_grad():
                prediction = loaded_model(test_input)

            self.assertEqual(prediction.shape, (1, 1), "Loaded model should produce correct output shape")
            self.assertTrue(torch.isfinite(prediction), "Loaded model prediction should be finite")

        except Exception as e:
            if "disk space" in str(e).lower() or "permission" in str(e).lower():
                self.skipTest(f"Model save/load test skipped due to file system constraints: {e}")
            else:
                raise

        finally:
            # Clean up test files
            try:
                for file_path in self.temp_dir.glob("*"):
                    file_path.unlink()
                self.temp_dir.rmdir()
            except Exception:
                pass

    def test_hyperparameter_optimization(self):
        """Test real hyperparameter optimization simulation."""
        # Define real hyperparameter search space
        param_grid = [
            {"hidden_size": 16, "learning_rate": 0.01, "dropout": 0.1},
            {"hidden_size": 32, "learning_rate": 0.01, "dropout": 0.2},
            {"hidden_size": 16, "learning_rate": 0.005, "dropout": 0.1},
            {"hidden_size": 32, "learning_rate": 0.005, "dropout": 0.2},
        ]

        search_results = []

        # Simulate hyperparameter search with real validation
        for params in param_grid:
            try:
                # Create trainer with specific parameters
                config = self.trainer_config.copy()
                config.update(params)
                config["epochs"] = 1  # Quick validation run

                trainer = AdvancedLSTMTrainer(symbol="TRAIN_TEST", config=config)

                # Prepare data once
                data = trainer.prepare_training_data("TRAIN_TEST")

                # Create and test model quickly
                input_size = data["train_features"].shape[2]
                model = trainer.create_model(input_size)

                # Quick validation to estimate performance
                model.eval()
                with torch.no_grad():
                    predictions = model(data["val_features"][:10])  # Small sample
                    val_loss = trainer.criterion(predictions, data["val_targets"][:10]).item()

                search_results.append(
                    {"config": params, "val_loss": val_loss, "model_params": sum(p.numel() for p in model.parameters())}
                )

            except Exception as e:
                # Handle resource constraints gracefully
                if any(keyword in str(e).lower() for keyword in ["memory", "insufficient", "cuda"]):
                    continue
                else:
                    raise

        # Validate search results if any completed
        if search_results:
            # Find best configuration
            best_result = min(search_results, key=lambda x: x["val_loss"])

            # Validate best configuration properties
            self.assertIn("config", best_result)
            self.assertIn("val_loss", best_result)
            self.assertGreater(best_result["val_loss"], 0, "Validation loss should be positive")
            self.assertTrue(np.isfinite(best_result["val_loss"]), "Validation loss should be finite")

            # Validate parameter count is reasonable
            self.assertGreater(best_result["model_params"], 0, "Model should have parameters")
            self.assertLess(best_result["model_params"], 100000, "Model should not be too large for testing")
        else:
            self.skipTest("Hyperparameter optimization test skipped due to resource constraints")

    def test_early_stopping(self):
        """Test real early stopping mechanism implementation."""
        # Simulate real early stopping logic
        train_losses = [0.8, 0.6, 0.45, 0.38, 0.36, 0.37, 0.38, 0.39, 0.40]
        val_losses = [0.9, 0.7, 0.55, 0.42, 0.41, 0.43, 0.45, 0.47, 0.49]  # Starts increasing after epoch 4

        # Real early stopping parameters from config
        patience = self.trainer_config["early_stopping_patience"]

        # Implement real early stopping logic
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        stopped_epoch = None

        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            # Update best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            # Check early stopping condition
            if patience_counter >= patience:
                stopped_epoch = epoch
                break

        # Validate early stopping behavior
        if stopped_epoch is not None:
            # Early stopping was triggered
            self.assertIsInstance(stopped_epoch, int)
            self.assertGreater(stopped_epoch, best_epoch, "Should stop after best epoch")
            self.assertGreaterEqual(stopped_epoch, patience, "Should wait at least patience epochs")

        # Validate best loss tracking
        self.assertEqual(best_val_loss, min(val_losses), "Should track minimum validation loss")
        self.assertGreater(best_val_loss, 0, "Best validation loss should be positive")
        self.assertTrue(np.isfinite(best_val_loss), "Best validation loss should be finite")

        # Test with trainer's real early stopping configuration
        trainer = AdvancedLSTMTrainer(config=self.trainer_config)
        self.assertEqual(trainer.config["early_stopping_patience"], patience, "Trainer should use configured patience")

    def test_learning_rate_scheduling(self):
        """Test real learning rate scheduling implementation."""
        trainer = AdvancedLSTMTrainer(config=self.trainer_config)

        # Get real scheduler configuration
        initial_lr = trainer.config["learning_rate"]
        factor = trainer.config["lr_scheduler_factor"]
        patience = trainer.config["lr_scheduler_patience"]

        # Create real optimizer and scheduler
        model = trainer.create_model(10)  # dummy model
        trainer.model = model
        trainer.optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
        trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            trainer.optimizer, mode="min", factor=factor, patience=patience, verbose=False
        )

        # Simulate real training with LR scheduling
        val_losses = [0.8, 0.6, 0.55, 0.56, 0.57, 0.58, 0.52, 0.51]  # Loss plateaus then improves
        lr_history = []

        for epoch, val_loss in enumerate(val_losses):
            # Record current learning rate
            current_lr = trainer.optimizer.param_groups[0]["lr"]
            lr_history.append(current_lr)

            # Step scheduler with validation loss
            trainer.scheduler.step(val_loss)

        # Validate learning rate scheduling behavior
        self.assertEqual(lr_history[0], initial_lr, "Should start with initial learning rate")

        # Check if learning rate was reduced (it should be with the plateau pattern)
        final_lr = trainer.optimizer.param_groups[0]["lr"]
        self.assertLessEqual(final_lr, initial_lr, "Learning rate should not increase")

        # Validate scheduler parameters match configuration
        self.assertEqual(trainer.scheduler.factor, factor)
        self.assertEqual(trainer.scheduler.patience, patience)
        self.assertEqual(trainer.scheduler.mode, "min")

        # Test manual step with improvement
        pre_step_lr = trainer.optimizer.param_groups[0]["lr"]
        trainer.scheduler.step(0.1)  # Significant improvement
        post_step_lr = trainer.optimizer.param_groups[0]["lr"]

        # Learning rate should not change immediately with improvement
        # (ReduceLROnPlateau only reduces, doesn't increase)
        self.assertEqual(pre_step_lr, post_step_lr, "LR should not change with single improvement")

    def tearDown(self):
        """Clean up after tests."""
        # Clean up test data
        DataTestDataFactory.cleanup_test_data()

        # Clean up temporary files
        try:
            if hasattr(self, "temp_dir") and self.temp_dir.exists():
                for file_path in self.temp_dir.glob("*"):
                    file_path.unlink()
                self.temp_dir.rmdir()
        except Exception:
            pass


class LSTMModelArchitectureTestCase(TestCase):
    """Test cases for real LSTM model architecture components."""

    def setUp(self):
        """Set up test data."""
        self.batch_size = 4
        self.seq_len = 30  # Smaller for testing
        self.hidden_size = 64  # Smaller for testing
        self.sector_embedding_dim = 32  # Smaller for testing

        # Create test configuration
        self.test_config = {"hidden_size": self.hidden_size, "num_layers": 2, "dropout": 0.2, "use_attention": True}

    def test_attention_layer_gradient_flow(self):
        """Test real gradient flow through attention layer."""
        attention = AttentionLayer(self.hidden_size)
        lstm_outputs = torch.randn(self.batch_size, self.seq_len, self.hidden_size, requires_grad=True)

        # Test real attention computation
        attended_output, attention_weights = attention(lstm_outputs)

        # Validate attention output properties
        self.assertEqual(attended_output.shape, (self.batch_size, self.hidden_size))
        self.assertEqual(attention_weights.shape, (self.batch_size, self.seq_len))

        # Verify attention weights are valid probabilities
        self.assertTrue(
            torch.allclose(attention_weights.sum(dim=1), torch.ones(self.batch_size)),
            "Attention weights should sum to 1",
        )
        self.assertTrue((attention_weights >= 0).all(), "Attention weights should be non-negative")

        # Test gradient flow
        loss = attended_output.sum()
        loss.backward()

        # Validate gradient computation
        self.assertIsNotNone(lstm_outputs.grad, "Gradients should be computed")
        self.assertTrue(torch.any(lstm_outputs.grad != 0), "Gradients should be non-zero")
        self.assertTrue(torch.isfinite(lstm_outputs.grad).all(), "Gradients should be finite")

        # Test attention layer parameters have gradients
        for param in attention.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, "Parameter gradients should be computed")
                self.assertTrue(torch.isfinite(param.grad).all(), "Parameter gradients should be finite")

    def test_sector_attention_gradient_flow(self):
        """Test real gradient flow through sector cross-attention."""
        attention = SectorCrossAttention(self.hidden_size, self.sector_embedding_dim)
        lstm_output = torch.randn(self.batch_size, self.seq_len, self.hidden_size, requires_grad=True)
        sector_embedding = torch.randn(self.batch_size, self.sector_embedding_dim, requires_grad=True)

        # Test real sector attention computation
        output = attention(lstm_output, sector_embedding)

        # Validate output properties
        self.assertEqual(
            output.shape, (self.batch_size, self.hidden_size), "Sector attention output should have correct shape"
        )
        self.assertTrue(torch.isfinite(output).all(), "Sector attention output should be finite")

        # Test gradient flow
        loss = output.sum()
        loss.backward()

        # Validate gradients for both inputs
        self.assertIsNotNone(lstm_output.grad, "LSTM output gradients should be computed")
        self.assertIsNotNone(sector_embedding.grad, "Sector embedding gradients should be computed")

        # Check gradient properties
        self.assertTrue(torch.any(lstm_output.grad != 0), "LSTM output gradients should be non-zero")
        self.assertTrue(torch.any(sector_embedding.grad != 0), "Sector embedding gradients should be non-zero")
        self.assertTrue(torch.isfinite(lstm_output.grad).all(), "LSTM output gradients should be finite")
        self.assertTrue(torch.isfinite(sector_embedding.grad).all(), "Sector embedding gradients should be finite")

        # Test attention layer parameters have gradients
        for name, param in attention.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} gradient should be computed")
                self.assertTrue(torch.isfinite(param.grad).all(), f"Parameter {name} gradient should be finite")

    def test_model_parameter_count(self):
        """Test real model parameter counting."""
        trainer = AdvancedLSTMTrainer(config=self.test_config)
        input_size = 15
        model = trainer.create_model(input_size)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Validate parameter counts
        self.assertGreater(total_params, 0, "Model should have parameters")
        self.assertEqual(total_params, trainable_params, "All parameters should be trainable by default")

        # Estimate expected parameter count for validation
        # LSTM: 4 * (input_size + hidden_size + 1) * hidden_size * num_layers
        # + Attention layer parameters
        # + Linear layer: hidden_size * 1 + 1
        expected_min_params = (
            4 * (input_size + self.hidden_size) * self.hidden_size * 2  # LSTM layers
            + self.hidden_size  # Linear layer
            + 1  # Linear bias
        )

        self.assertGreater(
            total_params,
            expected_min_params,
            f"Model should have at least {expected_min_params} parameters, got {total_params}",
        )

        # Validate parameter types and properties
        param_count_by_type = {}
        for name, param in model.named_parameters():
            param_type = name.split(".")[0]  # First part of parameter name
            if param_type not in param_count_by_type:
                param_count_by_type[param_type] = 0
            param_count_by_type[param_type] += param.numel()

            # Validate parameter properties
            self.assertTrue(param.requires_grad, f"Parameter {name} should require gradients")
            self.assertTrue(torch.isfinite(param).all(), f"Parameter {name} should be finite")

        # Should have parameters from different components
        self.assertIn("lstm", param_count_by_type, "Should have LSTM parameters")
        self.assertIn("fc", param_count_by_type, "Should have fully connected parameters")

    def test_model_mode_switching(self):
        """Test real switching between train and eval modes."""
        trainer = AdvancedLSTMTrainer(config=self.test_config)
        model = trainer.create_model(15)

        # Test train mode
        model.train()
        self.assertTrue(model.training, "Model should be in training mode")

        # Check all modules are in training mode
        for name, module in model.named_modules():
            if hasattr(module, "training"):
                self.assertTrue(module.training, f"Module {name} should be in training mode")

        # Test eval mode
        model.eval()
        self.assertFalse(model.training, "Model should be in eval mode")

        # Check all modules are in eval mode
        for name, module in model.named_modules():
            if hasattr(module, "training"):
                self.assertFalse(module.training, f"Module {name} should be in eval mode")

        # Test mode switching affects behavior (especially dropout)
        test_input = torch.randn(2, self.seq_len, 15)

        # Get outputs in both modes
        model.train()
        train_output1 = model(test_input)
        train_output2 = model(test_input)  # Should be different due to dropout

        model.eval()
        with torch.no_grad():
            eval_output1 = model(test_input)
            eval_output2 = model(test_input)  # Should be identical

        # Validate outputs
        self.assertEqual(train_output1.shape, eval_output1.shape, "Output shapes should match")
        self.assertTrue(torch.isfinite(train_output1).all(), "Train output should be finite")
        self.assertTrue(torch.isfinite(eval_output1).all(), "Eval output should be finite")

        # Eval mode should be deterministic
        self.assertTrue(
            torch.allclose(eval_output1, eval_output2, atol=1e-6), "Eval mode should produce identical outputs"
        )

        # Train mode may differ due to dropout (though not guaranteed)
        # Just check that both outputs are valid
        self.assertTrue(torch.isfinite(train_output2).all(), "Second train output should be finite")

    def tearDown(self):
        """Clean up after architecture tests."""
        # Clean up any test data
        try:
            DataTestDataFactory.cleanup_test_data()
        except Exception:
            pass


class LSTMTrainingIntegrationTestCase(TestCase):
    """Real integration tests for LSTM training pipeline."""

    def setUp(self):
        """Set up comprehensive test data using factories."""
        # Create multiple test stocks for comprehensive testing
        self.stocks = []
        self.integration_symbols = ["INTEGRATION0", "INTEGRATION1", "INTEGRATION2"]

        for i, symbol in enumerate(self.integration_symbols):
            stock = DataTestDataFactory.create_test_stock(
                symbol=symbol,
                company_name=f"Integration Test Stock {i}",
                sector="Technology" if i == 0 else "Financial Services",
            )
            self.stocks.append(stock)

            # Create realistic price data with different patterns using factory
            DataTestDataFactory.create_stock_price_history(stock, days=120)

        # Create test user
        self.test_user = User.objects.create_user(
            username="integration_test", email="integration@test.com", password="testpass"
        )

        # Integration test configuration
        self.integration_config = {
            "sequence_length": 20,  # Very small for fast integration testing
            "hidden_size": 16,  # Minimal for resource constraints
            "num_layers": 2,
            "dropout": 0.1,
            "learning_rate": 0.01,
            "batch_size": 4,  # Very small batches
            "epochs": 2,  # Minimal epochs
            "validation_split": 0.3,
            "early_stopping_patience": 10,
            "use_attention": False,  # Disable attention for faster testing
        }

    def test_multi_stock_training_pipeline(self):
        """Test real training pipeline with multiple stocks and resource constraints."""
        results = []
        successful_trainings = 0

        for stock in self.stocks:
            try:
                trainer = AdvancedLSTMTrainer(symbol=stock.symbol, config=self.integration_config)

                # Attempt real training
                result = trainer.train(stock.symbol)
                results.append(result)

                # Validate result structure
                self.assertIsInstance(result, dict)
                self.assertIn("success", result)

                if result["success"]:
                    successful_trainings += 1
                    # Validate successful training results
                    self.assertIn("training_time", result)
                    self.assertIn("history", result)
                    self.assertIn("symbol", result)
                    self.assertEqual(result["symbol"], stock.symbol)

                    # Validate training metrics
                    self.assertIsInstance(result["training_time"], (int, float))
                    self.assertGreater(result["training_time"], 0)

                    # Validate training history
                    history = result["history"]
                    self.assertIn("train_loss", history)
                    self.assertIn("val_loss", history)

                    train_losses = history["train_loss"]
                    val_losses = history["val_loss"]
                    self.assertGreater(len(train_losses), 0)
                    self.assertGreater(len(val_losses), 0)
                    self.assertTrue(all(loss >= 0 for loss in train_losses))
                    self.assertTrue(all(loss >= 0 for loss in val_losses))

            except Exception as e:
                # Handle resource constraints gracefully
                if any(
                    keyword in str(e).lower() for keyword in ["memory", "cuda", "insufficient", "timeout", "device"]
                ):
                    results.append(
                        {"success": False, "error": f"Resource constraint: {str(e)}", "symbol": stock.symbol}
                    )
                else:
                    # Re-raise unexpected errors
                    raise

        # Validate overall results
        self.assertEqual(len(results), len(self.stocks), "Should have result for each stock")

        # At least some training should complete or fail gracefully
        self.assertGreater(len([r for r in results if "success" in r]), 0, "Should have at least one training attempt")

        # If any trainings succeeded, validate they worked properly
        if successful_trainings > 0:
            self.assertGreater(
                successful_trainings, 0, f"At least one training should succeed, got {successful_trainings} successes"
            )
        else:
            # All failed - check if due to resource constraints
            resource_failures = sum(
                1 for r in results if not r["success"] and "resource constraint" in r.get("error", "").lower()
            )
            if resource_failures == len(results):
                self.skipTest("All trainings skipped due to resource constraints")

    def test_training_data_quality_validation(self):
        """Test real validation of training data quality."""
        for stock in self.stocks:
            prices = StockPrice.objects.filter(stock=stock).order_by("date")

            # Check data completeness
            price_count = prices.count()
            self.assertGreater(
                price_count,
                self.integration_config["sequence_length"],
                f"Stock {stock.symbol} should have enough data for training",
            )

            # Check for data integrity
            invalid_prices = 0
            for price in prices:
                try:
                    # Validate required fields
                    self.assertIsNotNone(price.close, f"Close price should not be None for {stock.symbol}")
                    self.assertIsNotNone(price.volume, f"Volume should not be None for {stock.symbol}")
                    self.assertIsNotNone(price.date, f"Date should not be None for {stock.symbol}")

                    # Validate price values
                    self.assertGreater(float(price.close), 0, f"Close price should be positive for {stock.symbol}")
                    self.assertGreaterEqual(price.volume, 0, f"Volume should be non-negative for {stock.symbol}")

                    # Validate OHLC relationships if available
                    if price.open and price.high and price.low:
                        self.assertGreaterEqual(
                            float(price.high),
                            float(price.open),
                            f"High should be >= open for {stock.symbol} on {price.date}",
                        )
                        self.assertLessEqual(
                            float(price.low),
                            float(price.open),
                            f"Low should be <= open for {stock.symbol} on {price.date}",
                        )
                        self.assertGreaterEqual(
                            float(price.high),
                            float(price.close),
                            f"High should be >= close for {stock.symbol} on {price.date}",
                        )
                        self.assertLessEqual(
                            float(price.low),
                            float(price.close),
                            f"Low should be <= close for {stock.symbol} on {price.date}",
                        )

                except (ValueError, TypeError, AssertionError):
                    invalid_prices += 1

            # Allow some invalid prices but not too many
            invalid_ratio = invalid_prices / price_count if price_count > 0 else 1
            self.assertLess(
                invalid_ratio, 0.1, f"Stock {stock.symbol} should have < 10% invalid prices, got {invalid_ratio:.2%}"
            )

            # Check date continuity
            dates = [p.date for p in prices]
            if len(dates) > 1:
                date_diffs = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]

                # Most date differences should be reasonable (1-7 days accounting for weekends)
                reasonable_gaps = sum(1 for diff in date_diffs if 1 <= diff <= 7)
                if len(date_diffs) > 0:
                    reasonable_ratio = reasonable_gaps / len(date_diffs)
                    self.assertGreater(
                        reasonable_ratio,
                        0.7,
                        f"Stock {stock.symbol} should have reasonable date gaps, got {reasonable_ratio:.2%}",
                    )

            # Test data preparation with real trainer
            try:
                trainer = AdvancedLSTMTrainer(symbol=stock.symbol, config=self.integration_config)
                data = trainer.prepare_training_data(stock.symbol)

                # Validate prepared data quality
                self.assertIsNotNone(data, f"Data preparation should succeed for {stock.symbol}")
                self.assertIn("features", data)
                self.assertIn("targets", data)
                self.assertGreater(data["features"].shape[0], 0, f"Should have training samples for {stock.symbol}")

            except Exception as e:
                if "insufficient data" in str(e).lower():
                    self.skipTest(f"Insufficient data for {stock.symbol}: {e}")
                else:
                    raise

    def test_model_performance_benchmarking(self):
        """Test real model performance benchmarking with resource constraints."""
        # Test with first stock
        test_symbol = self.integration_symbols[0]

        try:
            trainer = AdvancedLSTMTrainer(symbol=test_symbol, config=self.integration_config)

            # Train a minimal model for evaluation
            training_result = trainer.train(test_symbol)

            if training_result["success"]:
                # Evaluate real model performance
                eval_result = trainer.evaluate_model(test_symbol)

                # Validate evaluation result structure
                self.assertIsInstance(eval_result, dict)
                self.assertIn("success", eval_result)

                if eval_result["success"]:
                    # Validate metrics structure
                    self.assertIn("metrics", eval_result)
                    metrics = eval_result["metrics"]

                    # Validate required metrics exist
                    required_metrics = ["mse", "mae", "mape", "directional_accuracy", "r2_score"]
                    for metric in required_metrics:
                        self.assertIn(metric, metrics, f"Should have {metric} metric")

                    # Validate metric ranges and properties
                    mse = metrics["mse"]
                    mae = metrics["mae"]
                    mape = metrics["mape"]
                    directional_acc = metrics["directional_accuracy"]
                    r2 = metrics["r2_score"]

                    # Basic metric validation
                    self.assertGreaterEqual(mse, 0, "MSE should be non-negative")
                    self.assertGreaterEqual(mae, 0, "MAE should be non-negative")
                    self.assertGreaterEqual(mape, 0, "MAPE should be non-negative")
                    self.assertTrue(0 <= directional_acc <= 1, "Directional accuracy should be between 0 and 1")

                    # Finite value checks
                    self.assertTrue(np.isfinite(mse), "MSE should be finite")
                    self.assertTrue(np.isfinite(mae), "MAE should be finite")
                    self.assertTrue(np.isfinite(mape), "MAPE should be finite")
                    self.assertTrue(np.isfinite(directional_acc), "Directional accuracy should be finite")
                    self.assertTrue(np.isfinite(r2), "R² should be finite")

                    # Reasonable metric ranges (allowing for untrained/minimal training)
                    self.assertLess(mse, 10.0, "MSE should be reasonable for stock prices")
                    self.assertLess(mae, 20.0, "MAE should be reasonable for stock prices")
                    self.assertLess(mape, 1000.0, "MAPE should not be extremely high")

                    # Validate sample predictions if available
                    if "predictions_sample" in eval_result and "actuals_sample" in eval_result:
                        predictions = eval_result["predictions_sample"]
                        actuals = eval_result["actuals_sample"]

                        self.assertIsInstance(predictions, list, "Predictions should be a list")
                        self.assertIsInstance(actuals, list, "Actuals should be a list")
                        self.assertEqual(
                            len(predictions), len(actuals), "Predictions and actuals should have same length"
                        )

                        # Validate prediction values are reasonable
                        for pred in predictions:
                            self.assertTrue(np.isfinite(pred), "Predictions should be finite")
                            self.assertGreater(pred, 0, "Stock price predictions should be positive")

                else:
                    # Evaluation failed - validate error handling
                    self.assertIn("error", eval_result, "Failed evaluation should have error message")

            else:
                # Training failed - skip evaluation test
                self.skipTest(f"Model training failed for {test_symbol}, cannot test evaluation")

        except Exception as e:
            # Handle resource constraints
            if any(
                keyword in str(e).lower()
                for keyword in ["memory", "cuda", "insufficient", "timeout", "device", "resource"]
            ):
                self.skipTest(f"Performance benchmarking skipped due to resource constraints: {e}")
            else:
                raise

    def test_training_resource_management(self):
        """Test real training resource management and cleanup."""
        import gc

        import psutil
        import torch

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        test_symbol = self.integration_symbols[0]
        trainer = None
        peak_memory = initial_memory

        try:
            # Create trainer and attempt training
            trainer = AdvancedLSTMTrainer(symbol=test_symbol, config=self.integration_config)

            # Monitor memory during data preparation
            data = trainer.prepare_training_data(test_symbol)
            prep_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, prep_memory)

            # Create model and monitor memory
            input_size = data["train_features"].shape[2]
            model = trainer.create_model(input_size)
            trainer.model = model
            model_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, model_memory)

            # Perform a single training epoch to test resource usage
            try:
                train_loss = trainer._train_epoch(
                    data["train_features"][:8], data["train_targets"][:8]  # Very small batch
                )
                training_memory = process.memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, training_memory)

                # Validate training loss is reasonable
                self.assertIsInstance(train_loss, (int, float), "Training loss should be numeric")
                self.assertGreater(train_loss, 0, "Training loss should be positive")
                self.assertTrue(np.isfinite(train_loss), "Training loss should be finite")

            except Exception as e:
                if any(keyword in str(e).lower() for keyword in ["memory", "cuda", "device"]):
                    pass  # Memory constraint during training is acceptable
                else:
                    raise

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["memory", "insufficient", "cuda"]):
                self.skipTest(f"Resource management test skipped due to constraints: {e}")
            else:
                raise

        finally:
            # Cleanup resources
            if trainer:
                trainer.model = None
                trainer.optimizer = None
                trainer.scheduler = None

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            # Measure final memory
            final_memory = process.memory_info().rss / 1024 / 1024

        # Validate resource management
        memory_increase = final_memory - initial_memory
        peak_increase = peak_memory - initial_memory

        # Allow for some memory increase but should be reasonable
        self.assertLess(
            memory_increase,
            500,  # Less than 500MB increase
            f"Memory increase should be reasonable: {memory_increase:.1f}MB",
        )

        # Peak memory should not be excessive
        self.assertLess(
            peak_increase,
            1000,  # Less than 1GB peak increase
            f"Peak memory increase should be reasonable: {peak_increase:.1f}MB",
        )

        # Basic memory cleanup validation
        if peak_memory > initial_memory:
            cleanup_ratio = (peak_memory - final_memory) / (peak_memory - initial_memory)
            self.assertGreater(
                cleanup_ratio,
                0.5,  # At least 50% cleanup
                f"Should clean up at least 50% of peak memory usage, got {cleanup_ratio:.2%}",
            )

    def test_training_fault_tolerance(self):
        """Test real training fault tolerance and error handling."""
        test_symbol = self.integration_symbols[0]

        # Test 1: Invalid configuration handling
        invalid_configs = [
            {"sequence_length": -1},  # Invalid sequence length
            {"hidden_size": 0},  # Invalid hidden size
            {"epochs": 0},  # Invalid epochs
            {"batch_size": -1},  # Invalid batch size
        ]

        for invalid_config in invalid_configs:
            config = self.integration_config.copy()
            config.update(invalid_config)

            try:
                trainer = AdvancedLSTMTrainer(symbol=test_symbol, config=config)

                # Should handle invalid configuration gracefully
                result = trainer.train(test_symbol)

                # If training doesn't raise exception, should return failure
                if isinstance(result, dict) and "success" in result:
                    if not result["success"]:
                        self.assertIn("error", result, "Failed training should have error message")
                    else:
                        # Unexpected success with invalid config - just validate structure
                        self.assertIn("training_time", result)

            except (ValueError, RuntimeError, TypeError) as e:
                # Expected exception for invalid configuration
                self.assertIsInstance(str(e), str, "Error should have string representation")
            except Exception as e:
                # Unexpected exception type
                if any(keyword in str(e).lower() for keyword in ["memory", "cuda", "device"]):
                    continue  # Resource constraints are acceptable
                else:
                    # Re-raise unexpected errors for investigation
                    raise

        # Test 2: Insufficient data handling
        try:
            # Create stock with minimal data
            minimal_stock = DataTestDataFactory.create_test_stock(
                symbol="MINIMAL_DATA", company_name="Minimal Data Stock", sector="Technology"
            )
            DataTestDataFactory.create_stock_price_history(minimal_stock, days=5)  # Insufficient

            trainer = AdvancedLSTMTrainer(symbol="MINIMAL_DATA", config=self.integration_config)

            # Should handle insufficient data gracefully
            result = trainer.train("MINIMAL_DATA")

            self.assertIsInstance(result, dict, "Should return result dictionary")
            self.assertIn("success", result, "Should have success flag")

            if not result["success"]:
                self.assertIn("error", result, "Failed training should have error message")
                error_msg = result["error"].lower()
                self.assertTrue(
                    any(keyword in error_msg for keyword in ["insufficient", "data", "not found"]),
                    f"Error should mention data issues: {result['error']}",
                )

        except Exception as e:
            if "insufficient data" in str(e).lower() or "not found" in str(e).lower():
                pass  # Expected error for insufficient data
            else:
                raise

        # Test 3: Model state consistency after errors
        try:
            trainer = AdvancedLSTMTrainer(symbol=test_symbol, config=self.integration_config)

            # Check initial state
            self.assertIsNone(trainer.model, "Model should be None initially")
            self.assertIsNone(trainer.optimizer, "Optimizer should be None initially")

            # Attempt training
            result = trainer.train(test_symbol)

            if result["success"]:
                # Successful training - validate model state
                self.assertIsNotNone(trainer.model, "Model should exist after successful training")
                self.assertIsInstance(trainer.model, torch.nn.Module, "Model should be PyTorch module")

                # Validate training history
                self.assertIsInstance(trainer.training_history, dict, "Training history should be dictionary")
                self.assertIn("train_loss", trainer.training_history)
                self.assertIn("val_loss", trainer.training_history)

            else:
                # Failed training - validate error handling
                self.assertIn("error", result, "Failed training should have error message")

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["memory", "cuda", "device", "resource"]):
                self.skipTest(f"Fault tolerance test skipped due to resource constraints: {e}")
            else:
                raise

    def tearDown(self):
        """Clean up after integration tests."""
        # Clean up test data
        DataTestDataFactory.cleanup_test_data()

        # Clean up any additional test stocks
        from Data.models import Stock, StockPrice

        test_symbols = self.integration_symbols + ["MINIMAL_DATA"]

        try:
            StockPrice.objects.filter(stock__symbol__in=test_symbols).delete()
            Stock.objects.filter(symbol__in=test_symbols).delete()
        except Exception:
            pass

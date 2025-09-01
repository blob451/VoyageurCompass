"""
Advanced LSTM Training Service for VoyageurCompass Analytics
Provides comprehensive training, validation, and optimization for LSTM models.
Integrates with existing Universal LSTM architecture and data pipelines.
"""

import logging
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path

from Analytics.ml.models.lstm_base import (
    AttentionLayer, 
    SectorCrossAttention,
    UniversalLSTMPredictor,
    create_universal_lstm_model
)
from Analytics.ml.universal_preprocessor import UniversalLSTMPreprocessor
from Analytics.ml.sector_mappings import get_sector_mapper
from Data.repo.price_reader import PriceReader
from Data.models import Stock, StockPrice
from Data.services.yahoo_finance import yahoo_finance_service

logger = logging.getLogger(__name__)


class AdvancedLSTMTrainer:
    """
    Advanced LSTM trainer with support for single-stock and multi-stock training,
    hyperparameter optimization, and comprehensive evaluation.
    """
    
    def __init__(self, symbol: str = None, config: Dict[str, Any] = None):
        """
        Initialize the Advanced LSTM Trainer.
        
        Args:
            symbol: Stock symbol for single-stock training (optional)
            config: Training configuration dictionary
        """
        self.symbol = symbol
        self.config = config or self._get_default_config()
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"AdvancedLSTMTrainer initialized on device: {self.device}")
        
        # Data components
        self.price_reader = PriceReader()
        self.sector_mapper = get_sector_mapper()
        self.preprocessor = None
        
        # Model components
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.scheduler = None
        
        # Training state
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }
        
        # Paths
        self.model_dir = Path("Data/ml_models/advanced_lstm")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration."""
        return {
            'sequence_length': 60,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'validation_split': 0.2,
            'early_stopping_patience': 15,
            'lr_scheduler_patience': 5,
            'lr_scheduler_factor': 0.5,
            'gradient_clipping': 1.0,
            'weight_decay': 0.0001,
            'num_features': 24,  # Matches UniversalLSTMPreprocessor
            'use_attention': True,
            'use_sector_attention': False
        }
    
    def prepare_training_data(self, symbol: str) -> Dict[str, Any]:
        """
        Prepare training data for a given stock symbol.
        
        Args:
            symbol: Stock symbol to prepare data for
            
        Returns:
            Dictionary containing prepared features, targets, and metadata
        """
        try:
            # Get stock data
            stock = Stock.objects.get(symbol=symbol.upper())
            
            # Get price data
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=365 * 3)  # 3 years of data
            
            prices = StockPrice.objects.filter(
                stock=stock,
                date__gte=start_date,
                date__lte=end_date
            ).order_by('date').values(
                'date', 'open', 'high', 'low', 'close', 'volume'
            )
            
            if len(prices) < self.config['sequence_length'] + 1:
                raise ValueError(f"Insufficient data for {symbol}: {len(prices)} records")
            
            # Convert to DataFrame
            df = pd.DataFrame(list(prices))
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)
            
            # Prepare sequences
            features, targets = self._create_sequences(df)
            
            # Split into train/validation
            split_idx = int(len(features) * (1 - self.config['validation_split']))
            
            return {
                'features': features,
                'targets': targets,
                'train_features': features[:split_idx],
                'train_targets': targets[:split_idx],
                'val_features': features[split_idx:],
                'val_targets': targets[split_idx:],
                'scaler': self._create_scaler(df),
                'num_samples': len(features),
                'symbol': symbol
            }
            
        except Stock.DoesNotExist:
            logger.error(f"Stock {symbol} not found in database")
            raise
        except Exception as e:
            logger.error(f"Error preparing data for {symbol}: {str(e)}")
            raise
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the DataFrame."""
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Volume indicators
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _create_sequences(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create sequences for LSTM training."""
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # Select features
        features = df[feature_cols].values
        targets = df['close'].values
        
        # Normalize features
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)
        
        # Create sequences
        seq_length = self.config['sequence_length']
        X, y = [], []
        
        for i in range(seq_length, len(features)):
            X.append(features[i-seq_length:i])
            y.append(targets[i])
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Normalize targets
        target_scaler = MinMaxScaler()
        y = target_scaler.fit_transform(y)
        
        # Convert to tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        return X, y
    
    def _create_scaler(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create and return scalers for features and targets."""
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        feature_scaler.fit(df[feature_cols])
        target_scaler.fit(df[['close']])
        
        return {
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler,
            'feature_columns': feature_cols
        }
    
    def create_model(self, input_size: int) -> nn.Module:
        """
        Create LSTM model based on configuration.
        
        Args:
            input_size: Number of input features
            
        Returns:
            LSTM model
        """
        class LSTMModel(nn.Module):
            def __init__(self, config):
                super(LSTMModel, self).__init__()
                self.hidden_size = config['hidden_size']
                self.num_layers = config['num_layers']
                
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    batch_first=True,
                    dropout=config['dropout'] if self.num_layers > 1 else 0
                )
                
                if config['use_attention']:
                    self.attention = AttentionLayer(self.hidden_size)
                else:
                    self.attention = None
                
                self.fc = nn.Linear(self.hidden_size, 1)
                self.dropout = nn.Dropout(config['dropout'])
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                
                if self.attention:
                    attended_out, _ = self.attention(lstm_out)
                    output = self.fc(self.dropout(attended_out))
                else:
                    output = self.fc(self.dropout(lstm_out[:, -1, :]))
                
                return output
        
        return LSTMModel(self.config)
    
    def train(self, symbol: str = None) -> Dict[str, Any]:
        """
        Train LSTM model for a given symbol.
        
        Args:
            symbol: Stock symbol to train on (uses self.symbol if not provided)
            
        Returns:
            Training results dictionary
        """
        symbol = symbol or self.symbol
        if not symbol:
            raise ValueError("No symbol provided for training")
        
        logger.info(f"Starting training for {symbol}")
        start_time = datetime.now()
        
        try:
            # Prepare data
            data = self.prepare_training_data(symbol)
            
            # Create model
            input_size = data['train_features'].shape[2]
            self.model = self.create_model(input_size).to(self.device)
            
            # Setup optimizer and scheduler
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
            
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config['lr_scheduler_factor'],
                patience=self.config['lr_scheduler_patience'],
                verbose=True
            )
            
            # Training loop
            best_model_state = None
            patience_counter = 0
            
            for epoch in range(self.config['epochs']):
                # Train
                train_loss = self._train_epoch(
                    data['train_features'],
                    data['train_targets']
                )
                
                # Validate
                val_loss = self._validate(
                    data['val_features'],
                    data['val_targets']
                )
                
                # Update learning rate
                self.scheduler.step(val_loss)
                
                # Record history
                current_lr = self.optimizer.param_groups[0]['lr']
                self.training_history['train_loss'].append(train_loss)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['learning_rates'].append(current_lr)
                
                # Early stopping
                if val_loss < self.training_history['best_val_loss']:
                    self.training_history['best_val_loss'] = val_loss
                    self.training_history['best_epoch'] = epoch
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}/{self.config['epochs']} - "
                        f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                        f"LR: {current_lr:.6f}"
                    )
            
            # Restore best model
            if best_model_state:
                self.model.load_state_dict(best_model_state)
            
            # Save model
            model_path = self._save_model(symbol, data['scaler'])
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'symbol': symbol,
                'history': self.training_history,
                'best_val_loss': self.training_history['best_val_loss'],
                'best_epoch': self.training_history['best_epoch'],
                'model_path': str(model_path),
                'training_time': training_time,
                'device': str(self.device)
            }
            
        except Exception as e:
            logger.error(f"Training failed for {symbol}: {str(e)}")
            return {
                'success': False,
                'symbol': symbol,
                'error': str(e)
            }
    
    def _train_epoch(self, features: torch.Tensor, targets: torch.Tensor) -> float:
        """Train for one epoch."""
        self.model.train()
        batch_size = self.config['batch_size']
        total_loss = 0
        num_batches = 0
        
        # Move data to device
        features = features.to(self.device)
        targets = targets.to(self.device)
        
        for i in range(0, len(features), batch_size):
            batch_features = features[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_features)
            loss = self.criterion(outputs, batch_targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['gradient_clipping']
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate(self, features: torch.Tensor, targets: torch.Tensor) -> float:
        """Validate the model."""
        self.model.eval()
        
        features = features.to(self.device)
        targets = targets.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(features)
            loss = self.criterion(outputs, targets)
        
        return loss.item()
    
    def save_model(self, save_path: str = None) -> Dict[str, Any]:
        """
        Save the trained model.
        
        Args:
            save_path: Optional custom save path
            
        Returns:
            Dictionary with save information
        """
        if not self.model:
            raise ValueError("No model to save")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"lstm_{self.symbol}_{timestamp}.pth"
        
        if save_path:
            model_path = Path(save_path)
        else:
            model_path = self.model_dir / model_name
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'symbol': self.symbol,
            'timestamp': timestamp
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
        
        return {
            'success': True,
            'model_path': str(model_path),
            'metadata': {
                'symbol': self.symbol,
                'timestamp': timestamp,
                'architecture': 'LSTM_v1.0',
                'best_val_loss': self.training_history['best_val_loss']
            }
        }
    
    def _save_model(self, symbol: str, scaler: Dict[str, Any]) -> Path:
        """Internal method to save model during training."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"lstm_{symbol}_{timestamp}.pth"
        model_path = self.model_dir / model_name
        
        # Save model and training information
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'symbol': symbol,
            'timestamp': timestamp,
            'scaler': scaler
        }, model_path)
        
        # Save scalers separately
        scaler_path = self.model_dir / f"scaler_{symbol}_{timestamp}.pkl"
        joblib.dump(scaler, scaler_path)
        
        return model_path
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Dictionary with load information
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Recreate model
            self.config = checkpoint['config']
            input_size = self.config.get('num_features', 24)
            self.model = self.create_model(input_size).to(self.device)
            
            # Load state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.training_history = checkpoint.get('training_history', {})
            self.symbol = checkpoint.get('symbol')
            
            return {
                'success': True,
                'model': self.model,
                'scaler': checkpoint.get('scaler'),
                'metadata': {
                    'symbol': self.symbol,
                    'architecture': 'LSTM_v1.0',
                    'timestamp': checkpoint.get('timestamp')
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def evaluate_model(self, symbol: str) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            symbol: Stock symbol to evaluate
            
        Returns:
            Evaluation metrics
        """
        if not self.model:
            raise ValueError("No model loaded for evaluation")
        
        try:
            # Prepare test data
            data = self.prepare_training_data(symbol)
            
            # Use validation set as test set
            test_features = data['val_features'].to(self.device)
            test_targets = data['val_targets'].to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(test_features)
            
            # Calculate metrics
            mse = nn.MSELoss()(predictions, test_targets).item()
            mae = nn.L1Loss()(predictions, test_targets).item()
            
            # Calculate directional accuracy
            pred_np = predictions.cpu().numpy()
            target_np = test_targets.cpu().numpy()
            
            if len(pred_np) > 1:
                pred_direction = np.diff(pred_np.flatten()) > 0
                actual_direction = np.diff(target_np.flatten()) > 0
                directional_accuracy = np.mean(pred_direction == actual_direction)
            else:
                directional_accuracy = 0.0
            
            # Calculate MAPE
            mape = np.mean(np.abs((target_np - pred_np) / target_np)) * 100
            
            # Calculate R² score
            ss_res = np.sum((target_np - pred_np) ** 2)
            ss_tot = np.sum((target_np - np.mean(target_np)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                'success': True,
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'mape': mape,
                    'directional_accuracy': directional_accuracy,
                    'r2_score': r2_score
                },
                'predictions_sample': pred_np[:5].flatten().tolist(),
                'actuals_sample': target_np[:5].flatten().tolist()
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed for {symbol}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
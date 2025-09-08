"""
LSTM Base Model for Stock Price Prediction
Implements Long Short-Term Memory neural network for time series forecasting.
"""

import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, TYPE_CHECKING
import os
import joblib
from datetime import datetime
import math

# Conditional imports for ML dependencies to support CI environments
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

# Type hints for when torch is available
if TYPE_CHECKING and TORCH_AVAILABLE:
    from torch import Tensor, device
else:
    Tensor = Any
    device = Any

logger = logging.getLogger(__name__)

# Early exit if dependencies not available
if not TORCH_AVAILABLE:
    logger.warning("PyTorch not available - LSTM models disabled")

    # Fallback classes when PyTorch unavailable
    class SectorCrossAttention:
        """Fallback class for when PyTorch unavailable."""
        def __init__(self, *args, **kwargs):
            pass

    class AttentionLayer:
        """Fallback class for when PyTorch unavailable."""
        def __init__(self, *args, **kwargs):
            pass

    class MultiHeadAttention:
        """Fallback class for when PyTorch unavailable."""  
        def __init__(self, *args, **kwargs):
            pass

    class LSTMPricePredictor:
        """Fallback class for when PyTorch unavailable."""
        def __init__(self, *args, **kwargs):
            pass

        def get_model_info(self):
            return {"error": "PyTorch not available"}

    class UniversalLSTMPredictor:
        """Fallback class for when PyTorch unavailable."""
        def __init__(self, *args, **kwargs):
            pass

        def get_model_info(self):
            return {"error": "PyTorch not available"}

    class LSTMTrainingConfig:
        """Fallback class for when PyTorch unavailable."""
        def __init__(self, *args, **kwargs):
            pass

        def to_dict(self):
            return {"error": "PyTorch not available"}

    class DirectionalLoss:
        """Fallback class for when PyTorch unavailable."""
        def __init__(self, *args, **kwargs):
            pass

    class VolatilityAwareLoss:
        """Fallback class for when PyTorch unavailable."""
        def __init__(self, *args, **kwargs):
            pass

    # Fallback functions when PyTorch unavailable
    def save_model(*args, **kwargs):
        """Fallback function for when PyTorch unavailable."""
        logger.error("Cannot save model - PyTorch not available")
        return None

    def load_model(*args, **kwargs):
        """Fallback function for when PyTorch unavailable."""
        logger.error("Cannot load model - PyTorch not available")
        return None, {}, {}, {}

    def create_enhanced_lstm_model(*args, **kwargs):
        """Fallback function for when PyTorch unavailable."""
        logger.error("Cannot create LSTM model - PyTorch not available")
        return LSTMPricePredictor()

    def save_universal_model(*args, **kwargs):
        """Fallback function for when PyTorch unavailable."""
        logger.error("Cannot save universal model - PyTorch not available")  
        return None

    def create_universal_lstm_model(*args, **kwargs):
        """Fallback function for when PyTorch unavailable."""
        logger.error("Cannot create universal LSTM model - PyTorch not available")
        return UniversalLSTMPredictor()

    def get_model_complexity_score(*args, **kwargs):
        """Fallback function for when PyTorch unavailable."""
        return {'error': 'PyTorch not available'}

    def _recreate_scalers_from_file(*args, **kwargs):
        """Fallback function for when PyTorch unavailable."""
        logger.error("Cannot recreate scalers - PyTorch not available")
        return {}

else:
    # Full PyTorch implementations when available
    class SectorCrossAttention(nn.Module):
        """Cross-attention mechanism that allows the model to focus on sector-specific patterns."""

        def __init__(self, hidden_size: int, sector_embedding_dim: int):
            super(SectorCrossAttention, self).__init__()
            self.hidden_size = hidden_size
            self.sector_embedding_dim = sector_embedding_dim

            # Cross-attention layers
            self.query_projection = nn.Linear(hidden_size, hidden_size)
            self.key_projection = nn.Linear(sector_embedding_dim, hidden_size)
            self.value_projection = nn.Linear(sector_embedding_dim, hidden_size)

            # Output projection
            self.output_projection = nn.Linear(hidden_size, hidden_size)

        def forward(self, lstm_output: Tensor, sector_embedding: Tensor) -> Tensor:
            """Apply cross-attention between LSTM output and sector embedding."""
            batch_size, seq_len, _ = lstm_output.shape

            # Expand sector embedding to match sequence length
            sector_expanded = sector_embedding.unsqueeze(1).expand(-1, seq_len, -1)

            # Project inputs
            query = self.query_projection(lstm_output)
            key = self.key_projection(sector_expanded)
            value = self.value_projection(sector_expanded)

            # Compute attention scores
            attention_scores = torch.bmm(query, key.transpose(-2, -1)) / math.sqrt(self.hidden_size)
            attention_weights = F.softmax(attention_scores, dim=-1)

            # Apply attention
            attended_output = torch.bmm(attention_weights, value)

            # Residual connection and output projection
            output = self.output_projection(attended_output + lstm_output)

            return output

    class AttentionLayer(nn.Module):
        """Attention mechanism for LSTM outputs to focus on important time steps."""

        def __init__(self, hidden_size: int):
            super(AttentionLayer, self).__init__()
            self.hidden_size = hidden_size
            self.attn = nn.Linear(hidden_size, 1, bias=False)

        def forward(self, lstm_outputs: Tensor) -> Tuple[Tensor, Tensor]:
            """Apply attention to LSTM outputs."""
            # Calculate attention scores
            attn_scores = self.attn(lstm_outputs)
            attn_weights = F.softmax(attn_scores, dim=1)

            # Apply attention weights
            attended_output = torch.sum(lstm_outputs * attn_weights, dim=1)

            return attended_output, attn_weights.squeeze(-1)

    class MultiHeadAttention(nn.Module):
        """Multi-head attention mechanism for better pattern recognition."""

        def __init__(self, hidden_size: int, num_heads: int = 4):
            super(MultiHeadAttention, self).__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads

            assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

            self.query = nn.Linear(hidden_size, hidden_size)
            self.key = nn.Linear(hidden_size, hidden_size)
            self.value = nn.Linear(hidden_size, hidden_size)
            self.output = nn.Linear(hidden_size, hidden_size)

        def forward(self, x: Tensor) -> Tensor:
            """Apply multi-head attention."""
            batch_size, seq_len, _ = x.size()

            # Generate Q, K, V
            Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(scores, dim=-1)

            # Apply attention
            attended = torch.matmul(attn_weights, V)
            attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

            return self.output(attended)

    class LSTMPricePredictor(nn.Module):
        """Enhanced LSTM model for stock price prediction with 30-day sequence length."""

        def __init__(self, input_size: int = 38, hidden_size: int = 128, num_layers: int = 2, 
                     dropout: float = 0.3, sequence_length: int = 30, bidirectional: bool = True,
                     use_attention: bool = True, attention_type: str = 'single'):
            super(LSTMPricePredictor, self).__init__()

            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.sequence_length = sequence_length
            self.bidirectional = bidirectional
            self.use_attention = use_attention
            self.attention_type = attention_type

            # Calculate effective hidden size (double for bidirectional)
            self.effective_hidden_size = hidden_size * (2 if bidirectional else 1)

            # LSTM layer with bidirectional support
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirectional
            )

            # Attention mechanism
            if use_attention:
                if attention_type == 'multi':
                    self.attention = MultiHeadAttention(self.effective_hidden_size, num_heads=4)
                    self.attention_pool = AttentionLayer(self.effective_hidden_size)
                else:
                    self.attention = AttentionLayer(self.effective_hidden_size)

            # Dropout for regularization
            self.dropout = nn.Dropout(dropout)

            # Additional layers for better representation
            self.fc_intermediate = nn.Linear(self.effective_hidden_size, hidden_size)
            self.fc_output = nn.Linear(hidden_size, 1)

            # Model metadata  
            self.model_version = "3.0.0"
            self.created_at = datetime.now()

        def forward(self, x: Tensor) -> Tensor:
            """Enhanced forward pass through the bidirectional LSTM model with attention."""
            # Initialize hidden state (account for bidirectional)
            num_directions = 2 if self.bidirectional else 1
            h0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(x.device)

            # LSTM forward pass
            lstm_out, _ = self.lstm(x, (h0, c0))

            # Apply attention mechanism
            if self.use_attention:
                if self.attention_type == 'multi':
                    attended_features = self.attention(lstm_out)
                    final_output, attention_weights = self.attention_pool(attended_features)
                else:
                    final_output, attention_weights = self.attention(lstm_out)
            else:
                final_output = lstm_out[:, -1, :]

            # Apply dropout
            final_output = self.dropout(final_output)

            # Intermediate layer with activation
            intermediate = F.relu(self.fc_intermediate(final_output))
            intermediate = self.dropout(intermediate)

            # Final prediction
            prediction = self.fc_output(intermediate)

            return prediction

        def get_model_info(self) -> Dict[str, Any]:
            """Get model architecture and metadata information."""
            return {
                'model_type': 'LSTM',
                'version': self.model_version,
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'sequence_length': self.sequence_length,
                'total_params': sum(p.numel() for p in self.parameters()),
                'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'created_at': self.created_at.isoformat() if self.created_at else None
            }

    class UniversalLSTMPredictor(nn.Module):
        """Universal LSTM model for stock price prediction with sector-differentiation features."""

        def __init__(self, input_size: int = 25, sector_embedding_dim: int = 12,
                     industry_embedding_dim: int = 6, hidden_size: int = 128, num_layers: int = 3,
                     dropout: float = 0.3, sequence_length: int = 60, num_sectors: int = 11,
                     num_industries: int = 50, use_cross_attention: bool = False,
                     multi_task_output: bool = True):
            super(UniversalLSTMPredictor, self).__init__()

            # Store configuration
            self.input_size = input_size
            self.sector_embedding_dim = sector_embedding_dim
            self.industry_embedding_dim = industry_embedding_dim
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.sequence_length = sequence_length
            self.num_sectors = num_sectors
            self.num_industries = num_industries
            self.use_cross_attention = use_cross_attention
            self.multi_task_output = multi_task_output

            # Embedding layers for sector and industry
            self.sector_embedding = nn.Embedding(num_sectors, sector_embedding_dim)
            self.industry_embedding = nn.Embedding(num_industries, industry_embedding_dim)

            # Combined feature size
            combined_input_size = input_size + sector_embedding_dim + industry_embedding_dim

            # Bidirectional LSTM layers
            self.lstm = nn.LSTM(
                input_size=combined_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )

            # Effective hidden size (doubled for bidirectional)
            self.effective_hidden_size = hidden_size * 2

            # Attention pooling for sequence aggregation
            self.attention_pooling = AttentionLayer(self.effective_hidden_size)

            # Dropout layers
            self.dropout = nn.Dropout(dropout)

            # Price normalization layer
            self.price_norm_layer = nn.LayerNorm(self.effective_hidden_size)

            # Main price prediction head
            self.price_head = nn.Sequential(
                nn.Linear(self.effective_hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1)
            )

            # Model metadata
            self.model_version = "4.0.0"
            self.created_at = datetime.now()

            # Initialize embeddings with Xavier uniform
            nn.init.xavier_uniform_(self.sector_embedding.weight)
            nn.init.xavier_uniform_(self.industry_embedding.weight)

        def forward(self, x: Tensor, sector_ids: Tensor, 
                   industry_ids: Tensor) -> Dict[str, Tensor]:
            """Forward pass through the Universal LSTM model."""
            batch_size, seq_len, _ = x.shape

            # Get embeddings
            sector_emb = self.sector_embedding(sector_ids)
            industry_emb = self.industry_embedding(industry_ids)

            # Expand embeddings to sequence length
            sector_emb_expanded = sector_emb.unsqueeze(1).expand(-1, seq_len, -1)
            industry_emb_expanded = industry_emb.unsqueeze(1).expand(-1, seq_len, -1)

            # Concatenate features with embeddings
            combined_input = torch.cat([x, sector_emb_expanded, industry_emb_expanded], dim=-1)

            # LSTM forward pass
            h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)

            lstm_out, _ = self.lstm(combined_input, (h0, c0))

            # Apply attention pooling to aggregate sequence
            final_features, attention_weights = self.attention_pooling(lstm_out)

            # Apply price normalization and dropout
            final_features = self.price_norm_layer(final_features)
            final_features = self.dropout(final_features)

            # Generate predictions
            outputs = {}

            # Main price prediction
            price_pred = self.price_head(final_features)
            outputs['price'] = price_pred

            # Store attention weights for interpretability
            outputs['attention_weights'] = attention_weights

            return outputs

        def get_model_info(self) -> Dict[str, Any]:
            """Get universal model architecture and metadata information."""
            return {
                'model_type': 'UniversalLSTM',
                'version': self.model_version,
                'input_size': self.input_size,
                'sector_embedding_dim': self.sector_embedding_dim,
                'industry_embedding_dim': self.industry_embedding_dim,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'sequence_length': self.sequence_length,
                'num_sectors': self.num_sectors,
                'num_industries': self.num_industries,
                'use_cross_attention': self.use_cross_attention,
                'multi_task_output': self.multi_task_output,
                'total_params': sum(p.numel() for p in self.parameters()),
                'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'created_at': self.created_at.isoformat() if self.created_at else None
            }

    class LSTMTrainingConfig:
        """Configuration class for LSTM training parameters."""

        def __init__(self, learning_rate: float = 0.001, batch_size: int = 32,
                     num_epochs: int = 100, early_stopping_patience: int = 10,
                     validation_split: float = 0.15, test_split: float = 0.15,
                     sequence_length: int = 30, forecast_horizon: int = 1,
                     min_training_samples: int = 400):
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.early_stopping_patience = early_stopping_patience
            self.validation_split = validation_split
            self.test_split = test_split
            self.sequence_length = sequence_length
            self.forecast_horizon = forecast_horizon
            self.min_training_samples = min_training_samples
            self.train_split = 1.0 - validation_split - test_split

        def to_dict(self) -> Dict[str, Any]:
            """Convert configuration to dictionary."""
            return {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'early_stopping_patience': self.early_stopping_patience,
                'validation_split': self.validation_split,
                'test_split': self.test_split,
                'train_split': self.train_split,
                'sequence_length': self.sequence_length,
                'forecast_horizon': self.forecast_horizon,
                'min_training_samples': self.min_training_samples
            }

    class DirectionalLoss(nn.Module):
        """Production-ready loss function that prevents model collapse through diversity rewards."""

        def __init__(self, mse_weight: float = 0.4, direction_weight: float = 0.3, 
                     diversity_weight: float = 0.2, sector_weight: float = 0.1,
                     min_diversity_target: float = 0.001, diversity_epsilon: float = 1e-6):
            super(DirectionalLoss, self).__init__()
            self.mse_weight = mse_weight
            self.direction_weight = direction_weight
            self.diversity_weight = diversity_weight
            self.sector_weight = sector_weight
            self.min_diversity_target = min_diversity_target
            self.diversity_epsilon = diversity_epsilon

            # Validate weights sum to 1.0
            total_weight = mse_weight + direction_weight + diversity_weight + sector_weight
            if abs(total_weight - 1.0) > 1e-3:
                raise ValueError(f"Loss weights must sum to 1.0, got {total_weight}")

        def forward(self, predictions: Tensor, targets: Tensor, 
                   sector_ids: Tensor = None) -> Tensor:
            """Calculate production-ready combined loss that prevents model collapse."""
            predictions = predictions.squeeze()
            targets = targets.squeeze()

            # MSE Loss (40%)
            mse_loss = F.mse_loss(predictions, targets)

            # Directional Loss (30%)
            pred_direction = torch.sign(predictions)
            target_direction = torch.sign(targets)
            direction_accuracy = (pred_direction == target_direction).float().mean()
            direction_loss = 1.0 - direction_accuracy

            # Diversity Reward (20%)
            prediction_variance = torch.var(predictions)

            if prediction_variance < self.min_diversity_target:
                diversity_loss = torch.exp(-(prediction_variance / self.min_diversity_target))
            else:
                diversity_loss = 1.0 / (1.0 + prediction_variance / self.min_diversity_target)

            diversity_loss = torch.clamp(diversity_loss, 0.0, 1.0)

            # Sector Diversification (10%)
            sector_diversity_loss = 0.0
            if sector_ids is not None and len(torch.unique(sector_ids)) > 1:
                sector_diversity_loss = self._calculate_sector_diversity_loss(predictions, sector_ids)

            # Combine all losses
            total_loss = (
                self.mse_weight * mse_loss + 
                self.direction_weight * direction_loss + 
                self.diversity_weight * diversity_loss +
                self.sector_weight * sector_diversity_loss
            )

            # Numerical stability safeguards
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                fallback_loss = self.mse_weight * mse_loss + self.direction_weight * direction_loss
                if torch.isnan(fallback_loss) or torch.isinf(fallback_loss):
                    total_loss = mse_loss
                else:
                    total_loss = fallback_loss

            return total_loss

        def _calculate_sector_diversity_loss(self, predictions: Tensor, 
                                           sector_ids: Tensor) -> Tensor:
            """Calculate sector-aware diversity loss."""
            unique_sectors = torch.unique(sector_ids)

            if len(unique_sectors) <= 1:
                return torch.tensor(0.0, device=predictions.device)

            sector_means = []
            for sector_id in unique_sectors:
                sector_mask = (sector_ids == sector_id)
                if torch.sum(sector_mask) > 0:
                    sector_predictions = predictions[sector_mask]
                    sector_means.append(torch.mean(sector_predictions))

            if len(sector_means) > 1:
                sector_means_tensor = torch.stack(sector_means)
                sector_variance = torch.var(sector_means_tensor)
                sector_diversity_loss = 1.0 / (sector_variance + self.diversity_epsilon)

                max_sector_loss = 1.0 / (self.min_diversity_target + self.diversity_epsilon)
                sector_diversity_loss = torch.clamp(sector_diversity_loss, 0.0, max_sector_loss)
                sector_diversity_loss = sector_diversity_loss / max_sector_loss
            else:
                sector_diversity_loss = torch.tensor(0.0, device=predictions.device)

            return sector_diversity_loss

    class VolatilityAwareLoss(nn.Module):
        """Loss function that adapts based on market volatility."""

        def __init__(self, base_mse_weight: float = 0.6):
            super(VolatilityAwareLoss, self).__init__()
            self.base_mse_weight = base_mse_weight

        def forward(self, predictions: Tensor, targets: Tensor, 
                   volatility: Optional[Tensor] = None) -> Tensor:
            """Calculate volatility-aware loss."""
            mse_loss = F.mse_loss(predictions, targets)

            if volatility is None:
                volatility = torch.std(targets)

            volatility_factor = torch.clamp(volatility * 10, 0.1, 2.0)

            pred_direction = torch.sign(predictions)
            target_direction = torch.sign(targets)
            direction_loss = 1.0 - (pred_direction == target_direction).float().mean()

            mse_weight = self.base_mse_weight / volatility_factor
            direction_weight = (1.0 - self.base_mse_weight) * volatility_factor

            total_weight = mse_weight + direction_weight
            mse_weight = mse_weight / total_weight
            direction_weight = direction_weight / total_weight

            return mse_weight * mse_loss + direction_weight * direction_loss

    # Function definitions
    def save_model(model: LSTMPricePredictor, symbol: str, model_dir: str = "Analytics/ml/trained_models",
                   metadata: Optional[Dict[str, Any]] = None, scalers: Optional[Dict[str, Any]] = None,
                   preprocessing_params: Optional[Dict[str, Any]] = None) -> str:
        """Save trained LSTM model with metadata, scalers, and preprocessing parameters."""
        os.makedirs(model_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"lstm_{symbol}_{timestamp}.pth"
        model_path = os.path.join(model_dir, model_filename)

        model_data = {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_size': model.input_size,
                'hidden_size': model.hidden_size,
                'num_layers': model.num_layers,
                'sequence_length': model.sequence_length
            },
            'model_info': model.get_model_info(),
            'symbol': symbol,
            'saved_at': datetime.now().isoformat(),
            'metadata': metadata or {},
            'preprocessing_params': preprocessing_params or {},
            'model_version': '1.2.0'
        }

        if scalers and 'feature_scaler' in scalers and 'target_scaler' in scalers:
            scaler_filename = f"scalers_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            scaler_path = os.path.join(model_dir, scaler_filename)

            try:
                joblib.dump(scalers, scaler_path)
                model_data['scaler_file'] = scaler_filename
                model_data['scalers_saved_separately'] = True
                logger.info(f"Scalers saved separately using joblib for {symbol} at {scaler_path}")
            except Exception as e:
                logger.error(f"Failed to save scalers separately for {symbol}: {e}")
                model_data['scalers'] = scalers
                model_data['scalers_saved_separately'] = False
        else:
            model_data['scalers'] = {}
            model_data['scalers_saved_separately'] = False

        try:
            torch.save(model_data, model_path, _use_new_zipfile_serialization=False)
            logger.info(f"Enhanced model saved for {symbol} at {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model for {symbol}: {e}")
            raise

        return model_path

    def load_model(model_path: str, device: Optional[device] = None) -> Tuple[LSTMPricePredictor, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Load trained LSTM model from file with scalers and preprocessing parameters."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available - cannot load LSTM model")

        device = device or torch.device('cpu')

        try:
            model_data = torch.load(model_path, map_location=device, weights_only=False)
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

        config = model_data['model_config']
        model = LSTMPricePredictor(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            sequence_length=config['sequence_length']
        )

        model.load_state_dict(model_data['model_state_dict'])
        model.to(device)
        model.eval()

        scalers = model_data.get('scalers', {})
        preprocessing_params = model_data.get('preprocessing_params', {})

        return model, model_data, scalers, preprocessing_params

    def create_enhanced_lstm_model(input_size: int = 38, hidden_size: int = 128, num_layers: int = 2,
                                   dropout: float = 0.3, sequence_length: int = 30, 
                                   model_type: str = 'bidirectional_attention') -> LSTMPricePredictor:
        """Factory function to create enhanced LSTM models with different configurations."""
        if model_type == 'standard':
            return LSTMPricePredictor(input_size, hidden_size, num_layers, dropout, 
                                    sequence_length, False, False)
        elif model_type == 'bidirectional':
            return LSTMPricePredictor(input_size, hidden_size, num_layers, dropout, 
                                    sequence_length, True, False)
        elif model_type == 'attention':
            return LSTMPricePredictor(input_size, hidden_size, num_layers, dropout, 
                                    sequence_length, False, True, 'single')
        elif model_type == 'bidirectional_attention':
            return LSTMPricePredictor(input_size, hidden_size, num_layers, dropout, 
                                    sequence_length, True, True, 'single')
        elif model_type == 'multi_attention':
            return LSTMPricePredictor(input_size, hidden_size, num_layers, dropout, 
                                    sequence_length, True, True, 'multi')
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def save_universal_model(model: UniversalLSTMPredictor, model_name: str = "universal_lstm_v1.0",
                            model_dir: str = "Data/ml_models/universal_lstm", metadata: Optional[Dict[str, Any]] = None,
                            scalers: Optional[Dict[str, Any]] = None, sector_mappings: Optional[Dict[str, Any]] = None,
                            training_stocks: Optional[List[str]] = None) -> str:
        """Save Universal LSTM model with comprehensive metadata and mappings."""
        os.makedirs(model_dir, exist_ok=True)

        model_filename = f"{model_name}.pth"
        model_path = os.path.join(model_dir, model_filename)

        model_data = {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_size': model.input_size,
                'sector_embedding_dim': model.sector_embedding_dim,
                'industry_embedding_dim': model.industry_embedding_dim,
                'hidden_size': model.hidden_size,
                'num_layers': model.num_layers,
                'sequence_length': model.sequence_length,
                'num_sectors': model.num_sectors,
                'num_industries': model.num_industries,
                'use_cross_attention': model.use_cross_attention,
                'multi_task_output': model.multi_task_output
            },
            'model_info': model.get_model_info(),
            'model_type': 'UniversalLSTM',
            'saved_at': datetime.now().isoformat(),
            'metadata': metadata or {},
            'model_version': '4.0.0',
            'training_stocks': training_stocks or [],
            'sector_mappings': sector_mappings or {}
        }

        try:
            torch.save(model_data, model_path, _use_new_zipfile_serialization=False)
            logger.info(f"Universal LSTM model saved at {model_path}")
        except Exception as e:
            logger.error(f"Failed to save universal model: {e}")
            raise

        return model_path

    def create_universal_lstm_model(input_size: int = 24, sector_embedding_dim: int = 16,
                                   industry_embedding_dim: int = 8, hidden_size: int = 512,
                                   num_layers: int = 5, dropout: float = 0.4, sequence_length: int = 60,
                                   num_sectors: int = 11, num_industries: int = 50,
                                   model_type: str = 'full_universal') -> UniversalLSTMPredictor:
        """Factory function to create Universal LSTM models with different configurations."""
        if model_type == 'full_universal':
            return UniversalLSTMPredictor(input_size, sector_embedding_dim, industry_embedding_dim,
                                        hidden_size, num_layers, dropout, sequence_length,
                                        num_sectors, num_industries, False, True)
        elif model_type == 'price_only':
            return UniversalLSTMPredictor(input_size, sector_embedding_dim, industry_embedding_dim,
                                        hidden_size, num_layers, dropout, sequence_length,
                                        num_sectors, num_industries, True, False)
        elif model_type == 'lightweight':
            return UniversalLSTMPredictor(input_size, 8, 4, 128, 2, dropout, sequence_length,
                                        num_sectors, num_industries, False, False)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def get_model_complexity_score(model) -> Dict[str, Any]:
        """Calculate model complexity metrics for monitoring and selection."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        complexity_score = 1.0

        if hasattr(model, 'bidirectional') and model.bidirectional:
            complexity_score *= 1.5

        if hasattr(model, 'use_attention') and model.use_attention:
            complexity_score *= 1.3
            if hasattr(model, 'attention_type') and model.attention_type == 'multi':
                complexity_score *= 1.2

        if hasattr(model, 'num_layers'):
            complexity_score *= (model.num_layers / 2.0)
        if hasattr(model, 'hidden_size'):
            complexity_score *= (model.hidden_size / 128.0)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'complexity_score': complexity_score,
            'bidirectional': getattr(model, 'bidirectional', False),
            'attention_enabled': getattr(model, 'use_attention', False),
            'attention_type': getattr(model, 'attention_type', None),
            'num_layers': getattr(model, 'num_layers', 0),
            'hidden_size': getattr(model, 'hidden_size', 0),
            'model_version': getattr(model, 'model_version', 'unknown')
        }

    def _recreate_scalers_from_file(scaler_path: str) -> Dict[str, Any]:
        """Recreate scalers from saved parameters when joblib version incompatibility occurs."""
        from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
        import pickle

        try:
            with open(scaler_path, 'rb') as f:
                raw_data = pickle.load(f)

                scalers = {}

                for key, scaler_obj in raw_data.items():
                    if hasattr(scaler_obj, '__class__'):
                        scaler_class = scaler_obj.__class__.__name__

                        if scaler_class == 'MinMaxScaler':
                            new_scaler = MinMaxScaler()
                            if hasattr(scaler_obj, 'data_min_'):
                                new_scaler.data_min_ = scaler_obj.data_min_
                                new_scaler.data_max_ = scaler_obj.data_max_
                                new_scaler.data_range_ = scaler_obj.data_range_
                                new_scaler.scale_ = scaler_obj.scale_
                                new_scaler.min_ = scaler_obj.min_
                                new_scaler.n_features_in_ = scaler_obj.n_features_in_
                                new_scaler.feature_names_in_ = getattr(scaler_obj, 'feature_names_in_', None)
                            scalers[key] = new_scaler

                        elif scaler_class == 'StandardScaler':
                            new_scaler = StandardScaler()
                            if hasattr(scaler_obj, 'mean_'):
                                new_scaler.mean_ = scaler_obj.mean_
                                new_scaler.scale_ = scaler_obj.scale_
                                new_scaler.var_ = scaler_obj.var_
                                new_scaler.n_features_in_ = scaler_obj.n_features_in_
                                new_scaler.feature_names_in_ = getattr(scaler_obj, 'feature_names_in_', None)
                            scalers[key] = new_scaler

                        elif scaler_class == 'RobustScaler':
                            new_scaler = RobustScaler()
                            if hasattr(scaler_obj, 'center_'):
                                new_scaler.center_ = scaler_obj.center_
                                new_scaler.scale_ = scaler_obj.scale_
                                new_scaler.n_features_in_ = scaler_obj.n_features_in_
                                new_scaler.feature_names_in_ = getattr(scaler_obj, 'feature_names_in_', None)
                            scalers[key] = new_scaler

                        else:
                            scalers[key] = scaler_obj
                            logger.warning(f"Unknown scaler type: {scaler_class}, keeping original")
                    else:
                        scalers[key] = scaler_obj

                return scalers

        except Exception as e:
            logger.error(f"Failed to recreate scalers from {scaler_path}: {e}")
            raise

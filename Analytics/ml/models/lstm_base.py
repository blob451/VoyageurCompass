"""
LSTM Base Model for Stock Price Prediction
Implements Long Short-Term Memory neural network for time series forecasting.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import os
import joblib
from datetime import datetime
import math

logger = logging.getLogger(__name__)


class SectorCrossAttention(nn.Module):
    """
    Cross-attention mechanism that allows the model to focus on sector-specific patterns.
    """
    
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
        
    def forward(self, lstm_output: torch.Tensor, sector_embedding: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention between LSTM output and sector embedding.
        
        Args:
            lstm_output: LSTM output of shape (batch_size, seq_len, hidden_size)
            sector_embedding: Sector embedding of shape (batch_size, sector_embedding_dim)
            
        Returns:
            Attended output of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = lstm_output.shape
        
        # Expand sector embedding to match sequence length
        sector_expanded = sector_embedding.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Project inputs
        query = self.query_projection(lstm_output)  # (batch_size, seq_len, hidden_size)
        key = self.key_projection(sector_expanded)   # (batch_size, seq_len, hidden_size)
        value = self.value_projection(sector_expanded) # (batch_size, seq_len, hidden_size)
        
        # Compute attention scores
        attention_scores = torch.bmm(query, key.transpose(-2, -1)) / math.sqrt(self.hidden_size)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attended_output = torch.bmm(attention_weights, value)
        
        # Residual connection and output projection
        output = self.output_projection(attended_output + lstm_output)
        
        return output


class AttentionLayer(nn.Module):
    """
    Attention mechanism for LSTM outputs to focus on important time steps.
    """
    
    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention to LSTM outputs.
        
        Args:
            lstm_outputs: LSTM outputs of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        # Calculate attention scores
        attn_scores = self.attn(lstm_outputs)  # (batch_size, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch_size, seq_len, 1)
        
        # Apply attention weights
        attended_output = torch.sum(lstm_outputs * attn_weights, dim=1)  # (batch_size, hidden_size)
        
        return attended_output, attn_weights.squeeze(-1)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for better pattern recognition.
    """
    
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Attended output tensor
        """
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
    """
    Enhanced LSTM model for stock price prediction with 30-day sequence length.
    
    Architecture:
    - Input: 30-day sequences of OHLCV data + technical indicators
    - LSTM: 2 layers with 128 hidden units each (enhanced from single layer 50)
    - Dropout: 0.3 for improved regularization
    - Output: Predicted next-day price
    """
    
    def __init__(
        self,
        input_size: int = 38,    # Enhanced feature set (was 22, now 38)
        hidden_size: int = 128,  # Increased from 50 to 128 for better capacity
        num_layers: int = 2,     # Increased from 1 to 2 for better pattern recognition
        dropout: float = 0.3,    # Increased dropout for better regularization
        sequence_length: int = 30,
        bidirectional: bool = True,  # Enable bidirectional processing
        use_attention: bool = True,  # Enable attention mechanism
        attention_type: str = 'single'  # 'single' or 'multi'
    ):
        """
        Initialize enhanced LSTM model with bidirectional processing and attention.
        
        Args:
            input_size: Number of features per timestep
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
            sequence_length: Length of input sequences (30 days)
            bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to use attention mechanism
            attention_type: Type of attention ('single' or 'multi')
        """
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
        self.fc_output = nn.Linear(hidden_size, 1)  # Predict single price value
        
        # Residual connection (if dimensions match)
        self.use_residual = (self.effective_hidden_size == hidden_size)
        
        # Model metadata  
        self.model_version = "3.0.0"  # Enhanced architecture with attention and bidirectional
        self.created_at = datetime.now()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enhanced forward pass through the bidirectional LSTM model with attention.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Predicted prices of shape (batch_size, 1)
        """
        # Initialize hidden state (account for bidirectional)
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        # lstm_out shape: (batch_size, seq_len, effective_hidden_size)
        
        # Apply attention mechanism
        if self.use_attention:
            if self.attention_type == 'multi':
                # Multi-head attention followed by pooling attention
                attended_features = self.attention(lstm_out)  # Multi-head attention
                final_output, attention_weights = self.attention_pool(attended_features)
            else:
                # Single attention mechanism
                final_output, attention_weights = self.attention(lstm_out)
        else:
            # No attention - use last timestep
            final_output = lstm_out[:, -1, :]
        
        # Apply dropout
        final_output = self.dropout(final_output)
        
        # Intermediate layer with activation
        intermediate = F.relu(self.fc_intermediate(final_output))
        intermediate = self.dropout(intermediate)
        
        # Residual connection if dimensions match
        if self.use_residual:
            intermediate = intermediate + final_output
        
        # Final prediction
        prediction = self.fc_output(intermediate)
        
        return prediction
    
    def predict(self, x: torch.Tensor, return_confidence: bool = True) -> Dict[str, float]:
        """
        Make prediction with confidence estimation.
        
        Args:
            x: Input tensor for prediction
            return_confidence: Whether to estimate prediction confidence
            
        Returns:
            Dictionary with prediction and confidence
        """
        self.eval()
        with torch.no_grad():
            prediction = self.forward(x).item()
            
            # Simple confidence estimation based on model uncertainty
            # In production, this could be enhanced with ensemble methods
            confidence = 0.7  # Baseline confidence
            
            if return_confidence:
                # Enable training mode for dropout-based uncertainty estimation
                self.train()
                predictions = []
                for _ in range(10):  # Monte Carlo sampling
                    pred = self.forward(x).item()
                    predictions.append(pred)
                
                # Calculate confidence based on prediction variance
                std_dev = np.std(predictions)
                confidence = max(0.1, min(0.95, 1.0 - (std_dev / abs(prediction))))
                
                self.eval()
            
            return {
                'prediction': prediction,
                'confidence': confidence
            }
    
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
    """
    Universal LSTM model for stock price prediction with sector-differentiation features.
    
    This model can handle any stock by learning cross-sector patterns while maintaining
    sector-specific behaviors through embedding layers and attention mechanisms.
    
    Architecture:
    - Input: Multi-stock sequences with normalized features
    - Sector Embedding: Learned sector representations  
    - Industry Embedding: Fine-grained industry patterns
    - Bidirectional LSTM: Enhanced pattern recognition
    - Cross-Attention: Sector-specific attention patterns
    - Multi-Task Output: Price prediction + volatility + trend direction
    """
    
    def __init__(
        self,
        input_size: int = 25,            # Optimized universal features (excluding sector/industry)
        sector_embedding_dim: int = 12,   # REDUCED: Sector representation dimension
        industry_embedding_dim: int = 6,  # REDUCED: Industry sub-classification dimension
        hidden_size: int = 128,          # REDUCED: 256→128 for 75% parameter reduction
        num_layers: int = 3,             # REDUCED: 4→3 layers for simpler architecture
        dropout: float = 0.3,            # REDUCED: Lower dropout for smaller model
        sequence_length: int = 60,       # Longer sequences for better patterns
        num_sectors: int = 11,           # All major sectors + Unknown
        num_industries: int = 50,        # Industry fine-tuning
        use_cross_attention: bool = False, # DISABLED: Remove complexity initially
        multi_task_output: bool = True   # Enable multi-task learning
    ):
        """
        Initialize Universal LSTM model with sector-differentiation.
        
        Args:
            input_size: Number of normalized features per timestep
            sector_embedding_dim: Dimension of sector embeddings
            industry_embedding_dim: Dimension of industry embeddings
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
            sequence_length: Length of input sequences
            num_sectors: Number of sectors in vocabulary
            num_industries: Number of industries in vocabulary
            use_cross_attention: Whether to use sector cross-attention
            multi_task_output: Whether to use multi-task output heads
        """
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
        
        # Combined feature size (original features + embeddings)
        combined_input_size = input_size + sector_embedding_dim + industry_embedding_dim
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=combined_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # Always bidirectional for universal model
        )
        
        # Effective hidden size (doubled for bidirectional)
        self.effective_hidden_size = hidden_size * 2
        
        # Cross-attention mechanism for sector-aware processing
        if use_cross_attention:
            self.sector_cross_attention = SectorCrossAttention(
                self.effective_hidden_size, 
                sector_embedding_dim
            )
        
        # Attention pooling for sequence aggregation
        self.attention_pooling = AttentionLayer(self.effective_hidden_size)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        
        # Price normalization layer (handle different price scales)
        self.price_norm_layer = nn.LayerNorm(self.effective_hidden_size)
        
        # Multi-task output heads
        if multi_task_output:
            # Main price prediction head
            self.price_head = nn.Sequential(
                nn.Linear(self.effective_hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1)
            )
            
            # Volatility prediction head
            self.volatility_head = nn.Sequential(
                nn.Linear(self.effective_hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()  # Volatility is always positive
            )
            
            # Trend direction head (bullish/bearish)
            self.trend_head = nn.Sequential(
                nn.Linear(self.effective_hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 3),  # Up, Down, Sideways
                nn.Softmax(dim=-1)
            )
        else:
            # Single task: price prediction only
            self.price_head = nn.Sequential(
                nn.Linear(self.effective_hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1)
            )
        
        # Model metadata
        self.model_version = "4.0.0"  # Universal model version
        self.created_at = datetime.now()
        
        # Initialize embeddings with Xavier uniform
        nn.init.xavier_uniform_(self.sector_embedding.weight)
        nn.init.xavier_uniform_(self.industry_embedding.weight)
        
    def forward(
        self, 
        x: torch.Tensor, 
        sector_ids: torch.Tensor, 
        industry_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the Universal LSTM model.
        
        Args:
            x: Input features of shape (batch_size, sequence_length, input_size)
            sector_ids: Sector IDs of shape (batch_size,)
            industry_ids: Industry IDs of shape (batch_size,)
            
        Returns:
            Dictionary with prediction outputs
        """
        batch_size, seq_len, _ = x.shape
        
        # Get embeddings
        sector_emb = self.sector_embedding(sector_ids)  # (batch_size, sector_embedding_dim)
        industry_emb = self.industry_embedding(industry_ids)  # (batch_size, industry_embedding_dim)
        
        # Expand embeddings to sequence length
        sector_emb_expanded = sector_emb.unsqueeze(1).expand(-1, seq_len, -1)
        industry_emb_expanded = industry_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Concatenate features with embeddings
        combined_input = torch.cat([x, sector_emb_expanded, industry_emb_expanded], dim=-1)
        
        # LSTM forward pass
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(combined_input, (h0, c0))
        # lstm_out shape: (batch_size, seq_len, effective_hidden_size)
        
        # Apply sector cross-attention if enabled
        if self.use_cross_attention:
            lstm_out = self.sector_cross_attention(lstm_out, sector_emb)
        
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
        
        # Multi-task outputs if enabled
        if self.multi_task_output:
            volatility_pred = self.volatility_head(final_features)
            trend_pred = self.trend_head(final_features)
            
            outputs['volatility'] = volatility_pred
            outputs['trend'] = trend_pred
        
        # Store attention weights for interpretability
        outputs['attention_weights'] = attention_weights
        
        return outputs
    
    def predict_universal(
        self, 
        x: torch.Tensor, 
        sector_ids: torch.Tensor, 
        industry_ids: torch.Tensor,
        return_all_outputs: bool = False
    ) -> Dict[str, Any]:
        """
        Make universal prediction with confidence estimation.
        
        Args:
            x: Input tensor for prediction
            sector_ids: Sector IDs tensor
            industry_ids: Industry IDs tensor
            return_all_outputs: Whether to return all task outputs
            
        Returns:
            Dictionary with predictions and confidence
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, sector_ids, industry_ids)
            
            # Extract main price prediction
            price_prediction = outputs['price'].item()
            
            # Simple confidence estimation (can be enhanced with uncertainty quantification)
            confidence = 0.75  # Base confidence for universal model
            
            result = {
                'price_prediction': price_prediction,
                'confidence': confidence
            }
            
            # Add multi-task outputs if available and requested
            if return_all_outputs and self.multi_task_output:
                result['volatility_prediction'] = outputs['volatility'].item()
                result['trend_prediction'] = outputs['trend'].cpu().numpy().tolist()
                result['attention_weights'] = outputs['attention_weights'].cpu().numpy().tolist()
            
            return result
    
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
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
        validation_split: float = 0.15,
        test_split: float = 0.15,
        sequence_length: int = 30,
        forecast_horizon: int = 1,  # Days ahead to predict
        min_training_samples: int = 400
    ):
        """
        Initialize training configuration.
        
        Args:
            learning_rate: Adam optimizer learning rate
            batch_size: Training batch size
            num_epochs: Maximum training epochs
            early_stopping_patience: Epochs to wait for improvement
            validation_split: Fraction of data for validation
            test_split: Fraction of data for testing
            sequence_length: Input sequence length (30 days)
            forecast_horizon: Days ahead to predict
            min_training_samples: Minimum samples required for training
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.test_split = test_split
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.min_training_samples = min_training_samples
        
        # Derived training split
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


def save_model(
    model: LSTMPricePredictor,
    symbol: str,
    model_dir: str = "Analytics/ml/trained_models",
    metadata: Optional[Dict[str, Any]] = None,
    scalers: Optional[Dict[str, Any]] = None,
    preprocessing_params: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save trained LSTM model with metadata, scalers, and preprocessing parameters.
    
    Args:
        model: Trained LSTM model
        symbol: Stock symbol
        model_dir: Directory to save models
        metadata: Additional metadata to save
        scalers: Dictionary containing fitted scalers (feature_scaler, target_scaler)
        preprocessing_params: Preprocessing configuration parameters
        
    Returns:
        Path to saved model file
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate model filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"lstm_{symbol}_{timestamp}.pth"
    model_path = os.path.join(model_dir, model_filename)
    
    # Prepare model data with enhanced scaler persistence
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
        'model_version': '1.2.0'  # Increment version for joblib scaler persistence
    }
    
    # Enhanced scaler persistence using joblib for sklearn objects
    if scalers and 'feature_scaler' in scalers and 'target_scaler' in scalers:
        # Save scalers separately using joblib for reliable sklearn serialization
        scaler_filename = f"scalers_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        scaler_path = os.path.join(model_dir, scaler_filename)
        
        try:
            # Use joblib for sklearn scaler serialization (more reliable than pickle/torch.save)
            joblib.dump(scalers, scaler_path)
            logger.info(f"Scalers saved separately using joblib for {symbol} at {scaler_path}")
            
            # Store scaler file reference in model data
            model_data['scaler_file'] = scaler_filename
            model_data['scalers_saved_separately'] = True
        except Exception as e:
            logger.error(f"Failed to save scalers separately for {symbol}: {e}")
            # Fallback to embedding in model data (may not work reliably)
            model_data['scalers'] = scalers
            model_data['scalers_saved_separately'] = False
    else:
        model_data['scalers'] = {}
        model_data['scalers_saved_separately'] = False
    
    # Validate that scalers are provided for production models
    if scalers and 'feature_scaler' in scalers and 'target_scaler' in scalers:
        logger.info(f"Saving model for {symbol} with fitted scalers")
    else:
        logger.warning(f"Saving model for {symbol} WITHOUT fitted scalers - may cause inference issues")
    
    # Save model with enhanced error handling
    try:
        torch.save(model_data, model_path, _use_new_zipfile_serialization=False)
        logger.info(f"Enhanced model saved for {symbol} at {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model for {symbol}: {e}")
        # Clean up any partially saved scaler file
        if scalers and model_data.get('scalers_saved_separately', False):
            scaler_path = os.path.join(model_dir, model_data.get('scaler_file', ''))
            if os.path.exists(scaler_path):
                os.remove(scaler_path)
                logger.info(f"Cleaned up scaler file after model save failure: {scaler_path}")
        raise
    
    return model_path


def load_model(
    model_path: str,
    device: Optional[torch.device] = None
) -> Tuple[LSTMPricePredictor, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Load trained LSTM model from file with scalers and preprocessing parameters.
    
    Args:
        model_path: Path to saved model file
        device: Device to load model on
        
    Returns:
        Tuple of (loaded_model, metadata, scalers, preprocessing_params)
    """
    device = device or torch.device('cpu')
    
    try:
        # Load model data with enhanced error handling
        model_data = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise
    
    # Extract configuration
    config = model_data['model_config']
    
    # Create model instance - detect enhanced vs standard architecture
    model_info = model_data.get('model_info', {})
    model_version = model_data.get('model_version', '1.0.0')
    
    # Check if this is an enhanced model (version 3.0.0+ or has enhanced metadata)
    # Fix: Use proper semantic version comparison instead of string comparison
    def compare_version(version_str: str, target: str) -> bool:
        """Compare semantic versions properly."""
        try:
            v_parts = [int(x) for x in version_str.split('.')]
            t_parts = [int(x) for x in target.split('.')]
            # Pad shorter version with zeros
            max_len = max(len(v_parts), len(t_parts))
            v_parts.extend([0] * (max_len - len(v_parts)))
            t_parts.extend([0] * (max_len - len(t_parts)))
            return v_parts >= t_parts
        except (ValueError, AttributeError):
            # Fallback for invalid version strings
            return False
    
    # Enhanced detection logic with multiple fallback methods
    state_dict_keys = list(model_data.get('model_state_dict', {}).keys())
    
    # Method 1: Version-based detection (fixed)
    version_indicates_enhanced = compare_version(model_version, '3.0.0')
    
    # Method 2: Metadata-based detection  
    metadata_indicates_enhanced = (
        'bidirectional' in model_info or
        'attention_enabled' in model_info or
        model_info.get('bidirectional', False) or
        model_info.get('use_attention', False)
    )
    
    # Method 3: State dict key-based detection (most reliable)
    state_dict_indicates_enhanced = any(key in state_dict_keys for key in [
        'lstm.weight_ih_l0_reverse',  # Bidirectional LSTM
        'attention.attn.weight',      # Single attention
        'attention.query.weight',     # Multi-head attention
        'attention_pool.attn.weight'  # Attention pooling
    ])
    
    # Method 3.5: Universal model detection
    state_dict_indicates_universal = any(key in state_dict_keys for key in [
        'sector_embedding.weight',    # Sector embeddings
        'industry_embedding.weight',  # Industry embeddings
        'sector_cross_attention.',    # Cross-attention mechanism
        'price_head.',               # Multi-task heads
        'volatility_head.',
        'trend_head.'
    ])
    
    # Method 4: Architecture parameter detection
    try:
        # Check if the model was saved with enhanced architecture flags
        has_enhanced_params = (
            model_data.get('model_config', {}).get('bidirectional', False) or
            model_data.get('model_config', {}).get('use_attention', False)
        )
    except (KeyError, AttributeError):
        has_enhanced_params = False
    
    # Method 5: Universal model version detection
    version_indicates_universal = compare_version(model_version, '4.0.0')
    
    # Check for universal model type in metadata
    model_type = model_info.get('model_type', 'LSTM')
    metadata_indicates_universal = model_type == 'UniversalLSTM'
    
    # Combine detection methods with priority: universal > enhanced > standard
    is_universal_model = (
        state_dict_indicates_universal or
        metadata_indicates_universal or
        version_indicates_universal
    )
    
    is_enhanced_model = (
        not is_universal_model and (
            state_dict_indicates_enhanced or 
            metadata_indicates_enhanced or 
            has_enhanced_params or 
            version_indicates_enhanced
        )
    )
    
    logger.info(f"Model detection - Universal: {is_universal_model}, Enhanced: {is_enhanced_model}, "
                f"Version: {model_version}, Type: {model_type}")
    
    if is_universal_model:
        # Universal LSTM model - extract universal parameters
        logger.info(f"Detected Universal LSTM model (version: {model_version})")
        
        # Extract universal model parameters
        sector_embedding_dim = model_info.get('sector_embedding_dim', config.get('sector_embedding_dim', 16))
        industry_embedding_dim = model_info.get('industry_embedding_dim', config.get('industry_embedding_dim', 8))
        num_sectors = model_info.get('num_sectors', config.get('num_sectors', 11))
        num_industries = model_info.get('num_industries', config.get('num_industries', 50))
        use_cross_attention = model_info.get('use_cross_attention', config.get('use_cross_attention', True))
        multi_task_output = model_info.get('multi_task_output', config.get('multi_task_output', True))
        
        model = UniversalLSTMPredictor(
            input_size=config['input_size'],
            sector_embedding_dim=sector_embedding_dim,
            industry_embedding_dim=industry_embedding_dim,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            sequence_length=config['sequence_length'],
            num_sectors=num_sectors,
            num_industries=num_industries,
            use_cross_attention=use_cross_attention,
            multi_task_output=multi_task_output
        )
        logger.info(f"Created Universal LSTM: sectors={num_sectors}, industries={num_industries}, "
                   f"cross_attention={use_cross_attention}, multi_task={multi_task_output}")
        
    elif is_enhanced_model:
        # Enhanced model - extract architecture parameters
        logger.info(f"Detected enhanced model architecture (version: {model_version})")
        
        # Extract enhanced parameters with multiple fallback methods
        # Method 1: Try model_info first (most reliable if available)
        bidirectional = model_info.get('bidirectional', None)
        attention_enabled = model_info.get('attention_enabled', None) or model_info.get('use_attention', None)
        attention_type = model_info.get('attention_type', None)
        
        # Method 2: Try model_config as backup
        if bidirectional is None:
            bidirectional = config.get('bidirectional', None)
        if attention_enabled is None:
            attention_enabled = config.get('use_attention', None)
        if attention_type is None:
            attention_type = config.get('attention_type', None)
        
        # Method 3: Infer from state_dict keys (most reliable fallback)
        if bidirectional is None:
            bidirectional = any('reverse' in key for key in state_dict_keys)
        if attention_enabled is None:
            attention_enabled = any('attention' in key for key in state_dict_keys)
        
        # Method 4: Determine attention type from state_dict structure
        if attention_type is None and attention_enabled:
            if any('attention.query.weight' in key for key in state_dict_keys):
                attention_type = 'multi'
            elif any('attention.attn.weight' in key for key in state_dict_keys):
                attention_type = 'single'
            else:
                attention_type = 'single'  # Default fallback
        elif attention_type is None:
            attention_type = 'single'  # Default when no attention
            
        # Ensure we have valid boolean values
        bidirectional = bool(bidirectional) if bidirectional is not None else False
        attention_enabled = bool(attention_enabled) if attention_enabled is not None else False
            
        model = LSTMPricePredictor(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            sequence_length=config['sequence_length'],
            bidirectional=bidirectional,
            use_attention=attention_enabled,
            attention_type=attention_type
        )
        logger.info(f"Created enhanced model: bidirectional={bidirectional}, attention={attention_enabled}, type={attention_type}")
    else:
        # Standard model
        logger.info(f"Detected standard model architecture (version: {model_version})")
        model = LSTMPricePredictor(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            sequence_length=config['sequence_length']
        )
    
    # Load model weights
    model.load_state_dict(model_data['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Enhanced scaler loading with joblib support
    scalers = {}
    preprocessing_params = model_data.get('preprocessing_params', {})
    model_version = model_data.get('model_version', '1.0.0')
    
    # Check if scalers were saved separately using joblib
    if model_data.get('scalers_saved_separately', False) and 'scaler_file' in model_data:
        model_dir = os.path.dirname(model_path)
        scaler_path = os.path.join(model_dir, model_data['scaler_file'])
        
        try:
            if os.path.exists(scaler_path):
                # CRITICAL FIX: Version-independent scaler loading with multiple fallback methods
                scalers = None
                
                # Method 1: Try standard joblib loading (current sklearn version)
                try:
                    scalers = joblib.load(scaler_path)
                    logger.info(f"Scalers loaded using current joblib from {scaler_path}")
                except Exception as joblib_error:
                    logger.warning(f"Current joblib failed: {joblib_error}")
                    
                    # Method 2: Try loading with sklearn version tolerance
                    try:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning)
                            scalers = joblib.load(scaler_path)
                            logger.info(f"Scalers loaded with warnings suppressed from {scaler_path}")
                    except Exception as version_error:
                        logger.warning(f"Version-tolerant loading failed: {version_error}")
                        
                        # Method 3: Extract scaler parameters and recreate
                        try:
                            scalers = _recreate_scalers_from_file(scaler_path)
                            logger.info(f"Scalers recreated from parameters in {scaler_path}")
                        except Exception as recreate_error:
                            logger.error(f"Scaler recreation failed: {recreate_error}")
                            scalers = None
                
                if scalers is None:
                    logger.warning(f"All scaler loading methods failed for {scaler_path}")
                    scalers = model_data.get('scalers', {})
            else:
                logger.warning(f"Scaler file not found: {scaler_path}")
                scalers = model_data.get('scalers', {})
        except Exception as e:
            logger.error(f"Failed to load scalers from {scaler_path}: {e}")
            scalers = model_data.get('scalers', {})  # Fallback to embedded scalers
    else:
        # Fallback to embedded scalers (legacy models)
        scalers = model_data.get('scalers', {})
    
    # Validate scaler presence for inference consistency
    if scalers and 'feature_scaler' in scalers and 'target_scaler' in scalers:
        logger.info(f"Model loaded from {model_path} with fitted scalers (version: {model_version})")
    else:
        logger.warning(f"Model loaded from {model_path} WITHOUT fitted scalers - inference may be inconsistent")
    
    return model, model_data, scalers, preprocessing_params


def create_enhanced_lstm_model(
    input_size: int = 38,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    sequence_length: int = 30,
    model_type: str = 'bidirectional_attention'
) -> LSTMPricePredictor:
    """
    Factory function to create enhanced LSTM models with different configurations.
    
    Args:
        input_size: Number of input features
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        sequence_length: Sequence length
        model_type: Type of model ('standard', 'bidirectional', 'attention', 'bidirectional_attention')
        
    Returns:
        Configured LSTM model
    """
    if model_type == 'standard':
        # Standard unidirectional LSTM without attention
        return LSTMPricePredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            sequence_length=sequence_length,
            bidirectional=False,
            use_attention=False
        )
    elif model_type == 'bidirectional':
        # Bidirectional LSTM without attention
        return LSTMPricePredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            sequence_length=sequence_length,
            bidirectional=True,
            use_attention=False
        )
    elif model_type == 'attention':
        # Unidirectional LSTM with attention
        return LSTMPricePredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            sequence_length=sequence_length,
            bidirectional=False,
            use_attention=True,
            attention_type='single'
        )
    elif model_type == 'bidirectional_attention':
        # Bidirectional LSTM with single attention (default)
        return LSTMPricePredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            sequence_length=sequence_length,
            bidirectional=True,
            use_attention=True,
            attention_type='single'
        )
    elif model_type == 'multi_attention':
        # Bidirectional LSTM with multi-head attention
        return LSTMPricePredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            sequence_length=sequence_length,
            bidirectional=True,
            use_attention=True,
            attention_type='multi'
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from: 'standard', 'bidirectional', 'attention', 'bidirectional_attention', 'multi_attention'")


def save_universal_model(
    model: UniversalLSTMPredictor,
    model_name: str = "universal_lstm_v1.0",
    model_dir: str = "Data/ml_models/universal_lstm",
    metadata: Optional[Dict[str, Any]] = None,
    scalers: Optional[Dict[str, Any]] = None,
    sector_mappings: Optional[Dict[str, Any]] = None,
    training_stocks: Optional[List[str]] = None
) -> str:
    """
    Save Universal LSTM model with comprehensive metadata and mappings.
    
    Args:
        model: Trained Universal LSTM model
        model_name: Name for the model file
        model_dir: Directory to save model in
        metadata: Additional training metadata
        scalers: Universal scalers for feature/target normalization
        sector_mappings: Sector and industry mapping configurations
        training_stocks: List of stocks used in training
        
    Returns:
        Path to saved model file
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate model filename
    model_filename = f"{model_name}.pth"
    model_path = os.path.join(model_dir, model_filename)
    
    # Prepare comprehensive model data
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
    
    # Save scalers separately for universal model
    if scalers:
        scaler_filename = f"universal_scaler_{model_name.split('_')[-1]}.pkl"
        scaler_path = os.path.join(model_dir, scaler_filename)
        
        try:
            joblib.dump(scalers, scaler_path)
            logger.info(f"Universal scalers saved at {scaler_path}")
            model_data['scaler_file'] = scaler_filename
            model_data['scalers_saved_separately'] = True
        except Exception as e:
            logger.error(f"Failed to save universal scalers: {e}")
            model_data['scalers'] = scalers
            model_data['scalers_saved_separately'] = False
    
    # Save sector mappings separately
    if sector_mappings:
        mapping_filename = f"sector_mappings_{model_name.split('_')[-1]}.json"
        mapping_path = os.path.join(model_dir, mapping_filename)
        
        try:
            import json
            with open(mapping_path, 'w') as f:
                json.dump(sector_mappings, f, indent=2)
            logger.info(f"Sector mappings saved at {mapping_path}")
            model_data['mapping_file'] = mapping_filename
        except Exception as e:
            logger.error(f"Failed to save sector mappings: {e}")
    
    # Save the model
    try:
        torch.save(model_data, model_path, _use_new_zipfile_serialization=False)
        logger.info(f"Universal LSTM model saved at {model_path}")
        
        # Log model statistics
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Universal model parameters: {total_params:,}")
        
    except Exception as e:
        logger.error(f"Failed to save universal model: {e}")
        raise
    
    return model_path


def create_universal_lstm_model(
    input_size: int = 24,  # Fixed to match actual feature count
    sector_embedding_dim: int = 16,
    industry_embedding_dim: int = 8,
    hidden_size: int = 512,  # Doubled for higher capacity
    num_layers: int = 5,  # Increased depth for better learning
    dropout: float = 0.4,
    sequence_length: int = 60,
    num_sectors: int = 11,
    num_industries: int = 50,
    model_type: str = 'full_universal'
) -> UniversalLSTMPredictor:
    """
    Factory function to create Universal LSTM models with different configurations.
    
    Args:
        input_size: Number of input features (excluding embeddings)
        sector_embedding_dim: Sector embedding dimension
        industry_embedding_dim: Industry embedding dimension
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        sequence_length: Input sequence length
        num_sectors: Number of sectors
        num_industries: Number of industries
        model_type: Configuration type
        
    Returns:
        Configured Universal LSTM model
    """
    if model_type == 'full_universal':
        # Full universal model with all features - OPTIMIZED FOR CONVERGENCE
        return UniversalLSTMPredictor(
            input_size=input_size,
            sector_embedding_dim=sector_embedding_dim,
            industry_embedding_dim=industry_embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            sequence_length=sequence_length,
            num_sectors=num_sectors,
            num_industries=num_industries,
            use_cross_attention=False,  # OPTIMIZED: Disabled for simplicity and faster convergence
            multi_task_output=True
        )
    elif model_type == 'price_only':
        # Price prediction only (no multi-task)
        return UniversalLSTMPredictor(
            input_size=input_size,
            sector_embedding_dim=sector_embedding_dim,
            industry_embedding_dim=industry_embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            sequence_length=sequence_length,
            num_sectors=num_sectors,
            num_industries=num_industries,
            use_cross_attention=True,
            multi_task_output=False
        )
    elif model_type == 'lightweight':
        # Lightweight model for faster inference
        return UniversalLSTMPredictor(
            input_size=input_size,
            sector_embedding_dim=8,
            industry_embedding_dim=4,
            hidden_size=128,
            num_layers=2,
            dropout=dropout,
            sequence_length=sequence_length,
            num_sectors=num_sectors,
            num_industries=num_industries,
            use_cross_attention=False,
            multi_task_output=False
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from: 'full_universal', 'price_only', 'lightweight'")


def get_model_complexity_score(model) -> Dict[str, Any]:
    """
    Calculate model complexity metrics for monitoring and selection.
    
    Args:
        model: LSTM model instance
        
    Returns:
        Dictionary with complexity metrics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate complexity score based on architecture
    complexity_score = 1.0  # Base score
    
    if model.bidirectional:
        complexity_score *= 1.5  # Bidirectional adds complexity
    
    if model.use_attention:
        complexity_score *= 1.3  # Attention adds complexity
        if model.attention_type == 'multi':
            complexity_score *= 1.2  # Multi-head attention adds more complexity
    
    complexity_score *= (model.num_layers / 2.0)  # Layer scaling
    complexity_score *= (model.hidden_size / 128.0)  # Hidden size scaling
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'complexity_score': complexity_score,
        'bidirectional': model.bidirectional,
        'attention_enabled': model.use_attention,
        'attention_type': model.attention_type if model.use_attention else None,
        'num_layers': model.num_layers,
        'hidden_size': model.hidden_size,
        'model_version': model.model_version
    }


def _recreate_scalers_from_file(scaler_path: str) -> Dict[str, Any]:
    """
    Recreate scalers from saved parameters when joblib version incompatibility occurs.
    
    Args:
        scaler_path: Path to the scaler file
        
    Returns:
        Dictionary with recreated scalers
    """
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
    import pickle
    
    # Try to load just the scaler parameters using pickle directly
    try:
        # Open file and try to extract scaler type and parameters
        with open(scaler_path, 'rb') as f:
            # Read the raw data
            raw_data = pickle.load(f)
            
            scalers = {}
            
            # Handle different scaler types that might be saved
            for key, scaler_obj in raw_data.items():
                if hasattr(scaler_obj, '__class__'):
                    scaler_class = scaler_obj.__class__.__name__
                    
                    if scaler_class == 'MinMaxScaler':
                        # Recreate MinMaxScaler with saved parameters
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
                        # Recreate StandardScaler with saved parameters
                        new_scaler = StandardScaler()
                        if hasattr(scaler_obj, 'mean_'):
                            new_scaler.mean_ = scaler_obj.mean_
                            new_scaler.scale_ = scaler_obj.scale_
                            new_scaler.var_ = scaler_obj.var_
                            new_scaler.n_features_in_ = scaler_obj.n_features_in_
                            new_scaler.feature_names_in_ = getattr(scaler_obj, 'feature_names_in_', None)
                        scalers[key] = new_scaler
                        
                    elif scaler_class == 'RobustScaler':
                        # Recreate RobustScaler with saved parameters  
                        new_scaler = RobustScaler()
                        if hasattr(scaler_obj, 'center_'):
                            new_scaler.center_ = scaler_obj.center_
                            new_scaler.scale_ = scaler_obj.scale_
                            new_scaler.n_features_in_ = scaler_obj.n_features_in_
                            new_scaler.feature_names_in_ = getattr(scaler_obj, 'feature_names_in_', None)
                        scalers[key] = new_scaler
                        
                    else:
                        # Unknown scaler type, keep as-is and hope for the best
                        scalers[key] = scaler_obj
                        logger.warning(f"Unknown scaler type: {scaler_class}, keeping original")
                else:
                    # Not a scaler object
                    scalers[key] = scaler_obj
                    
            return scalers
            
    except Exception as e:
        logger.error(f"Failed to recreate scalers from {scaler_path}: {e}")
        raise


class DirectionalLoss(nn.Module):
    """
    Production-ready loss function that prevents model collapse through diversity rewards.
    Combines MSE, directional accuracy, diversity incentives, and sector-aware diversification.
    
    Key Innovation: REWARDS prediction diversity instead of penalizing it.
    """
    
    def __init__(
        self, 
        mse_weight: float = 0.4, 
        direction_weight: float = 0.3, 
        diversity_weight: float = 0.2,
        sector_weight: float = 0.1,
        min_diversity_target: float = 0.001,  # Minimum 0.1% spread required
        diversity_epsilon: float = 1e-6
    ):
        """
        Initialize production-ready directional loss function.
        
        Args:
            mse_weight: Weight for MSE loss component (default 0.4)
            direction_weight: Weight for directional accuracy component (default 0.3)
            diversity_weight: Weight for diversity reward component (default 0.2)
            sector_weight: Weight for sector diversification component (default 0.1)
            min_diversity_target: Minimum prediction spread target (default 0.001)
            diversity_epsilon: Small constant to prevent division by zero (default 1e-6)
        """
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
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, sector_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate production-ready combined loss that prevents model collapse.
        
        Args:
            predictions: Model predictions [batch_size, 1] or [batch_size]
            targets: True target values [batch_size, 1] or [batch_size]
            sector_ids: Optional sector IDs for sector-aware diversification [batch_size]
            
        Returns:
            Combined loss value that encourages diversity
        """
        # Ensure consistent tensor shapes
        predictions = predictions.squeeze()
        targets = targets.squeeze()
        
        # 1. MSE Loss (40%) - Core accuracy
        mse_loss = F.mse_loss(predictions, targets)
        
        # 2. Directional Loss (30%) - Trend accuracy
        pred_direction = torch.sign(predictions)
        target_direction = torch.sign(targets)
        direction_accuracy = (pred_direction == target_direction).float().mean()
        direction_loss = 1.0 - direction_accuracy
        
        # 3. Diversity Reward (20%) - CRITICAL FIX: Reward spread, don't penalize it
        prediction_variance = torch.var(predictions)
        
        # IMPROVED: More effective diversity loss calculation
        # Use exponential decay to provide stronger gradient signal
        if prediction_variance < self.min_diversity_target:
            # Strong penalty for insufficient diversity
            diversity_loss = torch.exp(-(prediction_variance / self.min_diversity_target)) 
        else:
            # Gentle reward for exceeding diversity target
            diversity_loss = 1.0 / (1.0 + prediction_variance / self.min_diversity_target)
        
        # Ensure reasonable range [0, 1] for stable training
        diversity_loss = torch.clamp(diversity_loss, 0.0, 1.0)
        
        # 4. Sector Diversification (10%) - Ensure cross-sector variety
        sector_diversity_loss = 0.0
        if sector_ids is not None and len(torch.unique(sector_ids)) > 1:
            sector_diversity_loss = self._calculate_sector_diversity_loss(predictions, sector_ids)
        
        # 5. Combine all losses with production weights
        total_loss = (
            self.mse_weight * mse_loss + 
            self.direction_weight * direction_loss + 
            self.diversity_weight * diversity_loss +
            self.sector_weight * sector_diversity_loss
        )
        
        # 6. Numerical stability safeguards
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            # Fallback hierarchy: MSE -> MSE + Direction -> MSE only
            fallback_loss = self.mse_weight * mse_loss + self.direction_weight * direction_loss
            if torch.isnan(fallback_loss) or torch.isinf(fallback_loss):
                total_loss = mse_loss
            else:
                total_loss = fallback_loss
        
        return total_loss
    
    def _calculate_sector_diversity_loss(self, predictions: torch.Tensor, sector_ids: torch.Tensor) -> torch.Tensor:
        """
        Calculate sector-aware diversity loss to ensure cross-sector prediction variety.
        
        Args:
            predictions: Model predictions [batch_size]
            sector_ids: Sector IDs for each prediction [batch_size]
            
        Returns:
            Sector diversity loss (lower = more diverse across sectors)
        """
        sector_diversity_loss = 0.0
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
            # Reward higher variance between sector means (inverse relationship)
            sector_variance = torch.var(sector_means_tensor)
            sector_diversity_loss = 1.0 / (sector_variance + self.diversity_epsilon)
            
            # Normalize to reasonable range
            max_sector_loss = 1.0 / (self.min_diversity_target + self.diversity_epsilon)
            sector_diversity_loss = torch.clamp(sector_diversity_loss, 0.0, max_sector_loss)
            sector_diversity_loss = sector_diversity_loss / max_sector_loss
        
        return sector_diversity_loss
    
    def get_component_losses(self, predictions: torch.Tensor, targets: torch.Tensor, sector_ids: torch.Tensor = None) -> Dict[str, float]:
        """
        Get individual component losses for monitoring the new loss structure.
        
        Args:
            predictions: Model predictions
            targets: True target values
            sector_ids: Optional sector IDs for sector diversity calculation
            
        Returns:
            Dictionary with all loss components and metrics
        """
        with torch.no_grad():
            predictions = predictions.squeeze()
            targets = targets.squeeze()
            
            # Core loss components
            mse_loss = F.mse_loss(predictions, targets).item()
            
            pred_direction = torch.sign(predictions)
            target_direction = torch.sign(targets)
            direction_accuracy = (pred_direction == target_direction).float().mean().item()
            direction_loss = 1.0 - direction_accuracy
            
            # Diversity metrics (matching forward method calculation)
            prediction_variance = torch.var(predictions).item()
            if prediction_variance < self.min_diversity_target:
                diversity_loss = torch.exp(-(torch.tensor(prediction_variance) / self.min_diversity_target)).item()
            else:
                diversity_loss = (1.0 / (1.0 + prediction_variance / self.min_diversity_target))
            diversity_loss = max(0.0, min(diversity_loss, 1.0))
            
            # Sector diversity
            sector_diversity_loss = 0.0
            sector_variance = 0.0
            if sector_ids is not None and len(torch.unique(sector_ids)) > 1:
                sector_loss_tensor = self._calculate_sector_diversity_loss(predictions, sector_ids)
                sector_diversity_loss = sector_loss_tensor.item()
                
                # Calculate actual sector variance for monitoring
                unique_sectors = torch.unique(sector_ids)
                sector_means = []
                for sector_id in unique_sectors:
                    sector_mask = (sector_ids == sector_id)
                    if torch.sum(sector_mask) > 0:
                        sector_predictions = predictions[sector_mask]
                        sector_means.append(torch.mean(sector_predictions))
                if len(sector_means) > 1:
                    sector_means_tensor = torch.stack(sector_means)
                    sector_variance = torch.var(sector_means_tensor).item()
            
            # Calculate total loss
            total_loss = (
                self.mse_weight * mse_loss + 
                self.direction_weight * direction_loss + 
                self.diversity_weight * diversity_loss +
                self.sector_weight * sector_diversity_loss
            )
            
        return {
            'mse_loss': mse_loss,
            'direction_loss': direction_loss,
            'direction_accuracy': direction_accuracy,
            'diversity_loss': diversity_loss,
            'prediction_variance': prediction_variance,
            'sector_diversity_loss': sector_diversity_loss,
            'sector_variance': sector_variance,
            'total_loss': total_loss,
            'diversity_target_met': prediction_variance >= self.min_diversity_target
        }


class VolatilityAwareLoss(nn.Module):
    """
    Loss function that adapts based on market volatility.
    Higher weight on accuracy during volatile periods.
    """
    
    def __init__(self, base_mse_weight: float = 0.6):
        super(VolatilityAwareLoss, self).__init__()
        self.base_mse_weight = base_mse_weight
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                volatility: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate volatility-aware loss.
        
        Args:
            predictions: Model predictions
            targets: True target values
            volatility: Market volatility indicator (optional)
            
        Returns:
            Volatility-adjusted loss
        """
        # Base MSE loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # Calculate volatility if not provided
        if volatility is None:
            volatility = torch.std(targets)
        
        # Adjust weights based on volatility
        # Higher volatility → More weight on directional accuracy
        volatility_factor = torch.clamp(volatility * 10, 0.1, 2.0)  # Scale volatility
        
        # Directional accuracy becomes more important in volatile markets
        pred_direction = torch.sign(predictions)
        target_direction = torch.sign(targets)
        direction_loss = 1.0 - (pred_direction == target_direction).float().mean()
        
        # Dynamic weighting
        mse_weight = self.base_mse_weight / volatility_factor
        direction_weight = (1.0 - self.base_mse_weight) * volatility_factor
        
        # Normalize weights
        total_weight = mse_weight + direction_weight
        mse_weight = mse_weight / total_weight
        direction_weight = direction_weight / total_weight
        
        return mse_weight * mse_loss + direction_weight * direction_loss
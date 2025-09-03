"""
Universal LSTM prediction service for real-time stock price forecasting.
"""

import logging
import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from django.core.cache import cache
from django.conf import settings

from Analytics.ml.models.lstm_base import UniversalLSTMPredictor, load_model
from Analytics.ml.universal_preprocessor import UniversalLSTMPreprocessor
from Analytics.ml.sector_mappings import get_sector_mapper
from Data.repo.price_reader import PriceReader
from Data.services.yahoo_finance import yahoo_finance_service

logger = logging.getLogger(__name__)


class UniversalLSTMAnalyticsService:
    """Universal LSTM prediction service with sector-aware processing architecture."""
    
    def __init__(self, model_dir: str = "Data/ml_models/universal_lstm"):
        """Initialise Universal LSTM service with model directory configuration."""
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.preprocessor = None
        self.model_metadata = None
        self.sector_mapper = get_sector_mapper()
        self.price_reader = PriceReader()
        
        self.sequence_length = 50
        self.cache_ttl = 300
        self.prediction_enabled = getattr(settings, 'ENABLE_UNIVERSAL_PREDICTIONS', True)
        
        self._load_universal_model()
        
        logger.info(f"Universal LSTM Analytics service initialized on device: {self.device}")
    
    def _load_universal_model(self) -> bool:
        """Load trained Universal LSTM model and preprocessor components."""
        if not os.path.exists(self.model_dir):
            logger.warning(f"Universal model directory not found: {self.model_dir}")
            return False
        
        try:
            model_files = [f for f in os.listdir(self.model_dir) if f.startswith('universal_lstm_') and f.endswith('.pth')]
        except OSError:
            logger.warning(f"Cannot access model directory: {self.model_dir}")
            return False
        
        if not model_files:
            logger.warning("No universal LSTM model files found")
            return False
        
        # Get the latest model file
        latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join(self.model_dir, x)))
        model_path = os.path.join(self.model_dir, latest_model)
        
        try:
            # Load universal model
            self.model, self.model_metadata, scalers, preprocessing_params = load_model(model_path, self.device)
            
            # Create preprocessor with fitted scalers
            self.preprocessor = UniversalLSTMPreprocessor(sequence_length=self.sequence_length)
            
            if scalers and 'feature_scaler' in scalers and 'target_scaler' in scalers:
                self.preprocessor.set_scalers(scalers)
                logger.info("Universal model loaded with fitted scalers")
            else:
                logger.warning("Universal model loaded WITHOUT fitted scalers")
            
            self.model.eval()
            logger.info(f"Universal LSTM model loaded successfully from {model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load universal model: {str(e)}")
            return False
    
    def _get_cache_key(self, symbol: str, horizon: str) -> str:
        """Generate cache key for universal predictions."""
        return f"universal_prediction:{symbol}:{horizon}"
    
    def _classify_stock(self, symbol: str) -> tuple:
        """
        Get sector/industry classification for a stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Tuple of (sector_id, industry_id)
        """
        # Try to classify from training universe first
        sector_name = self.sector_mapper.classify_stock_sector(symbol)
        
        if sector_name:
            sector_id = self.sector_mapper.get_sector_id(sector_name)
            industry_id = self.sector_mapper.infer_sector_from_industry(sector_id)
        else:
            # Unknown stock - use default classifications
            sector_id = 10  # Unknown sector
            industry_id = 0  # Default industry
            logger.debug(f"Stock {symbol} not in training universe, using Unknown sector")
        
        return sector_id, industry_id
    
    def _prepare_prediction_data(
        self,
        symbol: str,
        sector_id: int,
        industry_id: int
    ) -> Optional[tuple]:
        """
        Prepare recent price data for universal prediction.
        
        Args:
            symbol: Stock symbol
            sector_id: Sector classification ID
            industry_id: Industry classification ID
            
        Returns:
            Tuple of (features_tensor, sector_tensor, industry_tensor) or None if failed
        """
        try:
            # Fetch recent price data (convert datetime to date for price_reader)
            end_date = datetime.now().date()
            start_date = (datetime.now() - timedelta(days=self.sequence_length + 30)).date()  # Extra buffer
            
            price_data = self.price_reader.get_stock_prices(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if not price_data or len(price_data) < self.sequence_length + 5:
                logger.warning(f"Insufficient price data for {symbol}: got {len(price_data) if price_data else 0}, need {self.sequence_length + 5}")
                
                # Try auto-sync if no data
                if not price_data:
                    from Data.services.yahoo_finance import yahoo_finance_service
                    logger.info(f"Attempting auto-sync for {symbol}")
                    sync_result = yahoo_finance_service.get_stock_data(symbol, period='1y', sync_db=True)
                    if sync_result and 'error' not in sync_result:
                        # Retry fetching after sync
                        price_data = self.price_reader.get_stock_prices(
                            symbol=symbol,
                            start_date=start_date,
                            end_date=end_date
                        )
                        if price_data and len(price_data) >= self.sequence_length + 5:
                            logger.info(f"Auto-sync successful for {symbol}: {len(price_data)} data points")
                        else:
                            logger.warning(f"Auto-sync insufficient for {symbol}: {len(price_data) if price_data else 0} data points")
                            return None
                    else:
                        logger.error(f"Auto-sync failed for {symbol}")
                        return None
                else:
                    return None
            
            # Convert PriceData objects to DataFrame efficiently
            df = pd.DataFrame([{
                'date': p.date,
                'open': float(p.open),
                'high': float(p.high),
                'low': float(p.low),
                'close': float(p.close),
                'volume': int(p.volume)
            } for p in price_data])
            
            # Engineer universal features
            feature_df = self.preprocessor.engineer_universal_features(
                df, 
                symbol=symbol,
                sector_id=sector_id,
                industry_id=industry_id
            )
            
            # Extract the most recent sequence for prediction (optimized)
            feature_cols = [col for col in self.preprocessor.feature_set if col in feature_df.columns]
            
            if len(feature_df) < self.sequence_length:
                logger.warning(f"Insufficient features for {symbol}: {len(feature_df)} < {self.sequence_length}")
                return None
            
            # Get only the required sequence data directly to avoid copying full array
            features = feature_df[feature_cols].iloc[-self.sequence_length:].values
            
            # Transform using fitted scalers
            if self.preprocessor.fitted:
                features = self.preprocessor.transform_features(features)
            
            # Convert to tensors
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)  # Add batch dimension
            sector_tensor = torch.LongTensor([sector_id]).to(self.device)
            industry_tensor = torch.LongTensor([industry_id]).to(self.device)
            
            # Include current price for efficiency
            current_price = float(price_data[-1].close)
            return features_tensor, sector_tensor, industry_tensor, current_price
            
        except Exception as e:
            logger.error(f"Failed to prepare prediction data for {symbol}: {str(e)}")
            return None
    
    def predict_stock_price(
        self,
        symbol: str,
        horizon: str = '1d',
        use_cache: bool = True,
        return_all_outputs: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Predict stock price using Universal LSTM model for Analytics Engine.
        
        Args:
            symbol: Stock symbol (any valid Yahoo Finance symbol)
            horizon: Prediction horizon ('1d', '7d', '30d')
            use_cache: Whether to use cached predictions
            return_all_outputs: Whether to return all multi-task outputs
            
        Returns:
            Universal prediction result with sector-aware confidence
        """
        if not self.prediction_enabled:
            logger.debug("Universal LSTM predictions disabled")
            return None
        
        if not self.model:
            logger.error("Universal model not loaded")
            return None
        
        # Check cache first (faster cache lookup)
        if use_cache:
            cache_key = self._get_cache_key(symbol, horizon)
            cached_result = cache.get(cache_key)
            if cached_result:
                logger.debug(f"Using cached universal prediction for {symbol} {horizon}")
                return cached_result
        
        # Get sector/industry classification
        sector_id, industry_id = self._classify_stock(symbol)
        
        # Prepare prediction data
        prediction_result = self._prepare_prediction_data(symbol, sector_id, industry_id)
        if not prediction_result:
            logger.warning(f"Could not prepare prediction data for {symbol}")
            return None
        
        features_tensor, sector_tensor, industry_tensor, current_price = prediction_result
        
        try:
            # Make universal prediction
            with torch.no_grad():
                if isinstance(self.model, UniversalLSTMPredictor):
                    # Universal model with multi-task outputs
                    outputs = self.model(features_tensor, sector_tensor, industry_tensor)
                    raw_prediction = outputs['price'].item()
                    
                    # CRITICAL FIX: Apply inverse transformation to get percentage return
                    if self.preprocessor.fitted:
                        # Convert raw normalized output back to percentage return
                        predicted_return = self.preprocessor.inverse_transform_targets(np.array([raw_prediction]))[0]
                        
                        # Convert percentage return to absolute price
                        prediction_price = current_price * (1 + predicted_return)
                        
                        # Enhanced sanity check: ensure prediction is realistic
                        min_price = current_price * 0.5  # No more than 50% drop
                        max_price = current_price * 3.0  # No more than 200% increase
                        
                        if prediction_price <= min_price or prediction_price >= max_price:
                            logger.warning(f"Unrealistic prediction for {symbol}: ${prediction_price:.2f} "
                                         f"(raw: {raw_prediction:.6f}, return: {predicted_return:.4f}, "
                                         f"valid range: ${min_price:.2f}-${max_price:.2f})")
                            # Use a conservative prediction: current price ± 5% max
                            predicted_return = np.clip(predicted_return, -0.05, 0.05)
                            prediction_price = current_price * (1 + predicted_return)
                            logger.info(f"Clamped prediction for {symbol}: ${prediction_price:.2f} "
                                       f"(return: {predicted_return:.4f})")
                        
                        # Log successful transformation for debugging
                        logger.debug(f"Prediction transformation for {symbol}: "
                                   f"raw={raw_prediction:.6f} → return={predicted_return:.4f} → price=${prediction_price:.2f}")
                    else:
                        logger.warning(f"Scalers not fitted for {symbol} - using fallback prediction")
                        # Fallback: assume raw prediction is already a percentage return
                        predicted_return = raw_prediction
                        prediction_price = current_price * (1 + predicted_return)
                    
                    # Extract additional outputs if available
                    additional_outputs = {}
                    if return_all_outputs and 'volatility' in outputs:
                        additional_outputs['volatility'] = outputs['volatility'].item()
                    if return_all_outputs and 'trend' in outputs:
                        additional_outputs['trend_probabilities'] = outputs['trend'].cpu().numpy().tolist()
                    if return_all_outputs and 'attention_weights' in outputs:
                        additional_outputs['attention_weights'] = outputs['attention_weights'].cpu().numpy().tolist()
                else:
                    # Fallback for enhanced models
                    prediction_result = self.model.predict(features_tensor, return_confidence=True)
                    prediction_price = prediction_result['prediction']
                    additional_outputs = {}
            
            # Current price already extracted during data preparation for efficiency
            
            # Calculate percentage change using the corrected prediction
            price_change_pct = ((prediction_price - current_price) / current_price) * 100
            
            # Sector-aware confidence estimation
            sector_name = self.sector_mapper.get_sector_name(sector_id)
            base_confidence = 0.75  # Universal model base confidence
            
            # Adjust confidence based on sector classification
            if sector_id != 10:  # Not Unknown sector
                base_confidence += 0.1  # Higher confidence for known sectors
            
            # Prepare comprehensive result
            result = {
                'symbol': symbol,
                'horizon': horizon,
                'predicted_price': float(prediction_price),
                'current_price': float(current_price),
                'price_change': float(prediction_price - current_price),
                'price_change_pct': float(price_change_pct),
                'confidence': float(base_confidence),
                'sector_name': sector_name,
                'sector_id': int(sector_id),
                'industry_id': int(industry_id),
                'model_type': 'UniversalLSTM',
                'model_version': self.model_metadata.get('model_version', '4.0.0') if self.model_metadata else '4.0.0',
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            # Add multi-task outputs if requested
            result.update(additional_outputs)
            
            # Cache result
            if use_cache:
                cache_key = self._get_cache_key(symbol, horizon)
                cache.set(cache_key, result, self.cache_ttl)
            
            logger.debug(
                f"Universal prediction for {symbol} ({sector_name}): "
                f"${prediction_price:.2f} ({price_change_pct:+.2f}%) "
                f"confidence: {base_confidence:.3f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Universal prediction failed for {symbol}: {str(e)}")
            return None
    
    def normalize_prediction_score(
        self,
        prediction_result: Dict[str, Any],
        lookback_days: int = 30
    ) -> float:
        """
        Normalize universal prediction to [0, 1] score for TA engine integration.
        
        Args:
            prediction_result: Prediction result from predict_stock_price
            lookback_days: Days to look back for volatility calculation
            
        Returns:
            Normalized score (0=bearish, 0.5=neutral, 1=bullish)
        """
        if not prediction_result:
            return 0.5  # Neutral if no prediction
        
        price_change_pct = prediction_result.get('price_change_pct', 0)
        confidence = prediction_result.get('confidence', 0.5)
        
        # Enhanced normalization for universal model with sector awareness
        sector_id = prediction_result.get('sector_id', 10)
        
        # Sector-specific volatility adjustments
        sector_volatility_adjustment = {
            0: 1.2,   # Technology - higher volatility
            1: 0.8,   # Healthcare - lower volatility  
            2: 1.0,   # Financial Services - moderate
            3: 1.3,   # Energy - higher volatility
            4: 1.1,   # Consumer Cyclical - moderate-high
            5: 0.9,   # Industrials - moderate-low
            10: 1.0   # Unknown - neutral
        }.get(sector_id, 1.0)
        
        # Adjust prediction based on sector volatility
        adjusted_change = price_change_pct / sector_volatility_adjustment
        
        # Simple normalization based on adjusted price change and confidence
        if adjusted_change > 0:
            # Bullish prediction
            base_score = 0.5 + min(adjusted_change / 20.0, 0.4)  # Cap at 0.9
        else:
            # Bearish prediction
            base_score = 0.5 + max(adjusted_change / 20.0, -0.4)  # Floor at 0.1
        
        # Weight by confidence
        score = 0.5 + (base_score - 0.5) * confidence
        
        # Ensure score is in [0, 1]
        return max(0.0, min(1.0, score))
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the loaded Universal LSTM model."""
        if not self.model:
            return None
        
        model_info = {
            'model_type': 'UniversalLSTM',
            'device': str(self.device),
            'model_dir': self.model_dir,
            'sequence_length': self.sequence_length,
            'sectors_supported': 11,
            'industries_supported': 50,
            'cache_ttl': self.cache_ttl,
            'prediction_enabled': self.prediction_enabled
        }
        
        if self.model_metadata:
            model_info.update({
                'model_version': self.model_metadata.get('model_version', '4.0.0'),
                'saved_at': self.model_metadata.get('saved_at'),
                'training_stocks': len(self.model_metadata.get('training_stocks', [])),
                'model_config': self.model_metadata.get('model_config', {})
            })
        
        if hasattr(self.model, 'get_model_info'):
            model_info.update(self.model.get_model_info())
        
        return model_info
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear prediction cache for symbol or all symbols."""
        if symbol:
            # Clear cache for specific symbol
            for horizon in ['1d', '7d', '30d']:
                cache_key = self._get_cache_key(symbol, horizon)
                cache.delete(cache_key)
            logger.info(f"Cleared universal prediction cache for {symbol}")
        else:
            # Clear all cached predictions
            logger.info("Cleared all universal prediction cache")
    
    def reload_model(self) -> bool:
        """Reload the Universal LSTM model (useful for model updates)."""
        logger.info("Reloading Universal LSTM model...")
        return self._load_universal_model()


# Singleton instance for Analytics Engine
_universal_lstm_analytics_service = None


def get_universal_lstm_service() -> UniversalLSTMAnalyticsService:
    """Get or create singleton Universal LSTM Analytics service instance."""
    global _universal_lstm_analytics_service
    if _universal_lstm_analytics_service is None:
        _universal_lstm_analytics_service = UniversalLSTMAnalyticsService()
    return _universal_lstm_analytics_service
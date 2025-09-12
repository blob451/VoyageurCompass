"""
Universal LSTM Data Preprocessor for Multi-Stock Training
Implements enhanced feature engineering for cross-sector learning.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

# Conditional import for ML dependencies
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# Type hints for when torch is available
if TYPE_CHECKING and TORCH_AVAILABLE:
    from torch import Tensor
else:
    Tensor = Any

from Analytics.ml.sector_mappings import (
    UNIVERSAL_FEATURES,
    get_sector_mapper,
)

logger = logging.getLogger(__name__)


class UniversalLSTMPreprocessor:
    """
    Enhanced preprocessor for Universal LSTM model supporting multi-stock training.

    Features:
    - Normalized feature pipeline for cross-stock compatibility
    - Sector-relative metrics calculation
    - Price and volume normalization across different scales
    - Market regime detection
    - Cross-sector correlation features
    """

    def __init__(
        self,
        sequence_length: int = 60,
        feature_set: Optional[List[str]] = None,
        price_normalization: str = "percentage",  # 'percentage', 'log', 'minmax'
        sector_normalization: bool = True,
        market_regime_detection: bool = True,
    ):
        """
        Initialize universal preprocessor.

        Args:
            sequence_length: Length of input sequences (60 days for universal model)
            feature_set: List of features to use (defaults to UNIVERSAL_FEATURES)
            price_normalization: Method for price normalization
            sector_normalization: Whether to apply sector-relative normalization
            market_regime_detection: Whether to detect market regimes
        """
        self.sequence_length = sequence_length
        self.feature_set = feature_set or UNIVERSAL_FEATURES.copy()
        self.price_normalization = price_normalization
        self.sector_normalization = sector_normalization
        self.market_regime_detection = market_regime_detection

        # Scalers for different feature types
        self.feature_scaler = RobustScaler()  # Better for financial data with outliers
        self.target_scaler = StandardScaler()  # Better for percentage returns
        self.volume_scaler = RobustScaler()

        # Sector mapper
        self.sector_mapper = get_sector_mapper()

        # Fitted status
        self.fitted = False

        # Market regime thresholds
        self.volatility_regimes = {
            "low": 0.02,  # < 2% daily volatility
            "medium": 0.04,  # 2-4% daily volatility
            "high": 0.04,  # > 4% daily volatility
        }

        logger.info(f"Universal preprocessor initialized with {len(self.feature_set)} features")

    def engineer_universal_features(
        self, df: pd.DataFrame, symbol: str, sector_id: Optional[int] = None, industry_id: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Engineer all 42 universal features for LSTM model compatibility.

        Args:
            df: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
            symbol: Stock symbol for context
            sector_id: Sector ID (will be inferred if not provided)
            industry_id: Industry ID (will be inferred if not provided)

        Returns:
            DataFrame with all 42 engineered universal features
        """
        # Work on a copy
        feature_df = df.copy()

        # Ensure required columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in feature_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Infer sector/industry if not provided
        if sector_id is None:
            sector_name = self.sector_mapper.classify_stock_sector(symbol)
            sector_id = self.sector_mapper.get_sector_id(sector_name or "Unknown")

        if industry_id is None:
            industry_id = self.sector_mapper.infer_sector_from_industry(sector_id) if sector_id else 0

        logger.debug(f"Engineering all 42 features for {symbol} (sector: {sector_id}, industry: {industry_id})")

        # 1. Price dynamics (normalized) - 8 features
        feature_df["price_change_pct"] = feature_df["close"].pct_change().fillna(0)
        feature_df["price_range_pct"] = (feature_df["high"] - feature_df["low"]) / feature_df["close"]
        feature_df["volume_change_pct"] = feature_df["volume"].pct_change().fillna(0)
        feature_df["price_change_1d"] = feature_df["close"].pct_change(1).fillna(0)
        feature_df["price_change_3d"] = feature_df["close"].pct_change(3).fillna(0)
        feature_df["price_change_7d"] = feature_df["close"].pct_change(7).fillna(0)
        feature_df["high_low_pct"] = (feature_df["high"] - feature_df["low"]) / feature_df["low"]
        feature_df["open_close_pct"] = (feature_df["close"] - feature_df["open"]) / feature_df["open"]

        # 2. Moving averages (relative) - 8 features
        feature_df["sma10"] = feature_df["close"].rolling(10).mean()
        feature_df["sma20"] = feature_df["close"].rolling(20).mean()
        feature_df["sma50"] = feature_df["close"].rolling(50).mean()
        feature_df["sma200"] = feature_df["close"].rolling(200).mean()

        feature_df["price_vs_sma10"] = (feature_df["close"] - feature_df["sma10"]) / feature_df["sma10"]
        feature_df["price_vs_sma20"] = (feature_df["close"] - feature_df["sma20"]) / feature_df["sma20"]
        feature_df["price_vs_sma50"] = (feature_df["close"] - feature_df["sma50"]) / feature_df["sma50"]
        feature_df["price_vs_sma200"] = ((feature_df["close"] - feature_df["sma200"]) / feature_df["sma200"]).fillna(0)
        feature_df["sma10_vs_sma20"] = ((feature_df["sma10"] - feature_df["sma20"]) / feature_df["sma20"]).fillna(0)
        feature_df["sma20_vs_sma50"] = ((feature_df["sma20"] - feature_df["sma50"]) / feature_df["sma50"]).fillna(0)
        feature_df["sma50_vs_sma200"] = ((feature_df["sma50"] - feature_df["sma200"]) / feature_df["sma200"]).fillna(0)
        
        # EMA crossover signal
        ema12 = feature_df["close"].ewm(span=12).mean()
        ema26 = feature_df["close"].ewm(span=26).mean()
        feature_df["ema_crossover_signal"] = np.where(ema12 > ema26, 1.0, -1.0)

        # 3. Technical indicators (sector-normalized) - 8 features
        feature_df = self._calculate_rsi(feature_df)
        feature_df = self._calculate_macd(feature_df)
        feature_df = self._calculate_bollinger_bands(feature_df)
        feature_df = self._calculate_stochastic(feature_df)
        feature_df = self._calculate_williams_r(feature_df)
        feature_df = self._calculate_cci(feature_df)

        # 4. Volume patterns - 6 features
        feature_df["volume_sma20"] = feature_df["volume"].rolling(20).mean()
        feature_df["volume_ratio"] = (feature_df["volume"] / feature_df["volume_sma20"]).fillna(1.0)
        feature_df["volume_price_trend"] = feature_df["volume_change_pct"] * feature_df["price_change_pct"]
        feature_df["volume_sma_ratio"] = (feature_df["volume"] / feature_df["volume"].rolling(50).mean()).fillna(1.0)
        feature_df = self._calculate_volume_weighted_price(feature_df)
        feature_df = self._calculate_accumulation_distribution(feature_df)
        feature_df = self._calculate_obv(feature_df)

        # 5. Momentum (time-normalized) - 6 features
        feature_df["momentum_5d"] = feature_df["close"].pct_change(5).fillna(0)
        feature_df["momentum_20d"] = feature_df["close"].pct_change(20).fillna(0)
        feature_df["roc_10d"] = (
            (feature_df["close"] - feature_df["close"].shift(10)) / feature_df["close"].shift(10)
        ).fillna(0)
        feature_df["roc_20d"] = (
            (feature_df["close"] - feature_df["close"].shift(20)) / feature_df["close"].shift(20)
        ).fillna(0)
        feature_df = self._calculate_trix(feature_df)
        feature_df = self._calculate_ultimate_oscillator(feature_df)

        # 6. Volatility (regime-aware) - 6 features
        feature_df["volatility_10d"] = feature_df["price_change_pct"].rolling(10).std()
        feature_df["volatility_20d"] = feature_df["price_change_pct"].rolling(20).std()
        feature_df["volatility_ratio"] = feature_df["volatility_10d"] / feature_df["volatility_20d"]
        feature_df = self._calculate_atr(feature_df)
        feature_df = self._calculate_volatility_trend(feature_df)
        feature_df = self._calculate_volatility_breakout(feature_df)

        # GARCH volatility (simplified implementation)
        feature_df["garch_volatility"] = feature_df["price_change_pct"].rolling(20).std()

        # Add sector-relative normalization if enabled
        if self.sector_normalization:
            feature_df = self._apply_sector_normalization(feature_df, sector_id)

        # Fill any remaining NaN values
        feature_df = feature_df.ffill().fillna(0)

        # Add metadata columns
        feature_df["sector_id"] = sector_id
        feature_df["industry_id"] = industry_id
        feature_df["symbol"] = symbol

        # Debug: Log all columns created
        all_cols = list(feature_df.columns)
        feature_cols_only = [col for col in all_cols if col in self.feature_set]
        extra_cols = [col for col in all_cols if col not in self.feature_set and col not in ['sector_id', 'industry_id', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
        
        logger.debug(f"Feature engineering for {symbol} completed:")
        logger.debug(f"  Total columns: {len(all_cols)}")
        logger.debug(f"  Target features: {len(feature_cols_only)}/{len(self.feature_set)}")
        logger.debug(f"  Extra columns: {len(extra_cols)} - {extra_cols}")

        return feature_df

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate RSI with sector-relative normalization."""
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Normalize RSI to [-1, 1] range instead of [0, 100]
        df["rsi_sector_relative"] = (rsi - 50) / 50

        return df

    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD with normalization."""
        ema_fast = df["close"].ewm(span=fast).mean()
        ema_slow = df["close"].ewm(span=slow).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        # Normalize MACD histogram by current price
        df["macd_normalized"] = histogram / df["close"]

        return df

    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands position and width."""
        sma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        # Bollinger Band position (%B)
        df["bollinger_position"] = (df["close"] - lower_band) / (upper_band - lower_band)
        
        # Bollinger Band width (normalized by SMA)
        df["bollinger_width"] = (upper_band - lower_band) / sma

        return df

    def _calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Calculate support and resistance touch indicators."""
        # Rolling max/min as resistance/support levels
        resistance = df["high"].rolling(window).max()
        support = df["low"].rolling(window).min()

        # Touch indicators (1 if touching, 0 otherwise)
        df["resistance_touch"] = (df["high"] >= resistance * 0.99).astype(float)
        df["support_touch"] = (df["low"] <= support * 1.01).astype(float)

        return df

    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Calculate Stochastic oscillator."""
        lowest_low = df["low"].rolling(k_period).min()
        highest_high = df["high"].rolling(k_period).max()
        
        k_percent = 100 * ((df["close"] - lowest_low) / (highest_high - lowest_low))
        df["stochastic_k"] = (k_percent - 50) / 50  # Normalize to [-1, 1]
        df["stochastic_d"] = (k_percent.rolling(d_period).mean() - 50) / 50  # %D line
        
        return df

    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Williams %R."""
        highest_high = df["high"].rolling(period).max()
        lowest_low = df["low"].rolling(period).min()
        
        williams_r = -100 * ((highest_high - df["close"]) / (highest_high - lowest_low))
        df["williams_r"] = (williams_r + 50) / 50  # Normalize to [-1, 1]
        
        return df

    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Commodity Channel Index."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        sma_tp = typical_price.rolling(period).mean()
        mean_deviation = typical_price.rolling(period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        df["cci_normalized"] = np.tanh(cci / 100)  # Normalize using tanh
        
        return df

    def _calculate_volume_weighted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Weighted Average Price."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
        df["volume_weighted_price"] = (df["close"] - vwap) / vwap
        
        return df

    def _calculate_accumulation_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Accumulation/Distribution Line."""
        close_loc_value = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
        close_loc_value = close_loc_value.fillna(0)
        ad_line = (close_loc_value * df["volume"]).cumsum()
        ad_mean = ad_line.rolling(50).mean()
        df["accumulation_distribution"] = ((ad_line / ad_mean) - 1).fillna(0)  # Normalize
        
        return df

    def _calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume."""
        obv_values = np.where(df["close"] > df["close"].shift(1), df["volume"], 
                             np.where(df["close"] < df["close"].shift(1), -df["volume"], 0))
        obv_series = pd.Series(obv_values, index=df.index).cumsum()
        obv_mean = obv_series.rolling(50).mean()
        df["obv_normalized"] = ((obv_series / obv_mean) - 1).fillna(0)  # Normalize
        
        return df

    def _calculate_trix(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate TRIX indicator."""
        ema1 = df["close"].ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()
        ema3 = ema2.ewm(span=period).mean()
        trix = ema3.pct_change() * 10000  # Convert to basis points
        df["trix_signal"] = np.tanh(trix / 100)  # Normalize
        
        return df

    def _calculate_ultimate_oscillator(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ultimate Oscillator."""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift(1))
        low_close = np.abs(df["low"] - df["close"].shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        
        close_low = df["close"] - np.minimum(df["low"], df["close"].shift(1))
        
        bp7 = close_low.rolling(7).sum()
        tr7 = true_range.rolling(7).sum()
        bp14 = close_low.rolling(14).sum()
        tr14 = true_range.rolling(14).sum()
        bp28 = close_low.rolling(28).sum()
        tr28 = true_range.rolling(28).sum()
        
        uo = 100 * ((4 * bp7/tr7) + (2 * bp14/tr14) + (bp28/tr28)) / (4 + 2 + 1)
        df["ultimate_oscillator"] = (uo - 50) / 50  # Normalize to [-1, 1]
        
        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range."""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift(1))
        low_close = np.abs(df["low"] - df["close"].shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        df["atr_normalized"] = atr / df["close"]  # Normalize by price
        
        return df

    def _calculate_volatility_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility trend."""
        volatility = df["price_change_pct"].rolling(20).std()
        vol_trend = volatility.pct_change(5)  # 5-day volatility change
        df["volatility_trend"] = np.tanh(vol_trend * 100)  # Normalize
        
        return df

    def _calculate_volatility_breakout(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility breakout signal."""
        volatility = df["price_change_pct"].rolling(20).std()
        vol_mean = volatility.rolling(50).mean()
        vol_std = volatility.rolling(50).std()
        vol_breakout = (volatility - vol_mean) / vol_std
        df["volatility_breakout"] = np.tanh(vol_breakout)  # Normalize
        
        return df

    def _detect_market_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect market volatility and trend regimes."""
        # Volatility regime based on rolling volatility
        vol_10d = df["price_change_pct"].rolling(10).std()

        def volatility_regime(vol):
            if vol < self.volatility_regimes["low"]:
                return 0.0  # Low volatility
            elif vol < self.volatility_regimes["medium"]:
                return 0.5  # Medium volatility
            else:
                return 1.0  # High volatility

        df["volatility_regime"] = vol_10d.apply(volatility_regime)

        # Trend strength based on price momentum consistency
        momentum_5d = df["close"].pct_change(5)
        momentum_20d = df["close"].pct_change(20)

        # Trend strength: alignment between short and long-term momentum
        trend_alignment = np.sign(momentum_5d) * np.sign(momentum_20d)
        df["trend_strength"] = (trend_alignment + 1) / 2  # Normalize to [0, 1]

        return df

    def _apply_sector_normalization(self, df: pd.DataFrame, sector_id: int) -> pd.DataFrame:
        """Apply sector-relative normalization to certain features."""
        # This is a placeholder - in production, this would normalize features
        # relative to sector averages loaded from historical data

        # For now, apply a simple sector-based adjustment
        sector_adjustment = 1.0 + (sector_id - 5) * 0.05  # Simple linear adjustment

        # Apply to RSI (already sector-relative in name)
        if "rsi_sector_relative" in df.columns:
            df["rsi_sector_relative"] *= sector_adjustment

        return df

    def prepare_universal_sequences(
        self, feature_dfs: Dict[str, pd.DataFrame], target_column: str = "close"
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Prepare sequences for universal training from multiple stocks.
        Uses percentage returns as targets for better scaling and cross-stock compatibility.

        Args:
            feature_dfs: Dictionary mapping stock symbols to feature DataFrames
            target_column: Column name for prediction target (will be converted to returns)

        Returns:
            Tuple of (features, targets, sector_ids, industry_ids) tensors
        """
        all_sequences = []
        all_targets = []
        all_sector_ids = []
        all_industry_ids = []

        for symbol, df in feature_dfs.items():
            if len(df) < self.sequence_length + 1:
                logger.warning(f"Insufficient data for {symbol}: {len(df)} < {self.sequence_length + 1}")
                continue

            # Extract features (excluding metadata and target)
            feature_cols = [col for col in self.feature_set if col in df.columns]

            # Debug: Log actual feature count
            if len(feature_cols) != len(self.feature_set):
                missing_features = [col for col in self.feature_set if col not in df.columns]
                available_features = [col for col in df.columns if col not in ['sector_id', 'industry_id', 'symbol', 'sma10', 'sma20', 'sma50', 'sma200', 'volume_sma20']]
                logger.error(f"Feature mismatch for {symbol}:")
                logger.error(f"  Expected {len(self.feature_set)} features, got {len(feature_cols)}")
                logger.error(f"  Missing features: {missing_features}")
                logger.error(f"  Available features: {len(available_features)} - {available_features}")
                
                # For debugging, use only available features from the expected set
                features = df[feature_cols].values
            else:
                features = df[feature_cols].values

            # Convert targets to percentage returns for better scaling
            price_data = df[target_column].values
            returns = np.concatenate([[0], np.diff(price_data) / price_data[:-1]])  # Next-day returns
            returns = np.nan_to_num(returns, 0)  # Handle any NaN values

            # Get sector/industry IDs
            sector_id = df["sector_id"].iloc[0] if "sector_id" in df.columns else 10
            industry_id = df["industry_id"].iloc[0] if "industry_id" in df.columns else 0

            # Create sequences
            for i in range(len(features) - self.sequence_length):
                sequence = features[i : i + self.sequence_length]
                target = returns[i + self.sequence_length]  # Predict next-day return

                all_sequences.append(sequence)
                all_targets.append(target)
                all_sector_ids.append(sector_id)
                all_industry_ids.append(industry_id)

        if not all_sequences:
            raise ValueError("No valid sequences created from provided data")

        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available - cannot create tensors for training")
            raise RuntimeError("PyTorch required for sequence preparation")

        # Convert to tensors
        sequences_array = np.array(all_sequences)
        logger.info(f"Sequences array shape before tensor conversion: {sequences_array.shape}")
        
        features_tensor = torch.FloatTensor(sequences_array)
        targets_tensor = torch.FloatTensor(np.array(all_targets))
        sector_ids_tensor = torch.LongTensor(np.array(all_sector_ids))
        industry_ids_tensor = torch.LongTensor(np.array(all_industry_ids))

        logger.info(f"Created {len(all_sequences)} sequences from {len(feature_dfs)} stocks")
        logger.info(f"Features tensor shape: {features_tensor.shape}")
        logger.info(f"Expected shape: [batch_size, {self.sequence_length}, {len(self.feature_set)}]")

        return features_tensor, targets_tensor, sector_ids_tensor, industry_ids_tensor

    def fit_universal_scalers(self, feature_dfs: Dict[str, pd.DataFrame], target_column: str = "close") -> None:
        """
        Fit scalers on combined data from all stocks for universal normalization.

        Args:
            feature_dfs: Dictionary mapping stock symbols to feature DataFrames
            target_column: Column name for prediction target
        """
        all_features = []
        all_targets = []

        for symbol, df in feature_dfs.items():
            # Extract features
            feature_cols = [col for col in self.feature_set if col in df.columns]
            
            # Debug: Check feature count per stock
            if len(feature_cols) != len(self.feature_set):
                missing = [col for col in self.feature_set if col not in df.columns]
                available = [col for col in df.columns if col not in ['sector_id', 'industry_id', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                logger.error(f"Scaler fitting feature mismatch for {symbol}:")
                logger.error(f"  Expected: {len(self.feature_set)}, Got: {len(feature_cols)}")
                logger.error(f"  Missing: {missing}")
                logger.error(f"  Available non-raw: {len(available)} features")
            
            features = df[feature_cols].values

            # Convert targets to percentage returns for consistent scaling
            price_data = df[target_column].values
            returns = np.concatenate([[0], np.diff(price_data) / price_data[:-1]])
            returns = np.nan_to_num(returns, 0)

            all_features.append(features)
            all_targets.append(returns)

        if not all_features:
            raise ValueError("No feature data provided for fitting scalers")

        # Combine all features and targets
        combined_features = np.vstack(all_features)
        combined_targets = np.concatenate(all_targets)

        # Fit scalers
        self.feature_scaler.fit(combined_features)
        self.target_scaler.fit(combined_targets.reshape(-1, 1))

        self.fitted = True
        logger.info(f"Universal scalers fitted on {len(combined_features)} samples from {len(feature_dfs)} stocks")

    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler."""
        if not self.fitted:
            logger.warning("Scalers not fitted - returning unscaled features (prediction quality may be reduced)")
            return features  # Return unscaled features as fallback
        return self.feature_scaler.transform(features)

    def transform_targets(self, targets: np.ndarray) -> np.ndarray:
        """Transform targets using fitted scaler."""
        if not self.fitted:
            logger.warning("Target scaler not fitted - returning unscaled targets")
            return targets  # Return unscaled targets as fallback
        return self.target_scaler.transform(targets.reshape(-1, 1)).flatten()

    def inverse_transform_targets(self, scaled_targets: np.ndarray) -> np.ndarray:
        """Inverse transform scaled targets back to original scale."""
        if not self.fitted:
            raise ValueError("Scalers not fitted. Cannot inverse transform.")
        return self.target_scaler.inverse_transform(scaled_targets.reshape(-1, 1)).flatten()

    def set_scalers(self, scalers: Dict[str, Any]) -> None:
        """
        Set fitted scalers from loaded model.

        Args:
            scalers: Dictionary containing 'feature_scaler' and 'target_scaler'
        """
        if "feature_scaler" in scalers:
            feature_scaler = scalers["feature_scaler"]
            if isinstance(feature_scaler, str):
                # Handle case where scaler was saved as string name
                if feature_scaler == "UniversalRobustScaler":
                    self.feature_scaler = RobustScaler()
                    logger.warning("Recreated RobustScaler from string name (needs retraining for proper fit)")
                else:
                    logger.error(f"Unknown feature scaler string: {feature_scaler}")
                    self.feature_scaler = RobustScaler()
            else:
                self.feature_scaler = feature_scaler

        if "target_scaler" in scalers:
            target_scaler = scalers["target_scaler"]
            if isinstance(target_scaler, str):
                # Handle case where scaler was saved as string name
                if target_scaler == "UniversalMinMaxScaler":
                    self.target_scaler = StandardScaler()  # Use StandardScaler instead
                    logger.warning("Recreated StandardScaler from string name (needs retraining for proper fit)")
                else:
                    logger.error(f"Unknown target scaler string: {target_scaler}")
                    self.target_scaler = StandardScaler()
            else:
                self.target_scaler = target_scaler

        # For string-based scalers, we need to disable fitted status since they're not actually fitted
        if isinstance(scalers.get("feature_scaler"), str) or isinstance(scalers.get("target_scaler"), str):
            self.fitted = False
            logger.warning(
                "Scalers loaded from strings - not fitted. Predictions will use unfitted scalers (suboptimal)"
            )
        else:
            self.fitted = True
            logger.info("Fitted scalers loaded into preprocessor")

    def get_feature_importance_weights(self) -> Dict[str, float]:
        """
        Get feature importance weights for universal model training.

        Returns:
            Dictionary mapping feature names to importance weights
        """
        # Define importance based on financial domain knowledge
        importance_weights = {
            # High importance: core price and volume features
            "price_change_pct": 1.0,
            "volume_change_pct": 0.8,
            "price_vs_sma20": 0.9,
            "price_vs_sma50": 0.9,
            # Medium-high importance: technical indicators
            "rsi_sector_relative": 0.8,
            "macd_normalized": 0.8,
            "bollinger_position": 0.7,
            "momentum_20d": 0.7,
            # Medium importance: volatility and regime indicators
            "volatility_10d": 0.6,
            "volatility_ratio": 0.6,
            "volatility_regime": 0.6,
            "trend_strength": 0.6,
            # Lower importance: microstructure and correlation features
            "volume_ratio": 0.5,
            "sector_correlation": 0.4,
            "market_beta": 0.4,
            "bid_ask_spread_proxy": 0.3,
        }

        # Ensure all features have weights (default to 0.5)
        for feature in self.feature_set:
            if feature not in importance_weights:
                importance_weights[feature] = 0.5

        return importance_weights

"""
Adaptive Technical Indicators

Provides intelligent technical indicator calculations that adapt to available data.
Implements fallback strategies and confidence scoring for limited data scenarios.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class IndicatorResult:
    """Result of an indicator calculation"""
    value: float
    confidence: float  # 0-1 scale
    adapted: bool
    fallback_used: bool
    data_points_used: int
    original_name: str
    adapted_name: Optional[str] = None


class AdaptiveIndicatorEngine:
    """
    Adaptive technical indicator engine that adjusts calculations based on available data.
    """
    
    def __init__(self):
        self.adaptation_rules = {
            'sma50vs200': {
                'adaptations': [
                    {'min_days': 100, 'periods': [20, 50], 'name': 'SMA20vs50'},
                    {'min_days': 50, 'periods': [10, 20], 'name': 'SMA10vs20'},
                    {'min_days': 30, 'periods': [5, 15], 'name': 'SMA5vs15'}
                ],
                'fallback': 'momentum_trend'
            },
            'pricevs50': {
                'adaptations': [
                    {'min_days': 25, 'period': 20, 'name': 'PricevsSMA20'},
                    {'min_days': 15, 'period': 10, 'name': 'PricevsSMA10'},
                    {'min_days': 10, 'period': 5, 'name': 'PricevsSMA5'}
                ],
                'fallback': 'price_momentum'
            },
            'macd12269': {
                'adaptations': [
                    {'min_days': 30, 'fast': 8, 'slow': 17, 'signal': 6, 'name': 'MACD8176'},
                    {'min_days': 20, 'fast': 5, 'slow': 10, 'signal': 3, 'name': 'MACD5103'},
                    {'min_days': 15, 'fast': 3, 'slow': 7, 'signal': 2, 'name': 'MACD372'}
                ],
                'fallback': 'price_acceleration'
            },
            'srcontext': {
                'fallback': 'bollinger_levels'
            },
            'rel1y': {
                'adaptations': [
                    {'min_days': 120, 'period': 90, 'name': 'Rel90d'},
                    {'min_days': 60, 'period': 45, 'name': 'Rel45d'},
                    {'min_days': 30, 'period': 20, 'name': 'Rel20d'}
                ],
                'fallback': 'sector_relative'
            },
            'rel2y': {
                'adaptations': [
                    {'min_days': 250, 'period': 180, 'name': 'Rel180d'},
                    {'min_days': 120, 'period': 90, 'name': 'Rel90d'},
                    {'min_days': 60, 'period': 45, 'name': 'Rel45d'}
                ],
                'fallback': 'industry_relative'
            }
        }
        
    def calculate_adaptive_indicator(
        self, 
        indicator_name: str, 
        data: pd.DataFrame, 
        sector_data: Optional[pd.DataFrame] = None,
        industry_data: Optional[pd.DataFrame] = None
    ) -> IndicatorResult:
        """
        Calculate an indicator with adaptive fallback.
        
        Args:
            indicator_name: Name of the indicator to calculate
            data: Price data DataFrame
            sector_data: Sector composite data (optional)
            industry_data: Industry composite data (optional)
            
        Returns:
            IndicatorResult with calculation results
        """
        data_points = len(data)
        
        if indicator_name not in self.adaptation_rules:
            # No adaptation rules - try original calculation
            return self._calculate_original_indicator(indicator_name, data)
        
        rules = self.adaptation_rules[indicator_name]
        
        # Try adaptations in order of preference
        if 'adaptations' in rules:
            for adaptation in rules['adaptations']:
                if data_points >= adaptation['min_days']:
                    return self._calculate_adapted_indicator(
                        indicator_name, data, adaptation, data_points
                    )
        
        # Use fallback if no adaptations work
        if 'fallback' in rules:
            return self._calculate_fallback_indicator(
                indicator_name, rules['fallback'], data, sector_data, industry_data
            )
        
        # No adaptations or fallbacks available
        return IndicatorResult(
            value=0.5,  # Neutral score
            confidence=0.0,
            adapted=False,
            fallback_used=True,
            data_points_used=data_points,
            original_name=indicator_name,
            adapted_name='neutral_fallback'
        )
        
    def _calculate_original_indicator(self, indicator_name: str, data: pd.DataFrame) -> IndicatorResult:
        """Calculate original indicator without adaptation."""
        try:
            if indicator_name == 'obv20':
                return self._calculate_obv(data, 20)
            elif indicator_name == 'volsurge':
                return self._calculate_volume_surge(data)
            elif indicator_name == 'bbwidth20':
                return self._calculate_bb_width(data, 20)
            elif indicator_name == 'rsi14':
                return self._calculate_rsi(data, 14)
            elif indicator_name == 'candlerev':
                return self._calculate_candle_reversal(data)
            elif indicator_name == 'bbpos20':
                return self._calculate_bb_position(data, 20)
            else:
                return IndicatorResult(
                    value=0.5, confidence=0.0, adapted=False, fallback_used=True,
                    data_points_used=len(data), original_name=indicator_name
                )
        except Exception as e:
            logger.error(f"Error calculating {indicator_name}: {str(e)}")
            return IndicatorResult(
                value=0.5, confidence=0.0, adapted=False, fallback_used=True,
                data_points_used=len(data), original_name=indicator_name
            )
            
    def _calculate_adapted_indicator(
        self, 
        indicator_name: str, 
        data: pd.DataFrame, 
        adaptation: Dict[str, Any],
        data_points: int
    ) -> IndicatorResult:
        """Calculate adapted version of indicator."""
        try:
            if indicator_name == 'sma50vs200':
                return self._calculate_sma_comparison(data, adaptation['periods'], adaptation['name'])
            elif indicator_name == 'pricevs50':
                return self._calculate_price_vs_sma(data, adaptation['period'], adaptation['name'])
            elif indicator_name == 'macd12269':
                return self._calculate_macd(
                    data, adaptation['fast'], adaptation['slow'], adaptation['signal'], adaptation['name']
                )
            elif indicator_name == 'rel1y' or indicator_name == 'rel2y':
                return self._calculate_relative_performance(data, adaptation['period'], adaptation['name'])
            else:
                return IndicatorResult(
                    value=0.5, confidence=0.3, adapted=True, fallback_used=False,
                    data_points_used=data_points, original_name=indicator_name,
                    adapted_name=adaptation['name']
                )
        except Exception as e:
            logger.error(f"Error calculating adapted {indicator_name}: {str(e)}")
            return IndicatorResult(
                value=0.5, confidence=0.1, adapted=True, fallback_used=True,
                data_points_used=data_points, original_name=indicator_name,
                adapted_name=adaptation['name']
            )
            
    def _calculate_fallback_indicator(
        self, 
        indicator_name: str, 
        fallback_name: str, 
        data: pd.DataFrame,
        sector_data: Optional[pd.DataFrame] = None,
        industry_data: Optional[pd.DataFrame] = None
    ) -> IndicatorResult:
        """Calculate fallback indicator."""
        try:
            if fallback_name == 'momentum_trend':
                return self._calculate_momentum_trend(data)
            elif fallback_name == 'price_momentum':
                return self._calculate_price_momentum(data)
            elif fallback_name == 'price_acceleration':
                return self._calculate_price_acceleration(data)
            elif fallback_name == 'bollinger_levels':
                return self._calculate_bollinger_levels(data)
            elif fallback_name == 'sector_relative' and sector_data is not None:
                return self._calculate_sector_relative(data, sector_data)
            elif fallback_name == 'industry_relative' and industry_data is not None:
                return self._calculate_industry_relative(data, industry_data)
            else:
                return IndicatorResult(
                    value=0.5, confidence=0.2, adapted=False, fallback_used=True,
                    data_points_used=len(data), original_name=indicator_name,
                    adapted_name=fallback_name
                )
        except Exception as e:
            logger.error(f"Error calculating fallback {fallback_name}: {str(e)}")
            return IndicatorResult(
                value=0.5, confidence=0.0, adapted=False, fallback_used=True,
                data_points_used=len(data), original_name=indicator_name,
                adapted_name=fallback_name
            )
            
    # Specific indicator calculations
    def _calculate_obv(self, data: pd.DataFrame, period: int) -> IndicatorResult:
        """Calculate On-Balance Volume with confidence scoring."""
        if len(data) < period:
            confidence = len(data) / period * 0.5
        else:
            confidence = 0.9
            
        # Calculate OBV
        obv = (data['volume'] * np.where(data['close'] > data['close'].shift(1), 1, -1)).cumsum()
        
        current_obv = obv.iloc[-1]
        past_obv = obv.iloc[-period] if len(obv) >= period else obv.iloc[0]
        
        obv_delta = (current_obv - past_obv) / abs(past_obv) if past_obv != 0 else 0
        
        # Normalize to 0-1 scale
        normalized_value = max(0, min(1, (obv_delta + 1) / 2))
        
        return IndicatorResult(
            value=normalized_value,
            confidence=confidence,
            adapted=len(data) < period,
            fallback_used=False,
            data_points_used=len(data),
            original_name='obv20'
        )
        
    def _calculate_volume_surge(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate volume surge indicator."""
        if len(data) < 10:
            avg_volume = data['volume'].mean()
            confidence = 0.5
        else:
            avg_volume = data['volume'].rolling(20, min_periods=5).mean().iloc[-1]
            confidence = 0.8
            
        current_volume = data['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Price direction
        price_up = data['close'].iloc[-1] > data['close'].iloc[-2]
        
        # Score based on volume surge and price direction
        if volume_ratio > 1.5 and price_up:
            score = 0.8
        elif volume_ratio > 1.2 and price_up:
            score = 0.6
        elif volume_ratio > 1.0 and price_up:
            score = 0.55
        else:
            score = 0.4
            
        return IndicatorResult(
            value=score,
            confidence=confidence,
            adapted=len(data) < 20,
            fallback_used=False,
            data_points_used=len(data),
            original_name='volsurge'
        )
        
    def _calculate_sma_comparison(self, data: pd.DataFrame, periods: List[int], name: str) -> IndicatorResult:
        """Calculate SMA comparison with adaptive periods."""
        short_period, long_period = periods
        
        if len(data) < long_period:
            confidence = len(data) / long_period * 0.6
        else:
            confidence = 0.8
            
        short_sma = data['close'].rolling(short_period, min_periods=min(short_period, len(data))).mean()
        long_sma = data['close'].rolling(long_period, min_periods=min(long_period, len(data))).mean()
        
        current_short = short_sma.iloc[-1]
        current_long = long_sma.iloc[-1]
        
        if current_short > current_long:
            score = 0.7
        elif current_short < current_long:
            score = 0.3
        else:
            score = 0.5
            
        return IndicatorResult(
            value=score,
            confidence=confidence,
            adapted=True,
            fallback_used=False,
            data_points_used=len(data),
            original_name='sma50vs200',
            adapted_name=name
        )
        
    def _calculate_momentum_trend(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate momentum trend fallback."""
        if len(data) < 5:
            return IndicatorResult(
                value=0.5, confidence=0.1, adapted=False, fallback_used=True,
                data_points_used=len(data), original_name='momentum_trend'
            )
            
        # Simple momentum based on recent price changes
        recent_returns = data['close'].pct_change().tail(5)
        avg_return = recent_returns.mean()
        
        # Normalize to 0-1 scale
        if avg_return > 0.02:
            score = 0.8
        elif avg_return > 0.01:
            score = 0.6
        elif avg_return > 0:
            score = 0.55
        elif avg_return > -0.01:
            score = 0.45
        elif avg_return > -0.02:
            score = 0.4
        else:
            score = 0.2
            
        confidence = min(0.6, len(data) / 10)
            
        return IndicatorResult(
            value=score,
            confidence=confidence,
            adapted=False,
            fallback_used=True,
            data_points_used=len(data),
            original_name='momentum_trend'
        )
        
    def _calculate_rsi(self, data: pd.DataFrame, period: int) -> IndicatorResult:
        """Calculate RSI with confidence scoring."""
        if len(data) < period:
            confidence = len(data) / period * 0.7
            effective_period = max(2, len(data) - 1)
        else:
            confidence = 0.9
            effective_period = period
            
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(effective_period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(effective_period, min_periods=1).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        # Normalize RSI to 0-1 scale (50 = neutral = 0.5)
        normalized_rsi = current_rsi / 100
        
        return IndicatorResult(
            value=normalized_rsi,
            confidence=confidence,
            adapted=len(data) < period,
            fallback_used=False,
            data_points_used=len(data),
            original_name='rsi14'
        )
        
    def _calculate_candle_reversal(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate candle reversal patterns."""
        if len(data) < 3:
            return IndicatorResult(
                value=0.5, confidence=0.3, adapted=False, fallback_used=False,
                data_points_used=len(data), original_name='candlerev'
            )
            
        # Simple reversal pattern detection
        last_candle = data.iloc[-1]
        prev_candle = data.iloc[-2]
        
        # Bullish patterns
        is_hammer = (last_candle['close'] > last_candle['open'] and 
                    (last_candle['open'] - last_candle['low']) > 2 * abs(last_candle['close'] - last_candle['open']))
        
        is_engulfing_bull = (last_candle['close'] > last_candle['open'] and
                           prev_candle['close'] < prev_candle['open'] and
                           last_candle['close'] > prev_candle['open'] and
                           last_candle['open'] < prev_candle['close'])
        
        # Bearish patterns  
        is_shooting_star = (last_candle['close'] < last_candle['open'] and
                          (last_candle['high'] - last_candle['open']) > 2 * abs(last_candle['close'] - last_candle['open']))
        
        is_engulfing_bear = (last_candle['close'] < last_candle['open'] and
                           prev_candle['close'] > prev_candle['open'] and
                           last_candle['close'] < prev_candle['open'] and
                           last_candle['open'] > prev_candle['close'])
        
        if is_hammer or is_engulfing_bull:
            score = 0.8
        elif is_shooting_star or is_engulfing_bear:
            score = 0.2
        else:
            score = 0.5
            
        return IndicatorResult(
            value=score,
            confidence=0.7,
            adapted=False,
            fallback_used=False,
            data_points_used=len(data),
            original_name='candlerev'
        )
        
    def _calculate_bb_width(self, data: pd.DataFrame, period: int) -> IndicatorResult:
        """Calculate Bollinger Band width."""
        if len(data) < period:
            confidence = len(data) / period * 0.6
            effective_period = max(2, len(data))
        else:
            confidence = 0.8
            effective_period = period
            
        sma = data['close'].rolling(effective_period, min_periods=1).mean()
        std = data['close'].rolling(effective_period, min_periods=1).std()
        
        bandwidth = (std.iloc[-1] * 2) / sma.iloc[-1]
        
        # Normalize bandwidth (typical range 0.02-0.20)
        normalized_bandwidth = min(1.0, bandwidth / 0.10)
        
        return IndicatorResult(
            value=normalized_bandwidth,
            confidence=confidence,
            adapted=len(data) < period,
            fallback_used=False,
            data_points_used=len(data),
            original_name='bbwidth20'
        )
        
    def _calculate_bb_position(self, data: pd.DataFrame, period: int) -> IndicatorResult:
        """Calculate Bollinger Band position."""
        if len(data) < period:
            confidence = len(data) / period * 0.6
            effective_period = max(2, len(data))
        else:
            confidence = 0.8
            effective_period = period
            
        sma = data['close'].rolling(effective_period, min_periods=1).mean()
        std = data['close'].rolling(effective_period, min_periods=1).std()
        
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        
        current_price = data['close'].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        # Calculate %B (position within bands)
        percent_b = (current_price - current_lower) / (current_upper - current_lower)
        
        return IndicatorResult(
            value=max(0, min(1, percent_b)),
            confidence=confidence,
            adapted=len(data) < period,
            fallback_used=False,
            data_points_used=len(data),
            original_name='bbpos20'
        )


# Global service instance
adaptive_indicator_engine = AdaptiveIndicatorEngine()
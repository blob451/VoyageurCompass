"""
Dynamic TA-Weighted Prediction System
Uses technical indicators as dynamic weights for LSTM predictions
Revolutionary approach where TA indicators modulate prediction confidence
"""

import logging
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List
from decimal import Decimal

logger = logging.getLogger(__name__)


class DynamicTAPredictor:
    """
    Revolutionary prediction system that uses technical indicators as attention weights.
    Strong indicators → Higher prediction confidence
    Conflicting indicators → Lower prediction weight
    """
    
    # Indicator correlation matrix for agreement analysis
    INDICATOR_CORRELATIONS = {
        'sma50vs200': ['pricevs50', 'rel1y', 'rel2y'],  # Trend indicators
        'rsi14': ['macd12269', 'obv20'],  # Momentum indicators
        'bbpos20': ['bbwidth20', 'volsurge'],  # Volatility indicators
        'candlerev': ['srcontext'],  # Reversal indicators
    }
    
    # Market regime thresholds
    REGIME_THRESHOLDS = {
        'strong_bullish': 7.5,  # Score > 7.5
        'bullish': 6.0,         # Score 6.0-7.5
        'neutral': 4.0,          # Score 4.0-6.0
        'bearish': 2.5,          # Score 2.5-4.0
        'strong_bearish': 0      # Score < 2.5
    }
    
    def __init__(self):
        """Initialize dynamic TA predictor."""
        self.last_regime = 'neutral'
        self.confidence_history = []
        
    def calculate_dynamic_weights(self, indicators: Dict[str, Any]) -> Dict[str, float]:
        """
        Convert technical indicators to dynamic prediction weights.
        
        Args:
            indicators: Dictionary of indicator results with scores and weights
            
        Returns:
            Dictionary with dynamic weights and confidence metrics
        """
        try:
            # Extract indicator scores
            indicator_scores = {}
            for name, result in indicators.items():
                if hasattr(result, 'score'):
                    indicator_scores[name] = float(result.score)
                elif isinstance(result, dict) and 'score' in result:
                    indicator_scores[name] = float(result['score'])
            
            # Calculate indicator agreement/divergence
            agreement_score = self._calculate_agreement_score(indicator_scores)
            
            # Determine market regime
            composite_score = np.mean(list(indicator_scores.values()))
            market_regime = self._determine_market_regime(composite_score)
            
            # Calculate conviction level
            conviction = self._calculate_conviction(indicator_scores, agreement_score)
            
            # Generate dynamic weights
            weights = {
                'base_weight': 1.0,
                'agreement_multiplier': agreement_score,
                'conviction_multiplier': conviction,
                'regime_adjustment': self._get_regime_adjustment(market_regime),
                'final_weight': 1.0 * agreement_score * conviction * self._get_regime_adjustment(market_regime),
                'confidence': min(0.95, 0.5 + agreement_score * 0.25 + conviction * 0.2),
                'market_regime': market_regime,
                'composite_score': composite_score
            }
            
            logger.debug(f"Dynamic weights calculated: agreement={agreement_score:.2f}, "
                        f"conviction={conviction:.2f}, regime={market_regime}")
            
            return weights
            
        except Exception as e:
            logger.error(f"Error calculating dynamic weights: {str(e)}")
            return {
                'base_weight': 1.0,
                'agreement_multiplier': 1.0,
                'conviction_multiplier': 1.0,
                'regime_adjustment': 1.0,
                'final_weight': 1.0,
                'confidence': 0.5,
                'market_regime': 'neutral',
                'composite_score': 5.0
            }
    
    def _calculate_agreement_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate how well indicators agree with each other.
        High agreement = higher confidence in prediction.
        
        Args:
            scores: Dictionary of indicator scores
            
        Returns:
            Agreement score between 0.5 (high disagreement) and 1.5 (high agreement)
        """
        if len(scores) < 2:
            return 1.0
        
        # Check correlation groups
        agreement_scores = []
        
        for primary, correlated in self.INDICATOR_CORRELATIONS.items():
            if primary in scores:
                primary_score = scores[primary]
                
                # Check agreement with correlated indicators
                for indicator in correlated:
                    if indicator in scores:
                        secondary_score = scores[indicator]
                        
                        # Calculate agreement (closer scores = better agreement)
                        diff = abs(primary_score - secondary_score) / 10.0
                        agreement = 1.0 - min(diff, 0.5)  # Cap disagreement penalty at 0.5
                        agreement_scores.append(agreement)
        
        if agreement_scores:
            # Average agreement across all correlations
            avg_agreement = np.mean(agreement_scores)
            # Scale to 0.5-1.5 range
            return 0.5 + avg_agreement
        
        # Fallback: Calculate standard deviation of all scores
        score_std = np.std(list(scores.values()))
        # Lower std = better agreement
        # Scale: std of 3 → 0.5, std of 0 → 1.5
        return max(0.5, min(1.5, 1.5 - score_std / 6.0))
    
    def _calculate_conviction(self, scores: Dict[str, float], agreement: float) -> float:
        """
        Calculate conviction level based on indicator extremes and agreement.
        
        Args:
            scores: Dictionary of indicator scores
            agreement: Agreement score
            
        Returns:
            Conviction multiplier (0.5 to 1.5)
        """
        if not scores:
            return 1.0
        
        # Count extreme signals (very bullish >8 or very bearish <2)
        extreme_count = sum(1 for s in scores.values() if s > 8 or s < 2)
        extreme_ratio = extreme_count / len(scores)
        
        # High conviction when many extreme signals agree
        if extreme_ratio > 0.5 and agreement > 1.0:
            return 1.5  # High conviction
        elif extreme_ratio > 0.3:
            return 1.2  # Moderate conviction
        elif extreme_ratio < 0.1:
            return 0.8  # Low conviction (mixed signals)
        else:
            return 1.0  # Neutral conviction
    
    def _determine_market_regime(self, composite_score: float) -> str:
        """
        Determine current market regime based on composite score.
        
        Args:
            composite_score: Average of all indicator scores
            
        Returns:
            Market regime string
        """
        for regime, threshold in self.REGIME_THRESHOLDS.items():
            if composite_score >= threshold:
                return regime
        return 'strong_bearish'
    
    def _get_regime_adjustment(self, regime: str) -> float:
        """
        Get prediction weight adjustment based on market regime.
        
        Args:
            regime: Current market regime
            
        Returns:
            Regime adjustment multiplier
        """
        adjustments = {
            'strong_bullish': 1.3,   # Amplify bullish predictions
            'bullish': 1.1,          # Slight amplification
            'neutral': 1.0,          # No adjustment
            'bearish': 1.1,          # Slight amplification for shorts
            'strong_bearish': 1.3    # Amplify bearish predictions
        }
        return adjustments.get(regime, 1.0)
    
    def weighted_prediction(
        self, 
        lstm_output: float, 
        ta_weights: Dict[str, float],
        current_price: float
    ) -> Dict[str, Any]:
        """
        Apply TA-derived weights to LSTM predictions.
        
        Args:
            lstm_output: Raw LSTM prediction (price or return)
            ta_weights: Dynamic weights from TA analysis
            current_price: Current stock price
            
        Returns:
            Weighted prediction with confidence metrics
        """
        try:
            # Apply dynamic weight to prediction
            final_weight = ta_weights.get('final_weight', 1.0)
            confidence = ta_weights.get('confidence', 0.5)
            market_regime = ta_weights.get('market_regime', 'neutral')
            
            # Adjust prediction based on regime and confidence
            if market_regime in ['strong_bullish', 'bullish']:
                # In bullish regime, reduce bearish predictions
                if lstm_output < current_price:
                    lstm_output = current_price + (lstm_output - current_price) * 0.5
                    confidence *= 0.8  # Lower confidence for contra-trend
                    
            elif market_regime in ['strong_bearish', 'bearish']:
                # In bearish regime, reduce bullish predictions
                if lstm_output > current_price:
                    lstm_output = current_price + (lstm_output - current_price) * 0.5
                    confidence *= 0.8  # Lower confidence for contra-trend
            
            # Apply final weight
            weighted_prediction = current_price + (lstm_output - current_price) * final_weight
            
            # Calculate adjusted change
            price_change = weighted_prediction - current_price
            price_change_pct = (price_change / current_price) * 100
            
            # Bound predictions to reasonable range
            max_change_pct = 10.0  # Maximum 10% daily change
            if abs(price_change_pct) > max_change_pct:
                price_change_pct = np.sign(price_change_pct) * max_change_pct
                weighted_prediction = current_price * (1 + price_change_pct / 100)
                confidence *= 0.7  # Reduce confidence for clamped predictions
            
            return {
                'predicted_price': weighted_prediction,
                'original_prediction': lstm_output,
                'current_price': current_price,
                'price_change': weighted_prediction - current_price,
                'price_change_pct': price_change_pct,
                'confidence': confidence,
                'ta_weight': final_weight,
                'market_regime': market_regime,
                'agreement_score': ta_weights.get('agreement_multiplier', 1.0),
                'conviction': ta_weights.get('conviction_multiplier', 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error in weighted prediction: {str(e)}")
            # Fallback to original prediction
            return {
                'predicted_price': lstm_output,
                'original_prediction': lstm_output,
                'current_price': current_price,
                'price_change': lstm_output - current_price,
                'price_change_pct': ((lstm_output - current_price) / current_price) * 100,
                'confidence': 0.5,
                'ta_weight': 1.0,
                'market_regime': 'neutral',
                'agreement_score': 1.0,
                'conviction': 1.0
            }
    
    def get_indicator_importance(self, indicators: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Rank indicators by their importance in the current prediction.
        
        Args:
            indicators: Dictionary of indicator results
            
        Returns:
            List of (indicator_name, importance_score) tuples, sorted by importance
        """
        importance_scores = []
        
        for name, result in indicators.items():
            # Extract score
            if hasattr(result, 'score'):
                score = float(result.score)
            elif isinstance(result, dict) and 'score' in result:
                score = float(result['score'])
            else:
                continue
            
            # Calculate importance based on extremity and weight
            extremity = abs(score - 5.0) / 5.0  # How far from neutral
            
            # Get original weight if available
            if hasattr(result, 'weight'):
                weight = float(result.weight)
            elif isinstance(result, dict) and 'weight' in result:
                weight = float(result['weight'])
            else:
                weight = 0.1
            
            importance = extremity * weight
            importance_scores.append((name, importance))
        
        # Sort by importance (highest first)
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        
        return importance_scores


def create_dynamic_predictor() -> DynamicTAPredictor:
    """Factory function to create dynamic TA predictor."""
    return DynamicTAPredictor()
"""
Integrated Prediction Service
Combines TA Engine analysis with Dynamic LSTM predictions
Revolutionary approach using TA indicators as attention weights for ML predictions
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from Analytics.engine.ta_engine import TechnicalAnalysisEngine
from Analytics.engine.dynamic_predictor import DynamicTAPredictor
from Analytics.services.universal_predictor import UniversalLSTMAnalyticsService

logger = logging.getLogger(__name__)


class IntegratedPredictionService:
    """
    Service that integrates technical analysis with dynamic ML predictions.
    Uses TA indicators to dynamically weight and adjust LSTM predictions.
    """

    def __init__(self):
        """Initialize integrated prediction service."""
        self.ta_engine = TechnicalAnalysisEngine()
        self.dynamic_predictor = DynamicTAPredictor()
        self.lstm_service = UniversalLSTMAnalyticsService()

        logger.info("Integrated Prediction Service initialized")

    def predict_with_ta_context(
        self,
        symbol: str,
        horizon: str = '1d',
        include_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Generate prediction with full TA context and dynamic weighting.

        Args:
            symbol: Stock symbol
            horizon: Prediction horizon ('1d', '7d', '30d')
            include_analysis: Whether to include full TA analysis

        Returns:
            Comprehensive prediction with TA context
        """
        try:
            result = {
                'symbol': symbol,
                'horizon': horizon,
                'timestamp': datetime.now().isoformat(),
                'success': False
            }

            # Step 1: Get base LSTM prediction
            logger.info(f"Getting LSTM prediction for {symbol}")
            lstm_result = self.lstm_service.predict_stock_price(symbol, horizon=horizon)

            if not lstm_result:
                logger.warning(f"No LSTM prediction available for {symbol}")
                result['error'] = "LSTM prediction unavailable"
                return result

            current_price = lstm_result.get('current_price', 0)
            base_prediction = lstm_result.get('predicted_price', current_price)

            # Step 2: Perform technical analysis (without LSTM to avoid recursion)
            if include_analysis:
                logger.info(f"Performing technical analysis for {symbol}")
                ta_result = self._get_ta_indicators(symbol)

                if ta_result and ta_result.get('success'):
                    indicators = ta_result.get('indicators', {})

                    # Step 3: Calculate dynamic weights from TA indicators
                    logger.info(f"Calculating dynamic weights for {symbol}")
                    ta_weights = self.dynamic_predictor.calculate_dynamic_weights(indicators)

                    # Step 4: Apply dynamic weighting to prediction
                    logger.info(f"Applying TA weights to prediction for {symbol}")
                    weighted_result = self.dynamic_predictor.weighted_prediction(
                        lstm_output=base_prediction,
                        ta_weights=ta_weights,
                        current_price=current_price
                    )

                    # Step 5: Get indicator importance ranking
                    indicator_importance = self.dynamic_predictor.get_indicator_importance(indicators)

                    # Combine results
                    result.update({
                        'success': True,
                        'current_price': current_price,
                        'base_prediction': base_prediction,
                        'weighted_prediction': weighted_result['predicted_price'],
                        'price_change': weighted_result['price_change'],
                        'price_change_pct': weighted_result['price_change_pct'],
                        'confidence': weighted_result['confidence'],
                        'market_regime': weighted_result['market_regime'],
                        'ta_weight': weighted_result['ta_weight'],
                        'agreement_score': weighted_result['agreement_score'],
                        'conviction': weighted_result['conviction'],
                        'ta_composite_score': ta_result.get('composite_score', 5.0),
                        'top_indicators': indicator_importance[:5],  # Top 5 most important
                        'sector': lstm_result.get('sector_name', 'Unknown')
                    })

                    # Add detailed breakdown if requested
                    if include_analysis:
                        result['detailed_analysis'] = {
                            'indicators': self._format_indicators(indicators),
                            'ta_weights': ta_weights,
                            'lstm_details': {
                                'model_version': lstm_result.get('model_version', 'Unknown'),
                                'base_confidence': lstm_result.get('confidence', 0.5)
                            }
                        }

                    logger.info(f"Integrated prediction for {symbol}: ${weighted_result['predicted_price']:.2f} "
                               f"({weighted_result['price_change_pct']:+.2f}%), confidence: {weighted_result['confidence']:.2f}")
                else:
                    # Fallback to base LSTM prediction without TA weighting
                    logger.warning(f"TA analysis failed for {symbol}, using base prediction")
                    result.update({
                        'success': True,
                        'current_price': current_price,
                        'base_prediction': base_prediction,
                        'weighted_prediction': base_prediction,
                        'price_change': base_prediction - current_price,
                        'price_change_pct': ((base_prediction - current_price) / current_price) * 100,
                        'confidence': lstm_result.get('confidence', 0.5),
                        'market_regime': 'unknown',
                        'ta_weight': 1.0,
                        'sector': lstm_result.get('sector_name', 'Unknown')
                    })
            else:
                # Return base prediction without TA analysis
                result.update({
                    'success': True,
                    'current_price': current_price,
                    'base_prediction': base_prediction,
                    'weighted_prediction': base_prediction,
                    'price_change': base_prediction - current_price,
                    'price_change_pct': ((base_prediction - current_price) / current_price) * 100,
                    'confidence': lstm_result.get('confidence', 0.5),
                    'sector': lstm_result.get('sector_name', 'Unknown')
                })

            return result

        except Exception as e:
            logger.error(f"Error in integrated prediction for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'horizon': horizon,
                'success': False,
                'error': str(e)
            }

    def _get_ta_indicators(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get technical indicators without triggering LSTM prediction.

        Args:
            symbol: Stock symbol

        Returns:
            TA indicators and scores
        """
        try:
            # Temporarily disable LSTM predictions in TA engine
            # to avoid circular dependency
            from Analytics.engine import ta_engine as ta_module
            original_weights = ta_module.TechnicalAnalysisEngine.WEIGHTS.copy()

            # Set prediction weight to 0 to skip LSTM
            ta_module.TechnicalAnalysisEngine.WEIGHTS['prediction'] = 0.0

            # Perform analysis
            analysis = self.ta_engine.analyze_stock(symbol)

            # Restore original weights
            ta_module.TechnicalAnalysisEngine.WEIGHTS = original_weights

            if analysis:
                # Remove prediction indicator to avoid recursion
                indicators = analysis.get('indicators', {}).copy()
                indicators.pop('prediction', None)  # Remove prediction indicator

                return {
                    'success': True,
                    'indicators': indicators,
                    'composite_score': analysis.get('composite_score', 5.0)
                }

            return None

        except Exception as e:
            logger.error(f"Error getting TA indicators for {symbol}: {str(e)}")
            return None

    def _format_indicators(self, indicators: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Format indicators for output.

        Args:
            indicators: Raw indicator results

        Returns:
            Formatted indicator dictionary
        """
        formatted = {}

        for name, result in indicators.items():
            if hasattr(result, 'score'):
                formatted[name] = {
                    'score': float(result.score),
                    'weight': float(result.weight) if hasattr(result, 'weight') else 0.1,
                    'raw': result.raw if hasattr(result, 'raw') else None
                }
            elif isinstance(result, dict):
                formatted[name] = result

        return formatted

    def batch_predict(
        self,
        symbols: list,
        horizon: str = '1d'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate predictions for multiple symbols.

        Args:
            symbols: List of stock symbols
            horizon: Prediction horizon

        Returns:
            Dictionary mapping symbols to predictions
        """
        results = {}

        for symbol in symbols:
            logger.info(f"Processing batch prediction for {symbol}")
            results[symbol] = self.predict_with_ta_context(
                symbol, 
                horizon, 
                include_analysis=False  # Skip detailed analysis for batch
            )

        return results


# Singleton instance
_integrated_predictor_instance = None


def get_integrated_predictor() -> IntegratedPredictionService:
    """Get or create singleton integrated predictor instance."""
    global _integrated_predictor_instance

    if _integrated_predictor_instance is None:
        _integrated_predictor_instance = IntegratedPredictionService()

    return _integrated_predictor_instance

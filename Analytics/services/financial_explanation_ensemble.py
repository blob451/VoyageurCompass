"""
Financial Explanation Ensemble System.
Combines multiple models for optimal financial explanation generation.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from Analytics.services.local_llm_service import get_local_llm_service
from Analytics.services.sentiment_analyzer import get_sentiment_analyzer
from Analytics.services.hybrid_analysis_coordinator import get_hybrid_analysis_coordinator

logger = logging.getLogger(__name__)


class EnsembleStrategy(Enum):
    """Ensemble voting strategies."""
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    ADAPTIVE_WEIGHTED = "adaptive_weighted"


@dataclass
class ModelPrediction:
    """Container for individual model prediction."""
    model_name: str
    content: str
    confidence: float
    generation_time: float
    metadata: Dict[str, Any]
    recommendation: Optional[str] = None
    technical_coverage: float = 0.0
    quality_score: float = 0.0


class FinancialExplanationEnsemble:
    """
    Advanced ensemble system for financial explanation generation.
    Combines multiple models and strategies for optimal results.
    """

    def __init__(self):
        """Initialize the financial explanation ensemble."""
        # Core services
        self.llm_service = get_local_llm_service()
        self.sentiment_service = get_sentiment_analyzer()
        self.hybrid_coordinator = get_hybrid_analysis_coordinator()

        # Model registry with performance tracking
        self.models = {
            'base_8b': {
                'name': 'llama3.1:8b',
                'type': 'base',
                'strength': 'fast_generation',
                'weight': 1.0,
                'performance_history': []
            },
            'base_70b': {
                'name': 'llama3.1:70b',
                'type': 'base',
                'strength': 'complex_analysis',
                'weight': 1.2,
                'performance_history': []
            },
            'sentiment_enhanced': {
                'name': 'hybrid_sentiment',
                'type': 'hybrid',
                'strength': 'sentiment_integration',
                'weight': 1.1,
                'performance_history': []
            }
            # Fine-tuned models would be added here when available
        }

        # Ensemble configuration
        self.ensemble_config = {
            'default_strategy': EnsembleStrategy.CONFIDENCE_WEIGHTED,
            'parallel_execution': True,
            'timeout_per_model': 30.0,
            'min_models_required': 2,
            'quality_threshold': 0.6,
            'enable_adaptive_weighting': True,
            'performance_window_size': 100
        }

        # Performance tracking
        self.ensemble_metrics = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'average_generation_time': 0.0,
            'strategy_usage': {strategy: 0 for strategy in EnsembleStrategy},
            'model_usage_stats': {model: 0 for model in self.models.keys()}
        }

        logger.info("Financial Explanation Ensemble initialized")

    def generate_ensemble_explanation(self,
                                    analysis_data: Dict[str, Any],
                                    detail_level: str = 'standard',
                                    strategy: Optional[EnsembleStrategy] = None,
                                    return_all_predictions: bool = False) -> Dict[str, Any]:
        """
        Generate explanation using ensemble of models.

        Args:
            analysis_data: Technical analysis data
            detail_level: Level of detail for explanation
            strategy: Ensemble strategy to use
            return_all_predictions: Whether to return all model predictions

        Returns:
            Ensemble prediction result
        """
        start_time = time.time()
        self.ensemble_metrics['total_predictions'] += 1

        symbol = analysis_data.get('symbol', 'UNKNOWN')
        logger.info(f"[ENSEMBLE] Starting ensemble explanation for {symbol}")

        try:
            # Determine optimal strategy
            if strategy is None:
                strategy = self._select_optimal_strategy(analysis_data)

            self.ensemble_metrics['strategy_usage'][strategy] += 1

            # Get model predictions
            model_predictions = self._get_model_predictions(analysis_data, detail_level)

            if len(model_predictions) < self.ensemble_config['min_models_required']:
                logger.warning(f"[ENSEMBLE] Insufficient model predictions ({len(model_predictions)}) for {symbol}")
                return self._create_fallback_result(analysis_data, detail_level)

            # Apply ensemble strategy
            ensemble_result = self._apply_ensemble_strategy(
                model_predictions, 
                strategy, 
                analysis_data
            )

            # Add ensemble metadata
            generation_time = time.time() - start_time
            ensemble_result.update({
                'ensemble_metadata': {
                    'strategy_used': strategy.value,
                    'models_consulted': [pred.model_name for pred in model_predictions],
                    'total_generation_time': generation_time,
                    'model_count': len(model_predictions),
                    'consensus_strength': self._calculate_consensus_strength(model_predictions),
                    'quality_distribution': self._get_quality_distribution(model_predictions)
                }
            })

            # Include individual predictions if requested
            if return_all_predictions:
                ensemble_result['individual_predictions'] = [
                    {
                        'model': pred.model_name,
                        'content': pred.content,
                        'confidence': pred.confidence,
                        'recommendation': pred.recommendation,
                        'quality_score': pred.quality_score
                    }
                    for pred in model_predictions
                ]

            # Update performance metrics
            self._update_ensemble_metrics(generation_time, True)

            logger.info(f"[ENSEMBLE] Generated ensemble explanation for {symbol} in {generation_time:.2f}s")
            return ensemble_result

        except Exception as e:
            generation_time = time.time() - start_time
            self._update_ensemble_metrics(generation_time, False)
            logger.error(f"[ENSEMBLE] Error generating ensemble explanation for {symbol}: {str(e)}")
            return self._create_error_result(str(e))

    def _select_optimal_strategy(self, analysis_data: Dict[str, Any]) -> EnsembleStrategy:
        """
        Select optimal ensemble strategy based on analysis characteristics.

        Args:
            analysis_data: Technical analysis data

        Returns:
            Optimal ensemble strategy
        """
        score = analysis_data.get('score_0_10', 5.0)
        weighted_scores = analysis_data.get('weighted_scores', {})

        # Calculate complexity
        complexity = len(weighted_scores) / 12.0  # Normalized complexity
        score_extremity = abs(score - 5.0) / 5.0  # How extreme the score is

        # Strategy selection logic
        if complexity > 0.7 and score_extremity > 0.6:
            # Complex scenario with strong signals - use performance weighting
            return EnsembleStrategy.PERFORMANCE_WEIGHTED
        elif score_extremity < 0.3:
            # Unclear signals - use confidence weighting
            return EnsembleStrategy.CONFIDENCE_WEIGHTED
        elif self.ensemble_config['enable_adaptive_weighting']:
            # Adaptive strategy based on recent performance
            return EnsembleStrategy.ADAPTIVE_WEIGHTED
        else:
            # Default strategy
            return self.ensemble_config['default_strategy']

    def _get_model_predictions(self,
                             analysis_data: Dict[str, Any],
                             detail_level: str) -> List[ModelPrediction]:
        """
        Get predictions from all available models.

        Args:
            analysis_data: Technical analysis data
            detail_level: Level of detail

        Returns:
            List of model predictions
        """
        predictions = []

        if self.ensemble_config['parallel_execution']:
            # Parallel execution for faster results
            predictions = self._get_predictions_parallel(analysis_data, detail_level)
        else:
            # Sequential execution
            predictions = self._get_predictions_sequential(analysis_data, detail_level)

        # Filter and enhance predictions
        valid_predictions = []
        for prediction in predictions:
            if self._validate_prediction(prediction):
                enhanced_prediction = self._enhance_prediction(prediction, analysis_data)
                valid_predictions.append(enhanced_prediction)

        return valid_predictions

    def _get_predictions_parallel(self,
                                analysis_data: Dict[str, Any],
                                detail_level: str) -> List[ModelPrediction]:
        """Get model predictions in parallel."""
        predictions = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all model prediction tasks
            future_to_model = {}

            # Base 8B model
            if self._is_model_available('base_8b'):
                future = executor.submit(
                    self._get_base_model_prediction,
                    analysis_data, detail_level, 'llama3.1:8b'
                )
                future_to_model[future] = 'base_8b'

            # Sentiment-enhanced prediction
            if self._is_model_available('sentiment_enhanced'):
                future = executor.submit(
                    self._get_sentiment_enhanced_prediction,
                    analysis_data, detail_level
                )
                future_to_model[future] = 'sentiment_enhanced'

            # Base 70B model (for complex scenarios)
            complexity_score = len(analysis_data.get('weighted_scores', {})) / 12.0
            if complexity_score > 0.7 and self._is_model_available('base_70b'):
                future = executor.submit(
                    self._get_base_model_prediction,
                    analysis_data, detail_level, 'llama3.1:70b'
                )
                future_to_model[future] = 'base_70b'

            # Collect results
            for future in as_completed(future_to_model, timeout=self.ensemble_config['timeout_per_model']):
                model_name = future_to_model[future]
                try:
                    prediction = future.result(timeout=5.0)  # Additional timeout safety
                    if prediction:
                        predictions.append(prediction)
                        self.ensemble_metrics['model_usage_stats'][model_name] += 1
                except Exception as e:
                    logger.error(f"[ENSEMBLE] Error getting prediction from {model_name}: {str(e)}")

        return predictions

    def _get_predictions_sequential(self,
                                  analysis_data: Dict[str, Any],
                                  detail_level: str) -> List[ModelPrediction]:
        """Get model predictions sequentially."""
        predictions = []

        # Base 8B model (always try first)
        if self._is_model_available('base_8b'):
            prediction = self._get_base_model_prediction(analysis_data, detail_level, 'llama3.1:8b')
            if prediction:
                predictions.append(prediction)
                self.ensemble_metrics['model_usage_stats']['base_8b'] += 1

        # Sentiment-enhanced prediction
        if self._is_model_available('sentiment_enhanced'):
            prediction = self._get_sentiment_enhanced_prediction(analysis_data, detail_level)
            if prediction:
                predictions.append(prediction)
                self.ensemble_metrics['model_usage_stats']['sentiment_enhanced'] += 1

        return predictions

    def _get_base_model_prediction(self,
                                 analysis_data: Dict[str, Any],
                                 detail_level: str,
                                 model_name: str) -> Optional[ModelPrediction]:
        """Get prediction from base LLM model."""
        try:
            start_time = time.time()
            logger.info(f"[ENSEMBLE] Generating base model prediction for {analysis_data.get('symbol', 'UNKNOWN')}")

            result = self.llm_service.generate_explanation(
                analysis_data, detail_level
            )

            if not result:
                logger.warning(f"[ENSEMBLE] No result from base model {model_name}")
                return None

            generation_time = time.time() - start_time

            # Extract recommendation from content
            content = result.get('content', '')
            recommendation = self._extract_recommendation(content)

            prediction = ModelPrediction(
                model_name=model_name,
                content=content,
                confidence=result.get('confidence_score', 0.8),
                generation_time=generation_time,
                metadata={'cached': result.get('cached', False)},
                recommendation=recommendation,
                technical_coverage=self._calculate_technical_coverage(content, analysis_data),
                quality_score=self._assess_content_quality(content)
            )

            logger.info(f"[ENSEMBLE] Successfully generated prediction from {model_name} in {generation_time:.2f}s")
            return prediction

        except Exception as e:
            logger.error(f"Error getting base model prediction from {model_name}: {str(e)}")
            return None

    def _get_sentiment_enhanced_prediction(self,
                                         analysis_data: Dict[str, Any],
                                         detail_level: str) -> Optional[ModelPrediction]:
        """Get prediction from sentiment-enhanced system."""
        try:
            start_time = time.time()

            result = self.hybrid_coordinator.generate_enhanced_explanation(
                analysis_data, detail_level
            )

            if not result:
                return None

            generation_time = time.time() - start_time

            return ModelPrediction(
                model_name='hybrid_sentiment',
                content=result.get('content', ''),
                confidence=result.get('confidence_score', 0.5),
                generation_time=generation_time,
                metadata=result
            )

        except Exception as e:
            logger.error(f"Error getting sentiment-enhanced prediction: {str(e)}")
            return None

    def _is_model_available(self, model_key: str) -> bool:
        """Check if a model is available for predictions."""
        if model_key == 'base_8b' or model_key == 'base_70b':
            return self.llm_service.is_available()
        elif model_key == 'sentiment_enhanced':
            return (self.llm_service.is_available() and 
                   hasattr(self.sentiment_service, 'analyzeSentimentSingle'))
        return False

    def _validate_prediction(self, prediction: ModelPrediction) -> bool:
        """Validate individual model prediction."""
        if not prediction or not prediction.content:
            return False

        # Length validation
        if len(prediction.content) < 20:
            return False

        # Basic content validation
        if prediction.confidence <= 0:
            return False

        return True

    def _enhance_prediction(self,
                          prediction: ModelPrediction,
                          analysis_data: Dict[str, Any]) -> ModelPrediction:
        """Enhance prediction with additional analysis."""
        try:
            # Extract recommendation
            prediction.recommendation = self._extract_recommendation(prediction.content)

            # Calculate technical coverage
            prediction.technical_coverage = self._calculate_technical_coverage(
                prediction.content, analysis_data
            )

            # Calculate quality score
            prediction.quality_score = self._calculate_prediction_quality(prediction)

            return prediction

        except Exception as e:
            logger.error(f"Error enhancing prediction: {str(e)}")
            return prediction

    def _extract_recommendation(self, content: str) -> Optional[str]:
        """Extract investment recommendation from content."""
        content_upper = content.upper()

        if 'STRONG BUY' in content_upper:
            return 'STRONG BUY'
        elif 'STRONG SELL' in content_upper:
            return 'STRONG SELL'
        elif 'BUY' in content_upper:
            return 'BUY'
        elif 'SELL' in content_upper:
            return 'SELL'
        elif 'HOLD' in content_upper:
            return 'HOLD'

        return None

    def _calculate_technical_coverage(self,
                                    content: str,
                                    analysis_data: Dict[str, Any]) -> float:
        """Calculate how well content covers technical indicators."""
        weighted_scores = analysis_data.get('weighted_scores', {})
        if not weighted_scores:
            return 0.0

        # Get top indicators
        top_indicators = sorted(weighted_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        coverage_count = 0
        content_lower = content.lower()

        for indicator, _ in top_indicators:
            indicator_terms = []

            if 'sma' in indicator:
                indicator_terms.extend(['sma', 'moving average', 'ma'])
            elif 'rsi' in indicator:
                indicator_terms.extend(['rsi', 'relative strength'])
            elif 'macd' in indicator:
                indicator_terms.extend(['macd'])
            elif 'bb' in indicator:
                indicator_terms.extend(['bollinger', 'bands'])
            elif 'vol' in indicator:
                indicator_terms.extend(['volume', 'vol'])
            else:
                # Generic term
                clean_name = indicator.replace('w_', '').replace('_', ' ')
                indicator_terms.append(clean_name)

            if any(term in content_lower for term in indicator_terms):
                coverage_count += 1

        return coverage_count / len(top_indicators) if top_indicators else 0.0

    def _calculate_prediction_quality(self, prediction: ModelPrediction) -> float:
        """Calculate overall prediction quality score."""
        quality_factors = []

        # Content length factor
        length_score = min(len(prediction.content) / 200.0, 1.0)
        quality_factors.append(length_score * 0.2)

        # Recommendation clarity factor
        rec_score = 1.0 if prediction.recommendation else 0.0
        quality_factors.append(rec_score * 0.3)

        # Technical coverage factor
        quality_factors.append(prediction.technical_coverage * 0.3)

        # Confidence factor
        confidence_score = min(prediction.confidence, 1.0)
        quality_factors.append(confidence_score * 0.2)

        return sum(quality_factors)

    def _apply_ensemble_strategy(self,
                               predictions: List[ModelPrediction],
                               strategy: EnsembleStrategy,
                               analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply ensemble strategy to combine predictions."""

        if strategy == EnsembleStrategy.MAJORITY_VOTE:
            return self._majority_vote_ensemble(predictions)
        elif strategy == EnsembleStrategy.CONFIDENCE_WEIGHTED:
            return self._confidence_weighted_ensemble(predictions)
        elif strategy == EnsembleStrategy.PERFORMANCE_WEIGHTED:
            return self._performance_weighted_ensemble(predictions)
        elif strategy == EnsembleStrategy.ADAPTIVE_WEIGHTED:
            return self._adaptive_weighted_ensemble(predictions, analysis_data)
        else:
            # Default to confidence weighted
            return self._confidence_weighted_ensemble(predictions)

    def _confidence_weighted_ensemble(self, predictions: List[ModelPrediction]) -> Dict[str, Any]:
        """Combine predictions using confidence weighting."""
        if not predictions:
            return self._create_empty_result()

        # If only one prediction, return it
        if len(predictions) == 1:
            pred = predictions[0]
            return {
                'content': pred.content,
                'confidence_score': pred.confidence,
                'generation_time': pred.generation_time,
                'model_used': pred.model_name,
                'ensemble_method': 'single_model'
            }

        # Weight by confidence and quality
        total_weight = 0
        weighted_content_scores = {}

        for pred in predictions:
            weight = pred.confidence * pred.quality_score
            total_weight += weight

            # For recommendation consensus
            if pred.recommendation:
                if pred.recommendation not in weighted_content_scores:
                    weighted_content_scores[pred.recommendation] = 0
                weighted_content_scores[pred.recommendation] += weight

        # Select best prediction based on highest weighted score
        best_prediction = max(predictions, key=lambda p: p.confidence * p.quality_score)

        # Determine ensemble recommendation
        ensemble_recommendation = None
        if weighted_content_scores:
            ensemble_recommendation = max(weighted_content_scores.items(), key=lambda x: x[1])[0]

        # Calculate ensemble confidence
        ensemble_confidence = sum(p.confidence * p.quality_score for p in predictions) / len(predictions)

        # Generate ensemble content (use best prediction as base)
        ensemble_content = best_prediction.content

        # If recommendations differ significantly, add uncertainty note
        unique_recommendations = set(p.recommendation for p in predictions if p.recommendation)
        if len(unique_recommendations) > 1:
            ensemble_content += f"\n\nNote: Model consensus shows mixed signals. Primary recommendation: {ensemble_recommendation}."

        return {
            'content': ensemble_content,
            'confidence_score': ensemble_confidence,
            'generation_time': sum(p.generation_time for p in predictions) / len(predictions),
            'model_used': 'ensemble_confidence_weighted',
            'ensemble_method': 'confidence_weighted',
            'primary_model': best_prediction.model_name,
            'recommendation_consensus': ensemble_recommendation,
            'consensus_strength': len(predictions) - len(unique_recommendations) + 1
        }

    def _performance_weighted_ensemble(self, predictions: List[ModelPrediction]) -> Dict[str, Any]:
        """Combine predictions using historical performance weighting."""
        # Get performance weights
        weights = {}
        for pred in predictions:
            model_config = self.models.get(self._get_model_key(pred.model_name), {})
            weights[pred.model_name] = model_config.get('weight', 1.0)

        # Apply performance weighting (similar to confidence weighting but with performance weights)
        total_weight = 0
        weighted_scores = {}

        for pred in predictions:
            weight = weights.get(pred.model_name, 1.0) * pred.quality_score
            total_weight += weight

            if pred.recommendation:
                if pred.recommendation not in weighted_scores:
                    weighted_scores[pred.recommendation] = 0
                weighted_scores[pred.recommendation] += weight

        # Select based on performance weight
        best_prediction = max(predictions, key=lambda p: weights.get(p.model_name, 1.0) * p.quality_score)

        ensemble_recommendation = None
        if weighted_scores:
            ensemble_recommendation = max(weighted_scores.items(), key=lambda x: x[1])[0]

        return {
            'content': best_prediction.content,
            'confidence_score': best_prediction.confidence,
            'generation_time': best_prediction.generation_time,
            'model_used': 'ensemble_performance_weighted',
            'ensemble_method': 'performance_weighted',
            'primary_model': best_prediction.model_name,
            'recommendation_consensus': ensemble_recommendation
        }

    def _majority_vote_ensemble(self, predictions: List[ModelPrediction]) -> Dict[str, Any]:
        """Combine predictions using majority vote."""
        if not predictions:
            return self._create_empty_result()

        # Vote on recommendations
        recommendation_votes = {}
        for pred in predictions:
            if pred.recommendation:
                recommendation_votes[pred.recommendation] = recommendation_votes.get(pred.recommendation, 0) + 1

        # Get majority recommendation
        majority_recommendation = None
        if recommendation_votes:
            majority_recommendation = max(recommendation_votes.items(), key=lambda x: x[1])[0]

        # Find prediction with majority recommendation and highest quality
        majority_predictions = [p for p in predictions if p.recommendation == majority_recommendation]

        if majority_predictions:
            best_prediction = max(majority_predictions, key=lambda p: p.quality_score)
        else:
            best_prediction = max(predictions, key=lambda p: p.quality_score)

        return {
            'content': best_prediction.content,
            'confidence_score': best_prediction.confidence,
            'generation_time': best_prediction.generation_time,
            'model_used': 'ensemble_majority_vote',
            'ensemble_method': 'majority_vote',
            'primary_model': best_prediction.model_name,
            'recommendation_consensus': majority_recommendation,
            'vote_strength': recommendation_votes.get(majority_recommendation, 0) if majority_recommendation else 0
        }

    def _adaptive_weighted_ensemble(self,
                                  predictions: List[ModelPrediction],
                                  analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine predictions using adaptive weighting based on scenario."""
        # Analyze scenario characteristics
        score = analysis_data.get('score_0_10', 5.0)
        complexity = len(analysis_data.get('weighted_scores', {})) / 12.0

        # Adapt weights based on scenario
        adaptive_weights = {}

        for pred in predictions:
            base_weight = self.models.get(self._get_model_key(pred.model_name), {}).get('weight', 1.0)

            # Adjust based on scenario
            if pred.model_name == 'llama3.1:70b' and complexity > 0.7:
                # Boost 70B for complex scenarios
                adaptive_weights[pred.model_name] = base_weight * 1.3
            elif pred.model_name == 'hybrid_sentiment' and abs(score - 5.0) > 2.0:
                # Boost sentiment-enhanced for extreme scores
                adaptive_weights[pred.model_name] = base_weight * 1.2
            else:
                adaptive_weights[pred.model_name] = base_weight

        # Apply adaptive weights
        best_prediction = max(predictions, 
                            key=lambda p: adaptive_weights.get(p.model_name, 1.0) * p.quality_score)

        return {
            'content': best_prediction.content,
            'confidence_score': best_prediction.confidence,
            'generation_time': best_prediction.generation_time,
            'model_used': 'ensemble_adaptive_weighted',
            'ensemble_method': 'adaptive_weighted',
            'primary_model': best_prediction.model_name,
            'adaptive_weights': adaptive_weights
        }

    def _get_model_key(self, model_name: str) -> str:
        """Get model key from model name."""
        if 'llama3.1:8b' in model_name:
            return 'base_8b'
        elif 'llama3.1:70b' in model_name:
            return 'base_70b'
        elif 'hybrid_sentiment' in model_name:
            return 'sentiment_enhanced'
        return 'unknown'

    def _calculate_consensus_strength(self, predictions: List[ModelPrediction]) -> float:
        """Calculate consensus strength among predictions."""
        if len(predictions) <= 1:
            return 1.0

        recommendations = [p.recommendation for p in predictions if p.recommendation]
        if not recommendations:
            return 0.0

        # Calculate recommendation agreement
        unique_recommendations = len(set(recommendations))
        total_recommendations = len(recommendations)

        consensus = (total_recommendations - unique_recommendations + 1) / total_recommendations
        return consensus

    def _get_quality_distribution(self, predictions: List[ModelPrediction]) -> Dict[str, float]:
        """Get quality score distribution among predictions."""
        if not predictions:
            return {}

        quality_scores = [p.quality_score for p in predictions]

        return {
            'mean_quality': np.mean(quality_scores),
            'std_quality': np.std(quality_scores),
            'min_quality': np.min(quality_scores),
            'max_quality': np.max(quality_scores)
        }

    def _create_fallback_result(self, analysis_data: Dict[str, Any], detail_level: str) -> Dict[str, Any]:
        """Create fallback result when ensemble fails."""
        # Try to get at least one prediction
        try:
            result = self.llm_service.generate_explanation(analysis_data, detail_level)
            if result:
                result.update({
                    'ensemble_method': 'fallback_single',
                    'ensemble_metadata': {
                        'fallback_reason': 'insufficient_predictions',
                        'models_consulted': ['base_model']
                    }
                })
                return result
        except Exception:
            pass

        return self._create_empty_result()

    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result structure."""
        return {
            'content': 'Unable to generate explanation at this time.',
            'confidence_score': 0.0,
            'generation_time': 0.0,
            'model_used': 'none',
            'ensemble_method': 'failed',
            'error': True
        }

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result structure."""
        return {
            'content': 'Error generating explanation.',
            'confidence_score': 0.0,
            'generation_time': 0.0,
            'model_used': 'none',
            'ensemble_method': 'error',
            'error': True,
            'error_message': error_message
        }

    def _update_ensemble_metrics(self, generation_time: float, success: bool):
        """Update ensemble performance metrics."""
        if success:
            self.ensemble_metrics['successful_predictions'] += 1

        # Update average generation time
        current_avg = self.ensemble_metrics['average_generation_time']
        total_predictions = self.ensemble_metrics['total_predictions']

        self.ensemble_metrics['average_generation_time'] = (
            (current_avg * (total_predictions - 1) + generation_time) / total_predictions
        )

    def get_ensemble_status(self) -> Dict[str, Any]:
        """Get current ensemble system status."""
        return {
            'ensemble_active': True,
            'available_models': [key for key in self.models.keys() if self._is_model_available(key)],
            'ensemble_config': {k: v.value if isinstance(v, Enum) else v 
                              for k, v in self.ensemble_config.items()},
            'performance_metrics': self.ensemble_metrics.copy(),
            'model_registry': {k: {**v, 'available': self._is_model_available(k)} 
                             for k, v in self.models.items()}
        }


# Singleton instance
_ensemble_system = None


def get_financial_explanation_ensemble() -> FinancialExplanationEnsemble:
    """Get singleton instance of financial explanation ensemble."""
    global _ensemble_system
    if _ensemble_system is None:
        _ensemble_system = FinancialExplanationEnsemble()
    return _ensemble_system

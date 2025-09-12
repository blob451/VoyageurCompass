"""
Model Ensemble Support for Enhanced Sentiment Analysis
Implements multiple FinBERT model variants with weighted voting
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

# Conditional imports for ML dependencies to support CI environments
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    pipeline = None
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Early exit if dependencies not available
if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
    logger.warning("PyTorch or Transformers not available - Sentiment ensemble disabled")


class ModelVariant(Enum):
    """Available FinBERT model variants for ensemble."""

    PROSUS_FINBERT = "ProsusAI/finbert"  # Primary model
    YIYANG_FINBERT = "yiyanghkust/finbert-tone"  # Alternative tone analysis
    ABHILASH_FINBERT = "abhilash1910/finbert-sentiment"  # Sentiment focused
    FINBERT_BASE = "bert-base-uncased"  # Base BERT for fallback


@dataclass
class ModelConfig:
    """Configuration for individual models in ensemble."""

    name: str
    variant: ModelVariant
    weight: float
    min_confidence: float
    max_length: int = 512
    specialization: str = "general"


class SentimentEnsemble:
    """
    Ensemble of multiple FinBERT models for improved accuracy.
    Uses weighted voting based on model confidence and specialization.
    """

    def __init__(self, use_gpu: Optional[bool] = None):
        """
        Initialize the sentiment ensemble.

        Args:
            use_gpu: Whether to use GPU acceleration (defaults to torch.cuda.is_available())
        """
        if use_gpu is None:
            use_gpu = TORCH_AVAILABLE and torch.cuda.is_available() if torch else False
        self.use_gpu = use_gpu
        self.device = 0 if use_gpu else -1
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        self._initialized = False

        # Model configurations with specializations
        self.model_configs = [
            ModelConfig(
                name="primary",
                variant=ModelVariant.PROSUS_FINBERT,
                weight=0.4,
                min_confidence=0.6,
                specialization="general",
            ),
            ModelConfig(
                name="tone",
                variant=ModelVariant.YIYANG_FINBERT,
                weight=0.3,
                min_confidence=0.55,
                specialization="tone_analysis",
            ),
            ModelConfig(
                name="sentiment",
                variant=ModelVariant.ABHILASH_FINBERT,
                weight=0.3,
                min_confidence=0.6,
                specialization="sentiment_focus",
            ),
        ]

        # Voting strategies
        self.voting_strategies = {
            "weighted_average": self._weighted_average_voting,
            "confidence_weighted": self._confidence_weighted_voting,
            "majority": self._majority_voting,
            "max_confidence": self._max_confidence_voting,
        }

        self.default_strategy = "confidence_weighted"

    def initialize_models(self):
        """Initialize all models in the ensemble."""
        if self._initialized:
            return

        logger.info("Initializing sentiment ensemble models...")
        start_time = time.time()

        for config in self.model_configs:
            try:
                logger.info(f"Loading {config.name} model: {config.variant.value}")

                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(config.variant.value)
                model = AutoModelForSequenceClassification.from_pretrained(config.variant.value)

                # Move to GPU if available
                if self.use_gpu:
                    model = model.cuda()

                # Create pipeline
                pipe = pipeline(
                    "sentiment-analysis",
                    model=model,
                    tokenizer=tokenizer,
                    device=self.device,
                    max_length=config.max_length,
                    truncation=True,
                )

                # Store components
                self.models[config.name] = model
                self.tokenizers[config.name] = tokenizer
                self.pipelines[config.name] = pipe

                logger.info(f"Successfully loaded {config.name} model")

            except Exception as e:
                logger.error(f"Failed to load {config.name} model: {str(e)}")
                # Continue with other models

        self._initialized = True
        load_time = time.time() - start_time
        logger.info(f"Ensemble initialization completed in {load_time:.2f} seconds")

    def analyze_with_ensemble(
        self, text: str, voting_strategy: str = None, return_all_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze sentiment using model ensemble.

        Args:
            text: Text to analyze
            voting_strategy: Voting strategy to use
            return_all_predictions: Whether to return individual model predictions

        Returns:
            Ensemble sentiment result
        """
        if not self._initialized:
            self.initialize_models()

        if not text or not text.strip():
            return self._neutral_result()

        # Get predictions from all models
        predictions = []
        for config in self.model_configs:
            if config.name not in self.pipelines:
                continue

            try:
                # Get prediction from model
                result = self.pipelines[config.name](text)
                if result:
                    prediction = {
                        "model": config.name,
                        "specialization": config.specialization,
                        "label": result[0]["label"].lower(),
                        "score": result[0]["score"],
                        "weight": config.weight,
                        "min_confidence": config.min_confidence,
                    }
                    predictions.append(prediction)

            except Exception as e:
                logger.warning(f"Model {config.name} failed: {str(e)}")
                continue

        if not predictions:
            return self._neutral_result()

        # Apply voting strategy
        strategy = voting_strategy or self.default_strategy
        voting_func = self.voting_strategies.get(strategy, self._confidence_weighted_voting)

        ensemble_result = voting_func(predictions)

        # Add metadata
        ensemble_result["ensemble_size"] = len(predictions)
        ensemble_result["voting_strategy"] = strategy
        ensemble_result["models_used"] = [p["model"] for p in predictions]

        if return_all_predictions:
            ensemble_result["individual_predictions"] = predictions

        return ensemble_result

    def _weighted_average_voting(self, predictions: List[Dict]) -> Dict[str, Any]:
        """
        Simple weighted average of sentiment scores.

        Args:
            predictions: List of model predictions

        Returns:
            Ensemble result
        """
        total_weight = sum(p["weight"] for p in predictions)
        weighted_scores = []

        for pred in predictions:
            # Convert label to numeric score
            if pred["label"] == "positive":
                score = pred["score"]
            elif pred["label"] == "negative":
                score = -pred["score"]
            else:
                score = 0.0

            weighted_scores.append(score * pred["weight"])

        ensemble_score = sum(weighted_scores) / total_weight

        # Determine label
        if ensemble_score > 0.1:
            label = "positive"
        elif ensemble_score < -0.1:
            label = "negative"
        else:
            label = "neutral"

        # Calculate confidence as average of individual confidences
        avg_confidence = np.mean([p["score"] for p in predictions])

        return {
            "sentimentScore": float(ensemble_score),
            "sentimentLabel": label,
            "sentimentConfidence": float(avg_confidence),
        }

    def _confidence_weighted_voting(self, predictions: List[Dict]) -> Dict[str, Any]:
        """
        Weight votes by both model weight and confidence.

        Args:
            predictions: List of model predictions

        Returns:
            Ensemble result
        """
        # Filter by minimum confidence
        valid_predictions = [p for p in predictions if p["score"] >= p["min_confidence"]]

        if not valid_predictions:
            # Fall back to all predictions if none meet threshold
            valid_predictions = predictions

        # Calculate confidence-weighted scores
        weighted_scores = []
        total_weight = 0

        for pred in valid_predictions:
            # Combine model weight with confidence
            combined_weight = pred["weight"] * pred["score"]
            total_weight += combined_weight

            # Convert to numeric score
            if pred["label"] == "positive":
                score = pred["score"]
            elif pred["label"] == "negative":
                score = -pred["score"]
            else:
                score = 0.0

            weighted_scores.append(score * combined_weight)

        if total_weight > 0:
            ensemble_score = sum(weighted_scores) / total_weight
        else:
            ensemble_score = 0.0

        # Determine label with confidence adjustment
        confidence_adjusted_threshold = 0.1 * (1.0 - np.mean([p["score"] for p in valid_predictions]))

        if ensemble_score > confidence_adjusted_threshold:
            label = "positive"
        elif ensemble_score < -confidence_adjusted_threshold:
            label = "negative"
        else:
            label = "neutral"

        # Calculate ensemble confidence
        ensemble_confidence = np.average(
            [p["score"] for p in valid_predictions], weights=[p["weight"] for p in valid_predictions]
        )

        return {
            "sentimentScore": float(ensemble_score),
            "sentimentLabel": label,
            "sentimentConfidence": float(ensemble_confidence),
        }

    def _majority_voting(self, predictions: List[Dict]) -> Dict[str, Any]:
        """
        Simple majority voting across models.

        Args:
            predictions: List of model predictions

        Returns:
            Ensemble result
        """
        # Count votes for each label
        label_votes = {"positive": 0, "negative": 0, "neutral": 0}
        label_scores = {"positive": [], "negative": [], "neutral": []}

        for pred in predictions:
            label = pred["label"]
            if label in label_votes:
                label_votes[label] += pred["weight"]
                label_scores[label].append(pred["score"])

        # Find winning label
        winning_label = max(label_votes, key=label_votes.get)

        # Calculate score based on winning label
        if winning_label == "positive":
            ensemble_score = np.mean(label_scores["positive"]) if label_scores["positive"] else 0.5
        elif winning_label == "negative":
            ensemble_score = -np.mean(label_scores["negative"]) if label_scores["negative"] else -0.5
        else:
            ensemble_score = 0.0

        # Calculate confidence
        total_votes = sum(label_votes.values())
        vote_confidence = label_votes[winning_label] / total_votes if total_votes > 0 else 0.0

        return {
            "sentimentScore": float(ensemble_score),
            "sentimentLabel": winning_label,
            "sentimentConfidence": float(vote_confidence),
        }

    def _max_confidence_voting(self, predictions: List[Dict]) -> Dict[str, Any]:
        """
        Select prediction with highest confidence.

        Args:
            predictions: List of model predictions

        Returns:
            Ensemble result
        """
        # Find prediction with highest confidence
        best_pred = max(predictions, key=lambda p: p["score"])

        # Convert to result format
        if best_pred["label"] == "positive":
            score = best_pred["score"]
        elif best_pred["label"] == "negative":
            score = -best_pred["score"]
        else:
            score = 0.0

        return {
            "sentimentScore": float(score),
            "sentimentLabel": best_pred["label"],
            "sentimentConfidence": float(best_pred["score"]),
            "selected_model": best_pred["model"],
        }

    def _neutral_result(self) -> Dict[str, Any]:
        """Return neutral sentiment result."""
        return {"sentimentScore": 0.0, "sentimentLabel": "neutral", "sentimentConfidence": 0.0, "ensemble_size": 0}

    def compare_models(self, text: str) -> Dict[str, Any]:
        """
        Compare predictions from all models for analysis.

        Args:
            text: Text to analyze

        Returns:
            Comparison results
        """
        if not self._initialized:
            self.initialize_models()

        comparisons = {}

        for config in self.model_configs:
            if config.name not in self.pipelines:
                continue

            try:
                start_time = time.time()
                result = self.pipelines[config.name](text)
                inference_time = time.time() - start_time

                if result:
                    comparisons[config.name] = {
                        "model": config.variant.value,
                        "specialization": config.specialization,
                        "label": result[0]["label"].lower(),
                        "score": result[0]["score"],
                        "inference_time": inference_time,
                        "weight": config.weight,
                    }

            except Exception as e:
                comparisons[config.name] = {"error": str(e)}

        # Calculate agreement metrics
        labels = [c["label"] for c in comparisons.values() if "label" in c]
        unique_labels = set(labels)

        agreement_score = 1.0 if len(unique_labels) == 1 else 1.0 / len(unique_labels)

        return {
            "models": comparisons,
            "agreement_score": agreement_score,
            "consensus": len(unique_labels) == 1,
            "unique_predictions": list(unique_labels),
        }

    def unload_models(self):
        """Unload all models from memory."""
        try:
            for name in list(self.models.keys()):
                del self.models[name]
                del self.tokenizers[name]
                del self.pipelines[name]

            self.models.clear()
            self.tokenizers.clear()
            self.pipelines.clear()

            if self.use_gpu:
                torch.cuda.empty_cache()

            self._initialized = False
            logger.info("Ensemble models unloaded successfully")

        except Exception as e:
            logger.error(f"Error unloading ensemble models: {str(e)}")


# Singleton instance
_ensemble = None


def get_sentiment_ensemble() -> SentimentEnsemble:
    """Get or create singleton sentiment ensemble instance."""
    global _ensemble
    if _ensemble is None:
        _ensemble = SentimentEnsemble()
    return _ensemble

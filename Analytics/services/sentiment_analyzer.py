"""
Financial sentiment analysis service implementing FinBERT with confidence scoring and batch processing.
Provides comprehensive sentiment analysis with fallback mechanisms for enhanced reliability.
"""

import hashlib
import logging
import random
import re
import threading
import time
import unicodedata
from datetime import datetime
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
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch not available - ML-based sentiment analysis disabled")

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    pipeline = None
    TRANSFORMERS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Transformers not available - FinBERT sentiment analysis disabled")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None
    SKLEARN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("scikit-learn not available - semantic caching disabled")

from django.core.cache import cache

logger = logging.getLogger(__name__)


# Stocks requiring fallback due to processing timeouts
PROBLEMATIC_STOCKS = {"TSLA", "GM", "F", "NIO", "RIVN", "LCID", "URI", "PLUG", "FCEL", "BE", "TROW", "NVDA"}

# Global FinBERT availability flag
_FINBERT_DISABLED = False


class ErrorType(Enum):
    """Error classification for intelligent retry strategy implementation."""

    TOKENIZER_ERROR = "tokenizer"  # Text processing issues
    MODEL_ERROR = "model"  # Model inference errors
    TIMEOUT_ERROR = "timeout"  # Processing timeout
    RESOURCE_ERROR = "resource"  # Memory/GPU issues
    NETWORK_ERROR = "network"  # Model download/API issues
    UNKNOWN_ERROR = "unknown"  # Unclassified errors


class CircuitBreaker:
    """Circuit breaker pattern implementation for failure management."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False

    def record_success(self):
        """Record successful operation and reset failure state."""
        self.failure_count = 0
        self.is_open = False

    def record_failure(self):
        """Record operation failure and evaluate circuit state."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

    def can_attempt(self) -> bool:
        """Evaluate operation attempt eligibility based on circuit state."""
        if not self.is_open:
            return True

        if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
            self.is_open = False
            self.failure_count = 0
            logger.info("Circuit breaker reset after recovery timeout")
            return True

        return False


class SentimentMetrics:
    """Enhanced metrics collection with performance monitoring and anomaly detection."""

    def __init__(self):
        self.reset_metrics()

        self.performance_window = []
        self.window_size = 100
        self.anomaly_threshold = 2.5  # Z-score threshold for anomalies
        self.degradation_callbacks = []  # Callbacks for performance degradation

    def reset_metrics(self):
        """Reset all metrics counters."""
        self.total_requests = 0
        self.successful_analyses = 0
        self.failed_analyses = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_processing_time = 0.0
        self.total_articles_processed = 0
        self.confidence_distribution = {"high": 0, "medium": 0, "low": 0}
        self.sentiment_distribution = {"positive": 0, "negative": 0, "neutral": 0}

    def log_request(self, symbol: str, article_count: int):
        """Log a sentiment analysis request."""
        self.total_requests += 1
        self.total_articles_processed += article_count
        logger.info(
            "Sentiment analysis request",
            extra={
                "event_type": "sentiment_request",
                "symbol": symbol,
                "article_count": article_count,
                "total_requests": self.total_requests,
            },
        )

    def log_success(self, symbol: str, score: float, confidence: float, processing_time: float):
        """Log successful sentiment analysis with performance monitoring."""
        self.successful_analyses += 1
        self.total_processing_time += processing_time

        # Update performance window
        self._update_performance_window(processing_time)

        # Performance anomaly detection
        is_anomaly = self._detect_performance_anomaly(processing_time)

        # Categorize confidence
        if confidence >= 0.8:
            confidence_category = "high"
        elif confidence >= 0.6:
            confidence_category = "medium"
        else:
            confidence_category = "low"
        self.confidence_distribution[confidence_category] += 1

        # Categorize sentiment
        if score > 0.1:
            sentiment_category = "positive"
        elif score < -0.1:
            sentiment_category = "negative"
        else:
            sentiment_category = "neutral"
        self.sentiment_distribution[sentiment_category] += 1

        logger.info(
            "Sentiment analysis completed successfully",
            extra={
                "event_type": "sentiment_success",
                "symbol": symbol,
                "sentiment_score": score,
                "confidence": confidence,
                "confidence_category": confidence_category,
                "sentiment_category": sentiment_category,
                "processing_time_ms": processing_time * 1000,
                "success_rate": self.successful_analyses / self.total_requests if self.total_requests > 0 else 0,
                "is_anomaly": is_anomaly,
            },
        )

    def log_failure(self, symbol: str, error: str, processing_time: float = 0):
        """Log failed sentiment analysis."""
        self.failed_analyses += 1
        self.total_processing_time += processing_time

        logger.error(
            "Sentiment analysis failed",
            extra={
                "event_type": "sentiment_failure",
                "symbol": symbol,
                "error": error,
                "processing_time_ms": processing_time * 1000,
                "failure_rate": self.failed_analyses / self.total_requests if self.total_requests > 0 else 0,
            },
        )

    def log_cache_hit(self, symbol: str):
        """Log cache hit."""
        self.cache_hits += 1
        logger.debug(
            "Sentiment cache hit",
            extra={
                "event_type": "sentiment_cache_hit",
                "symbol": symbol,
                "cache_hit_rate": (
                    self.cache_hits / (self.cache_hits + self.cache_misses)
                    if (self.cache_hits + self.cache_misses) > 0
                    else 0
                ),
            },
        )

    def log_cache_miss(self, symbol: str):
        """Log cache miss."""
        self.cache_misses += 1
        logger.debug(
            "Sentiment cache miss",
            extra={
                "event_type": "sentiment_cache_miss",
                "symbol": symbol,
                "cache_hit_rate": (
                    self.cache_hits / (self.cache_hits + self.cache_misses)
                    if (self.cache_hits + self.cache_misses) > 0
                    else 0
                ),
            },
        )

    def _detect_performance_anomaly(self, processing_time: float) -> bool:
        """
        Detect performance anomalies using z-score.

        Args:
            processing_time: Current processing time

        Returns:
            True if anomaly detected
        """
        if len(self.performance_window) < 10:
            return False  # Need minimum samples

        # Calculate z-score
        mean_time = np.mean(self.performance_window)
        std_time = np.std(self.performance_window)

        if std_time == 0:
            return False

        z_score = abs((processing_time - mean_time) / std_time)

        if z_score > self.anomaly_threshold:
            logger.warning(
                "Performance anomaly detected",
                extra={
                    "event_type": "performance_anomaly",
                    "processing_time": processing_time,
                    "mean_time": mean_time,
                    "std_time": std_time,
                    "z_score": z_score,
                },
            )

            # Trigger callbacks
            for callback in self.degradation_callbacks:
                try:
                    callback(processing_time, mean_time, z_score)
                except Exception as e:
                    logger.error(f"Error in degradation callback: {str(e)}")

            return True

        return False

    def _update_performance_window(self, processing_time: float):
        """Update sliding window of performance metrics."""
        self.performance_window.append(processing_time)

        # Maintain window size
        if len(self.performance_window) > self.window_size:
            self.performance_window.pop(0)

    def register_degradation_callback(self, callback):
        """Register callback for performance degradation events."""
        self.degradation_callbacks.append(callback)

    def get_performance_trends(self) -> Dict[str, Any]:
        """Get performance trend analysis."""
        if len(self.performance_window) < 2:
            return {}

        window_array = np.array(self.performance_window)

        # Calculate trend using linear regression
        x = np.arange(len(window_array))
        coefficients = np.polyfit(x, window_array, 1)
        trend_slope = coefficients[0]

        # Determine trend direction
        if abs(trend_slope) < 0.001:
            trend = "stable"
        elif trend_slope > 0:
            trend = "degrading"
        else:
            trend = "improving"

        return {
            "trend": trend,
            "slope": float(trend_slope),
            "current_avg": (
                float(np.mean(window_array[-10:])) if len(window_array) >= 10 else float(np.mean(window_array))
            ),
            "overall_avg": float(np.mean(window_array)),
            "std_dev": float(np.std(window_array)),
            "sample_count": len(window_array),
        }

    def export_metrics_for_monitoring(self) -> Dict[str, Any]:
        """Export metrics in format suitable for monitoring systems (Prometheus/Grafana)."""
        base_stats = self.get_summary_stats()
        performance_trends = self.get_performance_trends()

        # Format for monitoring
        metrics = {
            # Counters
            "sentiment_requests_total": self.total_requests,
            "sentiment_successes_total": self.successful_analyses,
            "sentiment_failures_total": self.failed_analyses,
            "sentiment_cache_hits_total": self.cache_hits,
            "sentiment_cache_misses_total": self.cache_misses,
            # Gauges
            "sentiment_success_rate": base_stats["success_rate"],
            "sentiment_cache_hit_rate": base_stats["cache_hit_rate"],
            "sentiment_avg_processing_time_ms": base_stats["avg_processing_time_ms"],
            # Histograms
            "sentiment_confidence_high": self.confidence_distribution["high"],
            "sentiment_confidence_medium": self.confidence_distribution["medium"],
            "sentiment_confidence_low": self.confidence_distribution["low"],
            # Performance trends
            "sentiment_performance_trend": 1 if performance_trends.get("trend") == "degrading" else 0,
            "sentiment_performance_slope": performance_trends.get("slope", 0),
            "sentiment_current_avg_ms": performance_trends.get("current_avg", 0) * 1000,
        }

        return metrics

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_requests": self.total_requests,
            "successful_analyses": self.successful_analyses,
            "failed_analyses": self.failed_analyses,
            "success_rate": self.successful_analyses / self.total_requests if self.total_requests > 0 else 0,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0
                else 0
            ),
            "avg_processing_time_ms": (
                (self.total_processing_time / self.successful_analyses * 1000) if self.successful_analyses > 0 else 0
            ),
            "total_articles_processed": self.total_articles_processed,
            "confidence_distribution": self.confidence_distribution,
            "sentiment_distribution": self.sentiment_distribution,
        }


# Global metrics instance
sentiment_metrics = SentimentMetrics()


class SentimentAnalyzer:
    """
    Financial sentiment analyzer using FinBERT model.
    Provides single and batch analysis with confidence filtering.
    Supports GPU acceleration when available.
    """

    # Model configuration with versioning support
    MODEL_REGISTRY = {
        "finbert_v1": {
            "name": "ProsusAI/finbert",
            "version": "1.0.0",
            "description": "Primary FinBERT model for financial sentiment",
            "specialization": "general",
            "confidence_threshold": 0.6,
            "active": True,
        },
        "finbert_tone": {
            "name": "yiyanghkust/finbert-tone",
            "version": "1.1.0",
            "description": "FinBERT specialized for tone analysis",
            "specialization": "tone",
            "confidence_threshold": 0.55,
            "active": False,
        },
        "finbert_sentiment": {
            "name": "abhilash1910/finbert-sentiment",
            "version": "1.2.0",
            "description": "FinBERT specialized for sentiment analysis",
            "specialization": "sentiment",
            "confidence_threshold": 0.6,
            "active": False,
        },
    }

    # Default model selection
    DEFAULT_MODEL_KEY = "finbert_v1"
    MODEL_NAME = MODEL_REGISTRY[DEFAULT_MODEL_KEY]["name"]

    MAX_LENGTH = 512
    DEFAULT_BATCH_SIZE = 16
    MIN_BATCH_SIZE = 4
    MAX_BATCH_SIZE = 32
    CONFIDENCE_THRESHOLD = 0.6  # Base confidence threshold
    DYNAMIC_CONFIDENCE = True  # Enable dynamic confidence calibration
    CACHE_TTL_RECENT = 300  # 5 minutes for recent queries
    CACHE_TTL_HISTORICAL = 86400  # 24 hours for historical data

    # GPU configuration
    USE_GPU = TORCH_AVAILABLE and torch.cuda.is_available()
    DEVICE = 0 if USE_GPU else -1  # Use first GPU if available, else CPU
    GPU_BATCH_MULTIPLIER = 2  # Increase batch size on GPU

    # Performance monitoring
    MAX_PROCESSING_TIME = 30.0  # Maximum seconds per batch
    ERROR_THRESHOLD = 0.2  # Reduce batch size if error rate > 20%

    # Model lifecycle management
    MODEL_IDLE_TIMEOUT = 600  # 10 minutes in seconds
    MODEL_MAX_USAGE = 1000  # Unload after this many uses to prevent memory leaks

    # Sentiment mapping
    LABEL_MAP = {"positive": "positive", "negative": "negative", "neutral": "neutral"}

    def __init__(self):
        """Initialize the sentiment analyzer with FinBERT model."""
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._initialised = False
        self._last_used = None
        self._usage_count = 0
        self._model_lock = threading.Lock()

        # Smart caching components
        self._cache_embeddings = {}  # Store text embeddings for similarity
        self._cache_results = {}  # Store sentiment results
        if SKLEARN_AVAILABLE:
            self._tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        else:
            self._tfidf_vectorizer = None
        self._cache_warm_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]  # Frequently analyzed stocks
        self._similarity_threshold = 0.95  # Cosine similarity threshold for cache hits

        # Advanced error recovery
        self._circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        self._error_counts = {error_type: 0 for error_type in ErrorType}

        # Dynamic confidence calibration
        self._confidence_history = []  # Store confidence scores for calibration
        self._confidence_thresholds = {
            "short_text": 0.7,  # Higher threshold for short texts
            "medium_text": 0.6,  # Standard threshold
            "long_text": 0.55,  # Lower threshold for long texts
            "technical": 0.65,  # Financial jargon heavy texts
            "news": 0.6,  # Standard news articles
            "social": 0.7,  # Social media or informal texts
        }

        # Model version management
        self._current_model_key = self.DEFAULT_MODEL_KEY
        self._model_performance = {}  # Track performance per model
        self._ab_test_config = None  # A/B testing configuration

        # Adaptive batch processing
        self.current_batch_size = self.DEFAULT_BATCH_SIZE
        if self.USE_GPU:
            self.current_batch_size *= self.GPU_BATCH_MULTIPLIER
            logger.info(f"GPU detected - using CUDA device {self.DEVICE} with batch size {self.current_batch_size}")
        else:
            logger.info("No GPU detected - using CPU for inference")
        self.recent_processing_times = []
        self.recent_error_count = 0
        self.total_batches_processed = 0

    @property
    def is_initialised(self) -> bool:
        """Check if model is loaded and ready."""
        return self._initialised

    def _check_model_lifecycle(self):
        """Check if model should be unloaded based on usage or idle time."""
        if not self._initialised:
            return

        with self._model_lock:
            current_time = time.time()

            # Idle timeout verification
            if self._last_used and (current_time - self._last_used) > self.MODEL_IDLE_TIMEOUT:
                logger.info(f"Model idle for {self.MODEL_IDLE_TIMEOUT}s, unloading to save memory")
                self._unload_model()
                return

            # Usage count verification
            if self._usage_count > self.MODEL_MAX_USAGE:
                logger.info(f"Model usage count ({self._usage_count}) exceeded, unloading to prevent memory leaks")
                self._unload_model()
                return

    def _unload_model(self):
        """Unload model from memory to free resources."""
        try:
            if self.model:
                del self.model
                self.model = None
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            if self.pipeline:
                del self.pipeline
                self.pipeline = None

            # Clear GPU cache if using GPU
            if self.USE_GPU and TORCH_AVAILABLE:
                torch.cuda.empty_cache()

            # Force garbage collection
            import gc

            gc.collect()

            self._initialised = False
            self._last_used = None
            self._usage_count = 0

            logger.info("Model unloaded successfully, memory freed")

        except Exception as e:
            logger.error(f"Error unloading model: {str(e)}")

    def _preprocess_text(self, text: str) -> str:
        """
        Advanced text preprocessing with financial domain normalization.

        Args:
            text: Raw text input

        Returns:
            Cleaned and normalized text safe for tokenization
        """
        if not text:
            return ""

        # Step 1: Basic character normalization
        # Replace smart quotes and special characters using explicit Unicode code points
        text = re.sub(r"[\u201C\u201D]", '"', text)  # Smart quotes
        text = re.sub(r"[\u2018\u2019]", "'", text)  # Smart apostrophes
        text = re.sub(r"[\u2013\u2014]", "-", text)  # Em/en dashes
        text = re.sub(r"\u2026", "...", text)  # Ellipsis

        # Step 2: Financial entity recognition and normalization
        text = self._normalize_financial_entities(text)

        # Step 3: Abbreviation expansion
        text = self._expand_financial_abbreviations(text)

        # Step 4: Unicode normalization
        text = unicodedata.normalize("NFKD", text)

        # Step 5: Handle special financial patterns
        text = self._handle_financial_patterns(text)

        # Step 6: Remove or replace non-ASCII characters
        text = text.encode("ascii", "ignore").decode("ascii")

        # Step 7: Sentence boundary detection
        text = self._fix_sentence_boundaries(text)

        # Step 8: Clean up extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _normalize_financial_entities(self, text: str) -> str:
        """
        Normalize financial entities like tickers, currencies, and percentages.

        Args:
            text: Text to normalize

        Returns:
            Text with normalized financial entities
        """
        # Normalize ticker symbols (e.g., $AAPL -> AAPL stock)
        text = re.sub(r"\$([A-Z]{1,5})\b", r"\1 stock", text)

        # Normalize currency amounts (e.g., $1.5M -> 1.5 million dollars)
        text = re.sub(r"\$([0-9.]+)M\b", r"\1 million dollars", text)
        text = re.sub(r"\$([0-9.]+)B\b", r"\1 billion dollars", text)
        text = re.sub(r"\$([0-9.]+)T\b", r"\1 trillion dollars", text)
        text = re.sub(r"\$([0-9.]+)K\b", r"\1 thousand dollars", text)

        # Normalize percentages (e.g., +5.2% -> increased 5.2 percent)
        text = re.sub(r"\+([0-9.]+)%", r"increased \1 percent", text)
        text = re.sub(r"-([0-9.]+)%", r"decreased \1 percent", text)
        text = re.sub(r"([0-9.]+)%", r"\1 percent", text)

        # Normalize dates (Q1 2024 -> first quarter 2024)
        text = re.sub(r"Q1", "first quarter", text, flags=re.IGNORECASE)
        text = re.sub(r"Q2", "second quarter", text, flags=re.IGNORECASE)
        text = re.sub(r"Q3", "third quarter", text, flags=re.IGNORECASE)
        text = re.sub(r"Q4", "fourth quarter", text, flags=re.IGNORECASE)

        return text

    def _expand_financial_abbreviations(self, text: str) -> str:
        """
        Expand common financial abbreviations.

        Args:
            text: Text with abbreviations

        Returns:
            Text with expanded abbreviations
        """
        abbreviations = {
            "CEO": "chief executive officer",
            "CFO": "chief financial officer",
            "CTO": "chief technology officer",
            "IPO": "initial public offering",
            "M&A": "mergers and acquisitions",
            "P/E": "price to earnings",
            "EPS": "earnings per share",
            "YoY": "year over year",
            "QoQ": "quarter over quarter",
            "EBITDA": "earnings before interest taxes depreciation and amortization",
            "ROI": "return on investment",
            "ROE": "return on equity",
            "ETF": "exchange traded fund",
            "SEC": "securities and exchange commission",
            "GAAP": "generally accepted accounting principles",
            "YTD": "year to date",
            "FY": "fiscal year",
            "Rev": "revenue",
            "Avg": "average",
            "Est": "estimated",
        }

        for abbr, expansion in abbreviations.items():
            # Case-sensitive replacement for acronyms
            text = re.sub(r"\b" + abbr + r"\b", expansion, text)
            # Also try case-insensitive for common variations
            text = re.sub(r"\b" + abbr + r"\b", expansion, text, flags=re.IGNORECASE)

        return text

    def _handle_financial_patterns(self, text: str) -> str:
        """
        Handle special financial patterns and formatting.

        Args:
            text: Text with financial patterns

        Returns:
            Normalized text
        """
        # Handle ratios (e.g., 3:1 -> 3 to 1)
        text = re.sub(r"([0-9]+):([0-9]+)", r"\1 to \2", text)

        # Handle ranges (e.g., $10-$20 -> 10 to 20 dollars)
        text = re.sub(r"\$([0-9]+)-\$([0-9]+)", r"\1 to \2 dollars", text)

        # Handle fractions (e.g., 1/4 -> one quarter)
        text = text.replace("1/4", "one quarter")
        text = text.replace("1/2", "one half")
        text = text.replace("3/4", "three quarters")
        text = text.replace("1/3", "one third")
        text = text.replace("2/3", "two thirds")

        # Handle basis points (e.g., 25bps -> 25 basis points)
        text = re.sub(r"([0-9]+)bps", r"\1 basis points", text, flags=re.IGNORECASE)

        return text

    def _fix_sentence_boundaries(self, text: str) -> str:
        """
        Fix sentence boundaries for better text segmentation.

        Args:
            text: Text with potential sentence boundary issues

        Returns:
            Text with corrected sentence boundaries
        """
        # Add spaces after periods if missing
        text = re.sub(r"\.([A-Z])", r". \1", text)

        # Fix common sentence boundary errors
        text = re.sub(r"([a-z])([A-Z])", r"\1. \2", text)

        # Ensure proper spacing around punctuation
        text = re.sub(r"\s*,\s*", ", ", text)
        text = re.sub(r"\s*;\s*", "; ", text)
        text = re.sub(r"\s*:\s*", ": ", text)

        return text

    def _has_problematic_content(self, text: str) -> bool:
        """
        Check if text has content that might cause tokenizer hangs.

        Args:
            text: Text to check

        Returns:
            True if text might be problematic
        """
        if not text:
            return False

        # Check for patterns that cause issues
        problematic_patterns = [
            r"[^\x00-\x7F]",  # Non-ASCII characters
            r"[\u2018\u2019\u201C\u201D]",  # Smart quotes
            r"[\u2013\u2014]",  # Em/en dashes
            r"\u2026",  # Ellipsis
        ]

        for pattern in problematic_patterns:
            if re.search(pattern, text):
                return True

        return False

    def _fallback_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Simple rule-based sentiment fallback for problematic text.

        Args:
            text: Text to analyze

        Returns:
            Basic sentiment result
        """
        if not text:
            return self._neutral_sentiment()

        # Simple word-based sentiment
        positive_words = ["good", "great", "excellent", "positive", "gain", "up", "rise", "bull", "buy", "strong"]
        negative_words = ["bad", "poor", "negative", "loss", "down", "fall", "bear", "sell", "weak", "decline"]

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            score = 0.3  # Mild positive
            label = "positive"
            confidence = 0.6
        elif neg_count > pos_count:
            score = -0.3  # Mild negative
            label = "negative"
            confidence = 0.6
        else:
            score = 0.0
            label = "neutral"
            confidence = 0.5

        return {
            "sentimentScore": score,
            "sentimentLabel": label,
            "sentimentConfidence": confidence,
            "analysisTime": 0.001,
            "textLength": len(text),
            "timestamp": datetime.now().isoformat(),
            "fallback": True,
        }

    def _fallback_sentiment_result(self, text: str) -> Dict[str, Any]:
        """
        Generate fallback sentiment result in FinBERT pipeline format.

        Args:
            text: Text to analyze

        Returns:
            Result in FinBERT pipeline format (label, score)
        """
        if not text:
            return {"label": "neutral", "score": 0.5}

        # Simple word-based sentiment
        positive_words = ["good", "great", "excellent", "positive", "gain", "up", "rise", "bull", "buy", "strong"]
        negative_words = ["bad", "poor", "negative", "loss", "down", "fall", "bear", "sell", "weak", "decline"]

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            return {"label": "positive", "score": 0.7}
        elif neg_count > pos_count:
            return {"label": "negative", "score": 0.7}
        else:
            return {"label": "neutral", "score": 0.6}

    def _lazy_init(self):
        """Lazy initialization of the model to save memory with thread safety."""
        global _FINBERT_DISABLED

        if _FINBERT_DISABLED:
            logger.info("FinBERT is disabled, skipping model initialization")
            return

        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            logger.info("PyTorch or Transformers not available, disabling FinBERT")
            _FINBERT_DISABLED = True
            return

        # Check without lock first (double-checked locking pattern)
        if self._initialized:
            return

        with self._model_lock:
            # Check again inside the lock to prevent race conditions
            if self._initialized:
                logger.debug("Model already initialized by another thread")
                return

            try:
                logger.info("Loading FinBERT model for sentiment analysis...")
                start_time = time.time()

                # Load tokenizer and model from trusted FinBERT source
                self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)  # nosec B902
                self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)  # nosec B902

                # Move model to appropriate device if GPU available
                if self.USE_GPU:
                    self.model = self.model.cuda()
                    logger.info(f"Model moved to GPU (CUDA device {self.DEVICE})")

                # Create pipeline for easier inference
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.DEVICE,
                    batch_size=self.current_batch_size,
                )

                # Clear GPU cache if using GPU
                if self.USE_GPU:
                    torch.cuda.empty_cache()

                self._initialized = True
                self._last_used = time.time()
                self._usage_count = 0
                load_time = time.time() - start_time
                device_type = "GPU" if self.USE_GPU else "CPU"
                logger.info(f"FinBERT model loaded successfully on {device_type} in {load_time:.2f} seconds")

            except Exception as e:
                logger.error(f"Failed to load FinBERT model, disabling: {str(e)}")
                _FINBERT_DISABLED = True
                raise RuntimeError(f"Model initialization failed: {str(e)}")

    def analyzeSentimentSingle(self, text: str, use_cache: bool = True, timeout_seconds: int = 15) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text.

        Args:
            text: Financial text to analyze
            use_cache: Whether to use cache for results

        Returns:
            Dictionary with sentiment score, label, and confidence
        """
        global _FINBERT_DISABLED

        # Check model lifecycle
        self._check_model_lifecycle()

        if not text or not text.strip():
            return self._neutral_sentiment()

        # Check if FinBERT is globally disabled or text has problematic content
        if _FINBERT_DISABLED or self._has_problematic_content(text):
            if _FINBERT_DISABLED:
                logger.info("Using fallback sentiment analysis - FinBERT disabled")
            else:
                logger.info("Using fallback sentiment analysis for problematic text")
            return self._fallback_sentiment(text)

        # Preprocess text to handle problematic characters
        text = self._preprocess_text(text)
        if not text:
            return self._neutral_sentiment()

        # Check semantic cache first
        if use_cache:
            # Try semantic similarity cache
            semantic_cached = self._get_semantic_cache(text)
            if semantic_cached:
                return semantic_cached

            # Fall back to exact match cache
            cache_key = f"sentiment:single:{hashlib.blake2b(text.encode(), digest_size=16).hexdigest()}"
            cached_result = cache.get(cache_key)
            if cached_result:
                sentiment_metrics.log_cache_hit("single_text")
                return cached_result
            else:
                sentiment_metrics.log_cache_miss("single_text")

        # Ensure model is loaded
        try:
            self._lazy_init()
        except Exception as e:
            logger.warning(f"Model initialization failed, using fallback: {str(e)}")
            return self._fallback_sentiment(text)

        if _FINBERT_DISABLED:
            return self._fallback_sentiment(text)

        try:
            # Truncate text if too long
            if len(text) > 5000:
                text = text[:5000]

            # Run sentiment analysis
            start_time = time.time()

            # Update usage tracking
            with self._model_lock:
                self._usage_count += 1
                self._last_used = time.time()

            results = self.pipeline(text, truncation=True, max_length=self.MAX_LENGTH)

            if not results:
                return self._neutral_sentiment()

            result = results[0]

            # Map label and calculate score
            label = self.LABEL_MAP.get(result["label"].lower(), "neutral")
            raw_confidence = result["score"]

            # Calibrate confidence
            confidence = self._calibrate_confidence(raw_confidence, text)

            # Get dynamic confidence threshold
            dynamic_threshold = self._get_dynamic_confidence_threshold(text)

            # Apply dynamic confidence threshold
            if confidence < dynamic_threshold:
                logger.debug(
                    f"Confidence {confidence:.2f} below dynamic threshold {dynamic_threshold:.2f}, "
                    f"returning neutral (text type: {self._classify_text_type(text)})"
                )
                return self._neutral_sentiment()

            # Calculate sentiment score (-1 to 1)
            if label == "positive":
                score = confidence
            elif label == "negative":
                score = -confidence
            else:
                score = 0.0

            analysis_time = time.time() - start_time

            result_dict = {
                "sentimentScore": round(score, 4),
                "sentimentLabel": label,
                "sentimentConfidence": round(confidence, 4),
                "analysisTime": round(analysis_time, 3),
                "textLength": len(text),
                "timestamp": datetime.now().isoformat(),
            }

            # Cache result with adaptive TTL
            if use_cache:
                adaptive_ttl = self.get_adaptive_cache_ttl()
                cache.set(cache_key, result_dict, adaptive_ttl)
                # Also update semantic cache
                self._update_semantic_cache(text, result_dict)

            # Log successful analysis
            sentiment_metrics.log_success("single_text", score, confidence, analysis_time)
            return result_dict

        except Exception as e:
            # Log failure
            sentiment_metrics.log_failure("single_text", str(e), time.time() - start_time)
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return self._neutral_sentiment()

    def analyzeSentimentBatch(self, texts: List[str], use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of multiple texts in batch.

        Args:
            texts: List of financial texts to analyze
            use_cache: Whether to use cache for results

        Returns:
            List of sentiment analysis results
        """
        if not texts:
            return []

        # Ensure model is loaded
        self._lazy_init()

        results = []
        valid_texts = []
        valid_indices = []

        # Preprocess texts and prepare cache keys for bulk operations
        cache_keys = []
        processed_texts = []
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results.append(self._neutral_sentiment())
                cache_keys.append(None)
                processed_texts.append(None)
            else:
                # Preprocess text to handle problematic characters
                processed_text = self._preprocess_text(text)
                if not processed_text:
                    results.append(self._neutral_sentiment())
                    cache_keys.append(None)
                    processed_texts.append(None)
                    continue

                # Truncate if needed
                if len(processed_text) > 5000:
                    processed_text = processed_text[:5000]

                processed_texts.append(processed_text)
                cache_key = f"sentiment:single:{hashlib.blake2b(processed_text.encode(), digest_size=16).hexdigest()}" if use_cache else None
                cache_keys.append(cache_key)
                results.append(None)  # Placeholder

        # Bulk cache lookup for better performance
        if use_cache:
            valid_cache_keys = [k for k in cache_keys if k is not None]
            if valid_cache_keys:
                cached_results = cache.get_many(valid_cache_keys)
                for i, cache_key in enumerate(cache_keys):
                    if cache_key and cache_key in cached_results:
                        results[i] = cached_results[cache_key]
                    elif cache_key:  # Not in cache, need processing
                        valid_texts.append(processed_texts[i])
                        valid_indices.append(i)
            else:
                # No cache keys, process all valid texts
                for i, processed_text in enumerate(processed_texts):
                    if processed_text:
                        valid_texts.append(processed_text)
                        valid_indices.append(i)
        else:
            # No caching, process all valid texts
            for i, processed_text in enumerate(processed_texts):
                if processed_text:
                    valid_texts.append(processed_text)
                    valid_indices.append(i)

        # Process valid texts in adaptive batches
        if valid_texts:
            try:
                start_time = time.time()

                # Process in adaptive chunks to manage memory and performance
                for batch_start in range(0, len(valid_texts), self.current_batch_size):
                    batch_end = min(batch_start + self.current_batch_size, len(valid_texts))
                    batch_texts = valid_texts[batch_start:batch_end]

                    # Run batch analysis with retry logic
                    batch_results = self._process_batch_with_retry(batch_texts)

                    # Process batch results
                    for j, result in enumerate(batch_results):
                        idx = valid_indices[batch_start + j]

                        if not result:
                            results[idx] = self._neutral_sentiment()
                            continue

                        # Map label and calculate score
                        label = self.LABEL_MAP.get(result["label"].lower(), "neutral")
                        confidence = result["score"]

                        # Apply confidence threshold
                        if confidence < self.CONFIDENCE_THRESHOLD:
                            results[idx] = self._neutral_sentiment()
                            continue

                        # Calculate sentiment score
                        if label == "positive":
                            score = confidence
                        elif label == "negative":
                            score = -confidence
                        else:
                            score = 0.0

                        result_dict = {
                            "sentimentScore": round(score, 4),
                            "sentimentLabel": label,
                            "sentimentConfidence": round(confidence, 4),
                            "textLength": len(batch_texts[j]),
                            "timestamp": datetime.now().isoformat(),
                        }

                        results[idx] = result_dict

                        # Results will be cached in bulk after processing all batches

                batch_time = time.time() - start_time
                logger.info(f"Batch sentiment analysis completed: {len(valid_texts)} texts in {batch_time:.2f}s")

                # Bulk cache setting for better performance
                if use_cache:
                    cache_data = {}
                    for i, cache_key in enumerate(cache_keys):
                        if cache_key and results[i] and results[i].get("sentimentScore") is not None:
                            cache_data[cache_key] = results[i]
                    if cache_data:
                        cache.set_many(cache_data, self.CACHE_TTL_RECENT)
                        logger.debug(f"Cached {len(cache_data)} sentiment results in bulk")

            except Exception as e:
                logger.error(f"Error in batch sentiment analysis: {str(e)}")
                # Fill remaining None values with neutral sentiment
                for i in range(len(results)):
                    if results[i] is None:
                        results[i] = self._neutral_sentiment()

        return results

    def aggregateSentiment(self, sentiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple sentiment scores into overall sentiment.

        Args:
            sentiments: List of sentiment analysis results

        Returns:
            Aggregated sentiment with statistics
        """
        if not sentiments:
            return self._neutral_sentiment()

        scores = []
        labels = {"positive": 0, "negative": 0, "neutral": 0}
        total_confidence = 0

        for sentiment in sentiments:
            if sentiment and "sentimentScore" in sentiment:
                scores.append(sentiment["sentimentScore"])
                label = sentiment.get("sentimentLabel", "neutral")
                labels[label] = labels.get(label, 0) + 1
                total_confidence += sentiment.get("sentimentConfidence", 0)

        if not scores:
            return self._neutral_sentiment()

        # Calculate aggregate metrics
        avg_score = sum(scores) / len(scores)
        avg_confidence = total_confidence / len(scores)

        # Determine overall label
        if avg_score > 0.1:
            overall_label = "positive"
        elif avg_score < -0.1:
            overall_label = "negative"
        else:
            overall_label = "neutral"

        return {
            "sentimentScore": round(avg_score, 4),
            "sentimentLabel": overall_label,
            "sentimentConfidence": round(avg_confidence, 4),
            "distribution": labels,
            "sampleCount": len(scores),
            "minScore": round(min(scores), 4) if scores else 0,
            "maxScore": round(max(scores), 4) if scores else 0,
            "timestamp": datetime.now().isoformat(),
        }

    def analyzeSentimentWithEnsemble(self, text: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Analyze sentiment using model ensemble for improved accuracy.

        Args:
            text: Text to analyze
            use_cache: Whether to use cache

        Returns:
            Ensemble sentiment result
        """
        try:
            # Import ensemble module
            from Analytics.services.sentiment_ensemble import get_sentiment_ensemble

            # Check cache first
            if use_cache:
                cache_key = f"sentiment:ensemble:{hashlib.blake2b(text.encode(), digest_size=16).hexdigest()}"
                cached_result = cache.get(cache_key)
                if cached_result:
                    sentiment_metrics.log_cache_hit("ensemble")
                    return cached_result

            # Get ensemble instance
            ensemble = get_sentiment_ensemble()

            # Preprocess text
            processed_text = self._preprocess_text(text)
            if not processed_text:
                return self._neutral_sentiment()

            # Analyze with ensemble
            start_time = time.time()
            result = ensemble.analyze_with_ensemble(
                processed_text, voting_strategy="confidence_weighted", return_all_predictions=False
            )

            # Add metadata
            result["analysisTime"] = time.time() - start_time
            result["textLength"] = len(text)
            result["timestamp"] = datetime.now().isoformat()
            result["ensemble"] = True

            # Cache result
            if use_cache:
                cache.set(cache_key, result, self.get_adaptive_cache_ttl())

            # Log success
            sentiment_metrics.log_success(
                "ensemble", result["sentimentScore"], result["sentimentConfidence"], result["analysisTime"]
            )

            return result

        except Exception as e:
            logger.error(f"Ensemble analysis failed, falling back to single model: {str(e)}")
            # Fall back to single model analysis
            return self.analyzeSentimentSingle(text, use_cache)

    def analyzeNewsArticles(
        self, articles: List[Dict[str, str]], aggregate: bool = True, symbol: str = None
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of news articles.

        Args:
            articles: List of article dicts with 'title' and 'summary' keys
            aggregate: Whether to return aggregated or individual results
            symbol: Stock symbol (used for problematic stock handling)

        Returns:
            Sentiment analysis results
        """
        if not articles:
            return self._neutral_sentiment()

        # For problematic stocks, use fallback approach
        if symbol and symbol.upper() in PROBLEMATIC_STOCKS:
            logger.info(f"Using fallback sentiment analysis for problematic stock: {symbol}")
            texts = []
            for article in articles:
                title = article.get("title", "")
                summary = article.get("summary", "")
                text = f"{title}. {summary}".strip()
                if text:
                    texts.append(text)

            if not texts:
                return self._neutral_sentiment()

            # Use fallback method for each text
            sentiments = [self._fallback_sentiment(text) for text in texts]

            if aggregate:
                result = self.aggregateSentiment(sentiments)
                result["newsCount"] = len(articles)
                result["fallback"] = True
                return result
            else:
                return {
                    "articles": sentiments,
                    "newsCount": len(articles),
                    "aggregate": self.aggregateSentiment(sentiments),
                    "fallback": True,
                }

        # Extract text from articles
        texts = []
        for article in articles:
            # Combine title and summary for analysis
            title = article.get("title", "")
            summary = article.get("summary", "")
            text = f"{title}. {summary}".strip()
            if text:
                texts.append(text)

        if not texts:
            return self._neutral_sentiment()

        # Analyze all texts
        sentiments = self.analyzeSentimentBatch(texts)

        if aggregate:
            result = self.aggregateSentiment(sentiments)
            result["newsCount"] = len(articles)
            return result
        else:
            return {
                "articles": sentiments,
                "newsCount": len(articles),
                "aggregate": self.aggregateSentiment(sentiments),
            }

    def _classify_text_type(self, text: str) -> str:
        """
        Classify text type for dynamic confidence calibration.

        Args:
            text: Text to classify

        Returns:
            Text type classification
        """
        if not text:
            return "medium_text"

        # Text length classification
        text_length = len(text)
        if text_length < 100:
            length_type = "short_text"
        elif text_length < 500:
            length_type = "medium_text"
        else:
            length_type = "long_text"

        # Content type detection
        text_lower = text.lower()

        # Check for financial technical terms
        technical_terms = ["ebitda", "p/e", "eps", "roi", "dcf", "wacc", "beta", "alpha"]
        technical_count = sum(1 for term in technical_terms if term in text_lower)

        # Check for news patterns
        news_patterns = ["reported", "announced", "according to", "press release", "statement"]
        news_count = sum(1 for pattern in news_patterns if pattern in text_lower)

        # Check for social media patterns
        social_patterns = ["$", "#", "@", "!", "?", "...", "lol", "omg", "btw"]
        social_count = sum(1 for pattern in social_patterns if pattern in text_lower)

        # Determine primary type
        if technical_count >= 3:
            return "technical"
        elif news_count >= 2:
            return "news"
        elif social_count >= 3:
            return "social"
        else:
            return length_type

    def _get_dynamic_confidence_threshold(self, text: str) -> float:
        """
        Get dynamic confidence threshold based on text characteristics.

        Args:
            text: Text to analyze

        Returns:
            Adjusted confidence threshold
        """
        if not self.DYNAMIC_CONFIDENCE:
            return self.CONFIDENCE_THRESHOLD

        # Classify text type
        text_type = self._classify_text_type(text)
        base_threshold = self._confidence_thresholds.get(text_type, self.CONFIDENCE_THRESHOLD)

        # Adjust based on confidence history (Platt scaling)
        if len(self._confidence_history) >= 100:
            # Calculate recent confidence distribution
            recent_confidences = self._confidence_history[-100:]
            avg_confidence = np.mean(recent_confidences)
            std_confidence = np.std(recent_confidences)

            # Adjust threshold based on distribution
            if avg_confidence < 0.5:  # Model is generally uncertain
                adjusted_threshold = base_threshold * 0.9  # Lower threshold
            elif avg_confidence > 0.8:  # Model is generally confident
                adjusted_threshold = base_threshold * 1.1  # Raise threshold
            else:
                # Scale based on standard deviation
                if std_confidence > 0.2:  # High variance
                    adjusted_threshold = base_threshold
                else:  # Low variance, model is consistent
                    adjusted_threshold = base_threshold * 0.95

            # Ensure threshold stays within reasonable bounds
            adjusted_threshold = max(0.5, min(0.85, adjusted_threshold))

            logger.debug(
                f"Dynamic confidence threshold: {adjusted_threshold:.3f} "
                f"(base: {base_threshold:.3f}, type: {text_type})"
            )

            return adjusted_threshold

        return base_threshold

    def _update_confidence_history(self, confidence: float):
        """
        Update confidence history for calibration.

        Args:
            confidence: Confidence score to add to history
        """
        self._confidence_history.append(confidence)

        # Limit history size
        if len(self._confidence_history) > 1000:
            self._confidence_history = self._confidence_history[-1000:]

    def _calibrate_confidence(self, raw_confidence: float, text: str) -> float:
        """
        Calibrate confidence score using Platt scaling.

        Args:
            raw_confidence: Raw confidence from model
            text: Original text for context

        Returns:
            Calibrated confidence score
        """
        # Simple Platt scaling approximation
        # Adjust based on text characteristics
        text_type = self._classify_text_type(text)

        if text_type == "short_text":
            # Short texts tend to have inflated confidence
            calibrated = raw_confidence * 0.9
        elif text_type == "technical":
            # Technical texts are more reliable
            calibrated = raw_confidence * 1.05
        elif text_type == "social":
            # Social media texts are less reliable
            calibrated = raw_confidence * 0.85
        else:
            calibrated = raw_confidence

        # Ensure within [0, 1] range
        calibrated = max(0.0, min(1.0, calibrated))

        # Update history
        self._update_confidence_history(calibrated)

        return calibrated

    def _neutral_sentiment(self) -> Dict[str, Any]:
        """Return neutral sentiment result."""
        return {
            "sentimentScore": 0.0,
            "sentimentLabel": "neutral",
            "sentimentConfidence": 0.0,
            "timestamp": datetime.now().isoformat(),
        }

    def _get_semantic_cache(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Check semantic cache for similar text using TF-IDF similarity.

        Args:
            text: Text to check for cached similar results

        Returns:
            Cached sentiment result if similar text found, None otherwise
        """
        if not SKLEARN_AVAILABLE:
            return None

        try:
            # Generate text embedding
            text_embedding = self._get_text_embedding(text)

            # Check similarity with cached embeddings
            for cached_text_hash, cached_embedding in self._cache_embeddings.items():
                similarity = cosine_similarity([text_embedding], [cached_embedding])[0][0]

                if similarity >= self._similarity_threshold:
                    # Found similar text in cache
                    if cached_text_hash in self._cache_results:
                        result = self._cache_results[cached_text_hash].copy()
                        result["cache_similarity"] = float(similarity)
                        result["cache_type"] = "semantic"
                        sentiment_metrics.log_cache_hit("semantic")
                        logger.debug(f"Semantic cache hit with similarity {similarity:.3f}")
                        return result

            sentiment_metrics.log_cache_miss("semantic")
            return None

        except Exception as e:
            logger.error(f"Error in semantic cache lookup: {str(e)}")
            return None

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate text embedding using TF-IDF for semantic caching.

        Args:
            text: Text to embed

        Returns:
            Text embedding vector
        """
        if not SKLEARN_AVAILABLE or self._tfidf_vectorizer is None:
            # Fallback to simple hash-based embedding
            text_hash = hashlib.blake2b(text.encode(), digest_size=16).hexdigest()
            return np.array([float(ord(c)) for c in text_hash[:100]])

        try:
            # Fit or transform based on whether vectorizer is fitted
            if not hasattr(self._tfidf_vectorizer, "vocabulary_"):
                # First time - fit the vectorizer
                embedding = self._tfidf_vectorizer.fit_transform([text]).toarray()[0]
            else:
                # Already fitted - just transform
                embedding = self._tfidf_vectorizer.transform([text]).toarray()[0]

            return embedding

        except Exception as e:
            # Fallback to simple hash-based embedding
            logger.warning(f"TF-IDF embedding failed, using hash fallback: {str(e)}")
            text_hash = hashlib.blake2b(text.encode(), digest_size=16).hexdigest()
            return np.array([float(ord(c)) for c in text_hash[:100]])

    def _update_semantic_cache(self, text: str, result: Dict[str, Any]):
        """
        Update semantic cache with new text and result.

        Args:
            text: Original text
            result: Sentiment analysis result
        """
        if not SKLEARN_AVAILABLE:
            return

        try:
            text_hash = hashlib.blake2b(text.encode(), digest_size=16).hexdigest()
            text_embedding = self._get_text_embedding(text)

            # Store in memory cache
            self._cache_embeddings[text_hash] = text_embedding
            self._cache_results[text_hash] = result.copy()

            # Limit cache size
            if len(self._cache_embeddings) > 1000:
                # Remove oldest entries
                oldest = list(self._cache_embeddings.keys())[:100]
                for key in oldest:
                    del self._cache_embeddings[key]
                    if key in self._cache_results:
                        del self._cache_results[key]

        except Exception as e:
            logger.error(f"Error updating semantic cache: {str(e)}")

    def warm_cache(self, symbols: Optional[List[str]] = None):
        """
        Pre-warm cache for frequently analyzed stocks.

        Args:
            symbols: List of stock symbols to pre-warm cache for
        """
        symbols = symbols or self._cache_warm_stocks

        logger.info(f"Warming cache for stocks: {symbols}")

        for symbol in symbols:
            try:
                # Generate cache key
                cache_key = f"sentiment:stock:{symbol}:warm"

                # Check if already cached
                if cache.get(cache_key):
                    continue

                # Fetch recent news (this would normally call Yahoo Finance)
                # For now, we'll just mark it as warmed
                cache.set(
                    cache_key,
                    {"warmed": True, "timestamp": datetime.now().isoformat()},
                    timeout=self.CACHE_TTL_HISTORICAL,
                )

            except Exception as e:
                logger.error(f"Error warming cache for {symbol}: {str(e)}")

    def get_adaptive_cache_ttl(self, symbol: str = None) -> int:
        """
        Get adaptive cache TTL based on market hours and volatility.

        Args:
            symbol: Stock symbol for volatility-based TTL

        Returns:
            Cache TTL in seconds
        """
        now = datetime.now()
        weekday = now.weekday()
        hour = now.hour

        # Market hours (9:30 AM - 4:00 PM EST, weekdays)
        is_market_hours = weekday < 5 and 9 <= hour <= 16

        if is_market_hours:
            # Shorter cache during market hours
            base_ttl = self.CACHE_TTL_RECENT // 2  # 2.5 minutes

            # Even shorter for volatile stocks
            if symbol and symbol in ["TSLA", "GME", "AMC", "NVDA"]:
                return base_ttl // 2  # 1.25 minutes

            return base_ttl
        else:
            # Longer cache outside market hours
            return self.CACHE_TTL_HISTORICAL  # 24 hours

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available models from registry."""
        return self.MODEL_REGISTRY.copy()

    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about currently loaded model."""
        model_info = self.MODEL_REGISTRY.get(self._current_model_key, {}).copy()
        model_info["current"] = True
        model_info["key"] = self._current_model_key
        model_info["initialized"] = self._initialized

        # Add performance stats if available
        if self._current_model_key in self._model_performance:
            model_info["performance"] = self._model_performance[self._current_model_key]

        return model_info

    def switch_model(self, model_key: str, force_reload: bool = False) -> bool:
        """
        Switch to a different model variant.

        Args:
            model_key: Key of model in registry
            force_reload: Whether to force reload even if same model

        Returns:
            True if switch successful
        """
        if model_key not in self.MODEL_REGISTRY:
            logger.error(f"Model {model_key} not found in registry")
            return False

        if model_key == self._current_model_key and not force_reload:
            logger.info(f"Model {model_key} already active")
            return True

        try:
            # Unload current model
            if self._initialized:
                self._unload_model()

            # Update configuration
            old_model_key = self._current_model_key
            self._current_model_key = model_key
            model_config = self.MODEL_REGISTRY[model_key]

            # Update class variables
            self.MODEL_NAME = model_config["name"]
            self.CONFIDENCE_THRESHOLD = model_config["confidence_threshold"]

            # Initialize new model
            self._lazy_init()

            logger.info(
                f"Successfully switched from {old_model_key} to {model_key}",
                extra={
                    "event_type": "model_switch",
                    "old_model": old_model_key,
                    "new_model": model_key,
                    "model_name": model_config["name"],
                    "version": model_config["version"],
                },
            )

            return True

        except Exception as e:
            logger.error(f"Failed to switch to model {model_key}: {str(e)}")
            return False

    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance comparison across all models."""
        return self._model_performance.copy()

    def clearCache(self, pattern: str = "sentiment:*"):
        """Clear sentiment cache entries."""
        try:
            cache.delete_pattern(pattern)

            # Also clear in-memory semantic cache
            self._cache_embeddings.clear()
            self._cache_results.clear()

            logger.info(f"Cleared cache entries matching pattern: {pattern}")
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")

    def getCachedSentiment(self, cache_key: str, symbol: str, is_recent: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get cached sentiment with metrics logging.

        Args:
            cache_key: Cache key to lookup
            symbol: Stock symbol for logging
            is_recent: Whether this is a recent query (affects TTL)

        Returns:
            Cached sentiment result or None
        """
        try:
            cached_result = cache.get(cache_key)
            if cached_result:
                sentiment_metrics.log_cache_hit(symbol)

                # Add cache metadata
                cached_result["cached"] = True
                cached_result["cache_key"] = cache_key
                cached_result["cache_type"] = "recent" if is_recent else "historical"

                return cached_result
            else:
                sentiment_metrics.log_cache_miss(symbol)
                return None

        except Exception as e:
            logger.error(f"Error retrieving cache for {cache_key}: {str(e)}")
            sentiment_metrics.log_cache_miss(symbol)
            return None

    def setCachedSentiment(self, cache_key: str, result: Dict[str, Any], symbol: str, is_recent: bool = True) -> bool:
        """
        Set cached sentiment with appropriate TTL.

        Args:
            cache_key: Cache key
            result: Sentiment result to cache
            symbol: Stock symbol for logging
            is_recent: Whether this is a recent query (affects TTL)

        Returns:
            True if cached successfully
        """
        try:
            ttl = self.CACHE_TTL_RECENT if is_recent else self.CACHE_TTL_HISTORICAL

            # Add cache metadata
            cache_data = result.copy()
            cache_data["cached_at"] = datetime.now().isoformat()
            cache_data["cache_ttl"] = ttl

            cache.set(cache_key, cache_data, ttl)

            logger.debug(
                f"Cached sentiment for {symbol}",
                extra={
                    "event_type": "sentiment_cached",
                    "symbol": symbol,
                    "cache_key": cache_key,
                    "ttl_seconds": ttl,
                    "cache_type": "recent" if is_recent else "historical",
                },
            )
            return True

        except Exception as e:
            logger.error(f"Error caching sentiment for {cache_key}: {str(e)}")
            return False

    def generateCacheKey(
        self, symbol: str = None, text_hash: str = None, days: int = None, analysis_type: str = "sentiment"
    ) -> str:
        """
        Generate standardized cache keys for different sentiment operations.

        Args:
            symbol: Stock symbol
            text_hash: Hash of text content
            days: Number of days for news analysis
            analysis_type: Type of analysis

        Returns:
            Standardized cache key or None for invalid inputs
        """
        if text_hash:
            # For single text analysis
            return f"sentiment:text:{text_hash}"
        elif symbol and days:
            # For stock news analysis
            return f"sentiment:stock:{symbol}:{days}d"
        elif symbol:
            # For general stock sentiment
            return f"sentiment:stock:{symbol}"
        else:
            # Invalid inputs - don't cache to avoid unbounded cache growth
            logger.warning(f"Invalid cache key inputs for {analysis_type}, skipping cache")
            return None

    def _update_batch_performance(self, processing_time: float, had_error: bool = False):
        """
        Update batch processing performance metrics and adapt batch size.

        Args:
            processing_time: Time taken to process the batch
            had_error: Whether the batch had an error
        """
        self.total_batches_processed += 1
        self.recent_processing_times.append(processing_time)

        if had_error:
            self.recent_error_count += 1

        # Keep only last 10 measurements
        if len(self.recent_processing_times) > 10:
            self.recent_processing_times.pop(0)

        # Reset error count every 20 batches
        if self.total_batches_processed % 20 == 0:
            self.recent_error_count = 0

        # Adapt batch size based on performance
        self._adapt_batch_size()

    def _extract_batch_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract features from text batch for ML-based optimization.

        Args:
            texts: List of texts to analyze

        Returns:
            Feature vector for batch
        """
        if not texts:
            return np.zeros(6)

        # Extract text characteristics
        lengths = [len(text) for text in texts]
        avg_length = np.mean(lengths)
        std_length = np.std(lengths)
        min_length = min(lengths)
        max_length = max(lengths)

        # Complexity features
        complex_chars = sum(len(re.findall(r"[^\x00-\x7F]", text)) for text in texts) / len(texts)
        financial_terms = sum(
            len(re.findall(r"\b(?:USD|EUR|GBP|\$|%|Q[1-4]|CEO|CFO|IPO|M&A)\b", text, re.IGNORECASE)) for text in texts
        ) / len(texts)

        return np.array(
            [
                len(texts),  # Batch size
                avg_length,
                std_length,
                max_length - min_length,  # Length range
                complex_chars,
                financial_terms,
            ]
        )

    def _predict_optimal_batch_size(self, texts: List[str]) -> int:
        """
        Predict optimal batch size using simple ML model.

        Args:
            texts: List of texts to process

        Returns:
            Predicted optimal batch size
        """
        if len(self.recent_processing_times) < 10:
            # Not enough data for prediction, use heuristics
            return self._heuristic_batch_size(texts)

        # Extract features
        features = self._extract_batch_features(texts)

        # Simple linear model based on historical performance
        # This is a simplified version - in production, you'd use scikit-learn

        # Feature weights (learned from historical data)
        weights = np.array(
            [
                -0.1,  # Batch size (negative - larger batches take longer)
                -0.0001,  # Average length (negative - longer texts slower)
                -0.0002,  # Std length (negative - variability adds overhead)
                -0.00005,  # Length range (negative - range adds complexity)
                -0.01,  # Complex characters (negative - complex chars slower)
                0.002,  # Financial terms (slightly positive - model optimized for these)
            ]
        )

        # Predict processing time
        predicted_time = np.dot(features, weights) + 5.0  # Base time

        # Calculate optimal batch size based on predicted time
        if predicted_time > self.MAX_PROCESSING_TIME:
            # Reduce batch size
            optimal_size = max(self.MIN_BATCH_SIZE, int(len(texts) * 0.7))
        elif predicted_time < self.MAX_PROCESSING_TIME / 3:
            # Can increase batch size
            optimal_size = min(self.MAX_BATCH_SIZE, int(len(texts) * 1.3))
        else:
            optimal_size = len(texts)

        return optimal_size

    def _heuristic_batch_size(self, texts: List[str]) -> int:
        """
        Heuristic batch size calculation based on text characteristics.

        Args:
            texts: List of texts to process

        Returns:
            Heuristic batch size
        """
        if not texts:
            return self.MIN_BATCH_SIZE

        # Calculate text complexity
        avg_length = np.mean([len(text) for text in texts])

        # Adjust based on text characteristics
        if avg_length > 1000:  # Long texts
            return max(self.MIN_BATCH_SIZE, self.current_batch_size // 2)
        elif avg_length < 200:  # Short texts
            return min(self.MAX_BATCH_SIZE, self.current_batch_size * 2)
        else:
            return self.current_batch_size

    def _adapt_batch_size(self):
        """
        Enhanced batch size adaptation with ML-based prediction.
        """
        if len(self.recent_processing_times) < 3:
            return  # Need at least 3 samples

        avg_time = sum(self.recent_processing_times) / len(self.recent_processing_times)
        error_rate = self.recent_error_count / min(20, self.total_batches_processed)

        old_batch_size = self.current_batch_size

        # Enhanced adaptation logic
        if avg_time > self.MAX_PROCESSING_TIME or error_rate > self.ERROR_THRESHOLD:
            # Aggressive reduction for poor performance
            reduction_factor = 0.6 if error_rate > 0.3 else 0.8
            self.current_batch_size = max(self.MIN_BATCH_SIZE, int(self.current_batch_size * reduction_factor))
        elif avg_time < self.MAX_PROCESSING_TIME / 3 and error_rate < 0.02:
            # Conservative increase for good performance
            self.current_batch_size = min(self.MAX_BATCH_SIZE, int(self.current_batch_size * 1.1))
        elif len(self.recent_processing_times) >= 10:
            # Use trend analysis for fine-tuning
            trend_slope = np.polyfit(range(len(self.recent_processing_times)), self.recent_processing_times, 1)[0]

            if trend_slope > 0.1:  # Performance degrading
                self.current_batch_size = max(self.MIN_BATCH_SIZE, int(self.current_batch_size * 0.9))
            elif trend_slope < -0.1:  # Performance improving
                self.current_batch_size = min(self.MAX_BATCH_SIZE, int(self.current_batch_size * 1.05))

        if old_batch_size != self.current_batch_size:
            logger.info(
                f"Adapted batch size from {old_batch_size} to {self.current_batch_size}",
                extra={
                    "event_type": "batch_size_adapted",
                    "old_batch_size": old_batch_size,
                    "new_batch_size": self.current_batch_size,
                    "avg_processing_time": avg_time,
                    "error_rate": error_rate,
                    "adaptation_method": "ml_enhanced",
                },
            )

    def _classify_error(self, error: Exception) -> ErrorType:
        """
        Classify error type for intelligent retry strategy.

        Args:
            error: Exception to classify

        Returns:
            ErrorType classification
        """
        error_str = str(error).lower()

        if "tokenizer" in error_str or "encoding" in error_str or "unicode" in error_str:
            return ErrorType.TOKENIZER_ERROR
        elif "cuda" in error_str or "gpu" in error_str or "memory" in error_str:
            return ErrorType.RESOURCE_ERROR
        elif "timeout" in error_str or "timed out" in error_str:
            return ErrorType.TIMEOUT_ERROR
        elif "connection" in error_str or "download" in error_str or "network" in error_str:
            return ErrorType.NETWORK_ERROR
        elif "model" in error_str or "inference" in error_str:
            return ErrorType.MODEL_ERROR
        else:
            return ErrorType.UNKNOWN_ERROR

    def _get_retry_strategy(self, error_type: ErrorType, attempt: int) -> Dict[str, Any]:
        """
        Get retry strategy based on error type.

        Args:
            error_type: Type of error encountered
            attempt: Current retry attempt number

        Returns:
            Dictionary with retry strategy parameters
        """
        strategies = {
            ErrorType.TOKENIZER_ERROR: {
                "max_retries": 1,  # Limited retries for text issues
                "delay": 0.1,
                "backoff": 1.0,
                "jitter": False,
                "fallback": True,  # Use fallback sentiment
            },
            ErrorType.RESOURCE_ERROR: {
                "max_retries": 3,
                "delay": 2.0,  # Longer delay for resource issues
                "backoff": 2.0,
                "jitter": True,
                "fallback": False,
                "reduce_batch": True,  # Try smaller batch size
            },
            ErrorType.TIMEOUT_ERROR: {"max_retries": 2, "delay": 1.0, "backoff": 1.5, "jitter": True, "fallback": True},
            ErrorType.NETWORK_ERROR: {
                "max_retries": 5,
                "delay": 1.0,
                "backoff": 2.0,  # Exponential backoff for network
                "jitter": True,
                "fallback": False,
            },
            ErrorType.MODEL_ERROR: {"max_retries": 2, "delay": 0.5, "backoff": 1.5, "jitter": False, "fallback": True},
            ErrorType.UNKNOWN_ERROR: {"max_retries": 2, "delay": 0.5, "backoff": 1.5, "jitter": True, "fallback": True},
        }

        strategy = strategies.get(error_type, strategies[ErrorType.UNKNOWN_ERROR])

        # Calculate actual delay with backoff and jitter
        delay = strategy["delay"] * (strategy["backoff"] ** attempt)
        if strategy.get("jitter", False):
            delay = delay * (0.5 + random.random())  # Add 0-100% jitter

        strategy["calculated_delay"] = delay
        return strategy

    def _handle_error_with_recovery(self, error: Exception, text: str = None, batch: List[str] = None) -> Optional[Any]:
        """
        Handle error with intelligent recovery.

        Args:
            error: Exception that occurred
            text: Single text that caused error (if applicable)
            batch: Batch of texts that caused error (if applicable)

        Returns:
            Recovery result or None
        """
        error_type = self._classify_error(error)
        self._error_counts[error_type] += 1

        logger.error(
            f"Error classified as {error_type.value}",
            extra={
                "error_type": error_type.value,
                "error_message": str(error),
                "error_count": self._error_counts[error_type],
                "circuit_breaker_open": self._circuit_breaker.is_open,
            },
        )

        # Check circuit breaker
        if not self._circuit_breaker.can_attempt():
            logger.warning("Circuit breaker is open, using fallback")
            if text:
                return self._fallback_sentiment(text)
            elif batch:
                return [self._fallback_sentiment(t) for t in batch]
            return None

        # Get recovery strategy
        strategy = self._get_retry_strategy(error_type, 0)

        # Apply recovery based on error type
        if error_type == ErrorType.RESOURCE_ERROR and strategy.get("reduce_batch"):
            # Reduce batch size for resource errors
            self.current_batch_size = max(self.MIN_BATCH_SIZE, self.current_batch_size // 2)
            logger.info(f"Reduced batch size to {self.current_batch_size} due to resource error")

            # Clear GPU cache if using GPU
            if self.USE_GPU and TORCH_AVAILABLE:
                torch.cuda.empty_cache()

        elif error_type == ErrorType.TOKENIZER_ERROR:
            # For tokenizer errors, clean text more aggressively
            if text:
                # Remove all non-ASCII characters
                cleaned_text = "".join(char for char in text if ord(char) < 128)
                return cleaned_text

        # Return fallback if strategy suggests it
        if strategy.get("fallback"):
            if text:
                return self._fallback_sentiment(text)
            elif batch:
                return [self._fallback_sentiment(t) for t in batch]

        return None

    def _process_batch_with_retry(self, texts: List[str], max_retries: int = None) -> List[Dict[str, Any]]:
        """
        Process batch with intelligent retry logic and error handling.

        Args:
            texts: List of texts to process
            max_retries: Maximum number of retries per batch

        Returns:
            List of sentiment results
        """
        global _FINBERT_DISABLED

        # If FinBERT is disabled or pipeline not available, use fallback
        if _FINBERT_DISABLED or not self.pipeline:
            logger.info("Using fallback for batch sentiment processing")
            return [self._fallback_sentiment_result(text) for text in texts]

        # Classify error to determine max retries
        error_type = None

        for attempt in range(10):  # Max attempts across all strategies
            try:
                # Check circuit breaker
                if not self._circuit_breaker.can_attempt():
                    logger.warning("Circuit breaker is open, using fallback for batch")
                    return [self._fallback_sentiment_result(text) for text in texts]

                start_time = time.time()

                # Process batch with optimal batch size (not all texts at once)
                optimal_batch_size = min(self.current_batch_size, len(texts), 8)  # Cap at 8 for memory efficiency
                batch_results = self.pipeline(texts, truncation=True, max_length=self.MAX_LENGTH, batch_size=optimal_batch_size)

                processing_time = time.time() - start_time
                self._update_batch_performance(processing_time, had_error=False)
                self._circuit_breaker.record_success()

                return batch_results

            except Exception as e:
                processing_time = time.time() - start_time if "start_time" in locals() else 0
                self._update_batch_performance(processing_time, had_error=True)
                self._circuit_breaker.record_failure()

                # Classify error
                error_type = self._classify_error(e)
                strategy = self._get_retry_strategy(error_type, attempt)

                # Check if we should retry based on strategy
                if max_retries is None:
                    max_retries = strategy["max_retries"]

                if attempt >= max_retries:
                    logger.error(
                        f"Batch processing failed after {attempt + 1} attempts: {str(e)}",
                        extra={
                            "event_type": "batch_failed",
                            "batch_size": len(texts),
                            "error": str(e),
                            "error_type": error_type.value,
                        },
                    )

                    # Try recovery
                    recovery_result = self._handle_error_with_recovery(e, batch=texts)
                    if recovery_result:
                        return recovery_result
                    raise

                # Apply retry delay with jitter
                delay = strategy["calculated_delay"]
                logger.warning(
                    f"Batch processing attempt {attempt + 1} failed ({error_type.value}), "
                    f"retrying in {delay:.2f}s: {str(e)}",
                    extra={
                        "event_type": "batch_retry",
                        "attempt": attempt + 1,
                        "batch_size": len(texts),
                        "error": str(e),
                        "error_type": error_type.value,
                        "retry_delay": delay,
                    },
                )
                time.sleep(delay)

                # Apply error-specific recovery
                if error_type == ErrorType.RESOURCE_ERROR and strategy.get("reduce_batch"):
                    # Split batch for resource errors
                    if len(texts) > 1:
                        mid = len(texts) // 2
                        logger.info(f"Splitting batch of {len(texts)} into two batches")
                        batch1 = self._process_batch_with_retry(texts[:mid], max_retries - attempt - 1)
                        batch2 = self._process_batch_with_retry(texts[mid:], max_retries - attempt - 1)
                        return batch1 + batch2

        return []  # Should never reach here


# Singleton instance
_sentiment_analyzer = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get or create singleton sentiment analyzer instance."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer

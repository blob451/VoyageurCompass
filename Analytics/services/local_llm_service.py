"""
Local LLM service providing financial analysis explanations through Ollama integration.
Implements LLaMA 3.1 models with FinBERT sentiment enhancement and graceful degradation capabilities.
"""

import hashlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)

# Import monitoring components (graceful degradation if monitoring not available)
try:
    from Analytics.monitoring.llm_monitor import llm_operation_monitor, llm_logger
    MONITORING_AVAILABLE = True
except ImportError:
    logger.info("LLM monitoring not available - operating without monitoring")
    MONITORING_AVAILABLE = False
    llm_operation_monitor = lambda func: func  # No-op decorator

# Conditional import for ollama - graceful degradation in CI/testing environments
try:
    import ollama
    from ollama import Client

    OLLAMA_AVAILABLE = True
except ImportError:
    logger.warning("Ollama not available - LLM service will operate in fallback mode")
    ollama = None
    Client = None
    OLLAMA_AVAILABLE = False


class LLMCircuitBreaker:
    """Circuit breaker pattern for LLM service reliability."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call_llm(self, func, *args, **kwargs):
        """Circuit breaker wrapper for LLM calls."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("LLM service circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time > self.recovery_timeout

    def _on_success(self):
        """Handle successful request."""
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self):
        """Handle failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class LLMPerformanceMonitor:
    """Performance monitoring for LLM operations."""

    def __init__(self):
        self.metrics = {
            "generation_times": [],
            "cache_hits": 0,
            "cache_misses": 0,
            "error_count": 0,
            "total_requests": 0,
            "model_usage_stats": {},
        }
        self.start_time = time.time()

    def record_generation(self, duration: float, model: str, success: bool, cache_hit: bool = False):
        """Record generation performance metrics."""
        self.metrics["total_requests"] += 1

        if success:
            self.metrics["generation_times"].append(duration)
            if cache_hit:
                self.metrics["cache_hits"] += 1
            else:
                self.metrics["cache_misses"] += 1

            # Track model usage
            if model in self.metrics["model_usage_stats"]:
                self.metrics["model_usage_stats"][model] += 1
            else:
                self.metrics["model_usage_stats"][model] = 1
        else:
            self.metrics["error_count"] += 1

    def get_recent_performance(self, window_minutes: int = 10) -> Dict[str, float]:
        """Get recent performance metrics for load balancing decisions."""
        # For simplicity, use last N generations as "recent"
        recent_generations = self.metrics["generation_times"][-10:]  # Last 10 generations

        if not recent_generations:
            return {"avg_generation_time": 0, "recent_error_rate": 0, "sample_size": 0}

        return {
            "avg_generation_time": sum(recent_generations) / len(recent_generations),
            "recent_error_rate": self.metrics["error_count"] / max(1, self.metrics["total_requests"]),
            "sample_size": len(recent_generations),
        }

    def get_performance_summary(self) -> Dict[str, float]:
        """Calculate performance summary statistics."""
        generation_times = self.metrics["generation_times"]
        if not generation_times:
            return {"error": "No successful generations recorded"}

        total_cache_requests = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        total_requests = self.metrics["total_requests"]
        error_count = self.metrics["error_count"]

        return {
            "avg_generation_time": sum(generation_times) / len(generation_times),
            "p95_generation_time": self._percentile(generation_times, 95),
            "success_rate": (total_requests - error_count) / max(1, total_requests),
            "error_rate": error_count / max(1, total_requests),
            "cache_hit_rate": self.metrics["cache_hits"] / max(1, total_cache_requests),
            "total_requests": total_requests,
            "uptime_minutes": (time.time() - self.start_time) / 60,
        }

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


class SentimentEnhancedPromptBuilder:
    """Enhanced prompt builder that integrates sentiment context into LLM prompts."""

    def __init__(self):
        self.financial_terms_map = {
            "earnings": "quarterly earnings report",
            "revenue": "revenue performance",
            "guidance": "forward guidance",
            "eps": "earnings per share",
            "pe": "price-to-earnings ratio",
        }

    def build_sentiment_aware_prompt(
        self,
        analysis_data: Dict[str, Any],
        sentiment_data: Optional[Dict[str, Any]],
        detail_level: str,
        explanation_type: str = "technical_analysis",
    ) -> str:
        """
        Build prompt enriched with sentiment context for enhanced explanation generation.

        Args:
            analysis_data: Technical analysis data
            sentiment_data: FinBERT sentiment analysis results
            detail_level: Level of detail for explanation
            explanation_type: Type of explanation to generate

        Returns:
            Context-aware prompt string
        """
        symbol = analysis_data.get("symbol", "UNKNOWN")
        score = analysis_data.get("score_0_10", 0)

        # Technical context construction
        base_context = self._build_technical_context(analysis_data, detail_level)

        # Sentiment context integration
        sentiment_context = self._build_sentiment_context(sentiment_data) if sentiment_data else ""

        # Detail-level instruction construction
        instruction = self._build_enhanced_instruction(symbol, score, detail_level, sentiment_data)

        # Combine all components
        enhanced_prompt = f"{base_context}\n{sentiment_context}\n{instruction}"

        return enhanced_prompt.strip()

    def _build_technical_context(self, analysis_data: Dict[str, Any], detail_level: str) -> str:
        """Build technical analysis context section."""
        symbol = analysis_data.get("symbol", "UNKNOWN")
        score = analysis_data.get("score_0_10", 0)
        weighted_scores = analysis_data.get("weighted_scores", {})

        if detail_level == "summary":
            # Ultra-concise for summary
            top_indicators = self._get_top_indicators(analysis_data, limit=2)
            return f"{symbol} Technical Score: {score}/10\nKey indicators: {top_indicators}"

        elif detail_level == "detailed":
            # Comprehensive technical overview
            context = f"Technical Analysis for {symbol} (Score: {score}/10)\n\nKey Technical Indicators:"

            # Primary indicator integration
            if weighted_scores:
                sorted_indicators = sorted(weighted_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
                for indicator, value in sorted_indicators:
                    clean_name = indicator.replace("w_", "").upper()
                    strength = "Strong" if abs(value) > 0.15 else "Moderate" if abs(value) > 0.08 else "Weak"
                    direction = "Bullish" if value > 0 else "Bearish" if value < 0 else "Neutral"
                    context += f"\n- {clean_name}: {strength} {direction} signal ({value:+.3f})"

            return context

        else:  # standard
            top_indicators = self._get_top_indicators(analysis_data, limit=3)
            return f"{symbol} Analysis (Score: {score}/10)\nPrimary indicators: {top_indicators}"

    def _build_sentiment_context(self, sentiment_data: Dict[str, Any]) -> str:
        """Build sentiment analysis context section."""
        if not sentiment_data:
            return ""

        sentiment_score = sentiment_data.get("sentimentScore", 0)
        sentiment_confidence = sentiment_data.get("sentimentConfidence", 0)
        sentiment_label = sentiment_data.get("sentimentLabel", "neutral")
        news_count = sentiment_data.get("newsCount", 0)

        # Determine sentiment strength description
        if sentiment_confidence >= 0.8:
            confidence_desc = "highly confident"
            emphasis = "This strong sentiment signal should be considered alongside technical indicators."
        elif sentiment_confidence >= 0.6:
            confidence_desc = "moderately confident"
            emphasis = "Consider this sentiment context when evaluating technical signals."
        else:
            confidence_desc = "uncertain"
            emphasis = "Focus primarily on technical indicators due to unclear sentiment."

        sentiment_context = f"\nMarket Sentiment Analysis:"
        sentiment_context += f"\n- Current sentiment: {sentiment_label.upper()} ({sentiment_score:+.2f})"
        sentiment_context += f"\n- Confidence level: {confidence_desc} ({sentiment_confidence:.2f})"

        if news_count > 0:
            sentiment_context += f"\n- Based on {news_count} recent news articles"

        sentiment_context += f"\n- Integration guidance: {emphasis}"

        return sentiment_context

    def _build_enhanced_instruction(
        self, symbol: str, score: float, detail_level: str, sentiment_data: Optional[Dict[str, Any]]
    ) -> str:
        """Build enhanced instruction section with sentiment awareness and consistency rules."""

        # Determine consistent recommendation based on score
        if score >= 7:
            expected_rec = "BUY"
            score_desc = "strong bullish"
        elif score >= 4:
            expected_rec = "HOLD"
            score_desc = "mixed/neutral"
        else:
            expected_rec = "SELL"
            score_desc = "weak bearish"

        # Determine if sentiment should influence the instruction
        sentiment_influence = False
        sentiment_alignment = ""

        if sentiment_data:
            sentiment_confidence = sentiment_data.get("sentimentConfidence", 0)
            sentiment_score = sentiment_data.get("sentimentScore", 0)
            sentiment_label = sentiment_data.get("sentimentLabel", "neutral")

            if sentiment_confidence >= 0.7:
                sentiment_influence = True

                # Determine technical-sentiment alignment
                if score >= 7 and sentiment_score > 0.2:
                    sentiment_alignment = " Note how positive market sentiment aligns with strong technical indicators."
                elif score <= 4 and sentiment_score < -0.2:
                    sentiment_alignment = " Explain how negative sentiment reinforces weak technical signals."
                elif (score >= 6.5 and sentiment_score < -0.2) or (score <= 4.5 and sentiment_score > 0.2):
                    sentiment_alignment = " Address the divergence between technical analysis and market sentiment."

        # Consistency rules application
        consistency_rules = f"""
IMPORTANT: Based on the {score}/10 score, your recommendation MUST be {expected_rec}.
- Scores 7-10 = BUY
- Scores 4-6.9 = HOLD
- Scores 0-3.9 = SELL"""

        # Detail-level instruction with word count targets
        if detail_level == "summary":
            instruction = f"{consistency_rules}\n\nProvide a clear {expected_rec} recommendation for {symbol} with the primary reason. Target: 50-60 words for complete sentences."
            if sentiment_influence:
                instruction += sentiment_alignment

        elif detail_level == "detailed":
            instruction = f"""{consistency_rules}

Generate a comprehensive {expected_rec} analysis for {symbol} in 250-300 words using this structure:

**Investment Thesis:** Clear {expected_rec} recommendation with confidence level and core reasoning

**Technical Indicators:** Detailed analysis of key indicators supporting the {expected_rec} decision

**Risk Analysis:** Main risks, challenges, and risk mitigation strategies

**Market Context:** Price outlook, catalysts, and market environment factors"""

            if sentiment_influence:
                instruction += f"""

**Sentiment Impact:** How current market sentiment ({sentiment_data.get('sentimentLabel', 'neutral')}) affects the analysis"""

            instruction += f"\n\nUse professional investment research language. Ensure all sections support the {expected_rec} recommendation."
            if sentiment_influence and sentiment_alignment:
                instruction += sentiment_alignment

        else:  # standard
            instruction = f"""{consistency_rules}

Provide a professional {expected_rec} recommendation for {symbol} in 100-120 words including:
- Clear {expected_rec} decision with confidence level
- 2-3 key supporting technical factors
- Primary risk consideration
- Brief market outlook
Ensure complete sentences."""
            if sentiment_influence:
                instruction += "\n- How current market sentiment affects the outlook"
                instruction += sentiment_alignment

        return instruction


class ConfidenceAdaptiveGeneration:
    """Adaptive generation parameters based on sentiment confidence and analysis complexity."""

    def __init__(self):
        self.base_options = {
            "temperature": 0.4,
            "top_p": 0.7,
            "num_predict": 1000,  # Default token count - will be adjusted by detail level and strategy
            "stop": ["###", "END"],  # Removed \n\n\n to prevent early termination in detailed mode
            "repeat_penalty": 1.05,
            "top_k": 20,
        }

    def get_confidence_weighted_options(
        self, sentiment_data: Optional[Dict[str, Any]], complexity_score: float, detail_level: str, model_name: str
    ) -> Dict[str, Any]:
        """
        Generate adaptive generation options based on sentiment confidence and analysis complexity.

        Args:
            sentiment_data: FinBERT sentiment analysis results
            complexity_score: Technical analysis complexity score
            detail_level: Level of detail requested
            model_name: Target model name

        Returns:
            Optimized generation options
        """
        options = self.base_options.copy()

        # Extract sentiment metrics
        sentiment_confidence = 0.5  # Default neutral confidence
        sentiment_strength = 0.0

        if sentiment_data:
            sentiment_confidence = sentiment_data.get("sentimentConfidence", 0.5)
            sentiment_strength = abs(sentiment_data.get("sentimentScore", 0))

        # Determine generation strategy
        strategy = self._determine_generation_strategy(sentiment_confidence, sentiment_strength, complexity_score)

        # Apply strategy-specific adjustments
        if strategy == "high_confidence_definitive":
            # High confidence in both sentiment and technical - be definitive
            options["temperature"] = 0.25  # Very focused
            options["top_p"] = 0.6
            options["num_predict"] = self._get_token_count(detail_level) + 25  # Longer explanation
            options["top_k"] = 15  # More focused vocabulary

        elif strategy == "medium_confidence_balanced":
            # Moderate confidence - balanced approach
            options["temperature"] = 0.35
            options["top_p"] = 0.7
            options["num_predict"] = self._get_token_count(detail_level)
            options["top_k"] = 20

        elif strategy == "low_confidence_exploratory":
            # Low confidence - more exploratory and cautious
            options["temperature"] = 0.5
            options["top_p"] = 0.8
            options["num_predict"] = self._get_token_count(detail_level) - 15  # Shorter, more cautious
            options["top_k"] = 30  # Broader vocabulary for exploration

        elif strategy == "conflicting_signals":
            # Sentiment and technical analysis conflict - careful balanced approach
            options["temperature"] = 0.4
            options["top_p"] = 0.75
            options["num_predict"] = self._get_token_count(detail_level) + 10  # Slightly longer for nuance
            options["top_k"] = 25

        # Model-specific adjustments
        self._apply_model_specific_adjustments(options, model_name)

        return options

    def _determine_generation_strategy(
        self, sentiment_confidence: float, sentiment_strength: float, complexity_score: float
    ) -> str:
        """Determine the optimal generation strategy based on confidence metrics."""

        # High confidence scenario
        if sentiment_confidence >= 0.8 and complexity_score >= 0.7:
            return "high_confidence_definitive"

        # Medium confidence scenario
        elif sentiment_confidence >= 0.6 and complexity_score >= 0.5:
            return "medium_confidence_balanced"

        # Low confidence scenario
        elif sentiment_confidence < 0.6 or complexity_score < 0.4:
            return "low_confidence_exploratory"

        # Conflicting signals (high technical complexity but low sentiment confidence, or vice versa)
        elif abs(sentiment_confidence - complexity_score) > 0.3:
            return "conflicting_signals"

        # Default to balanced approach
        return "medium_confidence_balanced"

    def _get_token_count(self, detail_level: str) -> int:
        """Get base token count for detail level with proper word-to-token ratios.
        
        DEPRECATED: Use _get_optimized_tokens instead for consistency.
        """
        # Redirect to the optimized method to prevent conflicts
        return self._get_optimized_tokens(detail_level)

    def _apply_model_specific_adjustments(self, options: Dict[str, Any], model_name: str):
        """Apply model-specific parameter adjustments."""
        if "8b" in model_name.lower():
            # 8B model adjustments - optimized for detailed explanations
            options["num_ctx"] = 2048  # Increased context window for longer responses
            if options["temperature"] < 0.3:
                options["temperature"] = 0.35  # Slightly higher for more varied content
            # Ensure adequate generation for detailed explanations
            if options.get("num_predict", 0) >= 2500:  # For detailed mode
                options["top_p"] = 0.85  # Slightly more diverse for longer content
                options["top_k"] = 35    # Broader vocabulary for detailed analysis
        else:
            # 70B model adjustments
            options["num_ctx"] = 2048  # Increased context window
            if options["temperature"] > 0.3:
                options["temperature"] *= 0.85  # Slightly less aggressive reduction


class ModelHealthService:
    """Centralised model health monitoring and availability tracking."""
    
    def __init__(self):
        self.model_health = {}
        self.last_check_time = {}
        self.check_interval = getattr(settings, "OLLAMA_HEALTH_CHECK_INTERVAL", 30)
        self._lock = threading.Lock()
    
    def is_model_healthy(self, model_name: str, client=None) -> bool:
        """Check if a specific model is healthy with caching."""
        current_time = time.time()
        
        with self._lock:
            # Check cache first
            if (model_name in self.last_check_time and 
                current_time - self.last_check_time[model_name] < self.check_interval):
                return self.model_health.get(model_name, False)
            
            # Perform health check
            try:
                if not client:
                    return False
                
                # Quick ping test to the model
                response = client.generate(
                    model=model_name,
                    prompt="test",
                    options={"num_predict": 1, "temperature": 0.1}
                )
                
                is_healthy = bool(response and "response" in response)
                
                self.model_health[model_name] = is_healthy
                self.last_check_time[model_name] = current_time
                
                if is_healthy:
                    logger.debug(f"[MODEL HEALTH] {model_name} is healthy")
                else:
                    logger.warning(f"[MODEL HEALTH] {model_name} failed health check")
                
                return is_healthy
                
            except Exception as e:
                logger.error(f"[MODEL HEALTH] Health check failed for {model_name}: {str(e)}")
                self.model_health[model_name] = False
                self.last_check_time[model_name] = current_time
                return False
    
    def get_healthy_models(self, models: List[str], client=None) -> List[str]:
        """Get list of healthy models from provided list."""
        healthy_models = []
        for model in models:
            if self.is_model_healthy(model, client):
                healthy_models.append(model)
        return healthy_models


class LocalLLMService:
    """Enhanced LLM Service with multi-model configuration and performance optimisation."""

    def __init__(self):
        # Multi-model configuration for different detail levels
        self.summary_model = getattr(settings, "OLLAMA_SUMMARY_MODEL", "phi3:3.8b")
        self.standard_model = getattr(settings, "OLLAMA_STANDARD_MODEL", "phi3:3.8b")
        self.detailed_model = getattr(settings, "OLLAMA_DETAILED_MODEL", "llama3.1:8b")
        self.translation_model = getattr(settings, "OLLAMA_TRANSLATION_MODEL", "qwen2:3b")
        
        # Legacy model hierarchy for backward compatibility
        self.primary_model = getattr(settings, "OLLAMA_PRIMARY_MODEL", "llama3.1:8b")
        self.current_model = self.primary_model  # Start with fast model

        # Enhanced configuration - read from settings
        self._client = None
        self._client_lock = threading.Lock()
        self._availability_cache = {}
        self._availability_cache_timeout = 5  # Cache availability for 5 seconds

        self.max_retries = 3
        self.timeout = 60
        # Increased timeouts for different detail levels
        self.base_generation_timeout = getattr(settings, "OLLAMA_GENERATION_TIMEOUT", 60)  # Increased from 45
        self.timeout_multipliers = {
            "summary": 1.0,  # 60 seconds for summary
            "standard": 1.5,  # 90 seconds for standard
            "detailed": 2.0,  # 120 seconds for detailed
        }
        self.performance_mode = getattr(settings, "OLLAMA_PERFORMANCE_MODE", True)

        # Connection retry configuration
        self.connection_retry_attempts = getattr(settings, "OLLAMA_RETRY_ATTEMPTS", 3)
        self.connection_retry_delay = getattr(settings, "OLLAMA_RETRY_DELAY", 1)

        # Security and resource management integration
        self._security_validator = None
        self._resource_manager = None
        self.security_enabled = getattr(settings, "LLM_SECURITY_ENABLED", True)
        self.resource_management_enabled = getattr(settings, "LLM_RESOURCE_MANAGEMENT_ENABLED", True)

        # Circuit breaker, monitoring and health service initialisation
        self.circuit_breaker = LLMCircuitBreaker()
        self.performance_monitor = LLMPerformanceMonitor()
        self.model_health_service = ModelHealthService()

        # Thread executor for timeout handling with optimized resource allocation
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="llm-timeout")

        # Resource cleanup tracking
        self._active_requests = 0
        self._max_concurrent_requests = 5

        # Sentiment enhancement component initialisation
        self.sentiment_prompt_builder = SentimentEnhancedPromptBuilder()
        self.confidence_adaptive_generator = ConfidenceAdaptiveGeneration()

        # Sentiment integration configuration
        self.sentiment_integration_enabled = True
        self.sentiment_cache_prefix = "sentiment_enhanced:"

        # Multilingual support configuration
        self.multilingual_enabled = getattr(settings, "MULTILINGUAL_LLM_ENABLED", True)
        self.language_models = getattr(settings, "LLM_MODELS_BY_LANGUAGE", {
            'en': 'llama3.1:8b',
            'fr': 'qwen2:3b',
            'es': 'qwen2:3b',
        })
        self.supported_languages = list(self.language_models.keys())
        self.default_language = getattr(settings, "DEFAULT_USER_LANGUAGE", "en")

        # Translation service configuration
        self.translation_enabled = getattr(settings, "TRANSLATION_SERVICE_ENABLED", True)
        self.translation_cache_ttl = getattr(settings, "TRANSLATION_CACHE_TTL", 86400)
        self.translation_timeout = getattr(settings, "TRANSLATION_TIMEOUT", 30)

        # Financial terminology mapping
        self.financial_terminology = getattr(settings, "FINANCIAL_TERMINOLOGY_MAPPING", {})

        # Cultural formatting configuration
        self.cultural_formatting_enabled = getattr(settings, "CULTURAL_FORMATTING_ENABLED", True)
        self.financial_formatting = getattr(settings, "FINANCIAL_FORMATTING", {})

        # Client initialisation deferred to lazy property

    @property
    def client(self):
        """Lazy-loaded Ollama client with thread safety and availability checks."""
        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama client requested but not available")
            return None
        if self._client is None:
            with self._client_lock:
                if self._client is None:  # Double-check locking
                    self._client = self._initialise_client_with_retry()
        return self._client

    @client.setter
    def client(self, value):
        """Set the client (mainly for testing purposes)."""
        self._client = value

    def _initialise_client_with_retry(self):
        """Initialize Ollama client connection with retry logic."""
        for attempt in range(self.connection_retry_attempts):
            try:
                ollama_host = getattr(settings, "OLLAMA_HOST", "localhost")
                ollama_port = getattr(settings, "OLLAMA_PORT", 11434)

                client = Client(host=f"http://{ollama_host}:{ollama_port}")

                # Test connection by listing models
                try:
                    models = client.list()
                    logger.info(f"Local LLM client initialised successfully on attempt {attempt + 1}")

                    # Log available models without checking individual model availability yet
                    available_models = [m["name"] for m in models.get("models", [])]
                    logger.info(f"Available models: {available_models}")

                    return client

                except Exception as test_error:
                    logger.warning(f"Client connection test failed on attempt {attempt + 1}: {test_error}")
                    if attempt < self.connection_retry_attempts - 1:
                        time.sleep(self.connection_retry_delay * (2**attempt))  # Exponential backoff
                        continue
                    else:
                        raise test_error

            except Exception as e:
                logger.error(f"Failed to initialise Ollama client on attempt {attempt + 1}: {str(e)}")
                if attempt < self.connection_retry_attempts - 1:
                    time.sleep(self.connection_retry_delay * (2**attempt))  # Exponential backoff
                else:
                    logger.error(f"Failed to initialise Ollama client after {self.connection_retry_attempts} attempts")
                    return None

        return None

    def _verify_model_availability(self, model_name: str = None) -> bool:
        """Check if the specified model is available with caching."""
        target_model = model_name or self.current_model
        cache_key = f"model_availability_{target_model}"

        # Cache verification
        current_time = time.time()
        if cache_key in self._availability_cache:
            cached_result, cached_time = self._availability_cache[cache_key]
            if current_time - cached_time < self._availability_cache_timeout:
                return cached_result

        # Availability assessment
        try:
            if not self.client:
                logger.warning(f"No client available to check model {target_model}")
                # Cache negative result for shorter time
                self._availability_cache[cache_key] = (False, current_time)
                return False

            models = self.client.list()
            available_models = [model["name"] for model in models.get("models", [])]
            is_available = target_model in available_models

            # Cache the result
            self._availability_cache[cache_key] = (is_available, current_time)

            if not is_available:
                logger.warning(f"Model {target_model} not found in available models: {available_models}")

            return is_available

        except Exception as e:
            logger.error(f"Error checking model availability for {target_model}: {str(e)}")
            # Cache negative result for shorter time on error
            self._availability_cache[cache_key] = (False, current_time)
            return False

    @property
    def generation_timeout(self) -> int:
        """Get generation timeout (for backward compatibility)."""
        return self.base_generation_timeout

    @generation_timeout.setter
    def generation_timeout(self, value: int):
        """Set generation timeout (for testing purposes)."""
        self.base_generation_timeout = value

    def _get_timeout_for_detail_level(self, detail_level: str) -> int:
        """Get appropriate timeout for specific detail level."""
        multiplier = self.timeout_multipliers.get(detail_level, 1.5)
        return int(self.base_generation_timeout * multiplier)

    def _check_resource_availability(self) -> bool:
        """Check if resources are available for new request."""
        if self._active_requests >= self._max_concurrent_requests:
            logger.warning(
                f"[RESOURCE] Max concurrent requests reached ({self._active_requests}/{self._max_concurrent_requests})"
            )
            return False
        return True

    def _generate_with_timeout(self, model: str, prompt: str, options: dict, timeout: int) -> dict:
        """Generate LLM response with proper timeout handling and resource management."""
        # Resource availability assessment
        if not self._check_resource_availability():
            raise Exception("LLM service at maximum capacity, please try again later")

        def _generate():
            try:
                self._active_requests += 1
                logger.debug(f"[RESOURCE] Active requests: {self._active_requests}/{self._max_concurrent_requests}")

                return self.circuit_breaker.call_llm(self.client.generate, model=model, prompt=prompt, options=options)
            finally:
                self._active_requests -= 1
                logger.debug(f"[RESOURCE] Request completed, active: {self._active_requests}")

        try:
            # Submit the generation task to thread executor with timeout
            future = self._executor.submit(_generate)
            response = future.result(timeout=timeout)
            logger.info(f"[LLM TIMEOUT] Generation completed within {timeout}s timeout")
            return response

        except FutureTimeoutError:
            logger.warning(f"[LLM TIMEOUT] Generation exceeded {timeout}s timeout for model {model}")
            # Cancel the future if possible
            future.cancel()
            raise TimeoutError(f"LLM generation timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"[LLM TIMEOUT] Error in timeout-wrapped generation: {str(e)}")
            raise

    def _select_model_for_detail_level(self, detail_level: str) -> str:
        """
        Select the appropriate model based on detail level with health checks.
        
        Args:
            detail_level: The detail level requested ('summary', 'standard', 'detailed')
            
        Returns:
            Model name to use for the request
        """
        # Define model preferences based on detail level
        if detail_level == "summary":
            preferred_models = [self.summary_model, self.standard_model, self.primary_model]
        elif detail_level == "standard": 
            preferred_models = [self.standard_model, self.summary_model, self.primary_model]
        elif detail_level == "detailed":
            preferred_models = [self.detailed_model, self.standard_model, self.primary_model]
        else:
            # Default to standard
            preferred_models = [self.standard_model, self.summary_model, self.primary_model]
        
        # Get healthy models from preferences
        healthy_models = self.model_health_service.get_healthy_models(preferred_models, self.client)
        
        if healthy_models:
            selected_model = healthy_models[0]  # Use first healthy model
            logger.info(f"[MODEL SELECTION] Selected {selected_model} for {detail_level} (healthy models: {len(healthy_models)})")
            return selected_model
        else:
            # Fallback to first preference if no health data available
            fallback_model = preferred_models[0]
            logger.warning(f"[MODEL SELECTION] No healthy models found, using fallback {fallback_model} for {detail_level}")
            return fallback_model

    def _select_optimal_model(self, detail_level: str, complexity_score: float = 0.5) -> str:
        """Enhanced model selection with improved resource allocation and load balancing."""
        if not self.performance_mode:
            return self.detailed_model

        # Use new detail-level based selection with health checks
        return self._select_model_for_detail_level(detail_level)

    def _should_use_premium_model(self) -> bool:
        """
        Determine if premium 70B model should be used based on current load and time.
        Implements intelligent load balancing to optimize resource utilization.
        """
        try:
            # Time-based load balancing: avoid peak usage times
            current_hour = datetime.now().hour

            # Peak hour model preference definition
            peak_hours = range(9, 17)
            is_peak_time = current_hour in peak_hours

            # Performance monitoring: check recent generation times
            recent_performance = self.performance_monitor.get_recent_performance()
            avg_generation_time = recent_performance.get("avg_generation_time", 0)

            # Premium model selection criteria
            # 1. Off-peak hours OR
            # 2. Recent 8B performance is poor (slow) AND not severely overloaded
            use_premium = not is_peak_time or (avg_generation_time > 30 and avg_generation_time < 120)  # 30s-2min range

            if use_premium:
                logger.debug(
                    f"[LOAD BALANCING] Premium model recommended (peak_time: {is_peak_time}, avg_time: {avg_generation_time:.1f}s)"
                )

            return use_premium

        except Exception as e:
            logger.warning(f"[LOAD BALANCING] Error in load balancing logic: {str(e)}")
            return False  # Conservative fallback

    def _has_conflicting_signals(self, complexity_score: float) -> bool:
        """
        Detect if analysis has conflicting signals that require nuanced 70B model analysis.

        Args:
            complexity_score: Calculated complexity score

        Returns:
            True if conflicting signals detected
        """
        # If complexity is moderate but not extreme, may indicate mixed signals
        # This suggests indicators are pointing in different directions
        return 0.4 < complexity_score < 0.7

    def _calculate_complexity_score(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate analysis complexity score for model selection."""
        try:
            weighted_scores = analysis_data.get("weighted_scores", {})
            if not weighted_scores:
                return 0.3  # Low complexity for missing data

            # Count significant indicators (more indicators = higher complexity)
            significant_indicators = sum(1 for v in weighted_scores.values() if abs(v) > 0.05)
            total_indicators = len(weighted_scores)

            # Calculate score variance (higher variance = more complex analysis)
            scores = list(weighted_scores.values())
            if len(scores) > 1:
                mean_score = sum(scores) / len(scores)
                variance = sum((x - mean_score) ** 2 for x in scores) / len(scores)
            else:
                variance = 0

            # Detect conflicting signals (bullish vs bearish indicators)
            positive_signals = sum(1 for v in scores if v > 0.08)
            negative_signals = sum(1 for v in scores if v < -0.08)
            conflicting_signals = min(positive_signals, negative_signals) / max(
                1, max(positive_signals, negative_signals)
            )

            # Extreme score nuanced explanation assessment
            technical_score = analysis_data.get("score_0_10", 5)
            extreme_score_factor = 0
            if technical_score >= 8.5 or technical_score <= 1.5:
                extreme_score_factor = 0.3  # High complexity for extreme scores
            elif technical_score >= 7.5 or technical_score <= 2.5:
                extreme_score_factor = 0.2  # Medium complexity for strong scores

            # Calculate complexity score with enhanced factors
            indicator_complexity = significant_indicators / max(1, total_indicators)
            variance_complexity = min(variance * 20, 1.0)  # Scale variance

            # Combine factors with weights
            complexity = (
                indicator_complexity * 0.5  # Weight based on indicator coverage
                + variance_complexity * 0.2  # Weight based on score variance
                + conflicting_signals * 0.2  # Weight based on conflicting signals
                + extreme_score_factor  # Weight based on extreme scores
            )

            return min(1.0, max(0.1, complexity))  # Ensure reasonable bounds

        except Exception:
            return 0.5  # Default medium complexity

    def _pull_model(self, model_name: str = None):
        """Pull the specified model if not available."""
        try:
            if not self.client:
                raise Exception("Ollama client not initialised")

            target_model = model_name or self.primary_model
            logger.info(f"Pulling model {target_model}...")
            self.client.pull(target_model)
            logger.info(f"Model {target_model} pulled successfully")

        except Exception as e:
            logger.error(f"Failed to pull model {target_model}: {str(e)}")
            raise

    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        if not self.client:
            return False

        # Model availability verification
        return self._verify_model_availability(self.primary_model) or self._verify_model_availability(
            self.detailed_model
        )

    def generate_sentiment_aware_explanation(
        self,
        analysis_data: Dict[str, Any],
        sentiment_data: Optional[Dict[str, Any]] = None,
        detail_level: str = "standard",
        explanation_type: str = "technical_analysis",
    ) -> Optional[Dict[str, Any]]:
        """
        Generate sentiment-enhanced natural language explanation for financial analysis.
        This method integrates FinBERT sentiment analysis with LLaMA explanation generation.

        Args:
            analysis_data: Dictionary containing technical analysis results
            sentiment_data: Dictionary containing FinBERT sentiment analysis results
            detail_level: 'summary', 'standard', or 'detailed'
            explanation_type: Type of explanation to generate

        Returns:
            Dictionary with enhanced explanation content or None if failed
        """
        if not self.is_available():
            logger.warning("Local LLM service not available")
            return None

        # Sentiment-enhanced cache key generation
        cache_key = self._create_sentiment_enhanced_cache_key(
            analysis_data, sentiment_data, detail_level, explanation_type
        )
        dynamic_ttl = self._get_sentiment_aware_ttl(analysis_data, sentiment_data)

        # Cache verification
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"Retrieved sentiment-enhanced explanation from cache (TTL: {dynamic_ttl}s)")
            self.performance_monitor.record_generation(0, "cache", True, cache_hit=True)
            return cached_result

        try:
            # Calculate complexity and select optimal model
            complexity_score = self._calculate_complexity_score(analysis_data)
            selected_model = self._select_optimal_model(detail_level, complexity_score)

            # Sentiment-enhanced prompt construction
            enhanced_prompt = self.sentiment_prompt_builder.build_sentiment_aware_prompt(
                analysis_data, sentiment_data, detail_level, explanation_type
            )

            # Get confidence-weighted generation options
            generation_options = self.confidence_adaptive_generator.get_confidence_weighted_options(
                sentiment_data, complexity_score, detail_level, selected_model
            )

            logger.info(f"[LLM SENTIMENT] Generating {detail_level} explanation using model: {selected_model}")
            if sentiment_data:
                sentiment_label = sentiment_data.get("sentimentLabel", "neutral")
                sentiment_confidence = sentiment_data.get("sentimentConfidence", 0)
                logger.info(
                    f"[LLM SENTIMENT] Sentiment context: {sentiment_label} (confidence: {sentiment_confidence:.2f})"
                )
            logger.info(
                f"[LLM SENTIMENT] Target symbol: {analysis_data.get('symbol', 'Unknown')}, Score: {analysis_data.get('score_0_10', 0)}"
            )

            start_time = time.time()

            try:
                # Use circuit breaker pattern for reliability with ThreadPoolExecutor timeout
                response = self.circuit_breaker.call_llm(
                    self.client.generate, model=selected_model, prompt=enhanced_prompt, options=generation_options
                )

            except TimeoutError as e:
                timeout_duration = time.time() - start_time
                logger.warning(f"Sentiment-enhanced LLM generation timed out after {self.generation_timeout}s")
                self.performance_monitor.record_generation(timeout_duration, selected_model, False)
                return None

            except Exception as e:
                logger.error(f"Error during sentiment-enhanced LLM generation: {str(e)}")
                self.performance_monitor.record_generation(time.time() - start_time, selected_model, False)
                return None

            generation_time = time.time() - start_time

            if not response or "response" not in response:
                logger.error("Invalid response from sentiment-enhanced LLM")
                return None

            # Enhanced performance logging with sentiment context
            symbol = analysis_data.get("symbol", "Unknown")
            if generation_time > 60:
                logger.warning(
                    f"[LLM SENTIMENT PERFORMANCE] Slow generation for {symbol}: {generation_time:.2f}s using {selected_model}"
                )
            elif generation_time > 10:
                logger.info(
                    f"[LLM SENTIMENT PERFORMANCE] Moderate generation time for {symbol}: {generation_time:.2f}s using {selected_model}"
                )
            else:
                logger.info(
                    f"[LLM SENTIMENT PERFORMANCE] Fast generation for {symbol}: {generation_time:.2f}s using {selected_model}"
                )

            explanation_content = response["response"].strip()

            # Log content metrics
            word_count = len(explanation_content.split())
            logger.info(
                f"[LLM SENTIMENT CONTENT] Generated {word_count} words ({len(explanation_content)} chars) for {symbol}"
            )

            # Post-process content to fix any score format issues
            explanation_content = self._fix_score_format(explanation_content)
            
            # Format detailed sections for better structure
            explanation_content = self._format_detailed_sections(explanation_content, detail_level)

            # Enhanced result with sentiment integration metadata
            result = {
                "content": explanation_content,
                "detail_level": detail_level,
                "explanation_type": explanation_type,
                "generation_time": generation_time,
                "model_used": selected_model,
                "timestamp": time.time(),
                "word_count": word_count,
                "confidence_score": self._calculate_confidence_score(explanation_content),
                "performance_optimized": self.performance_mode,
                "sentiment_enhanced": True,
                "sentiment_integration": {
                    "sentiment_data_available": sentiment_data is not None,
                    "sentiment_confidence": sentiment_data.get("sentimentConfidence", 0) if sentiment_data else 0,
                    "sentiment_label": sentiment_data.get("sentimentLabel", "neutral") if sentiment_data else "neutral",
                    "generation_strategy": self.confidence_adaptive_generator._determine_generation_strategy(
                        sentiment_data.get("sentimentConfidence", 0.5) if sentiment_data else 0.5,
                        abs(sentiment_data.get("sentimentScore", 0)) if sentiment_data else 0,
                        complexity_score,
                    ),
                },
            }

            # Cache the result with dynamic TTL
            cache.set(cache_key, result, dynamic_ttl)
            self.performance_monitor.record_generation(generation_time, selected_model, True, cache_hit=False)

            logger.info(
                f"[LLM SENTIMENT SUCCESS] Generated {detail_level} explanation for {symbol} in {generation_time:.2f}s ({len(explanation_content)} chars)"
            )
            return result

        except Exception as e:
            logger.error(f"Error generating sentiment-enhanced explanation: {str(e)}")
            self.performance_monitor.record_generation(0, "unknown", False)
            return None

    @llm_operation_monitor
    def generate_explanation(
        self,
        analysis_data: Dict[str, Any],
        detail_level: str = "standard",
        explanation_type: str = "technical_analysis",
    ) -> Optional[Dict[str, Any]]:
        """
        Generate natural language explanation for financial analysis.

        Args:
            analysis_data: Dictionary containing analysis results
            detail_level: 'summary', 'standard', or 'detailed'
            explanation_type: Type of explanation to generate

        Returns:
            Dictionary with explanation content or None if failed
        """
        if not self.is_available():
            logger.warning("Local LLM service not available")
            return None

        # Validate input parameters
        if not analysis_data:
            logger.error("[VALIDATION] Analysis data is empty or None")
            return None
            
        if detail_level not in ["summary", "standard", "detailed"]:
            logger.warning(f"[VALIDATION] Invalid detail level '{detail_level}', defaulting to 'standard'")
            detail_level = "standard"
            
        symbol = analysis_data.get("symbol", "UNKNOWN")
        logger.info(f"[GENERATION START] Generating {detail_level} explanation for {symbol}")

        # Create cache key with dynamic TTL
        cache_key = self._create_cache_key(analysis_data, detail_level, explanation_type)
        dynamic_ttl = self._get_dynamic_ttl(analysis_data)

        # Cache verification
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"Retrieved explanation from cache (TTL: {dynamic_ttl}s)")
            self.performance_monitor.record_generation(0, "cache", True, cache_hit=True)
            return cached_result

        try:
            # Calculate complexity and select optimal model
            complexity_score = self._calculate_complexity_score(analysis_data)
            selected_model = self._select_optimal_model(detail_level, complexity_score)
            prompt = self._build_optimized_prompt(analysis_data, detail_level, explanation_type)

            logger.info(f"[LLM SERVICE] Generating {detail_level} explanation using model: {selected_model}")
            logger.info(
                f"[LLM SERVICE] Target symbol: {analysis_data.get('symbol', 'Unknown')}, Score: {analysis_data.get('score_0_10', 0)}"
            )
            start_time = time.time()

            # Determine expected recommendation for quality validation
            score = analysis_data.get("score_0_10", 5)
            if score >= 7:
                expected_rec = "BUY"
            elif score >= 4:
                expected_rec = "HOLD" 
            else:
                expected_rec = "SELL"
            
            try:
                # Use enhanced quality validation system 
                explanation_content, confidence, validation_result = self._retry_with_quality_check(
                    prompt=prompt,
                    detail_level=detail_level,
                    model_name=selected_model,
                    expected_recommendation=expected_rec,
                    max_retries=2
                )

            except Exception as e:
                logger.error(f"Error during LLM generation with quality check: {str(e)}")
                self.performance_monitor.record_generation(time.time() - start_time, selected_model, False)
                return None

            generation_time = time.time() - start_time

            if not explanation_content or "Error:" in explanation_content:
                logger.error("Failed to generate quality explanation after retries")
                return None

            # Enhanced performance logging with model context
            symbol = analysis_data.get("symbol", "Unknown")
            if generation_time > 60:
                logger.warning(
                    f"[LLM PERFORMANCE] Slow generation for {symbol}: {generation_time:.2f}s using {selected_model}"
                )
            elif generation_time > 10:
                logger.info(
                    f"[LLM PERFORMANCE] Moderate generation time for {symbol}: {generation_time:.2f}s using {selected_model}"
                )
            else:
                logger.info(
                    f"[LLM PERFORMANCE] Fast generation for {symbol}: {generation_time:.2f}s using {selected_model}"
                )

            # Log quality metrics
            word_count = validation_result.get("word_count", len(explanation_content.split()))
            quality_score = validation_result.get("quality_score", 0.0)
            logger.info(f"[LLM QUALITY] Generated {word_count} words, Quality: {quality_score:.2f}, Valid: {validation_result.get('is_valid', False)}")

            # Post-process content to fix any score format issues
            explanation_content = self._fix_score_format(explanation_content)
            
            # Format detailed sections for better structure
            explanation_content = self._format_detailed_sections(explanation_content, detail_level)

            result = {
                "content": explanation_content,
                "detail_level": detail_level,
                "explanation_type": explanation_type,
                "generation_time": generation_time,
                "model_used": selected_model,
                "timestamp": time.time(),
                "word_count": word_count,
                "confidence_score": confidence,
                "quality_score": quality_score,
                "quality_validation": validation_result,
                "performance_optimized": self.performance_mode,
            }

            # Cache the result with dynamic TTL
            cache.set(cache_key, result, dynamic_ttl)
            self.performance_monitor.record_generation(generation_time, selected_model, True, cache_hit=False)

            logger.info(
                f"[LLM SUCCESS] Generated {detail_level} explanation for {symbol} in {generation_time:.2f}s ({len(explanation_content)} chars)"
            )
            return result

        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            self.performance_monitor.record_generation(0, "unknown", False)
            return None

    def _get_dynamic_ttl(self, analysis_data: Dict[str, Any]) -> int:
        """Calculate dynamic cache TTL based on analysis characteristics."""
        try:
            # Base TTL
            base_ttl = 180  # 3 minutes default

            # Adjust based on score extremes (high confidence)
            score = analysis_data.get("score_0_10", 5)
            if score >= 8 or score <= 2:
                # Very high/low confidence scores can be cached longer
                return base_ttl * 2  # 6 minutes

            # Check for high volatility indicators
            weighted_scores = analysis_data.get("weighted_scores", {})
            volatility_indicators = ["w_bbwidth20", "w_volsurge"]
            high_volatility = any(abs(weighted_scores.get(ind, 0)) > 0.15 for ind in volatility_indicators)

            if high_volatility:
                return base_ttl // 2  # 90 seconds for volatile stocks

            # Check for mid-range scores (more uncertainty)
            if 4 <= score <= 6:
                return base_ttl // 2  # Shorter cache for uncertain scores

            return base_ttl

        except Exception:
            return 180  # Default 3 minutes

    def _build_optimized_prompt(self, analysis_data: Dict[str, Any], detail_level: str, explanation_type: str) -> str:
        """Build optimized prompt with explicit score-based consistency rules."""
        # Validate essential analysis data is present
        if not analysis_data:
            raise ValueError("Analysis data is empty or None")
        
        symbol = analysis_data.get("symbol", "UNKNOWN")
        if symbol == "UNKNOWN":
            logger.warning("[DATA VALIDATION] Missing symbol in analysis data")
            
        score = analysis_data.get("score_0_10")
        if score is None:
            logger.warning(f"[DATA VALIDATION] Missing score_0_10 for {symbol}, using default value 0")
            score = 0
        elif not isinstance(score, (int, float)) or score < 0 or score > 10:
            logger.warning(f"[DATA VALIDATION] Invalid score value {score} for {symbol}, using default 0")
            score = 0
            
        weighted_scores = analysis_data.get("weighted_scores", {})
        if not weighted_scores:
            logger.warning(f"[DATA VALIDATION] Missing weighted_scores for {symbol}")
            
        indicators = analysis_data.get("indicators", {})
        
        # Validate components data is present (this was the source of the SU issue)
        components = analysis_data.get("components", {})
        if not components:
            logger.warning(f"[DATA VALIDATION] Missing components data for {symbol} - some technical values may be unavailable")

        # Determine consistent recommendation based on score
        if score >= 7:
            expected_rec = "BUY"
            score_desc = "strong bullish"
        elif score >= 4:
            expected_rec = "HOLD"
            score_desc = "mixed/neutral"
        else:
            expected_rec = "SELL"
            score_desc = "weak bearish"
            
        # Extract key technical values for enhanced prompts with robust error handling
        # Use components field which contains the actual indicator data
        def safe_get_indicator(indicator_name: str, field_path: str, default: str = "N/A") -> str:
            """Safely extract indicator value with detailed logging."""
            try:
                indicator_data = components.get(indicator_name, {})
                if not indicator_data:
                    logger.warning(f"[INDICATOR MISSING] {indicator_name} not found in components for {symbol}")
                    return default
                    
                raw_data = indicator_data.get("raw", {})
                if not raw_data:
                    logger.warning(f"[INDICATOR DATA] Missing raw data for {indicator_name} in {symbol}")
                    return default
                    
                value = raw_data.get(field_path)
                if value is None:
                    logger.warning(f"[INDICATOR VALUE] Missing {field_path} in {indicator_name} for {symbol}")
                    return default
                    
                return value
            except Exception as e:
                logger.error(f"[INDICATOR ERROR] Failed to extract {indicator_name}.{field_path} for {symbol}: {e}")
                return default
        
        rsi_value = safe_get_indicator("rsi14", "rsi")
        macd_hist = safe_get_indicator("macd12269", "histogram") 
        sma50 = safe_get_indicator("sma50vs200", "sma50")
        sma200 = safe_get_indicator("sma50vs200", "sma200")
        current_price = safe_get_indicator("pricevs50", "price")
        support = safe_get_indicator("srcontext", "nearest_support")
        resistance = safe_get_indicator("srcontext", "nearest_resistance")

        # Enhanced prompts with consistency rules
        if detail_level == "summary":
            return f"""Stock Analysis: {symbol} scores {score:.1f}/10, indicating {score_desc} signals.

CRITICAL REQUIREMENTS:
- Generate EXACTLY 3 complete sentences (60-80 words total)
- First sentence: Clear {expected_rec} recommendation with the {score:.1f}/10 score
- Second sentence: Primary technical reason with one specific indicator value
- Third sentence: Key risk or opportunity
- NO formatting, NO bold text, NO sections
- Use simple, clear language
- Professional third-person tone (NO first-person narrative)
- NO disclaimers or investment advice warnings

Example format:
"Technical analysis indicates {symbol} warrants a {expected_rec} position based on its {score:.1f}/10 technical score. The [indicator] at [value] suggests [interpretation]. Key considerations include [risk/opportunity factor]."

Generate the 3-sentence recommendation:"""

        elif detail_level == "detailed":
            top_indicators = self._get_top_indicators(analysis_data, limit=4)
            # Pre-format technical values to avoid f-string format specifier issues
            rsi_display = f"{rsi_value:.2f}" if isinstance(rsi_value, (int, float)) else "N/A"
            macd_display = f"{macd_hist:.4f}" if isinstance(macd_hist, (int, float)) else "N/A"
            sma50_display = f"${sma50:.2f}" if isinstance(sma50, (int, float)) else "N/A"
            sma200_display = f"${sma200:.2f}" if isinstance(sma200, (int, float)) else "N/A"
            price_display = f"${current_price:.2f}" if isinstance(current_price, (int, float)) else "N/A"
            support_display = f"${support:.2f}" if isinstance(support, (int, float)) else "N/A"
            resistance_display = f"${resistance:.2f}" if isinstance(resistance, (int, float)) else "N/A"
            
            return f"""Premium Investment Analysis: {symbol} (Score: {score:.1f}/10)

Technical Data: RSI: {rsi_display}, MACD: {macd_display}, SMA50: {sma50_display}, Price: {price_display}, Support/Resistance: {support_display}/{resistance_display}

Key Indicators: {top_indicators}

CRITICAL FORMATTING REQUIREMENTS:
- MANDATORY: Each section MUST use EXACT format: **Section Name:** followed by content
- NO EXTRA asterisks or malformed headers
- Each section should be 120-150+ words
- TARGET: Generate 700-800 total words for comprehensive Premium analysis
- Continue writing until ALL 5 sections are complete and detailed

RECOMMENDATION: {expected_rec} (based on {score:.1f}/10 technical score)

Write comprehensive Premium analysis using this EXACT structure:

**Investment Summary:**
Provide definitive {expected_rec} recommendation with specific confidence percentage (e.g., 75% confidence). Explain primary technical reasoning using RSI ({rsi_display}) and moving average positioning. Detail investment thesis with expected price movement direction, percentage targets, and timeframe. Include position size recommendations and market context affecting this recommendation.

**Technical Analysis:**
Comprehensive indicator analysis: RSI {rsi_display} interpretation (overbought/oversold levels), moving average analysis (price vs SMA50 {sma50_display}), MACD momentum signals ({macd_display}), volume trend analysis, support/resistance levels ({support_display}/{resistance_display}). Provide signal strength ratings, convergence/divergence patterns, and quantitative technical insights with specific numerical thresholds.

**Risk Assessment:**  
Detailed risk-reward analysis with quantitative metrics: specific stop-loss price level, upside price targets with percentage gain potential, probability assessments for different scenarios, maximum drawdown considerations, volatility analysis based on technical indicators. Include risk management strategies and position sizing based on current technical setup.

**Entry Strategy:**
Specific entry points with exact price levels, position sizing methodology, scaling approach (initial vs. additional entries), optimal timing considerations, confirmation signals to monitor before entering. Detail order types, execution strategy, and technical levels that would invalidate the thesis.

**Market Outlook:**
Broader market context affecting this position, sector performance trends and correlations, relationship with major indices, upcoming earnings/catalysts, medium-term technical outlook for next 3-6 months. Include macroeconomic factors and industry-specific considerations that could impact price movement.

GENERATE COMPLETE ANALYSIS NOW - Write each section in full detail with specific values and percentages."""

        else:  # standard (enhanced)
            top_indicators = self._get_top_indicators(analysis_data, limit=3)
            # Pre-format technical values to avoid f-string format specifier issues
            rsi_display = f"{rsi_value:.2f}" if isinstance(rsi_value, (int, float)) else "N/A"
            price_display = f"${current_price:.2f}" if isinstance(current_price, (int, float)) else "N/A"
            sma50_display = f"${sma50:.2f}" if isinstance(sma50, (int, float)) else "N/A"
            support_display = f"${support:.2f}" if isinstance(support, (int, float)) else "N/A"
            resistance_display = f"${resistance:.2f}" if isinstance(resistance, (int, float)) else "N/A"
            
            return f"""Technical Analysis Report: {symbol} ({score:.1f}/10 score)

Key Technical Values:
- RSI: {rsi_display}
- Price: {price_display}
- SMA50: {sma50_display}
- Support/Resistance: {support_display}/{resistance_display}

Top Indicators: {top_indicators}

CRITICAL REQUIREMENTS:
- Generate 180-220 words of technical analysis
- Recommendation MUST be {expected_rec} based on {score:.1f}/10 score
- Include specific indicator values and percentages
- Professional yet accessible language
- Focus on technical analysis with quantitative insights

Provide a structured {expected_rec} recommendation using this format:

**Investment Recommendation:** (60 words)
Clear {expected_rec} decision with confidence level (as percentage), primary technical reasoning referencing the RSI and moving average values, and expected near-term price movement.

**Technical Indicators:** (80 words)
Detailed analysis of 3-4 key indicators with specific values. Include RSI interpretation (overbought/oversold levels), moving average analysis (above/below SMA50/200), MACD momentum, volume trends, and support/resistance levels. Use actual numbers.

**Risk & Outlook:** (60-80 words)
Key risk factors with percentage-based probabilities, stop-loss recommendation at specific price level, upside target with percentage gain, time horizon (days/weeks), and market conditions impact.

Use clear technical language with specific values throughout."""

    def _get_top_indicators(self, analysis_data: Dict[str, Any], limit: int = 3) -> str:
        """Extract top indicators for prompt optimisation."""
        weighted_scores = analysis_data.get("weighted_scores", {})
        if not weighted_scores:
            return "technical analysis"

        # Get top indicators by absolute value
        sorted_indicators = sorted(weighted_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:limit]
        if not sorted_indicators:
            return "technical analysis"

        # Clean indicator names
        indicator_names = []
        for indicator, value in sorted_indicators:
            clean_name = indicator.replace("w_", "").replace("_", " ")
            direction = " (bullish)" if value > 0 else " (bearish)" if value < 0 else " (neutral)"
            indicator_names.append(f"{clean_name}{direction}")

        return ", ".join(indicator_names)

    def _build_prompt_legacy(self, analysis_data: Dict[str, Any], detail_level: str, explanation_type: str) -> str:
        """Build the prompt for LLaMA 3.1 70B based on analysis data with consistency rules."""

        symbol = analysis_data.get("symbol", "UNKNOWN")
        score = analysis_data.get("score_0_10", 0)
        indicators = analysis_data.get("components", {})
        weighted_scores = analysis_data.get("weighted_scores", {})

        # Determine consistent recommendation based on score
        if score >= 7:
            expected_rec = "BUY"
            score_desc = "strong bullish"
        elif score >= 4:
            expected_rec = "HOLD"
            score_desc = "mixed/neutral"
        else:
            expected_rec = "SELL"
            score_desc = "weak bearish"

        # Optimized shorter base prompt
        base_prompt = f"""Financial Analysis for {symbol}: {score}/10 ({score_desc} signals)

IMPORTANT: Based on the {score}/10 score, your recommendation MUST be {expected_rec}.
- Scores 7-10 = BUY
- Scores 4-6.9 = HOLD
- Scores 0-3.9 = SELL

Key Indicators:
"""

        # Add top 3 most significant indicators only
        if weighted_scores:
            sorted_indicators = sorted(weighted_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            for indicator_name, weighted_score in sorted_indicators:
                indicator_value = indicators.get(indicator_name, "N/A")
                base_prompt += f"- {indicator_name}: {weighted_score:.2f}\n"

        # Word count-targeted instructions for complete explanations with consistency
        if detail_level == "summary":
            instruction = f"""

Provide a clean, conversational {expected_rec} recommendation in 50-60 words. Use simple paragraph format without section headers or formatting. Be direct, human-friendly, and concise with complete sentences."""

        elif detail_level == "detailed":
            instruction = f"""

Provide a comprehensive {expected_rec} analysis for {symbol} in 250-300 words using this structure:

**Investment Thesis:** Clear {expected_rec} recommendation with confidence level and core reasoning

**Technical Indicators:** Detailed analysis of key indicators supporting the {expected_rec} decision

**Risk Analysis:** Main risks, challenges, and risk mitigation strategies

**Market Context:** Price outlook, catalysts, and market environment factors

Use professional investment research language. Ensure all sections support the {expected_rec} recommendation."""

        else:  # standard
            instruction = f"""

Provide a professional {expected_rec} recommendation in 100-120 words using this structure:

**Investment Decision:** Clear {expected_rec} recommendation with confidence level

**Technical Analysis:** 2-3 key supporting technical factors

**Risk Assessment:** Primary risk consideration and brief market outlook

Use professional investment language with complete sentences."""

        return base_prompt + instruction + "\n\nAnalysis:"


    def _get_optimized_generation_options(self, detail_level: str, model_name: str) -> dict:
        """Get generation options optimized for enhanced explanation quality."""
        base_options = {
            "temperature": 0.4,  # Balanced creativity and consistency (will be adjusted for detailed mode)
            "top_p": 0.8,  # Allow broader vocabulary for quality (will be adjusted for detailed mode)
            "num_predict": self._get_optimized_tokens(detail_level),
            "stop": ["###", "END"] if detail_level != "detailed" else ["###"],  # Minimal stop sequences for detailed mode
            "repeat_penalty": 1.1,  # Prevent repetition
            "top_k": 30,  # Moderate vocabulary restriction (will be adjusted for detailed mode)
        }

        # Model-specific context optimization
        if "8b" in model_name.lower():
            base_options["num_ctx"] = 1024  # Adequate context for quality
            base_options["temperature"] = 0.4
        else:
            base_options["num_ctx"] = 2048  # Full context for 70B models
            base_options["temperature"] = 0.3

        # Detail level-specific tuning for enhanced output quality
        if detail_level == "summary":
            # Summary mode: Concise but informative
            base_options["temperature"] = 0.3  # Lower for precision
            base_options["top_p"] = 0.7  # Focused vocabulary
            base_options["top_k"] = 20  # More restrictive for conciseness
            
        elif detail_level == "standard":
            # Standard mode: Balanced detail and readability
            base_options["temperature"] = 0.4  # Moderate creativity
            base_options["top_p"] = 0.8  # Good vocabulary range
            base_options["top_k"] = 30  # Balanced restriction
            
        elif detail_level == "detailed":
            # Detailed mode: Comprehensive technical analysis with enhanced generation
            base_options["temperature"] = 0.6  # Increased for more elaborate content generation
            base_options["top_p"] = 0.95  # Broader vocabulary for comprehensive analysis
            base_options["top_k"] = 45  # Less restrictive for detailed coverage
            base_options["repeat_penalty"] = 1.02  # Allow technical repetition for thoroughness
            base_options["min_p"] = 0.05  # Minimum probability threshold for better quality

        return base_options

    def _get_optimized_tokens(self, detail_level: str) -> int:
        """Get optimized token counts based on enhanced word count requirements."""
        token_limits = {
            "summary": 200,    # ~80 words * 2.5 buffer = 200 tokens
            "standard": 500,   # ~200 words * 2.5 buffer = 500 tokens  
            "detailed": 7500,  # ~4500+ words * 1.67 tokens/word = 7500 tokens (increased for comprehensive Premium content)
        }
        return token_limits.get(detail_level, 450)

    def _fix_score_format(self, content: str) -> str:
        """Fix any incorrect score formats in the content (e.g., X.X/1 → X.X/10)."""
        import re
        
        try:
            # Pattern to match X.X/1 format and replace with X.X/10
            fixed_content = re.sub(r'(\d+\.?\d*)/1\b', r'\1/10', content)
            
            # Also fix common variants like "score of X.X/1"
            fixed_content = re.sub(r'score of (\d+\.?\d*)/1\b', r'score of \1/10', fixed_content)
            
            # Log if any fixes were made
            if fixed_content != content:
                logger.info("Fixed score format in LLM response (X.X/1 → X.X/10)")
            
            return fixed_content
        except Exception as e:
            logger.warning(f"Error fixing score format: {e}")
            return content

    def _format_detailed_sections(self, content: str, detail_level: str) -> str:
        """Post-process detailed explanations to improve formatting and structure."""
        if detail_level != "detailed":
            return content
            
        try:
            import re
            
            # Check if the content has section headers (flexible matching)
            import re
            content_lower = content.lower()
            
            # Look for various section header patterns (case-insensitive)
            section_patterns = [
                r'\*\*\s*executive\s+summary\s*:?\s*\*\*',
                r'\*\*\s*technical\s+analysis\s*:?\s*\*\*',
                r'\*\*\s*risk[/&\s]*reward\s*:?\s*\*\*',
                r'\*\*\s*entry[/\s]*exit\s*:?\s*\*\*',
                r'\*\*\s*market\s+context\s*:?\s*\*\*'
            ]
            
            section_matches = sum(1 for pattern in section_patterns if re.search(pattern, content_lower))
            has_proper_sections = section_matches >= 3  # At least 3 out of 5 sections
            
            # Also check for Enhanced-style headers that work well
            enhanced_patterns = [
                r'\*\*\s*investment\s+(recommendation|summary)\s*:?\s*\*\*',
                r'\*\*\s*technical\s+indicators?\s*:?\s*\*\*',
                r'\*\*\s*risk\s*[&\s]*outlook\s*:?\s*\*\*'
            ]
            enhanced_matches = sum(1 for pattern in enhanced_patterns if re.search(pattern, content_lower))
            
            # If we have proper sections OR it follows Enhanced format, don't force fallback
            has_sections = has_proper_sections or enhanced_matches >= 2
            
            if not has_sections:
                # LLM failed to follow instructions - force section structure
                logger.warning("LLM output lacks proper section headers - applying fallback formatting")
                return self._inject_section_structure(content)
            
            # Apply header correction for malformed sections
            content = self._correct_malformed_headers(content)
            
            # Content has sections - enhance formatting with better boundary detection
            formatted_content = self._fix_section_boundaries(content)
            
            # Final cleanup
            formatted_content = re.sub(r'\n\*\*', r'\n\n**', formatted_content)
            formatted_content = re.sub(r'\n{3,}', '\n\n', formatted_content)
            
            logger.info("Applied detailed section formatting post-processing")
            return formatted_content.strip()
            
        except Exception as e:
            logger.warning(f"Error formatting detailed sections: {e}")
            return content

    def _fix_section_boundaries(self, content: str) -> str:
        """Fix section boundaries to ensure all content is properly contained within sections."""
        try:
            import re
            
            # Find all section headers and their positions
            section_pattern = r'\*\*([^*]+?):\*\*'
            sections = []
            
            # Find all section headers with their positions
            for match in re.finditer(section_pattern, content):
                sections.append({
                    'header': match.group(0),
                    'title': match.group(1).strip(),
                    'start': match.start(),
                    'end': match.end()
                })
            
            if not sections:
                # No proper sections found, let fallback handle it
                return content
            
            # Reconstruct content with proper section boundaries
            result_parts = []
            
            # Handle content before first section (should be minimal)
            if sections[0]['start'] > 0:
                pre_content = content[:sections[0]['start']].strip()
                if pre_content:
                    # Move orphaned content to first section
                    pass  # We'll add this to the first section instead
            
            # Process each section
            for i, section in enumerate(sections):
                # Add section header
                result_parts.append(f"\n\n{section['header']}")
                
                # Determine content for this section
                content_start = section['end']
                content_end = sections[i + 1]['start'] if i + 1 < len(sections) else len(content)
                
                section_content = content[content_start:content_end].strip()
                
                # Clean up section content
                if section_content:
                    # Remove any stray ** markers
                    section_content = re.sub(r'\*\*([^:*]+)\*\*(?!\s*:)', r'\1', section_content)
                    
                    # Format paragraphs and bullets
                    section_content = self._format_section_content(section_content)
                    
                    result_parts.append(f"\n{section_content}")
                else:
                    # Empty section - add default content
                    default_content = self._get_default_section_content(section['title'])
                    result_parts.append(f"\n{default_content}")
            
            # Join all parts
            formatted_content = ''.join(result_parts).strip()
            
            # Final validation - check for orphaned content after last section
            orphaned_match = re.search(r'\*\*[^:*]+:\*\*.*?(?=\*\*[^:*]+:\*\*|$)', formatted_content, re.DOTALL)
            if orphaned_match:
                # Find text after the last section that might be orphaned
                last_section_end = formatted_content.rfind('**')
                if last_section_end != -1:
                    after_last = formatted_content[last_section_end:].split('**', 2)
                    if len(after_last) > 2:
                        # There's content after the last section header that doesn't belong to any section
                        orphaned_text = after_last[2].strip()
                        if orphaned_text and not orphaned_text.startswith('**'):
                            # Move orphaned content to the last section
                            formatted_content = formatted_content[:last_section_end] + after_last[0] + '**' + after_last[1] + '**\n' + orphaned_text
            
            logger.info("Successfully fixed section boundaries and contained all content")
            return formatted_content
            
        except Exception as e:
            logger.warning(f"Error fixing section boundaries: {e}")
            return content

    def _format_section_content(self, content: str) -> str:
        """Format content within a section for better readability."""
        try:
            import re
            
            # Enhance bullet points
            content = re.sub(r'^- ', '• ', content, flags=re.MULTILINE)
            content = re.sub(r'^(\d+)\. ', r'• ', content, flags=re.MULTILINE)
            content = re.sub(r'•\s*([^\n]+)', r'• \1', content)
            
            # Improve paragraph breaks for long sentences
            sentences = re.split(r'(\. [A-Z])', content)
            if len(sentences) > 4:  # Only for longer content
                formatted_content = ""
                sentence_count = 0
                for j, sent in enumerate(sentences):
                    if sent.startswith('. '):
                        formatted_content += sent
                        sentence_count += 1
                        # Add paragraph break after every 3 sentences
                        if sentence_count >= 3 and j < len(sentences) - 2:
                            formatted_content += "\n\n"
                            sentence_count = 0
                    else:
                        formatted_content += sent
                return formatted_content
            
            return content
            
        except Exception as e:
            logger.warning(f"Error formatting section content: {e}")
            return content

    def _get_default_section_content(self, section_title: str) -> str:
        """Provide default content for empty sections."""
        defaults = {
            'Investment Summary': 'Investment analysis based on technical indicators and market conditions.',
            'Technical Analysis': 'Technical indicators support the current market assessment and price action analysis.',
            'Risk Assessment': 'Risk-reward profile indicates balanced potential with appropriate position management strategies.',
            'Entry Strategy': 'Entry timing should be based on technical confirmation signals and optimal market conditions.',
            'Market Outlook': 'Market context and sector trends support the overall investment outlook and strategic positioning.'
        }
        
        # Try to match section title with defaults
        for key, value in defaults.items():
            if key.lower() in section_title.lower():
                return value
        
        # Generic default
        return f"{section_title} analysis based on current market data and technical indicators."

    def _inject_section_structure(self, content: str) -> str:
        """Force section structure when LLM fails to provide it."""
        try:
            import re
            
            # Split content into sentences for distribution
            sentences = re.split(r'(?<=\.)\s+', content.strip())
            total_sentences = len(sentences)
            
            # If content is very short, create minimal but proper structure
            if total_sentences < 3:
                return f"""**Investment Summary:**
{content}

**Technical Analysis:**
Technical indicators support this position based on current market conditions and price action analysis.

**Risk Assessment:**
Risk management strategies should be employed with appropriate position sizing and stop-loss levels.

**Entry Strategy:**
Consider entry points based on technical confirmation signals and market momentum.

**Market Outlook:**
Monitor broader market trends and sector performance for optimal timing."""
            
            # For short content, distribute minimally but ensure each section has content
            if total_sentences < 8:
                # Minimal distribution - ensure each section gets at least something
                investment_summary = sentences[0] if total_sentences > 0 else "Investment analysis based on technical indicators."
                
                tech_content = ' '.join(sentences[1:min(3, total_sentences)]) if total_sentences > 1 else "Technical analysis shows key indicators support the current outlook."
                
                risk_content = ' '.join(sentences[min(3, total_sentences):min(5, total_sentences)]) if total_sentences > 3 else "Risk assessment indicates balanced risk-reward potential with proper position management."
                
                entry_content = ' '.join(sentences[min(5, total_sentences):min(7, total_sentences)]) if total_sentences > 5 else "Entry strategy should focus on confirmation signals and appropriate timing."
                
                market_content = ' '.join(sentences[min(7, total_sentences):]) if total_sentences > 7 else "Market outlook supports the technical analysis and investment thesis."
                
                return f"""**Investment Summary:**
{investment_summary}

**Technical Analysis:**
{tech_content}

**Risk Assessment:**
{risk_content}

**Entry Strategy:**
{entry_content}

**Market Outlook:**
{market_content}"""
            
            # For longer content, use intelligent distribution
            # Investment Summary (first 15% or 2-3 sentences)
            exec_end = max(2, min(3, int(total_sentences * 0.15)))
            investment_summary = ' '.join(sentences[:exec_end])
            
            # Technical Analysis (next 40% of content)
            tech_start = exec_end
            tech_end = min(total_sentences - 3, tech_start + int(total_sentences * 0.4))
            technical_analysis = ' '.join(sentences[tech_start:tech_end])
            
            # Risk Assessment (next 20% of content)  
            risk_start = tech_end
            risk_end = min(total_sentences - 2, risk_start + int(total_sentences * 0.2))
            risk_assessment = ' '.join(sentences[risk_start:risk_end]) if risk_start < total_sentences else "Risk assessment based on current technical setup."
            
            # Entry Strategy (next 15% of content)
            entry_start = risk_end
            entry_end = min(total_sentences - 1, entry_start + int(total_sentences * 0.15))
            entry_strategy = ' '.join(sentences[entry_start:entry_end]) if entry_start < total_sentences else "Entry timing based on technical confirmation signals."
            
            # Market Outlook (remaining content)
            context_start = entry_end
            market_outlook = ' '.join(sentences[context_start:]) if context_start < total_sentences else "Market conditions support the overall investment thesis."
            
            # Construct properly formatted output with enhanced sections
            formatted_output = f"""**Investment Summary:**
{investment_summary}

**Technical Analysis:**
{technical_analysis}

**Risk Assessment:**
{risk_assessment}

**Entry Strategy:**
{entry_strategy}

**Market Outlook:**
{market_outlook}"""
            
            logger.info(f"Successfully injected section structure into content with {total_sentences} sentences")
            return formatted_output
            
        except Exception as e:
            logger.error(f"Error injecting section structure: {e}")
            # Last resort - basic but complete structure
            return f"""**Investment Summary:**
{content[:150]}...

**Technical Analysis:**
Technical indicators and price action analysis support the current assessment based on market data.

**Risk Assessment:**
Risk-reward profile should be evaluated with appropriate position sizing and risk management strategies.

**Entry Strategy:**
Entry points should be based on technical confirmation and optimal market timing conditions.

**Market Outlook:**
Broader market context and sector trends support the overall investment outlook and strategy."""

    def _correct_malformed_headers(self, content: str) -> str:
        """Correct malformed section headers to ensure proper formatting."""
        try:
            import re
            
            # Fix the most common malformed header issues
            # 1. **Title:****content -> **Title:** content
            content = re.sub(r'\*\*([^*]+?):\*\*+\s*([^*])', r'**\1:** \2', content)
            
            # 2. **Title:**content -> **Title:** content (missing space)
            content = re.sub(r'\*\*([^*]+?):\*\*([A-Za-z])', r'**\1:** \2', content)
            
            # 3. **Title:** ***content*** -> **Title:** content (extra asterisks in content)
            content = re.sub(r'(\*\*[^*]+:\*\*\s*)\*+([^*]+?)\*+', r'\1\2', content)
            
            # 4. ***Title:*** content -> **Title:** content (wrong asterisk count)
            content = re.sub(r'\*{3,}([^*]+?):\*{3,}', r'**\1:**', content)
            
            # Fix common header issues
            corrections = [
                # Executive Summary variations
                (r'\*\*\s*EXECUTIVE\s+SUMMARY\s*\*\*', '**Investment Summary:**'),
                (r'\*\*\s*Executive\s+Summary\s*\*\*', '**Investment Summary:**'),
                (r'\*\*\s*executive\s+summary\s*\*\*', '**Investment Summary:**'),
                
                # Technical Analysis variations
                (r'\*\*\s*TECHNICAL\s+ANALYSIS\s*\*\*', '**Technical Analysis:**'),
                (r'\*\*\s*Technical\s+Analysis\s*\*\*', '**Technical Analysis:**'),
                (r'\*\*\s*technical\s+analysis\s*\*\*', '**Technical Analysis:**'),
                
                # Risk/Reward variations
                (r'\*\*\s*RISK[/&\s]*REWARD\s+ASSESSMENT\s*\*\*', '**Risk Assessment:**'),
                (r'\*\*\s*Risk[/&\s]*Reward\s+Assessment\s*\*\*', '**Risk Assessment:**'),
                (r'\*\*\s*RISK\s*[&/]\s*OUTLOOK\s*\*\*', '**Risk Assessment:**'),
                
                # Entry/Exit variations
                (r'\*\*\s*ENTRY[/\s]*EXIT\s+STRATEGY\s*\*\*', '**Entry Strategy:**'),
                (r'\*\*\s*Entry[/\s]*Exit\s+Strategy\s*\*\*', '**Entry Strategy:**'),
                
                # Market Context variations
                (r'\*\*\s*MARKET\s+CONTEXT\s*[&\s]*CATALYSTS\s*\*\*', '**Market Outlook:**'),
                (r'\*\*\s*Market\s+Context\s*[&\s]*Catalysts\s*\*\*', '**Market Outlook:**'),
                
                # Add missing colons to any remaining headers
                (r'\*\*\s*([A-Z][^*]+?)\s*\*\*(?!\s*:)', r'**\1:**'),
            ]
            
            corrected_content = content
            corrections_made = 0
            
            for pattern, replacement in corrections:
                new_content = re.sub(pattern, replacement, corrected_content, flags=re.IGNORECASE)
                if new_content != corrected_content:
                    corrections_made += 1
                corrected_content = new_content
            
            if corrections_made > 0:
                logger.info(f"Applied {corrections_made} header corrections to detailed content")
            
            return corrected_content
            
        except Exception as e:
            logger.warning(f"Error correcting headers: {e}")
            return content

    def _calculate_confidence_score(self, content: str) -> float:
        """Calculate confidence score based on content quality (optimized)."""
        try:
            word_count = len(content.split())

            # Start with base confidence
            confidence = 0.6  # Increased base confidence

            # Length scoring (adjusted for shorter responses)
            if 10 <= word_count <= 300:  # Broader acceptable range
                confidence += 0.25

            # Check for key financial terms
            financial_terms = ["buy", "sell", "hold", "score", "trend", "indicator", "risk"]
            term_count = sum(1 for term in financial_terms if term.lower() in content.lower())
            confidence += min(0.15, term_count * 0.05)

            return min(1.0, confidence)

        except Exception:
            return 0.6  # Higher default confidence

    def _validate_explanation_quality(self, content: str, detail_level: str, expected_recommendation: str) -> Dict[str, Any]:
        """Enhanced quality validation for explanation content."""
        validation_result = {
            "is_valid": True,
            "quality_score": 0.0,
            "issues": [],
            "word_count": 0,
            "has_recommendation": False,
            "recommendation_matches": False,
        }
        
        try:
            words = content.split()
            validation_result["word_count"] = len(words)
            
            # Word count validation by detail level
            word_count_targets = {
                "summary": (45, 75),      # 50-70 words target
                "standard": (160, 200),   # 180-200 words target  
                "detailed": (420, 520),   # 450-500 words target
            }
            
            min_words, max_words = word_count_targets.get(detail_level, (50, 500))
            
            # Check word count
            if len(words) < min_words:
                validation_result["issues"].append(f"Content too short: {len(words)} words (minimum: {min_words})")
                validation_result["quality_score"] -= 0.3
            elif len(words) > max_words * 1.2:  # Allow 20% buffer
                validation_result["issues"].append(f"Content too long: {len(words)} words (maximum: {max_words})")
                validation_result["quality_score"] -= 0.2
            else:
                validation_result["quality_score"] += 0.3  # Good word count
                
            # Check for recommendation presence
            content_lower = content.lower()
            recommendations = ["buy", "sell", "hold"]
            found_recommendations = [rec for rec in recommendations if rec in content_lower]
            
            if found_recommendations:
                validation_result["has_recommendation"] = True
                validation_result["quality_score"] += 0.2
                
                # Check if recommendation matches expected
                if expected_recommendation.lower() in found_recommendations:
                    validation_result["recommendation_matches"] = True
                    validation_result["quality_score"] += 0.2
                else:
                    validation_result["issues"].append(f"Recommendation mismatch: expected {expected_recommendation}, found {found_recommendations}")
                    validation_result["quality_score"] -= 0.3
            else:
                validation_result["issues"].append("No clear recommendation found (BUY/SELL/HOLD)")
                validation_result["quality_score"] -= 0.3
                
            # Check for structured content (for standard and detailed modes)
            if detail_level in ["standard", "detailed"]:
                has_structure = "**" in content and ":" in content
                if has_structure:
                    validation_result["quality_score"] += 0.1
                else:
                    validation_result["issues"].append("Missing structured formatting (headers with **)")
                    validation_result["quality_score"] -= 0.1
                    
            # Check for key financial concepts
            financial_concepts = [
                "technical", "indicator", "rsi", "macd", "moving average", "volume",
                "trend", "support", "resistance", "momentum", "volatility", "risk"
            ]
            concept_count = sum(1 for concept in financial_concepts if concept in content_lower)
            
            if concept_count >= 2:
                validation_result["quality_score"] += 0.2
            elif concept_count == 1:
                validation_result["quality_score"] += 0.1
            else:
                validation_result["issues"].append("Lacks sufficient technical analysis terminology")
                validation_result["quality_score"] -= 0.1
                
            # Final quality score normalization
            validation_result["quality_score"] = max(0.0, min(1.0, validation_result["quality_score"] + 0.5))
            
            # Overall validation
            if validation_result["quality_score"] < 0.6 or len(validation_result["issues"]) > 2:
                validation_result["is_valid"] = False
                
        except Exception as e:
            logger.error(f"Error in quality validation: {e}")
            validation_result["is_valid"] = False
            validation_result["issues"].append(f"Validation error: {str(e)}")
            
        return validation_result

    def _retry_with_quality_check(
        self, 
        prompt: str, 
        detail_level: str, 
        model_name: str, 
        expected_recommendation: str,
        max_retries: int = 2
    ) -> Tuple[str, float, Dict[str, Any]]:
        """Retry generation with quality validation until acceptable result."""
        best_content = None
        best_quality = 0.0
        best_validation = None
        
        for attempt in range(max_retries + 1):
            try:
                # Generate content
                options = self._get_optimized_generation_options(detail_level, model_name)
                
                logger.info(f"[QUALITY RETRY] Attempt {attempt + 1}/{max_retries + 1} for {detail_level} mode")
                
                response = self.client.generate(
                    model=model_name,
                    prompt=prompt,
                    options=options
                )
                
                if not response or 'response' not in response:
                    continue
                    
                content = response['response'].strip()
                if not content:
                    continue
                    
                # Validate quality
                validation = self._validate_explanation_quality(content, detail_level, expected_recommendation)
                
                logger.info(f"[QUALITY CHECK] Attempt {attempt + 1}: Quality Score {validation['quality_score']:.2f}, Valid: {validation['is_valid']}")
                
                # If this is better than our best attempt, save it
                if validation['quality_score'] > best_quality:
                    best_content = content
                    best_quality = validation['quality_score']
                    best_validation = validation
                
                # If we got a good enough result, use it
                if validation['is_valid'] and validation['quality_score'] >= 0.7:
                    logger.info(f"[QUALITY SUCCESS] Achieved quality score {validation['quality_score']:.2f} on attempt {attempt + 1}")
                    break
                    
            except Exception as e:
                logger.error(f"[QUALITY RETRY] Attempt {attempt + 1} failed: {e}")
                continue
                
        # Return best result found
        if best_content:
            confidence = self._calculate_confidence_score(best_content)
            final_confidence = (confidence + best_quality) / 2  # Blend confidence scores
            
            logger.info(f"[QUALITY FINAL] Selected content with quality score {best_quality:.2f}, confidence {final_confidence:.2f}")
            return best_content, final_confidence, best_validation
        else:
            logger.warning(f"[QUALITY FAILURE] All retry attempts failed for {detail_level} mode")
            return "Error: Unable to generate quality explanation", 0.1, {"is_valid": False, "issues": ["All generation attempts failed"]}

    def _create_cache_key(self, analysis_data: Dict[str, Any], detail_level: str, explanation_type: str) -> str:
        """Create a cache key for explanation results."""
        # Use symbol, score, and key indicator values for cache key
        symbol = analysis_data.get("symbol", "UNKNOWN")
        score = analysis_data.get("score_0_10", 0)

        # Create a detailed hash of the analysis data
        data_str = f"{symbol}_{score:.2f}_{detail_level}_{explanation_type}"

        # Add all weighted scores for more specific caching
        indicators = analysis_data.get("weighted_scores", {})
        if indicators:
            # Sort indicators for consistent cache keys
            sorted_indicators = sorted(indicators.items())
            for key, value in sorted_indicators:
                # Include value with precision for uniqueness
                data_str += f"_{key}_{value:.4f}"

        # Add complexity score for additional uniqueness
        complexity = self._calculate_complexity_score(analysis_data)
        data_str += f"_complexity_{complexity:.3f}"

        return f"llm_explanation_{hashlib.blake2b(data_str.encode(), digest_size=16).hexdigest()}"

    def _create_sentiment_enhanced_cache_key(
        self,
        analysis_data: Dict[str, Any],
        sentiment_data: Optional[Dict[str, Any]],
        detail_level: str,
        explanation_type: str,
    ) -> str:
        """
        Create cache key for sentiment-enhanced explanations.

        Args:
            analysis_data: Technical analysis data
            sentiment_data: Sentiment analysis data
            detail_level: Level of detail
            explanation_type: Type of explanation

        Returns:
            Enhanced cache key string
        """
        # Start with base cache key
        base_key_data = self._create_cache_key(analysis_data, detail_level, explanation_type)

        # Add sentiment context
        sentiment_suffix = ""
        if sentiment_data:
            sentiment_score = sentiment_data.get("sentimentScore", 0)
            sentiment_confidence = sentiment_data.get("sentimentConfidence", 0)
            sentiment_label = sentiment_data.get("sentimentLabel", "neutral")

            # Create sentiment hash for cache differentiation using BLAKE2b
            sentiment_context = f"{sentiment_label}_{sentiment_score:.3f}_{sentiment_confidence:.3f}"
            sentiment_hash = hashlib.blake2b(sentiment_context.encode(), digest_size=16).hexdigest()
            sentiment_suffix = f"_sent_{sentiment_hash}"

        return f"{self.sentiment_cache_prefix}{base_key_data}{sentiment_suffix}"

    def _get_sentiment_aware_ttl(self, analysis_data: Dict[str, Any], sentiment_data: Optional[Dict[str, Any]]) -> int:
        """
        Calculate cache TTL considering both technical and sentiment factors.

        Args:
            analysis_data: Technical analysis data
            sentiment_data: Sentiment analysis data

        Returns:
            Cache TTL in seconds
        """
        # Start with base TTL from technical analysis
        base_ttl = self._get_dynamic_ttl(analysis_data)

        # Adjust based on sentiment characteristics
        if sentiment_data:
            sentiment_confidence = sentiment_data.get("sentimentConfidence", 0)
            sentiment_strength = abs(sentiment_data.get("sentimentScore", 0))

            # High confidence sentiment can be cached longer
            if sentiment_confidence >= 0.8:
                # Very confident sentiment - extend cache
                base_ttl = int(base_ttl * 1.5)
            elif sentiment_confidence < 0.5:
                # Low confidence sentiment - shorter cache
                base_ttl = int(base_ttl * 0.7)

            # Strong sentiment signals change more rapidly
            if sentiment_strength > 0.6:
                # Strong sentiment - shorter cache for volatility
                base_ttl = int(base_ttl * 0.8)

        # Ensure reasonable bounds
        return max(60, min(base_ttl, 1800))  # Between 1 minute and 30 minutes

    def generate_batch_explanations(
        self, analysis_results: List[Dict[str, Any]], detail_level: str = "standard"
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Generate explanations for multiple analysis results.

        Args:
            analysis_results: List of analysis data dictionaries
            detail_level: Detail level for all explanations

        Returns:
            List of explanation results (same order as input)
        """
        explanations = []

        for analysis_data in analysis_results:
            explanation = self.generate_explanation(analysis_data, detail_level)
            explanations.append(explanation)

        return explanations

    def _build_prompt(self, analysis_data: Dict[str, Any], detail_level: str = "summary", 
                     explanation_type: str = "comprehensive") -> str:
        """
        Compatibility shim for legacy test methods.
        
        This method maintains backward compatibility with tests that expect the old
        _build_prompt method interface. It delegates to the new _build_optimized_prompt.
        
        Args:
            analysis_data: Stock analysis data dictionary
            detail_level: Level of detail for explanation
            explanation_type: Type of explanation to generate
            
        Returns:
            Generated prompt string
        """
        return self._build_optimized_prompt(analysis_data, detail_level, explanation_type)

    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status and multi-model information."""
        try:
            if not self.client:
                return {"available": False, "error": "Client not initialised"}

            models = self.client.list()
            available_models = [model["name"] for model in models.get("models", [])]

            # Check availability of all configured models
            summary_available = self._verify_model_availability(self.summary_model)
            standard_available = self._verify_model_availability(self.standard_model)
            detailed_available = self._verify_model_availability(self.detailed_model)
            translation_available = self._verify_model_availability(self.translation_model)
            
            # Legacy model availability
            primary_available = self._verify_model_availability(self.primary_model)

            return {
                "available": any([summary_available, standard_available, detailed_available]),
                "multi_model_configuration": {
                    "summary_model": self.summary_model,
                    "standard_model": self.standard_model,
                    "detailed_model": self.detailed_model,
                    "translation_model": self.translation_model,
                    "summary_available": summary_available,
                    "standard_available": standard_available,
                    "detailed_available": detailed_available,
                    "translation_available": translation_available,
                },
                "legacy_configuration": {
                    "primary_model": self.primary_model,
                    "detailed_model": self.detailed_model,
                    "primary_model_available": primary_available,
                    "detailed_model_available": detailed_available,
                },
                "current_model": self.current_model,
                "available_models": available_models,
                "models_count": len(available_models),
                "client_initialised": True,
                "cache_enabled": hasattr(cache, "get"),
                "performance_mode": self.performance_mode,
                "generation_timeout": self.generation_timeout,
                "circuit_breaker_state": self.circuit_breaker.state,
                "health_check_interval": self.model_health_service.check_interval,
                "model_health_status": self.model_health_service.model_health.copy(),
                "performance_metrics": self.performance_monitor.get_performance_summary(),
            }

        except Exception as e:
            return {"available": False, "error": str(e), "client_initialised": self.client is not None}

    def warm_up_models(self) -> Dict[str, Any]:
        """
        Warm up all configured LLM models with test prompts.
        
        Returns:
            Dictionary with warm-up results and performance metrics.
        """
        if not self.client:
            return {
                "success": False, 
                "error": "Ollama client not available", 
                "models_tested": []
            }
        
        import time
        
        # Models to test
        models_to_warm = [
            self.summary_model,
            self.standard_model, 
            self.detailed_model
        ]
        
        # Remove duplicates while preserving order
        unique_models = []
        seen = set()
        for model in models_to_warm:
            if model not in seen:
                unique_models.append(model)
                seen.add(model)
        
        warm_up_results = []
        start_time = time.time()
        
        for model in unique_models:
            try:
                model_start = time.time()
                
                # Simple test prompt for warm-up
                test_prompt = "What is technical analysis in finance?"
                options = {
                    "temperature": 0.1,
                    "max_tokens": 50,
                    "top_p": 0.9
                }
                
                # Test the model with a simple prompt
                result = self._generate_with_timeout(
                    model=model,
                    prompt=test_prompt,
                    options=options,
                    timeout=30  # 30 second timeout for warm-up
                )
                
                model_time = time.time() - model_start
                
                if result and result.get('response'):
                    warm_up_results.append({
                        "model": model,
                        "success": True,
                        "response_time": round(model_time, 2),
                        "response_length": len(result.get('response', ''))
                    })
                    logger.info(f"Successfully warmed up model {model} in {model_time:.2f}s")
                else:
                    warm_up_results.append({
                        "model": model,
                        "success": False,
                        "response_time": round(model_time, 2),
                        "error": "No valid response received"
                    })
                    logger.warning(f"Failed to warm up model {model}")
                    
            except Exception as e:
                model_time = time.time() - model_start
                warm_up_results.append({
                    "model": model,
                    "success": False,
                    "response_time": round(model_time, 2),
                    "error": str(e)
                })
                logger.error(f"Error warming up model {model}: {str(e)}")
        
        total_time = time.time() - start_time
        successful_models = [r for r in warm_up_results if r["success"]]
        
        return {
            "success": len(successful_models) > 0,
            "total_time": round(total_time, 2),
            "models_tested": len(unique_models),
            "models_successful": len(successful_models),
            "results": warm_up_results,
            "timestamp": time.time()
        }

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a specific model is currently loaded and available."""
        try:
            if not OLLAMA_AVAILABLE or not self.client:
                return False

            # Check if model is in the list of available models
            models_response = self.client.list()
            available_models = [m["name"] for m in models_response.get("models", [])]

            return model_name in available_models

        except Exception as e:
            logger.error(f"Failed to check if model {model_name} is loaded: {str(e)}")
            return False

    def load_model(self, model_name: str) -> bool:
        """Load a specific model if not already loaded."""
        try:
            if not OLLAMA_AVAILABLE or not self.client:
                logger.warning("Ollama not available for model loading")
                return False

            # Check if model is already loaded
            if self.is_model_loaded(model_name):
                logger.debug(f"Model {model_name} already loaded")
                return True

            # Attempt to pull/load the model
            logger.info(f"Loading model: {model_name}")
            self.client.pull(model_name)

            # Verify the model was loaded
            return self.is_model_loaded(model_name)

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            return False

    def set_model(self, model_name: str) -> bool:
        """Set the current active model."""
        try:
            # Validate model availability
            if not self.is_model_loaded(model_name):
                logger.warning(f"Model {model_name} not available, attempting to load")
                if not self.load_model(model_name):
                    logger.error(f"Failed to load model {model_name}")
                    return False

            # Set as current model
            old_model = self.current_model
            self.current_model = model_name
            logger.info(f"Switched from {old_model} to {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to set model {model_name}: {str(e)}")
            return False

    def get_available_models(self) -> List[str]:
        """Get list of all available models."""
        try:
            if not OLLAMA_AVAILABLE or not self.client:
                return []

            models_response = self.client.list()
            return [m["name"] for m in models_response.get("models", [])]

        except Exception as e:
            logger.error(f"Failed to get available models: {str(e)}")
            return []

    # Multilingual Support Methods

    def generate_multilingual_explanation(
        self,
        analysis_data: Dict[str, Any],
        target_language: str = "en",
        detail_level: str = "standard",
        explanation_type: str = "technical_analysis",
    ) -> Optional[Dict[str, Any]]:
        """
        Generate explanations in the specified language.

        Args:
            analysis_data: Dictionary containing analysis results
            target_language: Target language code (en, fr, es)
            detail_level: 'summary', 'standard', or 'detailed'
            explanation_type: Type of explanation to generate

        Returns:
            Dictionary with multilingual explanation content or None if failed
        """
        if not self.multilingual_enabled:
            logger.warning("Multilingual support is disabled")
            return self.generate_explanation(analysis_data, detail_level, explanation_type)

        if target_language not in self.supported_languages:
            logger.warning(f"Unsupported language '{target_language}', falling back to {self.default_language}")
            target_language = self.default_language

        # Create multilingual cache key
        cache_key = self._create_multilingual_cache_key(
            analysis_data, target_language, detail_level, explanation_type
        )

        # Check cache first
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"Retrieved multilingual explanation from cache for language: {target_language}")
            return cached_result

        try:
            # Get language-specific model
            model_name = self.language_models.get(target_language, self.language_models[self.default_language])

            # Generate native language explanation
            if target_language == "en":
                result = self._generate_native_explanation(
                    analysis_data, detail_level, explanation_type, model_name, target_language
                )
            else:
                result = self._generate_translated_explanation(
                    analysis_data, detail_level, explanation_type, model_name, target_language
                )

            if result:
                # Apply cultural formatting
                if self.cultural_formatting_enabled:
                    result = self._apply_cultural_formatting(result, target_language)

                # Cache the result
                cache.set(cache_key, result, self.translation_cache_ttl)

            return result

        except Exception as e:
            logger.error(f"Error generating multilingual explanation for {target_language}: {str(e)}")
            return None

    def _generate_native_explanation(
        self,
        analysis_data: Dict[str, Any],
        detail_level: str,
        explanation_type: str,
        model_name: str,
        language: str,
    ) -> Optional[Dict[str, Any]]:
        """Generate explanation directly in the target language."""
        try:
            # Build language-specific prompt
            prompt = self._build_multilingual_prompt(
                analysis_data, detail_level, explanation_type, language
            )

            # Generate with language-specific model
            generation_options = self._get_language_specific_options(language, detail_level)

            result = self._generate_with_timeout(
                model=model_name,
                prompt=prompt,
                options=generation_options,
                timeout=self._get_timeout_for_detail_level(detail_level),
            )

            if result and result.get("response"):
                return {
                    "explanation": result["response"],
                    "language": language,
                    "model_used": model_name,
                    "detail_level": detail_level,
                    "explanation_type": explanation_type,
                    "generation_method": "native",
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"Error in native explanation generation for {language}: {str(e)}")

        return None

    def _generate_translated_explanation(
        self,
        analysis_data: Dict[str, Any],
        detail_level: str,
        explanation_type: str,
        model_name: str,
        target_language: str,
    ) -> Optional[Dict[str, Any]]:
        """Generate explanation and translate to target language."""
        try:
            # First generate in English
            english_result = self.generate_explanation(analysis_data, detail_level, explanation_type)

            if not english_result or not english_result.get("explanation"):
                return None

            # Translate to target language
            translated_explanation = self._translate_text(
                english_result["explanation"], target_language, model_name
            )

            if translated_explanation:
                return {
                    "explanation": translated_explanation,
                    "language": target_language,
                    "model_used": model_name,
                    "detail_level": detail_level,
                    "explanation_type": explanation_type,
                    "generation_method": "translated",
                    "original_language": "en",
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"Error in translated explanation generation for {target_language}: {str(e)}")

        return None

    def _translate_text(self, text: str, target_language: str, model_name: str) -> Optional[str]:
        """Translate text to target language using financial terminology mapping."""
        if not self.translation_enabled:
            return text

        try:
            # Build translation prompt with financial terminology context
            financial_terms = self.financial_terminology.get(target_language, {})

            prompt = f"""Translate the following financial analysis text to {target_language}.
            Use appropriate financial terminology and maintain professional tone.

            Financial term mappings for {target_language}:
            {chr(10).join([f"- {en}: {foreign}" for en, foreign in financial_terms.items()])}

            Text to translate:
            {text}

            Translation:"""

            result = self._generate_with_timeout(
                model=model_name,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "max_tokens": len(text) * 2,  # Allow for expansion
                    "top_p": 0.9,
                },
                timeout=self.translation_timeout,
            )

            if result and result.get("response"):
                return result["response"].strip()

        except Exception as e:
            logger.error(f"Error translating text to {target_language}: {str(e)}")

        return None

    def _build_multilingual_prompt(
        self,
        analysis_data: Dict[str, Any],
        detail_level: str,
        explanation_type: str,
        language: str,
    ) -> str:
        """Build language-specific prompt for native generation."""

        # Base analysis information
        symbol = analysis_data.get("symbol", "N/A")
        price = analysis_data.get("currentPrice", 0)

        # Language-specific prompt templates
        if language == "fr":
            return f"""Analysez les données financières suivantes pour {symbol} et fournissez une explication détaillée en français:

Prix actuel: {price}
Données d'analyse: {analysis_data}

Niveau de détail requis: {detail_level}
Type d'explication: {explanation_type}

Fournissez une analyse professionnelle en français utilisant la terminologie financière appropriée."""

        elif language == "es":
            return f"""Analice los siguientes datos financieros para {symbol} y proporcione una explicación detallada en español:

Precio actual: {price}
Datos de análisis: {analysis_data}

Nivel de detalle requerido: {detail_level}
Tipo de explicación: {explanation_type}

Proporcione un análisis profesional en español utilizando la terminología financiera apropiada."""

        else:  # Default to English
            return self._build_optimized_prompt(analysis_data, detail_level, explanation_type)

    def _get_language_specific_options(self, language: str, detail_level: str) -> Dict[str, Any]:
        """Get generation options optimized for specific language."""
        base_options = {
            "temperature": 0.4,
            "top_p": 0.9,
            "max_tokens": self._get_max_tokens_for_detail(detail_level),
        }

        # Language-specific adjustments
        if language in ["fr", "es"]:
            # Romance languages may need slightly more tokens for equivalent meaning
            base_options["max_tokens"] = int(base_options["max_tokens"] * 1.2)
            # Slightly lower temperature for more consistent terminology
            base_options["temperature"] = 0.3

        return base_options

    def _apply_cultural_formatting(self, result: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Apply cultural formatting preferences to the result."""
        if not self.cultural_formatting_enabled or language not in self.financial_formatting:
            return result

        try:
            formatting_config = self.financial_formatting[language]
            explanation = result.get("explanation", "")

            # Apply number formatting (basic pattern replacement)
            # This is a simple implementation - production would use more sophisticated formatting
            if language == "fr":
                # Replace comma thousands separators with spaces and dots with commas for decimals
                import re
                explanation = re.sub(r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                                   lambda m: f"{m.group(1).replace(',', ' ').replace('.', ',')} €",
                                   explanation)
            elif language == "es":
                # Similar for Spanish formatting
                import re
                explanation = re.sub(r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                                   lambda m: f"{m.group(1).replace(',', '.')},00 €",
                                   explanation)

            result["explanation"] = explanation
            result["cultural_formatting_applied"] = True

        except Exception as e:
            logger.error(f"Error applying cultural formatting for {language}: {str(e)}")

        return result

    def _create_multilingual_cache_key(
        self,
        analysis_data: Dict[str, Any],
        language: str,
        detail_level: str,
        explanation_type: str,
    ) -> str:
        """Create cache key for multilingual explanations."""
        base_key = self._create_cache_key(analysis_data, detail_level, explanation_type)
        return f"multilingual_{language}_{base_key}"

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.supported_languages

    def is_language_supported(self, language: str) -> bool:
        """Check if a language is supported."""
        return language in self.supported_languages

    def get_language_model(self, language: str) -> str:
        """Get the model assigned to a specific language."""
        return self.language_models.get(language, self.language_models[self.default_language])

    def _get_max_tokens_for_detail(self, detail_level: str) -> int:
        """Get maximum tokens based on detail level."""
        token_limits = {
            "summary": 150,
            "standard": 300,
            "detailed": 600,
        }
        return token_limits.get(detail_level, 300)


# Singleton instance
_llm_service = None


def get_local_llm_service() -> LocalLLMService:
    """Get singleton instance of LocalLLMService."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LocalLLMService()
    return _llm_service

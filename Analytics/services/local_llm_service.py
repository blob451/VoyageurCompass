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
from typing import Any, Dict, List, Optional

from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)

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

        # Build base technical context
        base_context = self._build_technical_context(analysis_data, detail_level)

        # Add sentiment context if available
        sentiment_context = self._build_sentiment_context(sentiment_data) if sentiment_data else ""

        # Build instruction based on detail level and sentiment confidence
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

            # Add top 4 indicators with values
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

        # Add consistency rules header
        consistency_rules = f"""
IMPORTANT: Based on the {score}/10 score, your recommendation MUST be {expected_rec}.
- Scores 7-10 = BUY
- Scores 4-6.9 = HOLD
- Scores 0-3.9 = SELL"""

        # Build instruction based on detail level with word count targets
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
            "num_predict": 180,  # Updated to match standard token count
            "stop": ["###", "END", "\n\n\n"],
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
        """Get base token count for detail level with proper word-to-token ratios."""
        token_counts = {
            "summary": 90,  # ~60 words * 1.5 tokens/word
            "standard": 180,  # ~120 words * 1.5 tokens/word
            "detailed": 450,  # ~300 words * 1.5 tokens/word
        }
        return token_counts.get(detail_level, 180)

    def _apply_model_specific_adjustments(self, options: Dict[str, Any], model_name: str):
        """Apply model-specific parameter adjustments."""
        if "8b" in model_name.lower():
            # 8B model adjustments
            options["num_ctx"] = 512
            if options["temperature"] < 0.3:
                options["temperature"] = 0.3  # Minimum temperature for 8B
        else:
            # 70B model adjustments
            options["num_ctx"] = 1024
            if options["temperature"] > 0.3:
                options["temperature"] *= 0.8  # Reduce temperature for 70B


class LocalLLMService:
    """Enhanced LLM Service with performance optimisation and monitoring."""

    def __init__(self):
        # Model hierarchy for performance optimisation - read from settings
        self.primary_model = getattr(settings, "OLLAMA_PRIMARY_MODEL", "llama3.1:8b")
        self.detailed_model = getattr(settings, "OLLAMA_DETAILED_MODEL", "llama3.1:70b")
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

        # Add circuit breaker and monitoring
        self.circuit_breaker = LLMCircuitBreaker()
        self.performance_monitor = LLMPerformanceMonitor()

        # Thread executor for timeout handling with optimized resource allocation
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="llm-timeout")

        # Resource cleanup tracking
        self._active_requests = 0
        self._max_concurrent_requests = 5

        # Add sentiment-enhanced components
        self.sentiment_prompt_builder = SentimentEnhancedPromptBuilder()
        self.confidence_adaptive_generator = ConfidenceAdaptiveGeneration()

        # Sentiment integration configuration
        self.sentiment_integration_enabled = True
        self.sentiment_cache_prefix = "sentiment_enhanced:"

        # Don't initialize client in __init__ - use lazy property instead

    @property
    def client(self):
        """Lazy-loaded Ollama client with thread safety and availability checks."""
        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama client requested but not available")
            return None
        if self._client is None:
            with self._client_lock:
                if self._client is None:  # Double-check locking
                    self._client = self._initialize_client_with_retry()
        return self._client

    @client.setter
    def client(self, value):
        """Set the client (mainly for testing purposes)."""
        self._client = value

    def _initialize_client_with_retry(self):
        """Initialize Ollama client connection with retry logic."""
        for attempt in range(self.connection_retry_attempts):
            try:
                ollama_host = getattr(settings, "OLLAMA_HOST", "localhost")
                ollama_port = getattr(settings, "OLLAMA_PORT", 11434)

                client = Client(host=f"http://{ollama_host}:{ollama_port}")

                # Test connection by listing models
                try:
                    models = client.list()
                    logger.info(f"Local LLM client initialized successfully on attempt {attempt + 1}")

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
                logger.error(f"Failed to initialize Ollama client on attempt {attempt + 1}: {str(e)}")
                if attempt < self.connection_retry_attempts - 1:
                    time.sleep(self.connection_retry_delay * (2**attempt))  # Exponential backoff
                else:
                    logger.error(f"Failed to initialize Ollama client after {self.connection_retry_attempts} attempts")
                    return None

        return None

    def _verify_model_availability(self, model_name: str = None) -> bool:
        """Check if the specified model is available with caching."""
        target_model = model_name or self.current_model
        cache_key = f"model_availability_{target_model}"

        # Check cache first
        current_time = time.time()
        if cache_key in self._availability_cache:
            cached_result, cached_time = self._availability_cache[cache_key]
            if current_time - cached_time < self._availability_cache_timeout:
                return cached_result

        # Check availability
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
        # Check resource availability
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

    def _select_optimal_model(self, detail_level: str, complexity_score: float = 0.5) -> str:
        """Enhanced model selection with improved resource allocation and load balancing."""
        if not self.performance_mode:
            return self.detailed_model

        # Check model load and availability first
        primary_available = self._verify_model_availability(self.primary_model)
        detailed_available = self._verify_model_availability(self.detailed_model)

        # If only one model is available, use it
        if not primary_available and detailed_available:
            logger.info(f"[MODEL SELECTION] Primary model unavailable, using 70B for {detail_level}")
            return self.detailed_model
        elif primary_available and not detailed_available:
            logger.info(f"[MODEL SELECTION] 70B model unavailable, using 8B for {detail_level}")
            return self.primary_model
        elif not primary_available and not detailed_available:
            logger.warning(f"[MODEL SELECTION] No models available, attempting primary")
            return self.primary_model

        # Enhanced resource allocation logic when both models available
        if detail_level == "summary":
            # Summary always uses fast 8B model for better resource utilization
            logger.debug(f"[MODEL SELECTION] Using 8B for summary (resource optimisation)")
            return self.primary_model

        elif detail_level == "detailed":
            # Detailed explanations benefit from 70B model's superior capabilities
            # Use 70B for complex detailed analysis, 8B for simpler ones
            if complexity_score > 0.6:  # Lowered threshold for detailed mode
                logger.info(
                    f"[MODEL SELECTION] Using 70B for complex detailed analysis (complexity: {complexity_score:.3f})"
                )
                return self.detailed_model
            else:
                logger.info(
                    f"[MODEL SELECTION] Using 8B for simple detailed analysis (complexity: {complexity_score:.3f})"
                )
                return self.primary_model

        else:  # standard
            # Standard mode: balanced approach based on complexity
            use_70b_model = (
                # High complexity standard analysis
                (complexity_score > 0.75)  # Increased threshold
                or
                # Mixed signals requiring nuanced analysis
                self._has_conflicting_signals(complexity_score)
                or
                # Time-based load balancing: use 70B less during peak times
                self._should_use_premium_model()
            )

            if use_70b_model:
                logger.info(
                    f"[MODEL SELECTION] Using 70B for complex standard analysis (complexity: {complexity_score:.3f})"
                )
                return self.detailed_model
            else:
                logger.debug(f"[MODEL SELECTION] Using 8B for standard analysis (complexity: {complexity_score:.3f})")
                return self.primary_model

        # Fallback to primary model
        return self.primary_model

    def _should_use_premium_model(self) -> bool:
        """
        Determine if premium 70B model should be used based on current load and time.
        Implements intelligent load balancing to optimize resource utilization.
        """
        try:
            # Time-based load balancing: avoid peak usage times
            current_hour = datetime.now().hour

            # Define peak hours when we should prefer 8B model (9 AM - 5 PM)
            peak_hours = range(9, 17)
            is_peak_time = current_hour in peak_hours

            # Performance monitoring: check recent generation times
            recent_performance = self.performance_monitor.get_recent_performance()
            avg_generation_time = recent_performance.get("avg_generation_time", 0)

            # Use premium model if:
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

            # Check for extreme technical score that may need nuanced explanation
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
                raise Exception("Ollama client not initialized")

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

        # Check if any model is available
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

        # Create enhanced cache key that includes sentiment context
        cache_key = self._create_sentiment_enhanced_cache_key(
            analysis_data, sentiment_data, detail_level, explanation_type
        )
        dynamic_ttl = self._get_sentiment_aware_ttl(analysis_data, sentiment_data)

        # Check cache first
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"Retrieved sentiment-enhanced explanation from cache (TTL: {dynamic_ttl}s)")
            self.performance_monitor.record_generation(0, "cache", True, cache_hit=True)
            return cached_result

        try:
            # Calculate complexity and select optimal model
            complexity_score = self._calculate_complexity_score(analysis_data)
            selected_model = self._select_optimal_model(detail_level, complexity_score)

            # Build sentiment-enhanced prompt
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

            # Add timeout handling
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError(f"LLM generation exceeded {self.generation_timeout} seconds")

            # Set timeout (Unix systems only)
            timeout_set = False
            if hasattr(signal, "SIGALRM"):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.generation_timeout)
                timeout_set = True

            try:
                # Use circuit breaker pattern for reliability
                response = self.circuit_breaker.call_llm(
                    self.client.generate, model=selected_model, prompt=enhanced_prompt, options=generation_options
                )

                if timeout_set:
                    signal.alarm(0)  # Cancel timeout

            except TimeoutError as e:
                if timeout_set:
                    signal.alarm(0)
                timeout_duration = time.time() - start_time
                logger.warning(f"Sentiment-enhanced LLM generation timed out after {self.generation_timeout}s")
                self.performance_monitor.record_generation(timeout_duration, selected_model, False)
                return None

            except Exception as e:
                if timeout_set:
                    signal.alarm(0)
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

        # Create cache key with dynamic TTL
        cache_key = self._create_cache_key(analysis_data, detail_level, explanation_type)
        dynamic_ttl = self._get_dynamic_ttl(analysis_data)

        # Check cache first
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

            try:
                # Use timeout-aware generation with circuit breaker pattern
                timeout = self._get_timeout_for_detail_level(detail_level)
                response = self._generate_with_timeout(
                    model=selected_model,
                    prompt=prompt,
                    options=self._get_optimized_generation_options(detail_level, selected_model),
                    timeout=timeout,
                )

            except Exception as e:
                logger.error(f"Error during LLM generation: {str(e)}")
                self.performance_monitor.record_generation(time.time() - start_time, selected_model, False)
                return None

            generation_time = time.time() - start_time

            if not response or "response" not in response:
                logger.error("Invalid response from LLM")
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

            explanation_content = response["response"].strip()

            # Log content metrics
            word_count = len(explanation_content.split())
            logger.info(f"[LLM CONTENT] Generated {word_count} words ({len(explanation_content)} chars) for {symbol}")

            result = {
                "content": explanation_content,
                "detail_level": detail_level,
                "explanation_type": explanation_type,
                "generation_time": generation_time,
                "model_used": selected_model,
                "timestamp": time.time(),
                "word_count": len(explanation_content.split()),
                "confidence_score": self._calculate_confidence_score(explanation_content),
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
        symbol = analysis_data.get("symbol", "UNKNOWN")
        score = analysis_data.get("score_0_10", 0)

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

        # Enhanced prompts with consistency rules
        if detail_level == "summary":
            return f"""Stock Analysis: {symbol} receives {score}/10 investment score, indicating {score_desc} signals.

IMPORTANT: Based on the {score}/10 score, your recommendation MUST be {expected_rec}.
- Scores 7-10 = BUY
- Scores 4-6.9 = HOLD
- Scores 0-3.9 = SELL

Provide a clean, conversational {expected_rec} recommendation in 50-60 words. Use simple paragraph format without section headers or formatting. Be direct, human-friendly, and concise with complete sentences."""

        elif detail_level == "detailed":
            top_indicators = self._get_top_indicators(analysis_data, limit=2)
            return f"""Financial Analysis: {symbol} scores {score}/10 based on technical indicators, indicating {score_desc} signals.
Key factors: {top_indicators}

IMPORTANT: Based on the {score}/10 score, your recommendation MUST be {expected_rec}.
- Scores 7-10 = BUY
- Scores 4-6.9 = HOLD
- Scores 0-3.9 = SELL

Provide a comprehensive {expected_rec} analysis in 250-300 words using this structure:

**Investment Thesis:** Clear {expected_rec} recommendation with confidence level and core reasoning (60 words)

**Technical Indicators:** Detailed analysis of key indicators supporting the {expected_rec} decision with specific insights (80 words)

**Risk Analysis:** Main risks, challenges, and risk mitigation strategies (60 words)

**Market Context:** Price outlook, catalysts, and market environment factors (50 words)

Use professional investment research language. Ensure all sections support the {expected_rec} recommendation."""
        else:  # standard
            top_indicators = self._get_top_indicators(analysis_data, limit=2)
            return f"""Investment Analysis: {symbol} receives {score}/10 score from technical analysis, indicating {score_desc} signals.
Key factors: {top_indicators}

IMPORTANT: Based on the {score}/10 score, your recommendation MUST be {expected_rec}.
- Scores 7-10 = BUY
- Scores 4-6.9 = HOLD
- Scores 0-3.9 = SELL

Provide a professional {expected_rec} recommendation in 100-120 words using this structure:

**Investment Decision:** Clear {expected_rec} recommendation with confidence level

**Technical Analysis:** 2-3 key supporting technical factors

**Risk Assessment:** Primary risk consideration and brief market outlook

Use professional investment language with complete sentences."""

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

    def _get_max_tokens_legacy(self, detail_level: str) -> int:
        """Get maximum tokens based on detail level with adequate limits."""
        token_limits = {
            "summary": 90,  # ~60 words * 1.5 tokens/word
            "standard": 180,  # ~120 words * 1.5 tokens/word
            "detailed": 450,  # ~300 words * 1.5 tokens/word
        }
        return token_limits.get(detail_level, 180)

    def _get_optimized_generation_options(self, detail_level: str, model_name: str) -> dict:
        """Get highly optimized generation options for maximum performance."""
        base_options = {
            "temperature": 0.3,  # Reduced for consistency and speed
            "top_p": 0.7,  # Focused sampling for faster generation
            "num_predict": self._get_optimized_tokens(detail_level),
            "stop": ["###", "END", "\n\n\n"],  # Simplified stop tokens
            "repeat_penalty": 1.05,  # Reduced penalty
            "top_k": 20,  # Limit vocabulary for speed
        }

        # Optimize context window based on model (smaller contexts for speed)
        if "8b" in model_name.lower():
            base_options["num_ctx"] = 512  # Reduced from 1024
            base_options["temperature"] = 0.4
        else:
            base_options["num_ctx"] = 1024  # Reduced from 2048
            base_options["temperature"] = 0.2  # Very focused for 70B

        # Detail level adjustments for 8B model
        if "8b" in model_name.lower():
            if detail_level == "summary":
                base_options["temperature"] = 0.5  # Higher for concise generation
                base_options["top_p"] = 0.6
            elif detail_level == "detailed":
                base_options["temperature"] = 0.3  # Focused but not too restrictive
                base_options["top_p"] = 0.8
            # For standard, keep the base values (0.4, 0.7)
        else:
            # 70B model adjustments
            if detail_level == "summary":
                base_options["temperature"] = 0.4
                base_options["top_p"] = 0.7
            elif detail_level == "detailed":
                base_options["temperature"] = 0.2  # Very focused for 70B
                base_options["top_p"] = 0.8

        return base_options

    def _get_optimized_tokens(self, detail_level: str) -> int:
        """Get optimized token counts with adequate limits for complete responses."""
        token_limits = {
            "summary": 90,  # ~60 words * 1.5 tokens/word
            "standard": 180,  # ~120 words * 1.5 tokens/word
            "detailed": 450,  # ~300 words * 1.5 tokens/word
        }
        return token_limits.get(detail_level, 180)

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
        """Get current service status and model information."""
        try:
            if not self.client:
                return {"available": False, "error": "Client not initialized"}

            models = self.client.list()
            model_available = self._verify_model_availability()

            primary_available = self._verify_model_availability(self.primary_model)
            detailed_available = self._verify_model_availability(self.detailed_model)

            return {
                "available": primary_available or detailed_available,
                "primary_model": self.primary_model,
                "detailed_model": self.detailed_model,
                "primary_model_available": primary_available,
                "detailed_model_available": detailed_available,
                "current_model": self.current_model,
                "models_count": len(models.get("models", [])),
                "client_initialized": True,
                "cache_enabled": hasattr(cache, "get"),
                "performance_mode": self.performance_mode,
                "generation_timeout": self.generation_timeout,
                "circuit_breaker_state": self.circuit_breaker.state,
                "performance_metrics": self.performance_monitor.get_performance_summary(),
            }

        except Exception as e:
            return {"available": False, "error": str(e), "client_initialized": self.client is not None}


# Singleton instance
_llm_service = None


def get_local_llm_service() -> LocalLLMService:
    """Get singleton instance of LocalLLMService."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LocalLLMService()
    return _llm_service

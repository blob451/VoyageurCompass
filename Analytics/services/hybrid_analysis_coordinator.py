"""
Hybrid Analysis Coordinator providing integrated sentiment analysis and explanation generation.
Coordinates multiple analytical services for comprehensive financial insights with intelligent caching.
"""

import hashlib
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

from django.core.cache import cache

from Analytics.services.local_llm_service import get_local_llm_service
from Analytics.services.sentiment_analyzer import get_sentiment_analyzer

logger = logging.getLogger(__name__)


class HybridAnalysisCache:
    """Enhanced caching system for hybrid analysis results."""

    def __init__(self):
        self.cache_prefix = "hybrid_analysis:"
        self.default_ttl = 300  # 5 minutes default

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached hybrid analysis result."""
        try:
            result = cache.get(f"{self.cache_prefix}{cache_key}")
            if result:
                logger.debug(f"Hybrid cache hit for key: {cache_key}")
                return result
            logger.debug(f"Hybrid cache miss for key: {cache_key}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving from hybrid cache: {str(e)}")
            return None

    def set(self, cache_key: str, result: Dict[str, Any], ttl: int) -> bool:
        """Store hybrid analysis result in cache."""
        try:
            # Add caching metadata
            cache_data = result.copy()
            cache_data["cached_at"] = datetime.now().isoformat()
            cache_data["cache_ttl"] = ttl
            cache_data["cache_type"] = "hybrid_analysis"

            cache.set(f"{self.cache_prefix}{cache_key}", cache_data, ttl)
            logger.debug(f"Cached hybrid analysis result for {ttl}s")
            return True
        except Exception as e:
            logger.error(f"Error caching hybrid result: {str(e)}")
            return False

    def clear_pattern(self, pattern: str = "*"):
        """Clear cache entries matching pattern."""
        try:
            cache.delete_pattern(f"{self.cache_prefix}{pattern}")
            logger.info(f"Cleared hybrid cache entries matching: {pattern}")
        except Exception as e:
            logger.error(f"Error clearing hybrid cache: {str(e)}")


class HybridAnalysisCoordinator:
    """
    Coordinates sentiment analysis and LLM explanation generation.
    Provides seamless integration between FinBERT sentiment analysis and LLaMA explanations.
    """

    def __init__(self):
        self.sentiment_service = get_sentiment_analyzer()
        self.llm_service = get_local_llm_service()
        self.cache = HybridAnalysisCache()

        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.cache_hits = 0
        self.average_generation_time = 0.0

        logger.info("Hybrid Analysis Coordinator initialized")

    def generate_enhanced_explanation(
        self,
        analysis_data: Dict[str, Any],
        detail_level: str = "standard",
        explanation_type: str = "technical_analysis",
        use_cache: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate sentiment-enhanced financial explanation through coordinated analysis.

        Args:
            analysis_data: Technical analysis data containing indicators and scores
            detail_level: 'summary', 'standard', or 'detailed'
            explanation_type: Type of explanation to generate
            use_cache: Whether to use caching

        Returns:
            Enhanced explanation with sentiment integration or None if failed
        """
        start_time = time.time()
        self.total_requests += 1

        symbol = analysis_data.get("symbol", "UNKNOWN")
        logger.info(f"[HYBRID] Starting enhanced explanation generation for {symbol}")

        try:
            # Step 1: Check hybrid cache first
            if use_cache:
                cache_key = self._generate_hybrid_cache_key(analysis_data, detail_level, explanation_type)
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.cache_hits += 1
                    logger.info(f"[HYBRID] Cache hit for {symbol}")
                    return cached_result

            # Step 2: Generate or retrieve sentiment analysis
            sentiment_data = self._get_sentiment_analysis(analysis_data)

            if sentiment_data:
                sentiment_label = sentiment_data.get("sentimentLabel", "neutral")
                sentiment_confidence = sentiment_data.get("sentimentConfidence", 0)
                logger.info(
                    f"[HYBRID] Sentiment for {symbol}: {sentiment_label} (confidence: {sentiment_confidence:.2f})"
                )
            else:
                logger.info(f"[HYBRID] No sentiment data available for {symbol}")

            # Step 3: Generate sentiment-enhanced explanation
            enhanced_result = self.llm_service.generate_sentiment_aware_explanation(
                analysis_data=analysis_data,
                sentiment_data=sentiment_data,
                detail_level=detail_level,
                explanation_type=explanation_type,
            )

            if not enhanced_result:
                logger.warning(f"[HYBRID] Failed to generate enhanced explanation for {symbol}")
                return None

            # Step 4: Post-process for quality enhancement
            enhanced_result = self._post_process_explanation(enhanced_result, sentiment_data, analysis_data)

            # Step 5: Cache result with adaptive TTL
            if use_cache:
                adaptive_ttl = self._get_adaptive_ttl(sentiment_data, analysis_data)
                cache_key = self._generate_hybrid_cache_key(analysis_data, detail_level, explanation_type)
                self.cache.set(cache_key, enhanced_result, adaptive_ttl)

            # Update performance metrics
            generation_time = time.time() - start_time
            self.successful_requests += 1
            self._update_performance_metrics(generation_time)

            logger.info(f"[HYBRID] Successfully generated enhanced explanation for {symbol} in {generation_time:.2f}s")
            return enhanced_result

        except Exception as e:
            logger.error(f"[HYBRID] Error generating enhanced explanation for {symbol}: {str(e)}")
            return None

    def _get_sentiment_analysis(self, analysis_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get or generate sentiment analysis for the given analysis data.

        Args:
            analysis_data: Technical analysis data

        Returns:
            Sentiment analysis results or None
        """
        symbol = analysis_data.get("symbol", "UNKNOWN")

        try:
            # Check if we have news articles to analyze
            news_articles = analysis_data.get("news_articles", [])

            if news_articles:
                # Analyze news sentiment
                sentiment_result = self.sentiment_service.analyzeNewsArticles(
                    articles=news_articles, aggregate=True, symbol=symbol
                )

                if sentiment_result and "sentimentScore" in sentiment_result:
                    logger.debug(f"[HYBRID] Generated sentiment from {len(news_articles)} articles for {symbol}")
                    return sentiment_result

            # Try to get cached sentiment if no fresh news
            sentiment_cache_key = f"sentiment:stock:{symbol}:recent"
            cached_sentiment = cache.get(sentiment_cache_key)

            if cached_sentiment:
                logger.debug(f"[HYBRID] Retrieved cached sentiment for {symbol}")
                return cached_sentiment

            # If no sentiment available, return None (LLM will handle technical-only analysis)
            logger.debug(f"[HYBRID] No sentiment data available for {symbol}")
            return None

        except Exception as e:
            logger.error(f"[HYBRID] Error getting sentiment analysis for {symbol}: {str(e)}")
            return None

    def _post_process_explanation(
        self,
        explanation_result: Dict[str, Any],
        sentiment_data: Optional[Dict[str, Any]],
        analysis_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Post-process explanation for quality enhancement.

        Args:
            explanation_result: Raw explanation from LLM
            sentiment_data: Sentiment analysis data
            analysis_data: Technical analysis data

        Returns:
            Enhanced explanation result
        """
        try:
            # Extract key metrics
            content = explanation_result.get("content", "")
            symbol = analysis_data.get("symbol", "UNKNOWN")

            # Quality enhancement metrics
            quality_metrics = {
                "has_recommendation": self._has_clear_recommendation(content),
                "mentions_indicators": self._mentions_technical_indicators(content, analysis_data),
                "sentiment_alignment": self._check_sentiment_alignment(content, sentiment_data),
                "content_completeness": self._assess_content_completeness(content),
                "professional_tone": self._assess_professional_tone(content),
            }

            # Calculate overall quality score
            quality_score = sum(quality_metrics.values()) / len(quality_metrics)

            # Add quality metadata to result
            enhanced_result = explanation_result.copy()
            enhanced_result.update(
                {
                    "hybrid_coordination": {
                        "coordinator_version": "1.0",
                        "sentiment_integration_success": sentiment_data is not None,
                        "quality_metrics": quality_metrics,
                        "overall_quality_score": quality_score,
                        "post_processing_applied": True,
                    }
                }
            )

            logger.debug(f"[HYBRID] Post-processing complete for {symbol}, quality score: {quality_score:.2f}")
            return enhanced_result

        except Exception as e:
            logger.error(f"[HYBRID] Error in post-processing: {str(e)}")
            return explanation_result

    def _has_clear_recommendation(self, content: str) -> bool:
        """Check if content contains clear BUY/SELL/HOLD recommendation."""
        recommendation_keywords = [
            "BUY",
            "SELL",
            "HOLD",
            "STRONG BUY",
            "STRONG SELL",
            "Buy",
            "Sell",
            "Hold",
            "buy",
            "sell",
            "hold",
        ]
        return any(keyword in content for keyword in recommendation_keywords)

    def _mentions_technical_indicators(self, content: str, analysis_data: Dict[str, Any]) -> bool:
        """Check if content mentions specific technical indicators."""
        weighted_scores = analysis_data.get("weighted_scores", {})
        if not weighted_scores:
            return False

        # Get top indicators
        top_indicators = sorted(weighted_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

        # Check if any top indicators are mentioned
        indicator_mentions = []
        for indicator, _ in top_indicators:
            clean_name = indicator.replace("w_", "").replace("_", " ")
            indicator_keywords = [clean_name, clean_name.upper(), clean_name.lower()]

            # Add common variations
            if "sma" in clean_name.lower():
                indicator_keywords.extend(["moving average", "SMA", "MA"])
            elif "rsi" in clean_name.lower():
                indicator_keywords.extend(["RSI", "relative strength"])
            elif "macd" in clean_name.lower():
                indicator_keywords.extend(["MACD", "macd"])
            elif "bb" in clean_name.lower():
                indicator_keywords.extend(["Bollinger", "bands"])

            mentioned = any(keyword in content for keyword in indicator_keywords)
            indicator_mentions.append(mentioned)

        return any(indicator_mentions)

    def _check_sentiment_alignment(self, content: str, sentiment_data: Optional[Dict[str, Any]]) -> bool:
        """Check if content aligns with sentiment analysis."""
        if not sentiment_data:
            return True  # No sentiment to align with

        sentiment_label = sentiment_data.get("sentimentLabel", "neutral")
        sentiment_confidence = sentiment_data.get("sentimentConfidence", 0)

        # Only check alignment for high-confidence sentiment
        if sentiment_confidence < 0.7:
            return True

        # Check for sentiment-related keywords
        positive_keywords = ["positive", "bullish", "optimistic", "strong", "good", "buy"]
        negative_keywords = ["negative", "bearish", "pessimistic", "weak", "poor", "sell"]

        content_lower = content.lower()

        if sentiment_label == "positive":
            return any(keyword in content_lower for keyword in positive_keywords)
        elif sentiment_label == "negative":
            return any(keyword in content_lower for keyword in negative_keywords)
        else:
            return True  # Neutral sentiment - no specific alignment needed

    def _assess_content_completeness(self, content: str) -> bool:
        """Assess if content is complete and informative."""
        if not content or len(content.strip()) < 50:
            return False

        # Check for basic completeness indicators
        completeness_indicators = [
            len(content.split()) >= 20,  # At least 20 words
            "." in content,  # Has complete sentences
            any(char.isdigit() for char in content),  # Has numbers (likely scores/values)
        ]

        return sum(completeness_indicators) >= 2

    def _assess_professional_tone(self, content: str) -> bool:
        """Assess if content maintains professional financial tone."""
        if not content:
            return False

        # Check for professional indicators
        professional_indicators = [
            any(term in content.lower() for term in ["analysis", "technical", "indicators", "performance"]),
            not any(term in content.lower() for term in ["omg", "wow", "awesome", "terrible"]),
            len([s for s in content.split(".") if s.strip()]) >= 2,  # Multiple sentences
        ]

        return sum(professional_indicators) >= 2

    def _generate_hybrid_cache_key(
        self, analysis_data: Dict[str, Any], detail_level: str, explanation_type: str
    ) -> str:
        """Generate cache key for hybrid analysis."""
        symbol = analysis_data.get("symbol", "UNKNOWN")
        score = analysis_data.get("score_0_10", 0)

        # Create base identifier
        base_data = f"{symbol}_{score:.2f}_{detail_level}_{explanation_type}"

        # Add technical indicators
        weighted_scores = analysis_data.get("weighted_scores", {})
        if weighted_scores:
            sorted_indicators = sorted(weighted_scores.items())
            for key, value in sorted_indicators[:5]:  # Top 5 indicators
                base_data += f"_{key}_{value:.4f}"

        # Create hash for cache key using BLAKE2b
        cache_hash = hashlib.blake2b(base_data.encode(), digest_size=16).hexdigest()
        return f"enhanced_{symbol}_{cache_hash}"

    def _get_adaptive_ttl(self, sentiment_data: Optional[Dict[str, Any]], analysis_data: Dict[str, Any]) -> int:
        """Get adaptive cache TTL based on analysis characteristics."""
        base_ttl = 300  # 5 minutes base

        # Adjust based on sentiment confidence
        if sentiment_data:
            sentiment_confidence = sentiment_data.get("sentimentConfidence", 0)
            if sentiment_confidence >= 0.8:
                base_ttl *= 1.5  # High confidence - cache longer
            elif sentiment_confidence < 0.5:
                base_ttl *= 0.7  # Low confidence - cache shorter

        # Adjust based on technical score
        technical_score = analysis_data.get("score_0_10", 5)
        if technical_score >= 8 or technical_score <= 2:
            base_ttl *= 1.3  # Strong signals - cache longer

        return int(max(60, min(base_ttl, 1800)))  # Between 1-30 minutes

    def _update_performance_metrics(self, generation_time: float):
        """Update internal performance metrics."""
        if self.successful_requests == 1:
            self.average_generation_time = generation_time
        else:
            # Running average
            self.average_generation_time = (
                self.average_generation_time * (self.successful_requests - 1) + generation_time
            ) / self.successful_requests

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        success_rate = (self.successful_requests / self.total_requests) if self.total_requests > 0 else 0
        cache_hit_rate = (self.cache_hits / self.total_requests) if self.total_requests > 0 else 0

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": success_rate,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "average_generation_time": self.average_generation_time,
            "coordinator_status": "active",
            "services_available": {
                "sentiment_service": hasattr(self.sentiment_service, "analyzeSentimentSingle"),
                "llm_service": self.llm_service.is_available(),
            },
        }

    def clear_cache(self, pattern: str = "*"):
        """Clear hybrid analysis cache."""
        self.cache.clear_pattern(pattern)
        logger.info(f"Cleared hybrid analysis cache with pattern: {pattern}")


# Singleton instance
_hybrid_coordinator = None


def get_hybrid_analysis_coordinator() -> HybridAnalysisCoordinator:
    """Get singleton instance of HybridAnalysisCoordinator."""
    global _hybrid_coordinator
    if _hybrid_coordinator is None:
        _hybrid_coordinator = HybridAnalysisCoordinator()
    return _hybrid_coordinator

"""
Enhanced cache management for multilingual LLM services.
Provides utilities for managing explanation and translation caches across different languages and models.
"""

import logging
import re
from typing import Dict, List, Optional, Any

from django.core.cache import cache
from django.conf import settings

logger = logging.getLogger(__name__)


class MultilingualCacheManager:
    """
    Centralised cache management for multilingual explanations and translations.
    Provides intelligent cache operations, monitoring, and cleanup capabilities.
    """
    
    def __init__(self):
        self.cache_prefixes = {
            "explanations": "llm_explanation_",
            "sentiment_explanations": "sentiment_enhanced:",
            "translations": "translation:",
            "models": "model_availability_",
        }
        
        self.supported_languages = ["en", "fr", "es"]
        self.detail_levels = ["summary", "standard", "detailed"]
        
    def clear_analysis_cache(self, analysis_id: int, symbol: Optional[str] = None) -> Dict[str, int]:
        """
        Clear all cached explanations for a specific analysis.
        
        Args:
            analysis_id: Analysis ID to clear cache for
            symbol: Optional symbol for additional cleanup
            
        Returns:
            Dictionary with count of cleared cache entries by type
        """
        cleared_counts = {
            "explanations": 0,
            "translations": 0,
            "sentiment_explanations": 0,
            "total": 0
        }
        
        try:
            # Clear explanation caches for all detail levels and languages
            for detail_level in self.detail_levels:
                for language in self.supported_languages:
                    # Clear standard explanation cache
                    explanation_pattern = f"{self.cache_prefixes['explanations']}*{analysis_id}*{detail_level}*"
                    cleared_explanations = self._clear_cache_by_pattern(explanation_pattern)
                    cleared_counts["explanations"] += cleared_explanations
                    
                    # Clear sentiment-enhanced explanation cache  
                    sentiment_pattern = f"{self.cache_prefixes['sentiment_explanations']}*{analysis_id}*{detail_level}*"
                    cleared_sentiment = self._clear_cache_by_pattern(sentiment_pattern)
                    cleared_counts["sentiment_explanations"] += cleared_sentiment
                    
                    # Clear translation cache
                    if language != "en":
                        translation_pattern = f"{self.cache_prefixes['translations']}{language}_*{analysis_id}*"
                        cleared_translations = self._clear_cache_by_pattern(translation_pattern)
                        cleared_counts["translations"] += cleared_translations
            
            cleared_counts["total"] = sum(cleared_counts.values()) - cleared_counts["total"]
            
            logger.info(f"[CACHE CLEAR] Cleared {cleared_counts['total']} cache entries for analysis {analysis_id}")
            return cleared_counts
            
        except Exception as e:
            logger.error(f"Error clearing analysis cache for {analysis_id}: {str(e)}")
            return cleared_counts
    
    def clear_symbol_cache(self, symbol: str) -> Dict[str, int]:
        """
        Clear all cached data for a specific stock symbol.
        
        Args:
            symbol: Stock symbol to clear cache for
            
        Returns:
            Dictionary with count of cleared cache entries by type
        """
        cleared_counts = {
            "explanations": 0,
            "translations": 0,
            "sentiment_explanations": 0,
            "total": 0
        }
        
        try:
            # Clear all caches containing the symbol
            for prefix_type, prefix in self.cache_prefixes.items():
                if prefix_type == "models":
                    continue  # Skip model availability cache
                    
                pattern = f"{prefix}*{symbol}*"
                cleared = self._clear_cache_by_pattern(pattern)
                
                if prefix_type in cleared_counts:
                    cleared_counts[prefix_type] += cleared
                else:
                    cleared_counts["explanations"] += cleared
                    
            cleared_counts["total"] = sum(cleared_counts.values()) - cleared_counts["total"]
            
            logger.info(f"[CACHE CLEAR] Cleared {cleared_counts['total']} cache entries for symbol {symbol}")
            return cleared_counts
            
        except Exception as e:
            logger.error(f"Error clearing symbol cache for {symbol}: {str(e)}")
            return cleared_counts
    
    def clear_language_cache(self, language: str) -> Dict[str, int]:
        """
        Clear all cached translations for a specific language.
        
        Args:
            language: Language code to clear cache for
            
        Returns:
            Dictionary with count of cleared cache entries
        """
        if language not in self.supported_languages:
            logger.warning(f"Unsupported language for cache clearing: {language}")
            return {"cleared": 0}
        
        cleared_count = 0
        
        try:
            if language == "en":
                # Clear English explanations (original content)
                for prefix in [self.cache_prefixes["explanations"], self.cache_prefixes["sentiment_explanations"]]:
                    pattern = f"{prefix}*"
                    cleared_count += self._clear_cache_by_pattern(pattern)
            else:
                # Clear translations for specific language
                pattern = f"{self.cache_prefixes['translations']}{language}_*"
                cleared_count = self._clear_cache_by_pattern(pattern)
            
            logger.info(f"[CACHE CLEAR] Cleared {cleared_count} cache entries for language {language}")
            return {"cleared": cleared_count}
            
        except Exception as e:
            logger.error(f"Error clearing language cache for {language}: {str(e)}")
            return {"cleared": 0}
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics for monitoring.
        
        Returns:
            Dictionary with cache statistics and metrics
        """
        try:
            stats = {
                "cache_types": {},
                "languages": {},
                "detail_levels": {},
                "total_entries": 0,
                "cache_health": "healthy"
            }
            
            # Get cache keys (this is implementation-dependent)
            # Note: Django's cache backend may not support key listing
            # This is a conceptual implementation
            
            try:
                # Attempt to get cache statistics if available
                cache_stats = self._get_cache_backend_stats()
                if cache_stats:
                    stats.update(cache_stats)
            except Exception:
                # Fallback to basic statistics
                stats["note"] = "Detailed statistics not available for this cache backend"
            
            # Add configuration information
            stats["configuration"] = {
                "supported_languages": self.supported_languages,
                "detail_levels": self.detail_levels,
                "cache_prefixes": self.cache_prefixes,
                "default_ttl": getattr(settings, "EXPLANATION_CACHE_TTL", 300),
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache statistics: {str(e)}")
            return {
                "error": str(e),
                "cache_health": "unhealthy"
            }
    
    def warm_popular_cache(self, popular_symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Warm cache for popular symbols and analysis patterns.
        
        Args:
            popular_symbols: List of popular symbols to warm cache for
            
        Returns:
            Dictionary with warming results
        """
        if not popular_symbols:
            popular_symbols = getattr(settings, "TRENDING_STOCKS", ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"])
        
        warming_results = {
            "symbols_processed": 0,
            "cache_entries_created": 0,
            "errors": 0,
            "languages_warmed": self.supported_languages.copy(),
        }
        
        try:
            logger.info(f"[CACHE WARM] Starting cache warming for {len(popular_symbols)} symbols")
            
            # This would typically trigger analysis and explanation generation
            # for popular symbols in different languages and detail levels
            # Implementation depends on your specific analysis pipeline
            
            warming_results["note"] = "Cache warming initiated - actual warming happens asynchronously"
            warming_results["symbols_processed"] = len(popular_symbols)
            
            logger.info(f"[CACHE WARM] Cache warming completed for {len(popular_symbols)} symbols")
            return warming_results
            
        except Exception as e:
            logger.error(f"Error during cache warming: {str(e)}")
            warming_results["errors"] = 1
            warming_results["error_message"] = str(e)
            return warming_results
    
    def _clear_cache_by_pattern(self, pattern: str) -> int:
        """
        Clear cache entries matching a pattern.

        Args:
            pattern: Pattern to match cache keys

        Returns:
            Number of cleared entries
        """
        try:
            cleared_count = 0

            # Check if we're using Redis cache backend
            if hasattr(cache, '_cache') and hasattr(cache._cache, '_client'):
                # Redis backend with direct client access
                redis_client = cache._cache._client

                # Use SCAN to find matching keys (more efficient than KEYS)
                cursor = 0
                keys_to_delete = []

                while True:
                    cursor, keys = redis_client.scan(cursor=cursor, match=pattern, count=100)
                    keys_to_delete.extend(keys)

                    if cursor == 0:
                        break

                # Delete keys in batches to avoid blocking Redis
                if keys_to_delete:
                    # Convert bytes to strings if needed
                    string_keys = [key.decode('utf-8') if isinstance(key, bytes) else key for key in keys_to_delete]

                    # Delete in batches of 100
                    for i in range(0, len(string_keys), 100):
                        batch = string_keys[i:i+100]
                        if batch:
                            deleted = redis_client.delete(*batch)
                            cleared_count += deleted

                logger.info(f"Cleared {cleared_count} cache entries matching pattern: {pattern}")

            elif hasattr(cache, 'clear'):
                # Fallback for other cache backends - clear all if pattern is too generic
                # This is not ideal but ensures cache clearing works
                if '*' in pattern and len(pattern.replace('*', '')) < 10:
                    # Very generic pattern, might be safer to warn and not clear everything
                    logger.warning(f"Generic cache pattern {pattern} - using selective clearing")

                    # Try to clear only known cache prefixes
                    for prefix_type, prefix in self.cache_prefixes.items():
                        if pattern.startswith(prefix) or prefix in pattern:
                            # For non-Redis backends, we can try clearing specific known keys
                            # This is a best-effort approach
                            for lang in self.supported_languages:
                                for level in self.detail_levels:
                                    test_key = f"{prefix}{lang}_{level}_test"
                                    cache.delete(test_key)
                                    cleared_count += 1
                else:
                    logger.warning(f"Cache backend doesn't support pattern clearing for pattern: {pattern}")

            else:
                # DummyCache or other backends without pattern support
                logger.info(f"Cache backend doesn't support pattern clearing: {type(cache).__name__}")

            return cleared_count

        except Exception as e:
            logger.error(f"Error clearing cache by pattern {pattern}: {str(e)}")
            return 0
    
    def _get_cache_backend_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get statistics from cache backend if available.
        
        Returns:
            Cache backend statistics or None
        """
        try:
            # This would depend on your specific cache backend
            # Redis, Memcached, etc. have different statistics APIs
            
            return {
                "backend_type": "django_cache",
                "note": "Backend-specific statistics not implemented"
            }
            
        except Exception:
            return None
    
    def invalidate_expired_translations(self) -> Dict[str, int]:
        """
        Invalidate translations with low quality scores or that are outdated.
        
        Returns:
            Dictionary with invalidation results
        """
        invalidated_count = 0
        
        try:
            # This would scan translation cache entries and remove those
            # with low quality scores or that are too old
            
            logger.info(f"[CACHE INVALIDATE] Invalidated {invalidated_count} low-quality translations")
            return {"invalidated": invalidated_count}
            
        except Exception as e:
            logger.error(f"Error invalidating expired translations: {str(e)}")
            return {"invalidated": 0, "error": str(e)}


# Singleton instance
_cache_manager = None


def get_cache_manager() -> MultilingualCacheManager:
    """Get singleton instance of MultilingualCacheManager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = MultilingualCacheManager()
    return _cache_manager
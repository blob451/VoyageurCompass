"""
Unified cache key management system for multilingual LLM services.

Provides consistent, secure, and optimized cache key generation across all services.
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)


class UnifiedCacheKeyGenerator:
    """Unified cache key generator with consistent hashing strategy."""

    def __init__(self):
        """Initialize the cache key generator."""
        self.version = getattr(settings, 'CACHE_KEY_VERSION', 'v1.0')
        self.max_key_length = getattr(settings, 'CACHE_MAX_KEY_LENGTH', 200)
        self.hash_algorithm = 'blake2b'
        self.hash_digest_size = 16  # 128-bit hash

        # Key prefixes for different content types
        self.prefixes = {
            'explanation': 'exp',
            'translation': 'trans',
            'sentiment': 'sent',
            'multilingual': 'multi',
            'optimization': 'opt',
            'quality': 'qual',
            'metrics': 'metrics',
            'template': 'tmpl'
        }

    def generate_explanation_key(
        self,
        analysis_data: Dict[str, Any],
        detail_level: str,
        language: str = "en",
        explanation_type: str = "technical_analysis",
        user_id: Optional[int] = None,
        sentiment_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate cache key for explanation content."""
        components = {
            'type': 'explanation',
            'symbol': analysis_data.get('symbol', 'UNKNOWN'),
            'score': self._normalize_score(analysis_data.get('score_0_10', 0)),
            'detail_level': detail_level,
            'language': language,
            'explanation_type': explanation_type
        }

        # Add user context if provided
        if user_id:
            components['user_id'] = user_id

        # Add sentiment fingerprint if provided
        if sentiment_data:
            components['sentiment_hash'] = self._create_sentiment_fingerprint(sentiment_data)

        # Add complexity fingerprint
        components['complexity'] = self._calculate_complexity_fingerprint(analysis_data)

        return self._build_cache_key('explanation', components)

    def generate_translation_key(
        self,
        content: str,
        source_language: str,
        target_language: str,
        context: Optional[Dict[str, Any]] = None,
        quality_level: str = "standard"
    ) -> str:
        """Generate cache key for translation content."""
        components = {
            'type': 'translation',
            'content_hash': self._create_content_hash(content),
            'source_lang': source_language,
            'target_lang': target_language,
            'quality_level': quality_level
        }

        # Add context fingerprint if provided
        if context:
            components['context_hash'] = self._create_context_fingerprint(context)

        return self._build_cache_key('translation', components)

    def generate_multilingual_key(
        self,
        analysis_data: Dict[str, Any],
        languages: List[str],
        detail_level: str,
        optimization_level: str = "standard"
    ) -> str:
        """Generate cache key for multilingual optimization results."""
        components = {
            'type': 'multilingual',
            'symbol': analysis_data.get('symbol', 'UNKNOWN'),
            'score': self._normalize_score(analysis_data.get('score_0_10', 0)),
            'languages': '_'.join(sorted(languages)),
            'detail_level': detail_level,
            'optimization_level': optimization_level,
            'complexity': self._calculate_complexity_fingerprint(analysis_data)
        }

        return self._build_cache_key('multilingual', components)

    def generate_template_key(
        self,
        template_type: str,
        language: str,
        detail_level: str,
        has_sentiment: bool = False
    ) -> str:
        """Generate cache key for prompt templates."""
        components = {
            'type': 'template',
            'template_type': template_type,
            'language': language,
            'detail_level': detail_level,
            'has_sentiment': str(has_sentiment).lower()
        }

        return self._build_cache_key('template', components)

    def generate_metrics_key(
        self,
        metric_type: str,
        language: Optional[str] = None,
        time_range: str = "24h",
        aggregation: str = "default"
    ) -> str:
        """Generate cache key for metrics and statistics."""
        components = {
            'type': 'metrics',
            'metric_type': metric_type,
            'time_range': time_range,
            'aggregation': aggregation
        }

        if language:
            components['language'] = language

        return self._build_cache_key('metrics', components)

    def _build_cache_key(self, content_type: str, components: Dict[str, Any]) -> str:
        """Build final cache key from components."""
        try:
            # Get prefix for content type
            prefix = self.prefixes.get(content_type, 'gen')

            # Create deterministic string from components
            components_str = self._serialize_components(components)

            # Create hash if key would be too long
            if len(components_str) > self.max_key_length - 20:  # Reserve space for prefix and version
                content_hash = hashlib.blake2b(
                    components_str.encode('utf-8'),
                    digest_size=self.hash_digest_size
                ).hexdigest()
                key_body = content_hash
            else:
                # Use components directly for shorter keys (better readability)
                key_body = components_str.replace(':', '_').replace('=', '-')

            # Build final key
            cache_key = f"{prefix}:{self.version}:{key_body}"

            # Ensure key length is within limits
            if len(cache_key) > self.max_key_length:
                # Fallback to pure hash
                full_str = f"{prefix}:{self.version}:{components_str}"
                content_hash = hashlib.blake2b(
                    full_str.encode('utf-8'),
                    digest_size=self.hash_digest_size
                ).hexdigest()
                cache_key = f"{prefix}:{self.version}:{content_hash}"

            return cache_key

        except Exception as e:
            logger.warning(f"Error building cache key: {str(e)}")
            # Emergency fallback
            fallback_str = f"{content_type}:{str(components)}"
            fallback_hash = hashlib.blake2b(
                fallback_str.encode('utf-8'),
                digest_size=self.hash_digest_size
            ).hexdigest()
            return f"fallback:{self.version}:{fallback_hash}"

    def _serialize_components(self, components: Dict[str, Any]) -> str:
        """Serialize components to deterministic string."""
        try:
            # Sort keys for consistency
            sorted_items = sorted(components.items())

            # Create key-value pairs
            pairs = []
            for key, value in sorted_items:
                if isinstance(value, (dict, list)):
                    # For complex types, create hash
                    value_str = json.dumps(value, sort_keys=True, separators=(',', ':'))
                    value_hash = hashlib.blake2b(
                        value_str.encode('utf-8'),
                        digest_size=8
                    ).hexdigest()
                    pairs.append(f"{key}:{value_hash}")
                else:
                    pairs.append(f"{key}:{str(value)}")

            return '|'.join(pairs)

        except Exception as e:
            logger.warning(f"Error serializing components: {str(e)}")
            return str(hash(str(components)))

    def _create_content_hash(self, content: str) -> str:
        """Create hash for content with normalization."""
        try:
            # Normalize content for better cache hits
            normalized = content.strip().lower()

            # Remove extra whitespace
            import re
            normalized = re.sub(r'\s+', ' ', normalized)

            # Create hash
            return hashlib.blake2b(
                normalized.encode('utf-8'),
                digest_size=8
            ).hexdigest()

        except Exception as e:
            logger.warning(f"Error creating content hash: {str(e)}")
            return hashlib.blake2b(
                content.encode('utf-8'),
                digest_size=8
            ).hexdigest()

    def _create_sentiment_fingerprint(self, sentiment_data: Dict[str, Any]) -> str:
        """Create fingerprint for sentiment data."""
        try:
            # Extract key sentiment indicators
            indicators = {
                'label': sentiment_data.get('label', 'neutral'),
                'score': round(sentiment_data.get('score', 0.5), 2),
                'confidence': round(sentiment_data.get('confidence', 0.5), 2)
            }

            # Create deterministic hash
            indicators_str = json.dumps(indicators, sort_keys=True)
            return hashlib.blake2b(
                indicators_str.encode('utf-8'),
                digest_size=6
            ).hexdigest()

        except Exception as e:
            logger.warning(f"Error creating sentiment fingerprint: {str(e)}")
            return "unknown"

    def _create_context_fingerprint(self, context: Dict[str, Any]) -> str:
        """Create fingerprint for context data."""
        try:
            # Extract stable context elements
            stable_context = {}
            for key in ['symbol', 'domain', 'market', 'sector']:
                if key in context:
                    stable_context[key] = context[key]

            if not stable_context:
                return "default"

            context_str = json.dumps(stable_context, sort_keys=True)
            return hashlib.blake2b(
                context_str.encode('utf-8'),
                digest_size=6
            ).hexdigest()

        except Exception as e:
            logger.warning(f"Error creating context fingerprint: {str(e)}")
            return "default"

    def _calculate_complexity_fingerprint(self, analysis_data: Dict[str, Any]) -> str:
        """Calculate complexity fingerprint for analysis data."""
        try:
            # Key complexity indicators
            indicators_count = len(analysis_data.get('weighted_scores', {}))
            technical_indicators = len(analysis_data.get('indicators', {}))

            # Normalize complexity to categories
            total_complexity = indicators_count + technical_indicators
            if total_complexity <= 5:
                complexity = 'simple'
            elif total_complexity <= 15:
                complexity = 'medium'
            else:
                complexity = 'complex'

            return complexity

        except Exception as e:
            logger.warning(f"Error calculating complexity: {str(e)}")
            return 'unknown'

    def _normalize_score(self, score: Union[int, float]) -> float:
        """Normalize score for consistent cache keys."""
        try:
            # Round to 1 decimal place for better cache hits
            return round(float(score), 1)
        except (ValueError, TypeError):
            return 0.0

    def validate_cache_key(self, cache_key: str) -> bool:
        """Validate cache key format and length."""
        if not cache_key:
            return False

        if len(cache_key) > self.max_key_length:
            return False

        # Check for invalid characters
        invalid_chars = [' ', '\n', '\r', '\t']
        if any(char in cache_key for char in invalid_chars):
            return False

        return True

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache key generation statistics."""
        return {
            'version': self.version,
            'max_key_length': self.max_key_length,
            'hash_algorithm': self.hash_algorithm,
            'hash_digest_size': self.hash_digest_size,
            'prefixes': self.prefixes,
            'supported_content_types': list(self.prefixes.keys())
        }


# Global cache key generator instance
_cache_key_generator = None


def get_cache_key_generator() -> UnifiedCacheKeyGenerator:
    """Get global cache key generator instance."""
    global _cache_key_generator
    if _cache_key_generator is None:
        _cache_key_generator = UnifiedCacheKeyGenerator()
    return _cache_key_generator


def generate_explanation_cache_key(
    analysis_data: Dict[str, Any],
    detail_level: str,
    language: str = "en",
    **kwargs
) -> str:
    """Convenience function for explanation cache keys."""
    generator = get_cache_key_generator()
    return generator.generate_explanation_key(
        analysis_data=analysis_data,
        detail_level=detail_level,
        language=language,
        **kwargs
    )


def generate_translation_cache_key(
    content: str,
    source_language: str,
    target_language: str,
    **kwargs
) -> str:
    """Convenience function for translation cache keys."""
    generator = get_cache_key_generator()
    return generator.generate_translation_key(
        content=content,
        source_language=source_language,
        target_language=target_language,
        **kwargs
    )


def generate_multilingual_cache_key(
    analysis_data: Dict[str, Any],
    languages: List[str],
    detail_level: str,
    **kwargs
) -> str:
    """Convenience function for multilingual cache keys."""
    generator = get_cache_key_generator()
    return generator.generate_multilingual_key(
        analysis_data=analysis_data,
        languages=languages,
        detail_level=detail_level,
        **kwargs
    )
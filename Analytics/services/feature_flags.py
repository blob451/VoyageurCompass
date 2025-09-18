"""
Feature flags system for multilingual LLM services.

Provides production-ready feature controls with gradual rollout capabilities,
A/B testing support, and emergency rollback functionality.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from django.conf import settings
from django.core.cache import cache
from django.contrib.auth.models import User

logger = logging.getLogger(__name__)


class MultilingualFeatureFlags:
    """Feature flags for multilingual LLM functionality."""

    # Core feature flags
    MULTILINGUAL_ENABLED = "multilingual_llm_enabled"
    DIRECT_GENERATION_ENABLED = "direct_generation_enabled"
    TRANSLATION_PIPELINE_ENABLED = "translation_pipeline_enabled"
    PARALLEL_PROCESSING_ENABLED = "parallel_processing_enabled"

    # Language-specific flags
    FRENCH_GENERATION_ENABLED = "french_generation_enabled"
    SPANISH_GENERATION_ENABLED = "spanish_generation_enabled"

    # Advanced features
    QUALITY_ENHANCEMENT_ENABLED = "quality_enhancement_enabled"
    ADAPTIVE_CACHING_ENABLED = "adaptive_caching_enabled"
    PERFORMANCE_MONITORING_ENABLED = "performance_monitoring_enabled"

    # Emergency controls
    EMERGENCY_FALLBACK_ENABLED = "emergency_fallback_enabled"
    CIRCUIT_BREAKER_ENABLED = "circuit_breaker_enabled"

    def __init__(self):
        """Initialize feature flags system."""
        self.cache_prefix = "feature_flags:"
        self.cache_ttl = getattr(settings, 'FEATURE_FLAGS_CACHE_TTL', 300)  # 5 minutes

        # Default flag states
        self.default_flags = {
            self.MULTILINGUAL_ENABLED: getattr(settings, 'MULTILINGUAL_LLM_ENABLED', True),
            self.DIRECT_GENERATION_ENABLED: getattr(settings, 'DIRECT_GENERATION_ENABLED', True),
            self.TRANSLATION_PIPELINE_ENABLED: getattr(settings, 'TRANSLATION_PIPELINE_ENABLED', True),
            self.PARALLEL_PROCESSING_ENABLED: getattr(settings, 'PARALLEL_MULTILINGUAL_GENERATION_ENABLED', True),
            self.FRENCH_GENERATION_ENABLED: getattr(settings, 'FRENCH_GENERATION_ENABLED', True),
            self.SPANISH_GENERATION_ENABLED: getattr(settings, 'SPANISH_GENERATION_ENABLED', True),
            self.QUALITY_ENHANCEMENT_ENABLED: getattr(settings, 'QUALITY_ENHANCEMENT_ENABLED', True),
            self.ADAPTIVE_CACHING_ENABLED: getattr(settings, 'ADAPTIVE_CACHING_ENABLED', True),
            self.PERFORMANCE_MONITORING_ENABLED: getattr(settings, 'PERFORMANCE_MONITORING_ENABLED', True),
            self.EMERGENCY_FALLBACK_ENABLED: getattr(settings, 'EMERGENCY_FALLBACK_ENABLED', False),
            self.CIRCUIT_BREAKER_ENABLED: getattr(settings, 'CIRCUIT_BREAKER_ENABLED', True),
        }

        # Gradual rollout percentages (0-100)
        self.rollout_percentages = {
            self.FRENCH_GENERATION_ENABLED: getattr(settings, 'FRENCH_ROLLOUT_PERCENTAGE', 100),
            self.SPANISH_GENERATION_ENABLED: getattr(settings, 'SPANISH_ROLLOUT_PERCENTAGE', 100),
            self.QUALITY_ENHANCEMENT_ENABLED: getattr(settings, 'QUALITY_ENHANCEMENT_ROLLOUT_PERCENTAGE', 100),
            self.PARALLEL_PROCESSING_ENABLED: getattr(settings, 'PARALLEL_PROCESSING_ROLLOUT_PERCENTAGE', 100),
        }

    @classmethod
    def is_multilingual_enabled(cls, language: str = None, user: User = None) -> bool:
        """
        Check if multilingual functionality is enabled for given language and user.

        Args:
            language: Target language code (fr, es, etc.)
            user: User for gradual rollout (optional)

        Returns:
            True if multilingual is enabled for this request
        """
        flags = cls()

        # Check master multilingual flag
        if not flags.is_enabled(cls.MULTILINGUAL_ENABLED):
            return False

        # Check emergency fallback
        if flags.is_enabled(cls.EMERGENCY_FALLBACK_ENABLED):
            logger.warning("Emergency fallback enabled - multilingual disabled")
            return False

        # Language-specific checks
        if language:
            if language == "fr":
                return flags.is_enabled_for_user(cls.FRENCH_GENERATION_ENABLED, user)
            elif language == "es":
                return flags.is_enabled_for_user(cls.SPANISH_GENERATION_ENABLED, user)

        return True

    @classmethod
    def is_feature_enabled(cls, feature_name: str, language: str = None, user: User = None) -> bool:
        """
        Check if a specific feature is enabled.

        Args:
            feature_name: Name of the feature flag
            language: Target language (optional)
            user: User for gradual rollout (optional)

        Returns:
            True if feature is enabled
        """
        flags = cls()

        # Check emergency fallback for core features
        if feature_name in [cls.MULTILINGUAL_ENABLED, cls.DIRECT_GENERATION_ENABLED,
                           cls.TRANSLATION_PIPELINE_ENABLED]:
            if flags.is_enabled(cls.EMERGENCY_FALLBACK_ENABLED):
                return False

        return flags.is_enabled_for_user(feature_name, user)

    def is_enabled(self, flag_name: str) -> bool:
        """Check if a feature flag is enabled (basic check without user context)."""
        try:
            # Check cache first
            cache_key = f"{self.cache_prefix}{flag_name}"
            cached_value = cache.get(cache_key)

            if cached_value is not None:
                return cached_value

            # Get from settings or default
            value = self.default_flags.get(flag_name, False)

            # Cache the result
            cache.set(cache_key, value, self.cache_ttl)

            return value

        except Exception as e:
            logger.warning(f"Error checking feature flag {flag_name}: {str(e)}")
            return self.default_flags.get(flag_name, False)

    def is_enabled_for_user(self, flag_name: str, user: User = None) -> bool:
        """
        Check if a feature flag is enabled for a specific user (supports gradual rollout).

        Args:
            flag_name: Name of the feature flag
            user: User for rollout calculation

        Returns:
            True if enabled for this user
        """
        try:
            # Basic flag check
            if not self.is_enabled(flag_name):
                return False

            # If no rollout percentage configured, return basic check
            if flag_name not in self.rollout_percentages:
                return True

            rollout_percentage = self.rollout_percentages[flag_name]

            # 100% rollout
            if rollout_percentage >= 100:
                return True

            # 0% rollout
            if rollout_percentage <= 0:
                return False

            # Calculate user bucket for gradual rollout
            if user and user.is_authenticated:
                user_bucket = self._get_user_bucket(user.id, flag_name)
            else:
                # For anonymous users, use IP-based bucketing or random
                user_bucket = hash(flag_name) % 100

            return user_bucket < rollout_percentage

        except Exception as e:
            logger.warning(f"Error checking user-specific flag {flag_name}: {str(e)}")
            return self.is_enabled(flag_name)

    def _get_user_bucket(self, user_id: int, flag_name: str) -> int:
        """Get consistent user bucket (0-99) for gradual rollout."""
        # Create deterministic hash from user ID and flag name
        hash_input = f"{user_id}:{flag_name}"
        return hash(hash_input) % 100

    def set_flag(self, flag_name: str, enabled: bool, ttl: int = None) -> bool:
        """
        Set a feature flag value (runtime override).

        Args:
            flag_name: Name of the feature flag
            enabled: Whether to enable the flag
            ttl: Cache TTL in seconds (optional)

        Returns:
            True if successfully set
        """
        try:
            cache_key = f"{self.cache_prefix}{flag_name}"
            cache_ttl = ttl or self.cache_ttl

            cache.set(cache_key, enabled, cache_ttl)

            logger.info(f"Feature flag {flag_name} set to {enabled} for {cache_ttl}s")
            return True

        except Exception as e:
            logger.error(f"Error setting feature flag {flag_name}: {str(e)}")
            return False

    def set_rollout_percentage(self, flag_name: str, percentage: int) -> bool:
        """
        Set rollout percentage for gradual deployment.

        Args:
            flag_name: Name of the feature flag
            percentage: Rollout percentage (0-100)

        Returns:
            True if successfully set
        """
        try:
            if not 0 <= percentage <= 100:
                raise ValueError("Percentage must be between 0 and 100")

            cache_key = f"{self.cache_prefix}rollout:{flag_name}"
            cache.set(cache_key, percentage, self.cache_ttl)

            # Update in-memory cache
            self.rollout_percentages[flag_name] = percentage

            logger.info(f"Rollout percentage for {flag_name} set to {percentage}%")
            return True

        except Exception as e:
            logger.error(f"Error setting rollout percentage for {flag_name}: {str(e)}")
            return False

    def emergency_disable_all(self, reason: str = "Emergency disable") -> Dict[str, bool]:
        """
        Emergency disable all multilingual features.

        Args:
            reason: Reason for emergency disable

        Returns:
            Dictionary of disabled flags
        """
        emergency_flags = [
            self.MULTILINGUAL_ENABLED,
            self.DIRECT_GENERATION_ENABLED,
            self.TRANSLATION_PIPELINE_ENABLED,
            self.PARALLEL_PROCESSING_ENABLED,
            self.FRENCH_GENERATION_ENABLED,
            self.SPANISH_GENERATION_ENABLED,
        ]

        disabled_flags = {}

        try:
            # Enable emergency fallback
            self.set_flag(self.EMERGENCY_FALLBACK_ENABLED, True, ttl=3600)  # 1 hour

            # Disable core features
            for flag in emergency_flags:
                if self.set_flag(flag, False, ttl=3600):
                    disabled_flags[flag] = False

            logger.critical(f"Emergency disable triggered: {reason}. Disabled flags: {list(disabled_flags.keys())}")

            return disabled_flags

        except Exception as e:
            logger.error(f"Error during emergency disable: {str(e)}")
            return {}

    def get_all_flags_status(self, user: User = None) -> Dict[str, Any]:
        """
        Get status of all feature flags for monitoring/debugging.

        Args:
            user: User for user-specific flag evaluation

        Returns:
            Dictionary with all flag statuses
        """
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'flags': {},
                'rollout_percentages': {},
                'emergency_status': {
                    'emergency_fallback_enabled': self.is_enabled(self.EMERGENCY_FALLBACK_ENABLED),
                    'circuit_breaker_enabled': self.is_enabled(self.CIRCUIT_BREAKER_ENABLED),
                }
            }

            # Check all flags
            for flag_name in self.default_flags.keys():
                status['flags'][flag_name] = {
                    'enabled': self.is_enabled(flag_name),
                    'user_enabled': self.is_enabled_for_user(flag_name, user) if user else None,
                    'default_value': self.default_flags[flag_name]
                }

            # Add rollout percentages
            for flag_name, percentage in self.rollout_percentages.items():
                status['rollout_percentages'][flag_name] = percentage

            return status

        except Exception as e:
            logger.error(f"Error getting flags status: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def clear_cache(self) -> bool:
        """Clear all cached feature flag values."""
        try:
            # Clear individual flags
            for flag_name in self.default_flags.keys():
                cache_key = f"{self.cache_prefix}{flag_name}"
                cache.delete(cache_key)

            # Clear rollout percentages
            for flag_name in self.rollout_percentages.keys():
                cache_key = f"{self.cache_prefix}rollout:{flag_name}"
                cache.delete(cache_key)

            logger.info("Feature flags cache cleared")
            return True

        except Exception as e:
            logger.error(f"Error clearing feature flags cache: {str(e)}")
            return False


# Global feature flags instance
_feature_flags_instance = None


def get_feature_flags() -> MultilingualFeatureFlags:
    """Get global feature flags instance."""
    global _feature_flags_instance
    if _feature_flags_instance is None:
        _feature_flags_instance = MultilingualFeatureFlags()
    return _feature_flags_instance


# Convenience functions
def is_multilingual_enabled(language: str = None, user: User = None) -> bool:
    """Check if multilingual is enabled for language/user."""
    return MultilingualFeatureFlags.is_multilingual_enabled(language, user)


def is_feature_enabled(feature_name: str, language: str = None, user: User = None) -> bool:
    """Check if specific feature is enabled."""
    return MultilingualFeatureFlags.is_feature_enabled(feature_name, language, user)


def emergency_disable_multilingual(reason: str = "Emergency disable") -> Dict[str, bool]:
    """Emergency disable all multilingual features."""
    flags = get_feature_flags()
    return flags.emergency_disable_all(reason)
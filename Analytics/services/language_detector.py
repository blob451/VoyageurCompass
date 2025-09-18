"""
Smart language detection service for multilingual financial analysis.
Intelligently determines the optimal language based on user preferences, browser settings, and request context.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from django.conf import settings
from django.core.cache import cache
from django.contrib.auth.models import User

logger = logging.getLogger(__name__)


class LanguagePreference:
    """Represents a user's language preference with confidence scoring."""

    def __init__(self, language: str, confidence: float, source: str, timestamp: datetime = None):
        self.language = language
        self.confidence = confidence  # 0.0 to 1.0
        self.source = source  # 'explicit', 'header', 'browser', 'inferred', 'default'
        self.timestamp = timestamp or datetime.now()

    def __repr__(self) -> str:
        return f"LanguagePreference({self.language}, {self.confidence:.2f}, {self.source})"


class SmartLanguageDetector:
    """Smart language detection with fallback hierarchy and user learning."""

    def __init__(self):
        """Initialize the language detector."""
        self.supported_languages = getattr(settings, 'LANGUAGES', [('en', 'English')])
        self.supported_codes = [lang[0] for lang in self.supported_languages]
        self.default_language = getattr(settings, 'DEFAULT_USER_LANGUAGE', 'en')

        # Language mapping for common variations
        self.language_mappings = {
            'en-US': 'en', 'en-GB': 'en', 'en-CA': 'en', 'en-AU': 'en',
            'fr-FR': 'fr', 'fr-CA': 'fr', 'fr-BE': 'fr', 'fr-CH': 'fr',
            'es-ES': 'es', 'es-MX': 'es', 'es-AR': 'es', 'es-CO': 'es',
        }

        # User preference cache settings
        self.preference_cache_ttl = getattr(settings, 'USER_LANGUAGE_CACHE_TTL', 86400)  # 24 hours
        self.preference_cache_prefix = 'user_lang_pref:'

        # Quality thresholds for confidence scoring
        self.high_confidence_threshold = 0.8
        self.medium_confidence_threshold = 0.5

        logger.info(f"Language detector initialized with supported languages: {self.supported_codes}")

    def detect_user_language(
        self,
        request=None,
        user: Optional[User] = None,
        explicit_language: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> LanguagePreference:
        """
        Detect the optimal language for a user using smart fallback hierarchy.

        Args:
            request: Django request object (for headers)
            user: Authenticated user (for preferences)
            explicit_language: Explicitly requested language
            context: Additional context (timezone, location, etc.)

        Returns:
            LanguagePreference with detected language and confidence
        """
        preferences = []

        # 1. Explicit language parameter (highest priority)
        if explicit_language:
            pref = self._evaluate_explicit_language(explicit_language)
            if pref:
                preferences.append(pref)

        # 2. User's stored preferences (high priority)
        if user and user.is_authenticated:
            pref = self._get_user_stored_preference(user)
            if pref:
                preferences.append(pref)

        # 3. Accept-Language header (medium priority)
        if request:
            header_prefs = self._parse_accept_language_header(request)
            preferences.extend(header_prefs)

        # 4. Context-based inference (low-medium priority)
        if context:
            context_pref = self._infer_from_context(context)
            if context_pref:
                preferences.append(context_pref)

        # 5. Default fallback (lowest priority)
        preferences.append(LanguagePreference(
            self.default_language, 0.3, 'default'
        ))

        # Select best preference
        best_preference = self._select_best_preference(preferences)

        # Store user preference if we have a high-confidence detection
        if (user and user.is_authenticated and
            best_preference.confidence >= self.high_confidence_threshold and
            best_preference.source != 'default'):
            self._store_user_preference(user, best_preference)

        logger.info(f"Language detected: {best_preference.language} (confidence: {best_preference.confidence:.2f}, source: {best_preference.source})")
        return best_preference

    def get_language_with_fallback(
        self,
        request=None,
        user: Optional[User] = None,
        explicit_language: Optional[str] = None
    ) -> str:
        """
        Simple interface to get language code with smart detection.

        Returns:
            Language code (guaranteed to be supported)
        """
        preference = self.detect_user_language(request, user, explicit_language)
        return preference.language

    def validate_language_support(self, language: str) -> Tuple[bool, str]:
        """
        Validate if a language is supported and return best match.

        Args:
            language: Language code to validate

        Returns:
            Tuple of (is_supported, best_match_code)
        """
        if not language:
            return False, self.default_language

        language = language.lower().strip()

        # Direct match
        if language in self.supported_codes:
            return True, language

        # Check language mappings (e.g., en-US -> en)
        if language in self.language_mappings:
            mapped = self.language_mappings[language]
            if mapped in self.supported_codes:
                return True, mapped

        # Check language family (e.g., fr-CA -> fr)
        if '-' in language:
            base_language = language.split('-')[0]
            if base_language in self.supported_codes:
                return True, base_language

        return False, self.default_language

    def get_user_preference_stats(self, user: User) -> Dict[str, Any]:
        """Get statistics about user's language preferences."""
        if not user or not user.is_authenticated:
            return {}

        cache_key = f"{self.preference_cache_prefix}stats:{user.id}"
        stats = cache.get(cache_key, {})

        return {
            'current_preference': self._get_user_stored_preference(user),
            'detection_history': stats.get('history', []),
            'confidence_trends': stats.get('confidence_trends', []),
            'last_updated': stats.get('last_updated'),
        }

    # Private methods

    def _evaluate_explicit_language(self, language: str) -> Optional[LanguagePreference]:
        """Evaluate explicitly requested language."""
        is_supported, best_match = self.validate_language_support(language)

        if is_supported:
            confidence = 1.0 if language == best_match else 0.9
            return LanguagePreference(best_match, confidence, 'explicit')

        return None

    def _get_user_stored_preference(self, user: User) -> Optional[LanguagePreference]:
        """Get user's stored language preference from cache."""
        cache_key = f"{self.preference_cache_prefix}{user.id}"
        stored_pref = cache.get(cache_key)

        if stored_pref:
            return LanguagePreference(
                stored_pref['language'],
                stored_pref['confidence'] * 0.9,  # Slightly decay confidence over time
                'stored'
            )

        return None

    def _parse_accept_language_header(self, request) -> List[LanguagePreference]:
        """Parse Accept-Language header into preferences."""
        accept_language = request.META.get('HTTP_ACCEPT_LANGUAGE', '')
        if not accept_language:
            return []

        preferences = []

        # Parse Accept-Language header format: "en-US,en;q=0.9,fr;q=0.8"
        language_entries = accept_language.split(',')

        for entry in language_entries:
            entry = entry.strip()
            if not entry:
                continue

            # Extract language and quality factor
            if ';q=' in entry:
                language, quality_str = entry.split(';q=', 1)
                try:
                    quality = float(quality_str)
                except ValueError:
                    quality = 1.0
            else:
                language, quality = entry, 1.0

            language = language.strip()
            is_supported, best_match = self.validate_language_support(language)

            if is_supported:
                # Scale quality to our confidence system
                confidence = quality * 0.7  # Header preferences get max 0.7 confidence
                preferences.append(LanguagePreference(best_match, confidence, 'header'))

        return sorted(preferences, key=lambda p: p.confidence, reverse=True)

    def _infer_from_context(self, context: Dict[str, Any]) -> Optional[LanguagePreference]:
        """Infer language from additional context."""
        # This could be extended with timezone, IP geolocation, etc.
        timezone = context.get('timezone', '')
        user_agent = context.get('user_agent', '')

        # Simple timezone-based inference
        if timezone:
            if 'Europe/Paris' in timezone or 'Europe/Brussels' in timezone:
                return LanguagePreference('fr', 0.4, 'inferred')
            elif 'Europe/Madrid' in timezone or 'America/Mexico_City' in timezone:
                return LanguagePreference('es', 0.4, 'inferred')

        return None

    def _select_best_preference(self, preferences: List[LanguagePreference]) -> LanguagePreference:
        """Select the best preference from a list based on confidence and recency."""
        if not preferences:
            return LanguagePreference(self.default_language, 0.3, 'default')

        # Sort by confidence (descending) and recency
        sorted_prefs = sorted(
            preferences,
            key=lambda p: (p.confidence, p.timestamp.timestamp()),
            reverse=True
        )

        return sorted_prefs[0]

    def _store_user_preference(self, user: User, preference: LanguagePreference) -> None:
        """Store user's language preference in cache."""
        cache_key = f"{self.preference_cache_prefix}{user.id}"

        preference_data = {
            'language': preference.language,
            'confidence': preference.confidence,
            'source': preference.source,
            'timestamp': preference.timestamp.isoformat(),
            'last_updated': datetime.now().isoformat(),
        }

        cache.set(cache_key, preference_data, self.preference_cache_ttl)

        # Update user preference statistics
        self._update_user_stats(user, preference)

        logger.info(f"Stored language preference for user {user.id}: {preference.language}")

    def _update_user_stats(self, user: User, preference: LanguagePreference):
        """Update user's language preference statistics."""
        stats_key = f"{self.preference_cache_prefix}stats:{user.id}"
        stats = cache.get(stats_key, {
            'history': [],
            'confidence_trends': [],
            'last_updated': None,
        })

        # Add to history (keep last 10 entries)
        history_entry = {
            'language': preference.language,
            'confidence': preference.confidence,
            'source': preference.source,
            'timestamp': preference.timestamp.isoformat(),
        }

        stats['history'].append(history_entry)
        stats['history'] = stats['history'][-10:]  # Keep last 10

        # Update confidence trends
        stats['confidence_trends'].append({
            'timestamp': preference.timestamp.isoformat(),
            'confidence': preference.confidence,
        })
        stats['confidence_trends'] = stats['confidence_trends'][-20:]  # Keep last 20

        stats['last_updated'] = datetime.now().isoformat()

        cache.set(stats_key, stats, self.preference_cache_ttl)


# Singleton instance
_language_detector = None


def get_language_detector() -> SmartLanguageDetector:
    """Get singleton instance of SmartLanguageDetector."""
    global _language_detector
    if _language_detector is None:
        _language_detector = SmartLanguageDetector()
    return _language_detector


def detect_request_language(
    request,
    user: Optional[User] = None,
    explicit_language: Optional[str] = None
) -> str:
    """
    Convenience function to detect language from Django request.

    Args:
        request: Django request object
        user: Authenticated user (optional)
        explicit_language: Explicitly requested language (optional)

    Returns:
        Language code (guaranteed to be supported)
    """
    detector = get_language_detector()
    return detector.get_language_with_fallback(request, user, explicit_language)


def validate_language(language: str) -> Tuple[bool, str]:
    """
    Convenience function to validate language support.

    Args:
        language: Language code to validate

    Returns:
        Tuple of (is_supported, best_match_code)
    """
    detector = get_language_detector()
    return detector.validate_language_support(language)
"""
Multilingual explanation pipeline for financial analysis.
Orchestrates the entire process of generating explanations in multiple languages
with quality validation, cultural formatting, and caching.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from django.conf import settings
from django.core.cache import cache

from Analytics.services.local_llm_service import get_local_llm_service
from Analytics.services.cultural_formatter import get_cultural_formatter

logger = logging.getLogger(__name__)


class MultilingualExplanationPipeline:
    """Pipeline for generating and managing multilingual financial explanations."""

    def __init__(self):
        """Initialize the multilingual explanation pipeline."""
        self.llm_service = get_local_llm_service()
        self.cultural_formatter = get_cultural_formatter()

        # Configuration from settings
        self.enabled = getattr(settings, "MULTILINGUAL_LLM_ENABLED", True)
        self.supported_languages = getattr(settings, "LANGUAGES", [])
        self.default_language = getattr(settings, "DEFAULT_USER_LANGUAGE", "en")

        # Quality and validation settings
        self.quality_threshold = getattr(settings, "TRANSLATION_QUALITY_THRESHOLD", 0.8)
        self.validation_enabled = getattr(settings, "FINANCIAL_TERMINOLOGY_VALIDATION", True)

        # Cache settings
        self.cache_ttl = getattr(settings, "TRANSLATION_CACHE_TTL", 86400)  # 24 hours
        self.pipeline_cache_prefix = "multilingual_pipeline:"

    def generate_explanation(
        self,
        analysis_data: Dict[str, Any],
        target_language: str = "en",
        detail_level: str = "standard",
        explanation_type: str = "technical_analysis",
        user_preferences: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a complete multilingual explanation with quality validation.

        Args:
            analysis_data: Financial analysis data
            target_language: Target language code
            detail_level: Level of detail for explanation
            explanation_type: Type of explanation to generate
            user_preferences: User-specific formatting preferences

        Returns:
            Complete explanation result with quality metrics
        """
        if not self.enabled:
            logger.warning("Multilingual pipeline is disabled")
            return self._generate_fallback_explanation(analysis_data, detail_level, explanation_type)

        # Validate input
        if not self._validate_input(analysis_data, target_language):
            return None

        # Create pipeline cache key
        cache_key = self._create_pipeline_cache_key(
            analysis_data, target_language, detail_level, explanation_type, user_preferences
        )

        # Check cache first
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"Retrieved pipeline result from cache for {target_language}")
            return cached_result

        try:
            # Step 1: Generate base explanation
            base_explanation = self._generate_base_explanation(
                analysis_data, target_language, detail_level, explanation_type
            )

            if not base_explanation:
                return None

            # Step 2: Apply cultural formatting
            formatted_explanation = self._apply_cultural_formatting(
                base_explanation, target_language, user_preferences
            )

            # Step 3: Validate quality
            quality_metrics = self._validate_explanation_quality(
                formatted_explanation, target_language, analysis_data
            )

            # Step 4: Create final result
            final_result = self._create_final_result(
                formatted_explanation, quality_metrics, target_language
            )

            # Step 5: Cache result if quality is acceptable
            if quality_metrics.get("overall_score", 0) >= self.quality_threshold:
                cache.set(cache_key, final_result, self.cache_ttl)
            else:
                logger.warning(f"Quality too low to cache: {quality_metrics.get('overall_score', 0)}")

            return final_result

        except Exception as e:
            logger.error(f"Error in multilingual pipeline for {target_language}: {str(e)}")
            return self._generate_fallback_explanation(analysis_data, detail_level, explanation_type)

    def generate_batch_explanations(
        self,
        analysis_results: List[Dict[str, Any]],
        target_languages: List[str],
        detail_level: str = "standard",
        explanation_type: str = "technical_analysis",
    ) -> Dict[str, List[Optional[Dict[str, Any]]]]:
        """
        Generate explanations for multiple analysis results in multiple languages.

        Args:
            analysis_results: List of analysis data dictionaries
            target_languages: List of target language codes
            detail_level: Detail level for all explanations
            explanation_type: Type of explanation to generate

        Returns:
            Dictionary mapping language codes to lists of explanation results
        """
        results = {}

        for language in target_languages:
            language_results = []
            for analysis_data in analysis_results:
                explanation = self.generate_explanation(
                    analysis_data, language, detail_level, explanation_type
                )
                language_results.append(explanation)
            results[language] = language_results

        return results

    def validate_translation_quality(
        self,
        original_text: str,
        translated_text: str,
        target_language: str,
        financial_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate the quality of a translation for financial content.

        Args:
            original_text: Original text in source language
            translated_text: Translated text
            target_language: Target language code
            financial_context: Financial context for terminology validation

        Returns:
            Quality metrics dictionary
        """
        try:
            # Initialize quality metrics
            quality_metrics = {
                "terminology_score": 0.0,
                "completeness_score": 0.0,
                "cultural_appropriateness": 0.0,
                "overall_score": 0.0,
                "issues": [],
                "timestamp": datetime.now().isoformat(),
            }

            # Validate financial terminology
            if self.validation_enabled:
                terminology_score = self._validate_financial_terminology(
                    translated_text, target_language, financial_context
                )
                quality_metrics["terminology_score"] = terminology_score

            # Validate completeness (basic length and content preservation)
            completeness_score = self._validate_completeness(original_text, translated_text)
            quality_metrics["completeness_score"] = completeness_score

            # Validate cultural appropriateness (formatting, conventions)
            cultural_score = self._validate_cultural_appropriateness(translated_text, target_language)
            quality_metrics["cultural_appropriateness"] = cultural_score

            # Calculate overall score
            quality_metrics["overall_score"] = (
                quality_metrics["terminology_score"] * 0.4 +
                quality_metrics["completeness_score"] * 0.3 +
                quality_metrics["cultural_appropriateness"] * 0.3
            )

            return quality_metrics

        except Exception as e:
            logger.error(f"Error validating translation quality: {str(e)}")
            return {"overall_score": 0.0, "error": str(e)}

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and configuration."""
        return {
            "enabled": self.enabled,
            "supported_languages": [lang[0] for lang in self.supported_languages],
            "default_language": self.default_language,
            "llm_service_available": self.llm_service.is_available(),
            "cultural_formatter_enabled": self.cultural_formatter.enabled,
            "quality_threshold": self.quality_threshold,
            "validation_enabled": self.validation_enabled,
            "timestamp": datetime.now().isoformat(),
        }

    # Private methods

    def _validate_input(self, analysis_data: Dict[str, Any], target_language: str) -> bool:
        """Validate input parameters."""
        if not analysis_data:
            logger.error("Analysis data is empty")
            return False

        if target_language not in [lang[0] for lang in self.supported_languages]:
            logger.warning(f"Unsupported language: {target_language}")
            return False

        return True

    def _generate_base_explanation(
        self,
        analysis_data: Dict[str, Any],
        target_language: str,
        detail_level: str,
        explanation_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Generate the base explanation using the LLM service."""
        try:
            return self.llm_service.generate_multilingual_explanation(
                analysis_data, target_language, detail_level, explanation_type
            )
        except Exception as e:
            logger.error(f"Error generating base explanation: {str(e)}")
            return None

    def _apply_cultural_formatting(
        self,
        explanation: Dict[str, Any],
        target_language: str,
        user_preferences: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Apply cultural formatting to the explanation."""
        try:
            if not explanation or not explanation.get("explanation"):
                return explanation

            # Apply cultural formatting to the explanation text
            formatted_text = self.cultural_formatter.format_financial_text(
                explanation["explanation"], target_language
            )

            # Update the explanation
            explanation["explanation"] = formatted_text
            explanation["cultural_formatting_applied"] = True
            explanation["formatting_language"] = target_language

            # Apply user-specific preferences if provided
            if user_preferences:
                explanation = self._apply_user_preferences(explanation, user_preferences)

            return explanation

        except Exception as e:
            logger.error(f"Error applying cultural formatting: {str(e)}")
            return explanation

    def _validate_explanation_quality(
        self,
        explanation: Dict[str, Any],
        target_language: str,
        analysis_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate the quality of the generated explanation."""
        try:
            explanation_text = explanation.get("explanation", "")

            # Use the translation quality validator if available
            if explanation.get("generation_method") == "translated" and explanation.get("original_text"):
                return self.validate_translation_quality(
                    explanation.get("original_text", ""),
                    explanation_text,
                    target_language,
                    analysis_data,
                )

            # For native generation, use simpler quality metrics
            return self._validate_native_explanation_quality(explanation_text, target_language, analysis_data)

        except Exception as e:
            logger.error(f"Error validating explanation quality: {str(e)}")
            return {"overall_score": 0.5, "error": str(e)}

    def _validate_native_explanation_quality(
        self,
        explanation_text: str,
        target_language: str,
        analysis_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate quality for natively generated explanations."""
        quality_metrics = {
            "length_appropriate": len(explanation_text) > 50,  # Minimum length check
            "contains_financial_terms": self._contains_financial_terminology(explanation_text, target_language),
            "language_appropriate": target_language in explanation_text.lower() or target_language == "en",
            "overall_score": 0.0,
        }

        # Calculate overall score
        score_components = [
            1.0 if quality_metrics["length_appropriate"] else 0.0,
            0.8 if quality_metrics["contains_financial_terms"] else 0.3,
            1.0 if quality_metrics["language_appropriate"] else 0.5,
        ]

        quality_metrics["overall_score"] = sum(score_components) / len(score_components)
        return quality_metrics

    def _validate_financial_terminology(
        self,
        text: str,
        target_language: str,
        financial_context: Dict[str, Any],
    ) -> float:
        """Validate financial terminology usage."""
        if target_language == "en":
            return 1.0  # Assume English terminology is correct

        try:
            # Get expected financial terms for the language
            terminology_mapping = getattr(settings, "FINANCIAL_TERMINOLOGY_MAPPING", {})
            expected_terms = terminology_mapping.get(target_language, {})

            if not expected_terms:
                return 0.8  # Give benefit of doubt if no mapping available

            # Check if appropriate financial terms are used
            found_terms = 0
            total_terms = len(expected_terms)

            for english_term, foreign_term in expected_terms.items():
                if foreign_term.lower() in text.lower():
                    found_terms += 1

            return found_terms / total_terms if total_terms > 0 else 0.8

        except Exception as e:
            logger.error(f"Error validating financial terminology: {str(e)}")
            return 0.5

    def _validate_completeness(self, original_text: str, translated_text: str) -> float:
        """Validate translation completeness."""
        try:
            original_length = len(original_text)
            translated_length = len(translated_text)

            # Check if translation is within reasonable length bounds
            if translated_length < original_length * 0.5:
                return 0.3  # Too short
            elif translated_length > original_length * 2.0:
                return 0.7  # Too long
            else:
                return 1.0  # Appropriate length

        except Exception:
            return 0.5

    def _validate_cultural_appropriateness(self, text: str, target_language: str) -> float:
        """Validate cultural appropriateness of the text."""
        try:
            # Basic checks for cultural formatting
            score = 0.8  # Base score

            # Check for proper currency formatting
            if target_language in ["fr", "es"] and "€" in text:
                score += 0.1
            elif target_language == "en" and "$" in text:
                score += 0.1

            # Check for proper number formatting (basic)
            if target_language == "fr" and " " in text:  # French uses spaces for thousands
                score += 0.1

            return min(score, 1.0)

        except Exception:
            return 0.8

    def _contains_financial_terminology(self, text: str, target_language: str) -> bool:
        """Check if text contains appropriate financial terminology."""
        financial_keywords = {
            "en": ["stock", "price", "market", "analysis", "trading", "investment"],
            "fr": ["action", "prix", "marché", "analyse", "trading", "investissement"],
            "es": ["acción", "precio", "mercado", "análisis", "trading", "inversión"],
        }

        keywords = financial_keywords.get(target_language, financial_keywords["en"])
        text_lower = text.lower()

        return any(keyword in text_lower for keyword in keywords)

    def _apply_user_preferences(
        self,
        explanation: Dict[str, Any],
        user_preferences: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply user-specific formatting preferences."""
        try:
            # Apply user's preferred number formatting, currency display, etc.
            # This is a placeholder for user preference application
            explanation["user_preferences_applied"] = True
            return explanation
        except Exception as e:
            logger.error(f"Error applying user preferences: {str(e)}")
            return explanation

    def _create_final_result(
        self,
        explanation: Dict[str, Any],
        quality_metrics: Dict[str, Any],
        target_language: str,
    ) -> Dict[str, Any]:
        """Create the final pipeline result."""
        return {
            **explanation,
            "pipeline_version": "1.0",
            "quality_metrics": quality_metrics,
            "pipeline_timestamp": datetime.now().isoformat(),
            "target_language": target_language,
            "pipeline_status": "completed",
        }

    def _create_pipeline_cache_key(
        self,
        analysis_data: Dict[str, Any],
        target_language: str,
        detail_level: str,
        explanation_type: str,
        user_preferences: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create cache key for pipeline results."""
        symbol = analysis_data.get("symbol", "unknown")
        price = analysis_data.get("currentPrice", 0)

        base_key = f"{symbol}_{price}_{target_language}_{detail_level}_{explanation_type}"

        if user_preferences:
            pref_hash = hash(str(sorted(user_preferences.items())))
            base_key += f"_{pref_hash}"

        return f"{self.pipeline_cache_prefix}{base_key}"

    def _generate_fallback_explanation(
        self,
        analysis_data: Dict[str, Any],
        detail_level: str,
        explanation_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Generate fallback explanation when pipeline fails."""
        try:
            return self.llm_service.generate_explanation(analysis_data, detail_level, explanation_type)
        except Exception as e:
            logger.error(f"Error generating fallback explanation: {str(e)}")
            return None


# Singleton instance
_multilingual_pipeline = None


def get_multilingual_pipeline() -> MultilingualExplanationPipeline:
    """Get singleton instance of MultilingualExplanationPipeline."""
    global _multilingual_pipeline
    if _multilingual_pipeline is None:
        _multilingual_pipeline = MultilingualExplanationPipeline()
    return _multilingual_pipeline
"""
Comprehensive tests for Translation Service.
Tests translation functionality, quality scoring, fallback mechanisms, and multilingual support.
"""

import time
from unittest.mock import Mock, patch

from django.test import TestCase
from django.test.utils import override_settings

from Analytics.services.translation_service import (
    TranslationService,
    FinancialTerminologyMapper,
    TranslationQualityScorer,
    get_translation_service
)


class FinancialTerminologyMapperTestCase(TestCase):
    """Test cases for financial terminology mapping."""

    def setUp(self):
        self.mapper = FinancialTerminologyMapper()

    def test_terminology_mapping_completeness(self):
        """Test that terminology mappings are comprehensive."""
        # Verify mappings exist for both languages
        self.assertGreater(len(self.mapper.en_to_fr), 15)
        self.assertGreater(len(self.mapper.en_to_es), 15)
        
        # Test key financial terms are present
        key_terms = ["technical indicators", "buy recommendation", "moving average", "volatility"]
        for term in key_terms:
            self.assertIn(term, self.mapper.en_to_fr)
            self.assertIn(term, self.mapper.en_to_es)

    def test_get_terminology_context(self):
        """Test terminology context generation."""
        # Test French context
        fr_context = self.mapper.get_terminology_context("fr")
        self.assertIn("French", fr_context)
        self.assertIn("technical indicators", fr_context)
        self.assertIn("indicateurs techniques", fr_context)
        
        # Test Spanish context
        es_context = self.mapper.get_terminology_context("es")
        self.assertIn("Spanish", es_context)
        self.assertIn("technical indicators", es_context)
        self.assertIn("indicadores técnicos", es_context)
        
        # Test unsupported language
        other_context = self.mapper.get_terminology_context("de")
        self.assertIn("accurate financial terminology", other_context)


class TranslationQualityScorerTestCase(TestCase):
    """Test cases for translation quality scoring."""

    def setUp(self):
        self.scorer = TranslationQualityScorer()

    def test_quality_score_calculation(self):
        """Test translation quality score calculation."""
        original = "This stock shows strong technical indicators with a BUY recommendation."
        translated_good = "Cette action montre des indicateurs techniques forts avec une recommandation d'ACHAT."
        translated_poor = "This stock shows strong technical indicators with a BUY recommendation."  # No translation
        
        # Test good translation score
        good_score = self.scorer.calculate_translation_quality(original, translated_good, "fr")
        self.assertIsInstance(good_score, float)
        self.assertGreaterEqual(good_score, 0.0)
        self.assertLessEqual(good_score, 1.0)
        
        # Test poor translation score
        poor_score = self.scorer.calculate_translation_quality(original, translated_poor, "fr")
        self.assertLess(poor_score, good_score)

    def test_financial_terminology_check(self):
        """Test financial terminology preservation checking."""
        original = "The moving average shows a bullish signal."
        translated_good = "La moyenne mobile montre un signal haussier."
        translated_poor = "The moving thing shows a happy signal."
        
        good_score = self.scorer._check_financial_terminology(original, translated_good, "fr")
        poor_score = self.scorer._check_financial_terminology(original, translated_poor, "fr")
        
        self.assertGreater(good_score, poor_score)

    def test_numerical_preservation(self):
        """Test numerical values preservation."""
        original = "Stock price: $150.25, Volume: 1,234,567"
        preserved = "Prix de l'action: $150.25, Volume: 1,234,567"
        not_preserved = "Prix de l'action: $150, Volume: 1,234"
        
        preserved_score = self.scorer._check_numerical_preservation(original, preserved)
        not_preserved_score = self.scorer._check_numerical_preservation(original, not_preserved)
        
        self.assertGreater(preserved_score, not_preserved_score)


class TranslationServiceTestCase(TestCase):
    """Test cases for TranslationService functionality."""

    def setUp(self):
        self.service = TranslationService()

    def test_service_initialisation(self):
        """Test translation service initialisation."""
        self.assertIsInstance(self.service.terminology_mapper, FinancialTerminologyMapper)
        self.assertIsInstance(self.service.quality_scorer, TranslationQualityScorer)
        self.assertIn("fr", self.service.language_names)
        self.assertIn("es", self.service.language_names)

    def test_language_validation(self):
        """Test language parameter validation."""
        self.assertTrue(self.service._validate_language("fr"))
        self.assertTrue(self.service._validate_language("es"))
        self.assertFalse(self.service._validate_language("de"))
        self.assertFalse(self.service._validate_language(""))
        self.assertFalse(self.service._validate_language(None))

    def test_prompt_building(self):
        """Test translation prompt construction."""
        text = "Strong BUY recommendation based on technical indicators."
        context = {"symbol": "AAPL", "score": 7.5}
        
        prompt = self.service._build_translation_prompt(text, "fr", context)
        
        # Verify prompt contains key elements
        self.assertIn("financial translator", prompt)
        self.assertIn("French", prompt)
        self.assertIn("AAPL", prompt)
        self.assertIn("7.5", prompt)
        self.assertIn(text, prompt)
        self.assertIn("technical indicators", prompt)
        self.assertIn("indicateurs techniques", prompt)

    def test_cache_key_generation(self):
        """Test translation cache key generation."""
        text = "Test translation content"
        context = {"symbol": "AAPL", "score": 8.0}
        
        key1 = self.service._create_translation_cache_key(text, "fr", context)
        key2 = self.service._create_translation_cache_key(text, "fr", context)
        key3 = self.service._create_translation_cache_key(text, "es", context)
        
        # Same inputs should generate same key
        self.assertEqual(key1, key2)
        # Different language should generate different key
        self.assertNotEqual(key1, key3)
        # Keys should have proper prefix
        self.assertTrue(key1.startswith(self.service.cache_prefix))

    def test_fallback_translation_creation(self):
        """Test fallback translation creation."""
        english_text = "This is a test financial explanation."
        
        fallback = self.service._create_fallback_translation(english_text, "fr")
        
        # Verify fallback structure
        self.assertEqual(fallback["original_text"], english_text)
        self.assertIn(english_text, fallback["translated_text"])
        self.assertEqual(fallback["target_language"], "fr")
        self.assertEqual(fallback["translation_model"], "fallback")
        self.assertEqual(fallback["quality_score"], 0.3)
        self.assertTrue(fallback["fallback"])

    def test_language_detection(self):
        """Test simple language detection."""
        # Test English detection
        english_text = "This stock shows strong technical indicators with buy recommendation."
        self.assertEqual(self.service.detect_language(english_text), "en")
        
        # Test French detection
        french_text = "Cette action montre des indicateurs techniques forts avec une recommandation."
        detected = self.service.detect_language(french_text)
        # Language detection is heuristic, so we check for reasonable results
        self.assertIn(detected, ["fr", "en", "unknown"])
        
        # Test empty text
        self.assertEqual(self.service.detect_language(""), "unknown")
        self.assertEqual(self.service.detect_language(None), "unknown")

    @patch('Analytics.services.translation_service.OLLAMA_AVAILABLE', False)
    def test_translate_explanation_fallback_mode(self):
        """Test translation when Ollama is not available."""
        text = "Strong BUY recommendation based on technical analysis."
        
        result = self.service.translate_explanation(text, "fr")
        
        # Should return fallback translation
        self.assertIsNotNone(result)
        self.assertTrue(result["fallback"])
        self.assertEqual(result["translation_model"], "fallback")
        self.assertIn(text, result["translated_text"])

    @patch('Analytics.services.translation_service.get_local_llm_service')
    def test_translate_explanation_with_mock_llm(self, mock_get_service):
        """Test translation with mocked LLM service."""
        # Setup mock LLM service
        mock_llm_service = Mock()
        mock_llm_service.is_available.return_value = True
        mock_llm_service.client = Mock()
        mock_llm_service.translation_model = "qwen2:3b"
        
        # Mock successful translation response
        mock_response = {
            "response": "Cette action montre une recommandation d'ACHAT forte basée sur l'analyse technique."
        }
        mock_llm_service.client.generate.return_value = mock_response
        mock_get_service.return_value = mock_llm_service
        
        # Test translation
        text = "This stock shows strong BUY recommendation based on technical analysis."
        result = self.service.translate_explanation(text, "fr")
        
        # Verify successful translation
        self.assertIsNotNone(result)
        self.assertFalse(result.get("fallback", False))
        self.assertIn("Cette action", result["translated_text"])
        self.assertGreater(result["quality_score"], 0.3)

    @patch('Analytics.services.translation_service.cache')
    def test_caching_behaviour(self, mock_cache):
        """Test translation caching behaviour."""
        # Setup cache mock
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True
        
        text = "Technical analysis shows positive momentum."
        context = {"symbol": "AAPL"}
        
        # Call translation (will use fallback since no LLM)
        result = self.service.translate_explanation(text, "fr", context)
        
        # Verify cache was checked and set
        self.assertTrue(mock_cache.get.called)
        # Only cache non-fallback translations with good quality
        if not result.get("fallback", False) and result.get("quality_score", 0) > 0.6:
            self.assertTrue(mock_cache.set.called)

    def test_translation_options(self):
        """Test translation generation options."""
        options = self.service._get_translation_options("fr")
        
        # Verify translation-specific options
        self.assertEqual(options["temperature"], 0.2)  # Low for accuracy
        self.assertLessEqual(options["top_p"], 1.0)
        self.assertGreater(options["num_predict"], 100)
        self.assertIn("repeat_penalty", options)

    def test_input_validation(self):
        """Test input validation for translation requests."""
        # Test empty text
        result = self.service.translate_explanation("", "fr")
        self.assertIsNone(result)
        
        # Test None text
        result = self.service.translate_explanation(None, "fr")
        self.assertIsNone(result)
        
        # Test invalid language
        result = self.service.translate_explanation("Test text", "de")
        self.assertIsNone(result)
        
        # Test English to English (should return None)
        result = self.service.translate_explanation("Test text", "en")
        self.assertIsNone(result)


class TranslationServiceSingletonTestCase(TestCase):
    """Test cases for translation service singleton pattern."""

    def test_singleton_pattern(self):
        """Test that get_translation_service returns singleton instance."""
        service1 = get_translation_service()
        service2 = get_translation_service()
        
        # Should be the same instance
        self.assertIs(service1, service2)
        self.assertIsInstance(service1, TranslationService)

    @override_settings(EXPLAINABILITY_ENABLED=False)
    def test_service_when_disabled(self):
        """Test service behaviour when explainability is disabled."""
        # Clear singleton cache
        if hasattr(get_translation_service, '_instance'):
            delattr(get_translation_service, '_instance')
            
        service = get_translation_service()
        self.assertIsInstance(service, TranslationService)


class TranslationIntegrationTestCase(TestCase):
    """Integration tests for translation service with realistic scenarios."""

    def setUp(self):
        self.service = get_translation_service()

    def test_financial_explanation_translation_structure(self):
        """Test translation of typical financial explanation structure."""
        explanation = """
        AAPL receives a 7.8/10 technical analysis score, indicating a STRONG BUY recommendation.
        
        Key Technical Indicators:
        - Moving Average (SMA 50/200): Bullish crossover signal
        - RSI(14): 65.2 (moderate overbought territory)
        - MACD: Positive momentum with strengthening histogram
        - Bollinger Bands: Price trading near upper band
        - Volume: 15% above average with buying pressure
        
        Risk Assessment: Moderate risk due to current market volatility.
        Investment Recommendation: Consider position sizing and entry timing.
        """
        
        # Test translation (will use fallback without LLM)
        result = self.service.translate_explanation(explanation, "fr")
        
        # Verify result structure
        self.assertIsNotNone(result)
        self.assertIn("original_text", result)
        self.assertIn("translated_text", result)
        self.assertIn("quality_score", result)
        self.assertIn("target_language", result)
        
        # Verify essential content preservation
        translated = result["translated_text"]
        self.assertIn("AAPL", translated)
        self.assertIn("7.8", translated)
        self.assertIn("10", translated)

    def test_error_handling_robustness(self):
        """Test error handling in various failure scenarios."""
        # Test with malformed input
        malformed_inputs = [
            "",
            None,
            "   ",
            "\n\n\n",
            "123456789" * 1000,  # Very long text
        ]
        
        for malformed_input in malformed_inputs:
            result = self.service.translate_explanation(malformed_input, "fr")
            # Should handle gracefully
            if malformed_input and malformed_input.strip():
                self.assertIsNotNone(result)
            else:
                self.assertIsNone(result)

    def test_performance_characteristics(self):
        """Test performance characteristics of translation service."""
        text = "Strong technical indicators suggest a BUY recommendation for this equity."
        
        start_time = time.time()
        result = self.service.translate_explanation(text, "fr")
        end_time = time.time()
        
        # Verify reasonable performance (fallback should be very fast)
        duration = end_time - start_time
        self.assertLess(duration, 5.0)  # Should complete within 5 seconds
        
        if result:
            self.assertIn("generation_time", result)
            self.assertGreaterEqual(result["generation_time"], 0.0)


if __name__ == "__main__":
    import unittest
    unittest.main()
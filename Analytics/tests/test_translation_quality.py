import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from django.test import TestCase
from django.conf import settings
from Analytics.services.translation_service import TranslationService
from Analytics.services.local_llm_service import LocalLLMService

class TranslationQualityTestCase(TestCase):
    """Test suite for translation quality validation and assurance"""

    def setUp(self):
        self.translation_service = TranslationService()
        self.test_phrases = {
            'financial_terms': [
                "The stock price has increased by 15% over the past quarter.",
                "Market capitalization reached $2.5 billion with strong earnings.",
                "Technical analysis indicates a bullish trend with high volume.",
                "The P/E ratio suggests the stock is undervalued compared to peers.",
                "Dividend yield remains attractive for income-focused investors."
            ],
            'ui_elements': [
                "Welcome back to your dashboard",
                "Search for stocks by symbol or name",
                "Your recent analysis results",
                "Settings and preferences",
                "Quick actions and tools"
            ],
            'explanations': [
                "This company shows strong financial performance with consistent revenue growth.",
                "The technical indicators suggest a potential buying opportunity.",
                "Risk factors include market volatility and sector-specific challenges.",
                "The recommendation is based on fundamental and technical analysis.",
                "Consider diversification when making investment decisions."
            ]
        }

    def test_financial_terminology_accuracy(self):
        """Test that financial terms are translated accurately"""
        for phrase in self.test_phrases['financial_terms']:
            with self.subTest(phrase=phrase):
                # Test French translation
                french_translation = self.translation_service.translate_text(
                    phrase, 'fr', context='financial'
                )
                self.assertIsNotNone(french_translation)
                self.assertNotEqual(french_translation, phrase)

                # Test Spanish translation
                spanish_translation = self.translation_service.translate_text(
                    phrase, 'es', context='financial'
                )
                self.assertIsNotNone(spanish_translation)
                self.assertNotEqual(spanish_translation, phrase)

                # Verify key financial terms are present
                if 'stock' in phrase.lower():
                    self.assertTrue(
                        any(term in french_translation.lower()
                            for term in ['action', 'titre', 'valeur']),
                        f"French translation missing stock terminology: {french_translation}"
                    )
                    self.assertTrue(
                        any(term in spanish_translation.lower()
                            for term in ['acción', 'valor', 'título']),
                        f"Spanish translation missing stock terminology: {spanish_translation}"
                    )

    def test_translation_quality_scoring(self):
        """Test the quality scoring mechanism for translations"""
        test_text = "The company's quarterly earnings exceeded expectations with revenue growth of 12%."

        # Test with mock LLM service
        with patch.object(LocalLLMService, 'generate_text') as mock_llm:
            mock_llm.return_value = "Les bénéfices trimestriels de l'entreprise ont dépassé les attentes avec une croissance des revenus de 12%."

            result = self.translation_service.translate_text(
                test_text, 'fr', context='financial'
            )

            # Verify quality assessment
            self.assertIsInstance(result, str)
            self.assertNotEqual(result, test_text)

            # Check cache entry includes quality score
            cache_key = f"translation:fr:{hash(test_text + 'financial')}"
            cached_result = self.translation_service.cache_manager.get(cache_key)
            if cached_result:
                self.assertIn('quality_score', str(cached_result))

    def test_terminology_consistency(self):
        """Test that financial terminology remains consistent across translations"""
        terminology_tests = {
            'stock price': {'fr': 'prix de l\'action', 'es': 'precio de la acción'},
            'market cap': {'fr': 'capitalisation boursière', 'es': 'capitalización de mercado'},
            'P/E ratio': {'fr': 'ratio P/E', 'es': 'ratio P/E'},
            'dividend': {'fr': 'dividende', 'es': 'dividendo'},
            'earnings': {'fr': 'bénéfices', 'es': 'ganancias'}
        }

        for term, expected_translations in terminology_tests.items():
            phrase = f"The {term} is important for analysis."

            for language, expected_term in expected_translations.items():
                translation = self.translation_service.translate_text(
                    phrase, language, context='financial'
                )

                # Check if expected terminology is present (case-insensitive)
                translation_lower = translation.lower()
                expected_term_lower = expected_term.lower()

                self.assertTrue(
                    expected_term_lower in translation_lower or
                    any(word in translation_lower for word in expected_term_lower.split()),
                    f"Expected '{expected_term}' terminology not found in {language} translation: {translation}"
                )

    def test_fallback_translation_quality(self):
        """Test quality of fallback translations when LLM is unavailable"""
        test_phrase = "Stock analysis shows bullish trends."

        # Mock LLM failure
        with patch.object(LocalLLMService, 'generate_text', side_effect=Exception("LLM unavailable")):
            # Test enhanced fallback
            french_fallback = self.translation_service._create_enhanced_fallback_translation(
                test_phrase, 'fr', 'financial'
            )

            self.assertIsNotNone(french_fallback)
            self.assertNotEqual(french_fallback, test_phrase)

            # Verify financial terms are mapped
            self.assertTrue(
                any(term in french_fallback.lower()
                    for term in ['action', 'analyse', 'tendance']),
                f"Fallback translation lacks financial terminology: {french_fallback}"
            )

    def test_translation_completeness(self):
        """Test that all UI elements have complete translations"""
        required_translation_keys = [
            'navigation.home',
            'navigation.search',
            'navigation.settings',
            'dashboard.welcome',
            'dashboard.quickActions',
            'auth.signIn',
            'auth.username',
            'auth.password',
            'recommendations.buy',
            'recommendations.hold',
            'recommendations.sell'
        ]

        for language in ['fr', 'es']:
            for key in required_translation_keys:
                with self.subTest(language=language, key=key):
                    # Load translation file
                    translation_file = f"Design/frontend/src/i18n/locales/{language}/common.json"
                    try:
                        with open(translation_file, 'r', encoding='utf-8') as f:
                            translations = json.load(f)

                        # Navigate to nested key
                        value = translations
                        for part in key.split('.'):
                            self.assertIn(part, value, f"Missing key '{key}' in {language} translations")
                            value = value[part]

                        # Verify translation is not empty and different from key
                        self.assertIsInstance(value, str)
                        self.assertTrue(len(value.strip()) > 0)
                        self.assertNotEqual(value, key)

                    except FileNotFoundError:
                        self.fail(f"Translation file not found: {translation_file}")
                    except json.JSONDecodeError:
                        self.fail(f"Invalid JSON in translation file: {translation_file}")

    def test_number_formatting_accuracy(self):
        """Test locale-specific number formatting accuracy"""
        test_numbers = [1234.56, 1000000, 0.025, 9876543.21]

        for number in test_numbers:
            with self.subTest(number=number):
                # Test French formatting (comma decimal, space thousands)
                french_formatted = self.translation_service._format_number_for_locale(number, 'fr')
                if '.' in str(number):
                    self.assertIn(',', french_formatted, f"French number {number} should use comma decimal: {french_formatted}")

                if number >= 1000:
                    self.assertTrue(
                        ' ' in french_formatted or len(french_formatted) <= 7,
                        f"French number {number} should use space thousands separator: {french_formatted}"
                    )

                # Test Spanish formatting (comma decimal, period thousands)
                spanish_formatted = self.translation_service._format_number_for_locale(number, 'es')
                if '.' in str(number):
                    self.assertIn(',', spanish_formatted, f"Spanish number {number} should use comma decimal: {spanish_formatted}")

    def test_date_formatting_localization(self):
        """Test that dates are formatted according to locale conventions"""
        from datetime import datetime
        test_date = datetime(2024, 3, 15, 14, 30)

        # Test French date format (DD/MM/YYYY)
        french_date = self.translation_service._format_date_for_locale(test_date, 'fr')
        self.assertTrue(french_date.startswith('15/03/'), f"French date should be DD/MM format: {french_date}")

        # Test Spanish date format (DD/MM/YYYY)
        spanish_date = self.translation_service._format_date_for_locale(test_date, 'es')
        self.assertTrue(spanish_date.startswith('15/03/'), f"Spanish date should be DD/MM format: {spanish_date}")

        # Test English date format (MM/DD/YYYY)
        english_date = self.translation_service._format_date_for_locale(test_date, 'en')
        self.assertTrue(english_date.startswith('03/15/'), f"English date should be MM/DD format: {english_date}")

    def test_translation_caching_efficiency(self):
        """Test that translation caching works efficiently"""
        test_text = "Quick test for caching efficiency"

        # First translation should hit LLM
        with patch.object(LocalLLMService, 'generate_text') as mock_llm:
            mock_llm.return_value = "Test rapide pour l'efficacité du cache"

            result1 = self.translation_service.translate_text(test_text, 'fr')
            self.assertEqual(mock_llm.call_count, 1)

            # Second identical translation should use cache
            result2 = self.translation_service.translate_text(test_text, 'fr')
            self.assertEqual(mock_llm.call_count, 1)  # Should not increase
            self.assertEqual(result1, result2)

    def test_context_aware_translation(self):
        """Test that translations adapt based on context"""
        # Test financial context
        financial_text = "The stock analysis shows positive indicators."
        financial_translation = self.translation_service.translate_text(
            financial_text, 'fr', context='financial'
        )

        # Test general context
        general_translation = self.translation_service.translate_text(
            financial_text, 'fr', context='general'
        )

        # Context should influence terminology choices
        self.assertIsNotNone(financial_translation)
        self.assertIsNotNone(general_translation)
        # Note: Actual difference validation would require more sophisticated analysis

    def test_error_handling_robustness(self):
        """Test translation service handles errors gracefully"""
        # Test with invalid input
        result = self.translation_service.translate_text(None, 'fr')
        self.assertEqual(result, "")

        result = self.translation_service.translate_text("", 'fr')
        self.assertEqual(result, "")

        # Test with unsupported language
        result = self.translation_service.translate_text("Hello", 'de')
        self.assertEqual(result, "Hello")  # Should return original

        # Test with extremely long text
        long_text = "This is a test. " * 1000
        result = self.translation_service.translate_text(long_text, 'fr')
        self.assertIsNotNone(result)
        self.assertTrue(len(result) > 0)

    @pytest.mark.performance
    def test_translation_performance_benchmarks(self):
        """Benchmark translation performance under various conditions"""
        import time

        test_cases = [
            ("Short phrase", "Hello world"),
            ("Medium text", "This is a medium length text for testing translation performance with financial terms like stock price and market analysis."),
            ("Long text", "This is a much longer text that contains multiple sentences and various financial terminology including stock prices, market capitalization, earnings reports, technical analysis, and investment recommendations. " * 5)
        ]

        performance_results = {}

        for case_name, text in test_cases:
            start_time = time.time()

            result = self.translation_service.translate_text(text, 'fr', context='financial')

            end_time = time.time()
            translation_time = end_time - start_time

            performance_results[case_name] = {
                'time': translation_time,
                'length': len(text),
                'result_length': len(result) if result else 0
            }

            # Performance assertions
            if case_name == "Short phrase":
                self.assertLess(translation_time, 2.0, f"Short phrase translation took too long: {translation_time}s")
            elif case_name == "Medium text":
                self.assertLess(translation_time, 5.0, f"Medium text translation took too long: {translation_time}s")
            elif case_name == "Long text":
                self.assertLess(translation_time, 10.0, f"Long text translation took too long: {translation_time}s")

        # Log performance results for analysis
        print(f"Translation Performance Results: {performance_results}")
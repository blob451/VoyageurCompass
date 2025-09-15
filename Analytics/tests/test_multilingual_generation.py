"""
Phase 3.2 Multilingual Generation Testing

Comprehensive validation of translation quality for French and Spanish financial explanations
including terminology preservation, cultural context adaptation, and accuracy consistency.
"""

import logging
import statistics
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase, TransactionTestCase
from django.utils import timezone

from Analytics.engine.ta_engine import TechnicalAnalysisEngine
from Analytics.services.local_llm_service import get_local_llm_service
from Analytics.services.translation_service import TranslationService, FinancialTerminologyMapper
from Analytics.prompts.multilingual_templates import MultilingualPromptTemplates
from Data.models import AnalyticsResults, DataIndustry, DataSector, Stock, StockPrice

logger = logging.getLogger(__name__)
User = get_user_model()


class MultilingualGenerationTestCase(TransactionTestCase):
    """Phase 3.2 multilingual generation and translation quality testing."""

    def setUp(self):
        """Set up comprehensive test data and services."""
        cache.clear()

        # Create test user
        self.user = User.objects.create_user(
            username="multilingual_user", 
            email="multilingual@test.com", 
            password="testpass123"
        )

        # Create sector and industry
        self.sector = DataSector.objects.create(
            sectorKey="tech_multi", 
            sectorName="Technology Multilingual", 
            data_source="test"
        )
        
        self.industry = DataIndustry.objects.create(
            industryKey="software_multi", 
            industryName="Software Multilingual",
            sector=self.sector,
            data_source="test"
        )

        # Create test stock
        self.stock = Stock.objects.create(
            symbol="MULTI",
            short_name="Multilingual Test Company",
            sector=self.sector.sectorName,
            industry=self.industry.industryName,
            market_cap=2000000000,
            shares_outstanding=200000000,
            is_active=True
        )

        # Create stock prices for analysis
        self.create_test_price_data()

        # Initialise services
        self.llm_service = get_local_llm_service()
        self.translation_service = TranslationService()
        self.terminology_mapper = FinancialTerminologyMapper()
        self.prompt_templates = MultilingualPromptTemplates()

    def create_test_price_data(self):
        """Create comprehensive test price data for multilingual analysis."""
        base_price = Decimal('150.00')
        base_date = datetime.now().date() - timedelta(days=90)
        
        prices = []
        for i in range(90):
            date = base_date + timedelta(days=i)
            # Create realistic bullish trend
            price_change = Decimal(str(0.8 * (i % 7 - 3)))  # Weekly oscillation
            current_price = base_price + price_change + Decimal(str(i * 0.15))
            
            prices.append(StockPrice(
                stock=self.stock,
                date=date,
                open_price=current_price - Decimal('0.75'),
                high_price=current_price + Decimal('1.25'),
                low_price=current_price - Decimal('1.75'),
                close_price=current_price,
                volume=1500000 + i * 15000,
                data_source="test"
            ))
        
        StockPrice.objects.bulk_create(prices)

    def create_test_analysis_data(self, scenario: str = "bullish") -> Dict[str, Any]:
        """Create test analysis data for different market scenarios."""
        base_data = {
            'symbol': 'MULTI',
            'analysis_date': datetime.now().isoformat(),
        }
        
        if scenario == "bullish":
            base_data.update({
                'technical_score': 8.2,
                'recommendation': 'BUY',
                'indicators': {
                    'rsi': {
                        'value': 62.4, 'signal': 'bullish', 
                        'interpretation': 'RSI indicates strong upward momentum with room for growth'
                    },
                    'macd': {
                        'value': 0.45, 'signal': 'bullish', 
                        'interpretation': 'MACD shows strong bullish crossover with increasing momentum'
                    },
                    'sma_50': {
                        'value': 148.2, 'signal': 'bullish', 
                        'interpretation': 'Price trading well above 50-day moving average'
                    },
                    'sma_200': {
                        'value': 142.8, 'signal': 'bullish', 
                        'interpretation': 'Strong uptrend confirmed by price above 200-day moving average'
                    },
                    'bollinger_bands': {
                        'upper': 155.5, 'lower': 145.2, 'position': 0.72,
                        'signal': 'bullish', 
                        'interpretation': 'Price positioned in upper Bollinger Band indicating strong momentum'
                    },
                    'volume_trend': {
                        'value': 1.35, 'signal': 'bullish', 
                        'interpretation': 'Volume 35% above average supporting price movement'
                    }
                }
            })
        elif scenario == "bearish":
            base_data.update({
                'technical_score': 3.1,
                'recommendation': 'SELL',
                'indicators': {
                    'rsi': {
                        'value': 28.5, 'signal': 'bearish', 
                        'interpretation': 'RSI indicates oversold conditions with downward pressure'
                    },
                    'macd': {
                        'value': -0.32, 'signal': 'bearish', 
                        'interpretation': 'MACD shows bearish crossover with accelerating decline'
                    },
                    'sma_50': {
                        'value': 152.1, 'signal': 'bearish', 
                        'interpretation': 'Price trading below 50-day moving average'
                    },
                    'sma_200': {
                        'value': 155.3, 'signal': 'bearish', 
                        'interpretation': 'Downtrend confirmed by price below 200-day moving average'
                    }
                }
            })
        else:  # neutral
            base_data.update({
                'technical_score': 5.5,
                'recommendation': 'HOLD',
                'indicators': {
                    'rsi': {
                        'value': 48.2, 'signal': 'neutral', 
                        'interpretation': 'RSI indicates balanced momentum with no clear direction'
                    },
                    'macd': {
                        'value': 0.02, 'signal': 'neutral', 
                        'interpretation': 'MACD shows minimal momentum with sideways movement'
                    }
                }
            })
        
        return base_data

    def test_french_translation_quality(self):
        """Test French translation quality and financial terminology preservation."""
        analysis_data = self.create_test_analysis_data("bullish")
        
        # Generate English explanation
        english_result = self.llm_service.generate_explanation(
            analysis_data=analysis_data,
            detail_level="standard"
        )
        
        self.assertIsNotNone(english_result)
        self.assertIn('explanation', english_result)
        
        # Translate to French
        french_result = self.translation_service.translate_explanation(
            explanation=english_result['explanation'],
            target_language="fr",
            analysis_data=analysis_data
        )
        
        self.assertIsNotNone(french_result)
        self.assertIn('translated_explanation', french_result)
        self.assertEqual(french_result['target_language'], 'fr')
        
        # Verify French content quality
        french_explanation = french_result['translated_explanation']
        
        # Check for key French financial terminology
        french_terms = [
            'ACHAT', 'recommandation', 'analyse', 'momentum', 
            'haussier', 'technique', 'indicateurs'
        ]
        
        found_terms = 0
        for term in french_terms:
            if term.lower() in french_explanation.lower():
                found_terms += 1
        
        # Should find at least 3 key financial terms
        self.assertGreaterEqual(found_terms, 3, 
                               f"Expected at least 3 French financial terms, found {found_terms}")
        
        # Verify translation quality score
        if 'quality_score' in french_result:
            self.assertGreaterEqual(french_result['quality_score'], 0.85,
                                   "Translation quality score should be ≥85%")

    def test_spanish_translation_quality(self):
        """Test Spanish translation quality and financial terminology preservation."""
        analysis_data = self.create_test_analysis_data("bearish")
        
        # Generate English explanation
        english_result = self.llm_service.generate_explanation(
            analysis_data=analysis_data,
            detail_level="standard"
        )
        
        self.assertIsNotNone(english_result)
        self.assertIn('explanation', english_result)
        
        # Translate to Spanish
        spanish_result = self.translation_service.translate_explanation(
            explanation=english_result['explanation'],
            target_language="es",
            analysis_data=analysis_data
        )
        
        self.assertIsNotNone(spanish_result)
        self.assertIn('translated_explanation', spanish_result)
        self.assertEqual(spanish_result['target_language'], 'es')
        
        # Verify Spanish content quality
        spanish_explanation = spanish_result['translated_explanation']
        
        # Check for key Spanish financial terminology
        spanish_terms = [
            'VENTA', 'recomendación', 'análisis', 'momentum', 
            'bajista', 'técnico', 'indicadores'
        ]
        
        found_terms = 0
        for term in spanish_terms:
            if term.lower() in spanish_explanation.lower():
                found_terms += 1
        
        # Should find at least 3 key financial terms
        self.assertGreaterEqual(found_terms, 3, 
                               f"Expected at least 3 Spanish financial terms, found {found_terms}")
        
        # Verify translation quality score
        if 'quality_score' in spanish_result:
            self.assertGreaterEqual(spanish_result['quality_score'], 0.85,
                                   "Translation quality score should be ≥85%")

    def test_financial_terminology_mapping(self):
        """Test comprehensive financial terminology mapping accuracy."""
        terminology_mapper = FinancialTerminologyMapper()
        
        # Test French terminology context
        french_context = terminology_mapper.get_terminology_context("fr")
        self.assertIn("technical indicators", french_context)
        self.assertIn("indicateurs techniques", french_context)
        self.assertIn("buy recommendation", french_context)
        self.assertIn("recommandation d'achat", french_context)
        
        # Test Spanish terminology context
        spanish_context = terminology_mapper.get_terminology_context("es")
        self.assertIn("technical indicators", spanish_context)
        self.assertIn("indicadores técnicos", spanish_context)
        self.assertIn("sell recommendation", spanish_context)
        self.assertIn("recomendación de venta", spanish_context)
        
        # Verify terminology coverage
        self.assertGreaterEqual(len(terminology_mapper.en_to_fr), 20,
                               "French terminology should have at least 20 mappings")
        self.assertGreaterEqual(len(terminology_mapper.en_to_es), 20,
                               "Spanish terminology should have at least 20 mappings")

    def test_number_formatting_localisation(self):
        """Test number and currency formatting for different locales."""
        prompt_templates = MultilingualPromptTemplates()
        
        # Test French formatting
        french_formatting = prompt_templates.number_formatting["fr"]
        self.assertEqual(french_formatting["decimal_separator"], ",")
        self.assertEqual(french_formatting["thousands_separator"], " ")
        self.assertEqual(french_formatting["currency_symbol"], "€")
        self.assertEqual(french_formatting["currency_position"], "after")
        
        # Test Spanish formatting
        spanish_formatting = prompt_templates.number_formatting["es"]
        self.assertEqual(spanish_formatting["decimal_separator"], ",")
        self.assertEqual(spanish_formatting["thousands_separator"], ".")
        self.assertEqual(spanish_formatting["currency_symbol"], "€")
        self.assertEqual(spanish_formatting["currency_position"], "after")

    def test_translation_consistency_across_detail_levels(self):
        """Test translation consistency across different explanation detail levels."""
        analysis_data = self.create_test_analysis_data("neutral")
        
        translations = {}
        detail_levels = ['summary', 'standard', 'detailed']
        
        for detail_level in detail_levels:
            try:
                # Generate English explanation
                english_result = self.llm_service.generate_explanation(
                    analysis_data=analysis_data,
                    detail_level=detail_level
                )
                
                if english_result and 'explanation' in english_result:
                    # Translate to French
                    french_result = self.translation_service.translate_explanation(
                        explanation=english_result['explanation'],
                        target_language="fr",
                        analysis_data=analysis_data
                    )
                    
                    if french_result and 'translated_explanation' in french_result:
                        translations[detail_level] = {
                            'english': english_result['explanation'],
                            'french': french_result['translated_explanation'],
                            'quality': french_result.get('quality_score', 0)
                        }
                        
            except Exception as e:
                logger.warning(f"Translation failed for {detail_level}: {str(e)}")
        
        # Verify we got at least 2 translations
        self.assertGreaterEqual(len(translations), 2,
                               "Should have successful translations for at least 2 detail levels")
        
        # Verify consistency in recommendation translation
        recommendations_found = 0
        for level, data in translations.items():
            french_text = data['french'].upper()
            if any(rec in french_text for rec in ['ACHAT', 'VENTE', 'CONSERVATION']):
                recommendations_found += 1
        
        # All translations should contain consistent recommendation terminology
        self.assertEqual(recommendations_found, len(translations),
                        "All translations should contain consistent recommendation terminology")

    def test_translation_performance_benchmarks(self):
        """Test translation performance meets benchmark requirements."""
        analysis_data = self.create_test_analysis_data("bullish")
        
        # Generate base English explanation
        english_result = self.llm_service.generate_explanation(
            analysis_data=analysis_data,
            detail_level="standard"
        )
        
        self.assertIsNotNone(english_result)
        self.assertIn('explanation', english_result)
        
        # Test French translation performance
        start_time = time.time()
        french_result = self.translation_service.translate_explanation(
            explanation=english_result['explanation'],
            target_language="fr",
            analysis_data=analysis_data
        )
        french_translation_time = time.time() - start_time
        
        # Test Spanish translation performance
        start_time = time.time()
        spanish_result = self.translation_service.translate_explanation(
            explanation=english_result['explanation'],
            target_language="es",
            analysis_data=analysis_data
        )
        spanish_translation_time = time.time() - start_time
        
        # Verify performance benchmarks (translation should add 2-3s max)
        self.assertLess(french_translation_time, 5.0,
                       f"French translation took {french_translation_time:.2f}s, should be <5s")
        self.assertLess(spanish_translation_time, 5.0,
                       f"Spanish translation took {spanish_translation_time:.2f}s, should be <5s")
        
        # Verify translation results
        self.assertIsNotNone(french_result)
        self.assertIsNotNone(spanish_result)

    def test_cultural_context_adaptation(self):
        """Test cultural context adaptation in financial explanations."""
        analysis_data = self.create_test_analysis_data("bullish")
        
        # Generate and translate explanation
        english_result = self.llm_service.generate_explanation(
            analysis_data=analysis_data,
            detail_level="detailed"
        )
        
        if english_result and 'explanation' in english_result:
            # Test French cultural adaptation
            french_result = self.translation_service.translate_explanation(
                explanation=english_result['explanation'],
                target_language="fr",
                analysis_data=analysis_data
            )
            
            if french_result and 'translated_explanation' in french_result:
                french_text = french_result['translated_explanation']
                
                # Check for appropriate formal tone in French
                # French financial language tends to be more formal
                formal_indicators = [
                    'analyse', 'évaluation', 'recommandation', 
                    'investissement', 'performance'
                ]
                
                formal_count = sum(1 for indicator in formal_indicators 
                                 if indicator in french_text.lower())
                
                self.assertGreaterEqual(formal_count, 2,
                                       "French translation should use formal financial language")

    def test_multilingual_fallback_mechanisms(self):
        """Test fallback mechanisms when translation fails."""
        analysis_data = self.create_test_analysis_data("neutral")
        
        # Test with invalid language code
        try:
            result = self.translation_service.translate_explanation(
                explanation="Test explanation for fallback testing.",
                target_language="invalid_lang",
                analysis_data=analysis_data
            )
            
            # Should either return None or original explanation
            if result:
                self.assertIn('error', result or {},
                             "Invalid language should return error information")
                
        except Exception as e:
            # Exception is acceptable for invalid language
            self.assertIn('language', str(e).lower(),
                         "Exception should mention language issue")

    def test_translation_caching_functionality(self):
        """Test translation caching for performance optimisation."""
        analysis_data = self.create_test_analysis_data("bullish")
        
        # Generate base explanation
        english_result = self.llm_service.generate_explanation(
            analysis_data=analysis_data,
            detail_level="standard"
        )
        
        if english_result and 'explanation' in english_result:
            explanation = english_result['explanation']
            
            # First translation (should be cached)
            start_time = time.time()
            first_result = self.translation_service.translate_explanation(
                explanation=explanation,
                target_language="fr",
                analysis_data=analysis_data
            )
            first_translation_time = time.time() - start_time
            
            # Second translation (should use cache)
            start_time = time.time()
            second_result = self.translation_service.translate_explanation(
                explanation=explanation,
                target_language="fr",
                analysis_data=analysis_data
            )
            second_translation_time = time.time() - start_time
            
            # Verify caching improves performance
            if first_result and second_result:
                self.assertEqual(
                    first_result.get('translated_explanation'),
                    second_result.get('translated_explanation'),
                    "Cached translation should match original"
                )
                
                # Second request should be significantly faster
                if second_translation_time > 0.1:  # Only test if measurable
                    cache_improvement = (first_translation_time - second_translation_time) / first_translation_time
                    self.assertGreater(cache_improvement, 0.1,
                                     f"Cache should improve performance by >10%, got {cache_improvement:.1%}")
#!/usr/bin/env python
"""
Multilingual Generation Testing Script for Phase 2

This script tests the Phase 2 implementation of direct multilingual generation
for detailed mode, specifically testing:
- French and Spanish generation with llama3.1:8b
- Financial terminology accuracy
- Recommendation consistency across languages
- Performance and reliability

Usage:
    python scripts/test_multilingual_generation.py [--symbols AAPL,MSFT] [--detail-levels detailed] [--languages fr,es]
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse

# Add Django setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'VoyageurCompass.settings')

import django
django.setup()

from django.core.cache import cache
from django.test import RequestFactory
from django.contrib.auth.models import User, AnonymousUser

from Analytics.services.local_llm_service import LocalLLMService


class MultilingualGenerationTester:
    """Comprehensive testing framework for multilingual generation in Phase 2."""

    def __init__(self, symbols: List[str] = None, detail_levels: List[str] = None, languages: List[str] = None):
        self.symbols = symbols or ['AAPL', 'MSFT', 'GOOGL']
        self.detail_levels = detail_levels or ['summary', 'standard', 'detailed']
        self.languages = languages or ['en', 'fr', 'es']

        self.llm_service = LocalLLMService()
        self.request_factory = RequestFactory()

        # Test results storage
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'phase': 'Phase 2 - Direct Multilingual Generation',
            'test_config': {
                'symbols': self.symbols,
                'detail_levels': self.detail_levels,
                'languages': self.languages,
            },
            'results': {}
        }

        print("=== Phase 2 Multilingual Generation Testing ===")
        print(f"Testing symbols: {', '.join(self.symbols)}")
        print(f"Detail levels: {', '.join(self.detail_levels)}")
        print(f"Languages: {', '.join(self.languages)}")
        print("-" * 60)

    def run_all_tests(self):
        """Execute all Phase 2 multilingual tests."""
        try:
            print("Starting Phase 2 multilingual generation tests...")

            # Test 1: Basic generation in all languages and detail levels
            self._test_basic_generation()

            # Test 2: Financial terminology accuracy
            self._test_financial_terminology()

            # Test 3: Recommendation consistency
            self._test_recommendation_consistency()

            # Test 4: Direct generation vs translation mode
            self._test_direct_vs_translation()

            # Test 5: Performance validation
            self._test_performance()

            # Generate summary report
            self._generate_test_report()

        except Exception as e:
            print(f"ERROR: Test suite failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def _test_basic_generation(self):
        """Test basic multilingual generation across all configurations."""
        print("Test 1: Basic multilingual generation...")

        test_data = {
            'symbol': 'AAPL',
            'score_0_10': 7.2,
            'currentPrice': 150.00,
            'weighted_scores': {
                'w_rsi14': 0.12,
                'w_sma_crossover': 0.08,
                'w_macd_histogram': -0.05
            }
        }

        results = {}

        for language in self.languages:
            results[language] = {}
            for detail_level in self.detail_levels:
                print(f"  Testing {language}/{detail_level}...")

                start_time = time.perf_counter()
                try:
                    if language == 'en':
                        result = self.llm_service.generate_explanation(
                            test_data, detail_level, "technical_analysis", language
                        )
                    else:
                        result = self.llm_service.generate_multilingual_explanation(
                            test_data, language, detail_level, "technical_analysis"
                        )

                    end_time = time.perf_counter()
                    generation_time = (end_time - start_time) * 1000

                    if result and result.get('content') or result.get('explanation'):
                        content = result.get('content') or result.get('explanation')
                        results[language][detail_level] = {
                            'success': True,
                            'content_length': len(content),
                            'generation_time_ms': round(generation_time, 2),
                            'content_preview': content[:100] + "..." if len(content) > 100 else content,
                            'generation_method': result.get('generation_method', 'unknown'),
                            'model_used': result.get('model_used', 'unknown')
                        }
                        print(f"    ✅ Success ({generation_time:.1f}ms)")
                    else:
                        results[language][detail_level] = {
                            'success': False,
                            'error': 'No content generated',
                            'generation_time_ms': round(generation_time, 2)
                        }
                        print(f"    ❌ Failed - No content")

                except Exception as e:
                    end_time = time.perf_counter()
                    generation_time = (end_time - start_time) * 1000
                    results[language][detail_level] = {
                        'success': False,
                        'error': str(e),
                        'generation_time_ms': round(generation_time, 2)
                    }
                    print(f"    ❌ Failed - {str(e)}")

        self.test_results['results']['basic_generation'] = results

    def _test_financial_terminology(self):
        """Test accuracy of financial terminology in different languages."""
        print("\\nTest 2: Financial terminology accuracy...")

        test_data = {
            'symbol': 'MSFT',
            'score_0_10': 6.8,
            'currentPrice': 320.00,
            'weighted_scores': {
                'w_rsi14': 0.15,
                'w_bollinger_position': -0.08,
                'w_volume_trend': 0.06
            }
        }

        terminology_tests = {
            'fr': {
                'expected_terms': ['CONSERVATION', 'analyse', 'indicateurs', 'technique', 'investissement', 'marché'],
                'forbidden_terms': ['BUY', 'SELL', 'HOLD']  # Should be translated
            },
            'es': {
                'expected_terms': ['MANTENER', 'análisis', 'indicadores', 'técnico', 'inversión', 'mercado'],
                'forbidden_terms': ['BUY', 'SELL', 'HOLD']  # Should be translated
            }
        }

        results = {}

        for language, terminology in terminology_tests.items():
            print(f"  Testing {language} terminology...")

            try:
                result = self.llm_service.generate_multilingual_explanation(
                    test_data, language, 'detailed', 'technical_analysis'
                )

                if result and (result.get('content') or result.get('explanation')):
                    content = result.get('content') or result.get('explanation')
                    content_lower = content.lower()

                    # Check for expected terms
                    found_expected = []
                    missing_expected = []
                    for term in terminology['expected_terms']:
                        if term.lower() in content_lower:
                            found_expected.append(term)
                        else:
                            missing_expected.append(term)

                    # Check for forbidden terms
                    found_forbidden = []
                    for term in terminology['forbidden_terms']:
                        if term in content:  # Case sensitive for exact English terms
                            found_forbidden.append(term)

                    accuracy_score = len(found_expected) / len(terminology['expected_terms'])

                    results[language] = {
                        'success': True,
                        'accuracy_score': round(accuracy_score, 2),
                        'found_expected': found_expected,
                        'missing_expected': missing_expected,
                        'found_forbidden': found_forbidden,
                        'content_preview': content[:200] + "..." if len(content) > 200 else content
                    }

                    print(f"    ✅ Accuracy: {accuracy_score:.1%}")
                    if found_forbidden:
                        print(f"    ⚠️  Found forbidden English terms: {found_forbidden}")

                else:
                    results[language] = {'success': False, 'error': 'No content generated'}
                    print(f"    ❌ Failed - No content")

            except Exception as e:
                results[language] = {'success': False, 'error': str(e)}
                print(f"    ❌ Failed - {str(e)}")

        self.test_results['results']['terminology_accuracy'] = results

    def _test_recommendation_consistency(self):
        """Test that recommendations are consistent across languages."""
        print("\\nTest 3: Recommendation consistency...")

        test_scenarios = [
            {'symbol': 'GOOGL', 'score_0_10': 8.2, 'expected_rec': {'en': 'BUY', 'fr': 'ACHAT', 'es': 'COMPRA'}},
            {'symbol': 'TSLA', 'score_0_10': 5.5, 'expected_rec': {'en': 'HOLD', 'fr': 'CONSERVATION', 'es': 'MANTENER'}},
            {'symbol': 'NVDA', 'score_0_10': 2.8, 'expected_rec': {'en': 'SELL', 'fr': 'VENTE', 'es': 'VENTA'}}
        ]

        results = {}

        for scenario in test_scenarios:
            symbol = scenario['symbol']
            score = scenario['score_0_10']
            expected_recs = scenario['expected_rec']

            print(f"  Testing {symbol} (score: {score}/10)...")

            test_data = {
                'symbol': symbol,
                'score_0_10': score,
                'currentPrice': 100.00,
                'weighted_scores': {'w_rsi14': 0.1, 'w_sma_crossover': 0.05}
            }

            scenario_results = {}

            for language in self.languages:
                try:
                    if language == 'en':
                        result = self.llm_service.generate_explanation(
                            test_data, 'detailed', 'technical_analysis', language
                        )
                    else:
                        result = self.llm_service.generate_multilingual_explanation(
                            test_data, language, 'detailed', 'technical_analysis'
                        )

                    if result and (result.get('content') or result.get('explanation')):
                        content = result.get('content') or result.get('explanation')
                        expected_rec = expected_recs[language]

                        # Check if expected recommendation is in content
                        has_expected_rec = expected_rec in content

                        scenario_results[language] = {
                            'success': True,
                            'expected_recommendation': expected_rec,
                            'has_expected_recommendation': has_expected_rec,
                            'content_preview': content[:150] + "..." if len(content) > 150 else content
                        }

                        status = "✅" if has_expected_rec else "⚠️"
                        print(f"    {status} {language}: Expected {expected_rec}, Found: {has_expected_rec}")

                    else:
                        scenario_results[language] = {
                            'success': False,
                            'error': 'No content generated'
                        }
                        print(f"    ❌ {language}: Failed - No content")

                except Exception as e:
                    scenario_results[language] = {
                        'success': False,
                        'error': str(e)
                    }
                    print(f"    ❌ {language}: Failed - {str(e)}")

            results[symbol] = scenario_results

        self.test_results['results']['recommendation_consistency'] = results

    def _test_direct_vs_translation(self):
        """Test Phase 2 direct generation vs translation for detailed mode."""
        print("\\nTest 4: Direct generation vs translation mode...")

        test_data = {
            'symbol': 'AAPL',
            'score_0_10': 7.5,
            'currentPrice': 155.00,
            'weighted_scores': {
                'w_rsi14': 0.14,
                'w_macd_histogram': 0.08,
                'w_bollinger_position': -0.03
            }
        }

        results = {}

        for language in ['fr', 'es']:
            print(f"  Testing {language} direct generation...")

            language_results = {}

            # Test detailed mode (should use direct generation)
            try:
                start_time = time.perf_counter()
                detailed_result = self.llm_service.generate_multilingual_explanation(
                    test_data, language, 'detailed', 'technical_analysis'
                )
                detailed_time = (time.perf_counter() - start_time) * 1000

                if detailed_result:
                    generation_method = detailed_result.get('generation_method', 'unknown')
                    model_used = detailed_result.get('model_used', 'unknown')

                    language_results['detailed'] = {
                        'success': True,
                        'generation_method': generation_method,
                        'model_used': model_used,
                        'generation_time_ms': round(detailed_time, 2),
                        'is_direct_generation': generation_method == 'native',
                        'uses_llama_model': 'llama' in model_used.lower()
                    }

                    direct_status = "✅" if generation_method == 'native' else "⚠️"
                    model_status = "✅" if 'llama' in model_used.lower() else "⚠️"
                    print(f"    {direct_status} Detailed mode: {generation_method} method")
                    print(f"    {model_status} Model used: {model_used}")

                else:
                    language_results['detailed'] = {
                        'success': False,
                        'error': 'No result generated'
                    }
                    print(f"    ❌ Detailed mode failed")

            except Exception as e:
                language_results['detailed'] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"    ❌ Detailed mode error: {str(e)}")

            # Test standard mode (should use translation)
            try:
                start_time = time.perf_counter()
                standard_result = self.llm_service.generate_multilingual_explanation(
                    test_data, language, 'standard', 'technical_analysis'
                )
                standard_time = (time.perf_counter() - start_time) * 1000

                if standard_result:
                    generation_method = standard_result.get('generation_method', 'unknown')

                    language_results['standard'] = {
                        'success': True,
                        'generation_method': generation_method,
                        'generation_time_ms': round(standard_time, 2),
                        'is_translation': generation_method != 'native'
                    }

                    translation_status = "✅" if generation_method != 'native' else "⚠️"
                    print(f"    {translation_status} Standard mode: {generation_method} method")

                else:
                    language_results['standard'] = {
                        'success': False,
                        'error': 'No result generated'
                    }
                    print(f"    ❌ Standard mode failed")

            except Exception as e:
                language_results['standard'] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"    ❌ Standard mode error: {str(e)}")

            results[language] = language_results

        self.test_results['results']['direct_vs_translation'] = results

    def _test_performance(self):
        """Test performance characteristics of Phase 2 implementation."""
        print("\\nTest 5: Performance validation...")

        test_data = {
            'symbol': 'MSFT',
            'score_0_10': 6.5,
            'currentPrice': 310.00,
            'weighted_scores': {
                'w_rsi14': 0.09,
                'w_sma_crossover': 0.12,
                'w_volume_trend': -0.04
            }
        }

        performance_results = {}
        iterations = 3

        for language in self.languages:
            print(f"  Performance testing {language}...")

            language_performance = {}

            for detail_level in self.detail_levels:
                times = []
                successes = 0

                for i in range(iterations):
                    try:
                        start_time = time.perf_counter()

                        if language == 'en':
                            result = self.llm_service.generate_explanation(
                                test_data, detail_level, 'technical_analysis', language
                            )
                        else:
                            result = self.llm_service.generate_multilingual_explanation(
                                test_data, language, detail_level, 'technical_analysis'
                            )

                        end_time = time.perf_counter()
                        generation_time = (end_time - start_time) * 1000
                        times.append(generation_time)

                        if result and (result.get('content') or result.get('explanation')):
                            successes += 1

                    except Exception as e:
                        print(f"    ⚠️  Iteration {i+1} failed: {str(e)}")

                if times:
                    language_performance[detail_level] = {
                        'avg_time_ms': round(sum(times) / len(times), 2),
                        'min_time_ms': round(min(times), 2),
                        'max_time_ms': round(max(times), 2),
                        'success_rate': round(successes / iterations, 2),
                        'iterations': iterations
                    }

                    avg_time = sum(times) / len(times)
                    success_rate = successes / iterations
                    print(f"    {detail_level}: {avg_time:.1f}ms avg, {success_rate:.1%} success")
                else:
                    language_performance[detail_level] = {
                        'error': 'No successful generations',
                        'success_rate': 0,
                        'iterations': iterations
                    }
                    print(f"    {detail_level}: All attempts failed")

            performance_results[language] = language_performance

        self.test_results['results']['performance'] = performance_results

    def _generate_test_report(self):
        """Generate comprehensive test report."""
        print("\\n" + "="*60)
        print("PHASE 2 MULTILINGUAL GENERATION TEST REPORT")
        print("="*60)

        # Summary statistics
        total_tests = 0
        passed_tests = 0

        # Basic generation summary
        basic_results = self.test_results['results'].get('basic_generation', {})
        print("\\n📊 BASIC GENERATION RESULTS:")
        for language in self.languages:
            for detail_level in self.detail_levels:
                total_tests += 1
                result = basic_results.get(language, {}).get(detail_level, {})
                if result.get('success'):
                    passed_tests += 1
                    method = result.get('generation_method', 'unknown')
                    model = result.get('model_used', 'unknown')
                    time_ms = result.get('generation_time_ms', 0)
                    print(f"  ✅ {language}/{detail_level}: {time_ms}ms ({method}, {model})")
                else:
                    error = result.get('error', 'Unknown error')
                    print(f"  ❌ {language}/{detail_level}: {error}")

        # Terminology accuracy summary
        terminology_results = self.test_results['results'].get('terminology_accuracy', {})
        print("\\n🔤 TERMINOLOGY ACCURACY:")
        for language, result in terminology_results.items():
            if result.get('success'):
                accuracy = result.get('accuracy_score', 0)
                forbidden = result.get('found_forbidden', [])
                status = "✅" if accuracy >= 0.7 and not forbidden else "⚠️"
                print(f"  {status} {language}: {accuracy:.1%} accuracy")
                if forbidden:
                    print(f"    ⚠️  Found forbidden terms: {forbidden}")
            else:
                print(f"  ❌ {language}: {result.get('error', 'Failed')}")

        # Recommendation consistency summary
        consistency_results = self.test_results['results'].get('recommendation_consistency', {})
        print("\\n🎯 RECOMMENDATION CONSISTENCY:")
        for symbol, symbol_results in consistency_results.items():
            print(f"  {symbol}:")
            for language, result in symbol_results.items():
                if result.get('success'):
                    has_rec = result.get('has_expected_recommendation', False)
                    expected = result.get('expected_recommendation', 'unknown')
                    status = "✅" if has_rec else "⚠️"
                    print(f"    {status} {language}: Expected {expected}, Found: {has_rec}")
                else:
                    print(f"    ❌ {language}: {result.get('error', 'Failed')}")

        # Direct generation summary
        direct_results = self.test_results['results'].get('direct_vs_translation', {})
        print("\\n🎭 DIRECT GENERATION (Phase 2):")
        for language, language_results in direct_results.items():
            print(f"  {language}:")
            detailed = language_results.get('detailed', {})
            if detailed.get('success'):
                is_direct = detailed.get('is_direct_generation', False)
                uses_llama = detailed.get('uses_llama_model', False)
                method = detailed.get('generation_method', 'unknown')
                model = detailed.get('model_used', 'unknown')

                direct_status = "✅" if is_direct else "❌"
                model_status = "✅" if uses_llama else "❌"
                print(f"    {direct_status} Detailed mode: {method} ({model})")
                print(f"    {model_status} Uses llama3.1:8b: {uses_llama}")
            else:
                print(f"    ❌ Detailed mode: {detailed.get('error', 'Failed')}")

        # Overall summary
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"\\n📈 OVERALL RESULTS:")
        print(f"  Total tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Success rate: {success_rate:.1f}%")

        # Save results
        results_file = f"scripts/phase2_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)

        print(f"\\nDetailed results saved to: {results_file}")
        print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Phase 2 Multilingual Generation Testing')
    parser.add_argument('--symbols',
                       default='AAPL,MSFT,GOOGL',
                       help='Comma-separated list of symbols to test (default: AAPL,MSFT,GOOGL)')
    parser.add_argument('--detail-levels',
                       default='summary,standard,detailed',
                       help='Comma-separated list of detail levels (default: summary,standard,detailed)')
    parser.add_argument('--languages',
                       default='en,fr,es',
                       help='Comma-separated list of languages (default: en,fr,es)')

    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',')]
    detail_levels = [d.strip() for d in args.detail_levels.split(',')]
    languages = [l.strip() for l in args.languages.split(',')]

    # Run Phase 2 tests
    tester = MultilingualGenerationTester(symbols=symbols, detail_levels=detail_levels, languages=languages)
    tester.run_all_tests()


if __name__ == '__main__':
    main()
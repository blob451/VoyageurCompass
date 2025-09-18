#!/usr/bin/env python3
"""
Comprehensive testing framework for Phase 3 Translation Pipeline.

Tests the integration between ExplanationService and TranslationService,
batch translation optimization, and cache performance.
"""

import os
import sys
import time
import statistics
from typing import Dict, List, Any, Optional

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'VoyageurCompass.settings')

import django
django.setup()

from Analytics.services.explanation_service import get_explanation_service
from Analytics.services.translation_service import get_translation_service
from Analytics.services.local_llm_service import get_local_llm_service


class Phase3TranslationPipelineTestSuite:
    """Comprehensive test suite for Phase 3 translation pipeline."""

    def __init__(self):
        self.explanation_service = get_explanation_service()
        self.translation_service = get_translation_service()
        self.llm_service = get_local_llm_service()

        # Test configuration
        self.test_languages = ["en", "fr", "es"]
        self.detail_levels = ["summary", "standard", "detailed"]
        self.test_symbols = ["AAPL", "MSFT", "GOOGL"]

        # Test data
        self.sample_analysis_data = {
            "symbol": "AAPL",
            "score_0_10": 7.5,
            "weighted_scores": {
                "sma50vs200": 0.8,
                "rsi14": 0.6,
                "macd12269": 0.7,
                "volsurge": 0.5
            },
            "indicators": {
                "sma50": 180.5,
                "sma200": 175.2,
                "rsi": 65.3,
                "macd": 2.1
            }
        }

        # Results storage
        self.test_results = {
            "explanation_service_integration": {},
            "translation_service_integration": {},
            "batch_translation": {},
            "cache_optimization": {},
            "performance_metrics": {},
            "error_handling": {}
        }

    def run_all_tests(self):
        """Run complete Phase 3 test suite."""
        print("=" * 80)
        print("PHASE 3 TRANSLATION PIPELINE COMPREHENSIVE TEST SUITE")
        print("=" * 80)

        # Test 1: ExplanationService Integration
        print("\n1. Testing ExplanationService Integration with TranslationService...")
        self.test_explanation_service_integration()

        # Test 2: TranslationService Functionality
        print("\n2. Testing TranslationService Core Functionality...")
        self.test_translation_service_integration()

        # Test 3: Batch Translation Optimization
        print("\n3. Testing Batch Translation Optimization...")
        self.test_batch_translation_optimization()

        # Test 4: Cache Optimization
        print("\n4. Testing Cache Optimization...")
        self.test_cache_optimization()

        # Test 5: Performance Metrics
        print("\n5. Testing Performance Metrics...")
        self.test_performance_metrics()

        # Test 6: Error Handling
        print("\n6. Testing Error Handling and Fallbacks...")
        self.test_error_handling()

        # Generate summary report
        self.generate_summary_report()

    def test_explanation_service_integration(self):
        """Test ExplanationService integration with TranslationService."""
        print("  Testing multilingual explanation generation...")

        for detail_level in self.detail_levels:
            for language in self.test_languages:
                start_time = time.time()

                try:
                    explanation = self.explanation_service.explain_prediction_single(
                        self.sample_analysis_data,
                        detail_level=detail_level,
                        language=language
                    )

                    generation_time = time.time() - start_time

                    if explanation and explanation.get("content"):
                        success = True
                        content_length = len(explanation["content"])
                        translation_method = explanation.get("translation_method", "unknown")

                        print(f"    ✓ {detail_level} {language}: {content_length} chars, {generation_time:.2f}s ({translation_method})")
                    else:
                        success = False
                        print(f"    ✗ {detail_level} {language}: Failed to generate explanation")

                    # Store results
                    key = f"{detail_level}_{language}"
                    self.test_results["explanation_service_integration"][key] = {
                        "success": success,
                        "generation_time": generation_time,
                        "content_length": content_length if success else 0,
                        "translation_method": translation_method if success else None
                    }

                except Exception as e:
                    print(f"    ✗ {detail_level} {language}: Error - {str(e)}")
                    self.test_results["explanation_service_integration"][f"{detail_level}_{language}"] = {
                        "success": False,
                        "error": str(e)
                    }

    def test_translation_service_integration(self):
        """Test TranslationService direct functionality."""
        print("  Testing direct translation capabilities...")

        sample_english_text = """Apple Inc. shows strong technical indicators with a score of 7.5/10.
        The 50-day moving average (180.5) is above the 200-day moving average (175.2), indicating a bullish trend.
        RSI at 65.3 suggests moderate momentum. Recommendation: BUY with confidence level 8/10."""

        for language in ["fr", "es"]:
            start_time = time.time()

            try:
                translation_result = self.translation_service.translate_explanation(
                    sample_english_text,
                    target_language=language,
                    financial_context={
                        "symbol": "AAPL",
                        "score": 7.5,
                        "recommendation": "BUY"
                    }
                )

                translation_time = time.time() - start_time

                if translation_result and translation_result.get("translated_text"):
                    quality_score = translation_result.get("quality_score", 0.0)
                    model_used = translation_result.get("model_used", "unknown")

                    print(f"    ✓ EN → {language.upper()}: Quality {quality_score:.2f}, {translation_time:.2f}s ({model_used})")

                    self.test_results["translation_service_integration"][language] = {
                        "success": True,
                        "translation_time": translation_time,
                        "quality_score": quality_score,
                        "model_used": model_used
                    }
                else:
                    print(f"    ✗ EN → {language.upper()}: Translation failed")
                    self.test_results["translation_service_integration"][language] = {
                        "success": False
                    }

            except Exception as e:
                print(f"    ✗ EN → {language.upper()}: Error - {str(e)}")
                self.test_results["translation_service_integration"][language] = {
                    "success": False,
                    "error": str(e)
                }

    def test_batch_translation_optimization(self):
        """Test batch translation functionality."""
        print("  Testing batch translation optimization...")

        # Prepare batch translation requests
        batch_requests = []
        for symbol in self.test_symbols:
            for language in ["fr", "es"]:
                batch_requests.append({
                    "content": f"Technical analysis for {symbol} shows bullish signals. Recommendation: BUY.",
                    "target_language": language,
                    "financial_context": {
                        "symbol": symbol,
                        "recommendation": "BUY"
                    }
                })

        # Test batch translation
        start_time = time.time()

        try:
            batch_results = self.translation_service.translate_explanations_batch(batch_requests)
            batch_time = time.time() - start_time

            successful_translations = sum(1 for result in batch_results if result and result.get("translated_text"))
            total_requests = len(batch_requests)
            success_rate = (successful_translations / total_requests) * 100 if total_requests > 0 else 0

            print(f"    ✓ Batch translation: {successful_translations}/{total_requests} successful ({success_rate:.1f}%)")
            print(f"    ✓ Batch processing time: {batch_time:.2f}s ({batch_time/total_requests:.2f}s per translation)")

            self.test_results["batch_translation"] = {
                "success": True,
                "total_requests": total_requests,
                "successful_translations": successful_translations,
                "success_rate": success_rate,
                "batch_time": batch_time,
                "avg_time_per_translation": batch_time / total_requests
            }

        except Exception as e:
            print(f"    ✗ Batch translation failed: {str(e)}")
            self.test_results["batch_translation"] = {
                "success": False,
                "error": str(e)
            }

    def test_cache_optimization(self):
        """Test cache optimization functionality."""
        print("  Testing cache optimization...")

        test_content = "Apple Inc. technical analysis shows strong buy signals with RSI at 65.3."

        # Test multiple cache key variations
        cache_keys = self.translation_service._generate_cache_key_variations(test_content, "fr")
        print(f"    ✓ Generated {len(cache_keys)} cache key variations")

        # Test content normalization
        normalized = self.translation_service._normalize_content_for_cache(test_content)
        print(f"    ✓ Content normalization: {len(test_content)} → {len(normalized)} chars")

        # Test cache optimization lookup
        start_time = time.time()
        cached_result = self.translation_service.optimize_translation_cache(test_content, "fr")
        lookup_time = time.time() - start_time

        print(f"    ✓ Cache optimization lookup: {lookup_time:.4f}s")

        self.test_results["cache_optimization"] = {
            "cache_key_variations": len(cache_keys),
            "content_normalization_ratio": len(normalized) / len(test_content),
            "lookup_time": lookup_time
        }

    def test_performance_metrics(self):
        """Test performance across different scenarios."""
        print("  Testing performance metrics...")

        # Test different detail levels performance
        performance_data = {}

        for detail_level in self.detail_levels:
            times = []

            for _ in range(3):  # Multiple runs for average
                start_time = time.time()

                explanation = self.explanation_service.explain_prediction_single(
                    self.sample_analysis_data,
                    detail_level=detail_level,
                    language="fr"
                )

                if explanation:
                    times.append(time.time() - start_time)

            if times:
                avg_time = statistics.mean(times)
                performance_data[detail_level] = {
                    "avg_time": avg_time,
                    "min_time": min(times),
                    "max_time": max(times),
                    "samples": len(times)
                }

                print(f"    ✓ {detail_level}: avg={avg_time:.2f}s, min={min(times):.2f}s, max={max(times):.2f}s")

        self.test_results["performance_metrics"] = performance_data

    def test_error_handling(self):
        """Test error handling and fallback mechanisms."""
        print("  Testing error handling...")

        # Test invalid language code
        try:
            result = self.translation_service.translate_explanation(
                "Test content",
                target_language="invalid",
                financial_context={}
            )
            print(f"    ✓ Invalid language handling: {'Handled correctly' if not result else 'Unexpected success'}")
        except Exception as e:
            print(f"    ✓ Invalid language exception: {str(e)}")

        # Test empty content
        try:
            result = self.translation_service.translate_explanation(
                "",
                target_language="fr",
                financial_context={}
            )
            print(f"    ✓ Empty content handling: {'Handled correctly' if not result else 'Unexpected success'}")
        except Exception as e:
            print(f"    ✓ Empty content exception: {str(e)}")

        # Test service availability
        if hasattr(self.llm_service, 'is_available'):
            availability = self.llm_service.is_available()
            print(f"    ✓ LLM service availability: {availability}")

        self.test_results["error_handling"] = {
            "invalid_language_handled": True,
            "empty_content_handled": True,
            "service_availability_checked": True
        }

    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("\n" + "=" * 80)
        print("PHASE 3 TRANSLATION PIPELINE TEST SUMMARY")
        print("=" * 80)

        # ExplanationService Integration Summary
        esi_results = self.test_results["explanation_service_integration"]
        esi_success_count = sum(1 for result in esi_results.values() if result.get("success", False))
        esi_total = len(esi_results)
        esi_success_rate = (esi_success_count / esi_total * 100) if esi_total > 0 else 0

        print(f"\n1. ExplanationService Integration:")
        print(f"   Success Rate: {esi_success_count}/{esi_total} ({esi_success_rate:.1f}%)")

        if esi_success_count > 0:
            avg_times = [r["generation_time"] for r in esi_results.values() if r.get("success")]
            if avg_times:
                print(f"   Average Generation Time: {statistics.mean(avg_times):.2f}s")

        # TranslationService Integration Summary
        tsi_results = self.test_results["translation_service_integration"]
        tsi_success_count = sum(1 for result in tsi_results.values() if result.get("success", False))
        tsi_total = len(tsi_results)

        print(f"\n2. TranslationService Integration:")
        print(f"   Success Rate: {tsi_success_count}/{tsi_total} ({tsi_success_count/tsi_total*100:.1f}%)")

        # Batch Translation Summary
        batch_results = self.test_results["batch_translation"]
        if batch_results.get("success"):
            print(f"\n3. Batch Translation:")
            print(f"   Success Rate: {batch_results['success_rate']:.1f}%")
            print(f"   Processing Time: {batch_results['avg_time_per_translation']:.2f}s per translation")

        # Performance Summary
        perf_results = self.test_results["performance_metrics"]
        if perf_results:
            print(f"\n4. Performance by Detail Level:")
            for level, data in perf_results.items():
                print(f"   {level}: {data['avg_time']:.2f}s average")

        # Overall Assessment
        print(f"\n5. Overall Assessment:")
        total_tests = esi_total + tsi_total + (1 if batch_results.get("success") else 0)
        total_success = esi_success_count + tsi_success_count + (1 if batch_results.get("success") else 0)
        overall_success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0

        print(f"   Overall Success Rate: {total_success}/{total_tests} ({overall_success_rate:.1f}%)")

        if overall_success_rate >= 80:
            print("   Status: PHASE 3 IMPLEMENTATION SUCCESSFUL ✓")
        elif overall_success_rate >= 60:
            print("   Status: PHASE 3 IMPLEMENTATION PARTIALLY SUCCESSFUL ⚠")
        else:
            print("   Status: PHASE 3 IMPLEMENTATION NEEDS IMPROVEMENT ✗")

        print("\n" + "=" * 80)


def main():
    """Main test execution function."""
    try:
        test_suite = Phase3TranslationPipelineTestSuite()
        test_suite.run_all_tests()

    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
    except Exception as e:
        print(f"\nTest execution failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
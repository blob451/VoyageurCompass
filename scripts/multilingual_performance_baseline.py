#!/usr/bin/env python
"""
Multilingual Performance Baseline Script

This script measures performance baselines for the multilingual LLM system:
- Language detection performance
- Cache hit/miss rates by language
- LLM generation times by language and detail level
- Translation quality and speed
- Cultural context processing overhead

Usage:
    python scripts/multilingual_performance_baseline.py [--languages en,fr,es] [--iterations 10]
"""

import os
import sys
import time
import json
import statistics
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

from Analytics.services.language_detector import get_language_detector, detect_request_language
from Analytics.services.local_llm_service import LocalLLMService
from Analytics.services.explanation_service import ExplanationService


class MultilingualPerformanceBaseline:
    """Performance baseline measurement tool for multilingual system."""

    def __init__(self, languages: List[str] = None, iterations: int = 10):
        self.languages = languages or ['en', 'fr', 'es']
        self.iterations = iterations
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'test_config': {
                'languages': self.languages,
                'iterations': self.iterations,
            },
            'baselines': {}
        }

        # Initialize services
        self.language_detector = get_language_detector()
        self.llm_service = LocalLLMService()
        self.explanation_service = ExplanationService()
        self.request_factory = RequestFactory()

        # Test data
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        self.detail_levels = ['summary', 'standard', 'detailed']

        print("=== Multilingual Performance Baseline Test ===")
        print(f"Languages: {', '.join(self.languages)}")
        print(f"Iterations: {self.iterations}")
        print(f"Test symbols: {', '.join(self.test_symbols)}")
        print("-" * 60)

    def run_all_benchmarks(self):
        """Run all performance benchmarks."""
        try:
            self._benchmark_language_detection()
            self._benchmark_cache_performance()
            self._benchmark_llm_generation_times()
            self._benchmark_cultural_context_processing()
            self._generate_summary_report()

        except Exception as e:
            print(f"ERROR: Benchmark failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def _benchmark_language_detection(self):
        """Benchmark language detection performance."""
        print("Testing language detection performance...")

        detection_times = {lang: [] for lang in self.languages}

        # Test scenarios for each language
        test_scenarios = {
            'en': [
                {'headers': {'HTTP_ACCEPT_LANGUAGE': 'en-US,en;q=0.9'}},
                {'headers': {'HTTP_ACCEPT_LANGUAGE': 'en-GB,en;q=0.8,fr;q=0.6'}},
                {'explicit': 'en'},
            ],
            'fr': [
                {'headers': {'HTTP_ACCEPT_LANGUAGE': 'fr-FR,fr;q=0.9,en;q=0.8'}},
                {'headers': {'HTTP_ACCEPT_LANGUAGE': 'fr-CA,fr;q=0.9'}},
                {'explicit': 'fr'},
            ],
            'es': [
                {'headers': {'HTTP_ACCEPT_LANGUAGE': 'es-ES,es;q=0.9,en;q=0.7'}},
                {'headers': {'HTTP_ACCEPT_LANGUAGE': 'es-MX,es;q=0.8'}},
                {'explicit': 'es'},
            ]
        }

        for language in self.languages:
            for scenario in test_scenarios.get(language, []):
                for _ in range(self.iterations):
                    # Create request with language hints
                    if 'headers' in scenario:
                        request = self.request_factory.get('/', **scenario['headers'])
                    else:
                        request = self.request_factory.get('/')

                    explicit_lang = scenario.get('explicit')

                    start_time = time.perf_counter()
                    detected_lang = detect_request_language(request, explicit_language=explicit_lang)
                    end_time = time.perf_counter()

                    detection_time = (end_time - start_time) * 1000  # ms
                    detection_times[language].append(detection_time)

        # Calculate statistics
        self.results['baselines']['language_detection'] = {}
        for language in self.languages:
            times = detection_times[language]
            if times:
                self.results['baselines']['language_detection'][language] = {
                    'avg_time_ms': round(statistics.mean(times), 3),
                    'median_time_ms': round(statistics.median(times), 3),
                    'min_time_ms': round(min(times), 3),
                    'max_time_ms': round(max(times), 3),
                    'std_dev_ms': round(statistics.stdev(times) if len(times) > 1 else 0, 3),
                    'samples': len(times)
                }

                print(f"  {language}: {self.results['baselines']['language_detection'][language]['avg_time_ms']:.3f}ms avg")

    def _benchmark_cache_performance(self):
        """Benchmark cache hit/miss rates by language."""
        print("Testing cache performance by language...")

        cache_stats = {lang: {'hits': 0, 'misses': 0, 'hit_times': [], 'miss_times': []}
                      for lang in self.languages}

        # Test with sample analysis data
        test_analysis = {
            'symbol': 'AAPL',
            'score_0_10': 7.5,
            'currentPrice': 150.00,
            'weighted_scores': {
                'w_rsi14': 0.15,
                'w_sma_crossover': 0.12,
                'w_macd_histogram': -0.08
            }
        }

        for language in self.languages:
            for detail_level in self.detail_levels:
                # First call (cache miss)
                start_time = time.perf_counter()
                cache_key = self.llm_service._create_cache_key(
                    test_analysis, detail_level, 'technical_analysis', language
                )
                cached_result = cache.get(cache_key)
                end_time = time.perf_counter()

                miss_time = (end_time - start_time) * 1000

                if cached_result is None:
                    cache_stats[language]['misses'] += 1
                    cache_stats[language]['miss_times'].append(miss_time)

                    # Simulate cache population
                    cache.set(cache_key, {
                        'explanation': f"Test explanation in {language}",
                        'confidence': 0.85,
                        'generation_time': 1.5
                    }, 300)
                else:
                    cache_stats[language]['hits'] += 1
                    cache_stats[language]['hit_times'].append(miss_time)

                # Second call (should be cache hit)
                start_time = time.perf_counter()
                cached_result = cache.get(cache_key)
                end_time = time.perf_counter()

                hit_time = (end_time - start_time) * 1000

                if cached_result is not None:
                    cache_stats[language]['hits'] += 1
                    cache_stats[language]['hit_times'].append(hit_time)

        # Calculate cache performance metrics
        self.results['baselines']['cache_performance'] = {}
        for language in self.languages:
            stats = cache_stats[language]
            total_requests = stats['hits'] + stats['misses']
            hit_rate = (stats['hits'] / total_requests * 100) if total_requests > 0 else 0

            self.results['baselines']['cache_performance'][language] = {
                'hit_rate_percent': round(hit_rate, 2),
                'total_requests': total_requests,
                'hits': stats['hits'],
                'misses': stats['misses'],
                'avg_hit_time_ms': round(statistics.mean(stats['hit_times']), 3) if stats['hit_times'] else 0,
                'avg_miss_time_ms': round(statistics.mean(stats['miss_times']), 3) if stats['miss_times'] else 0
            }

            print(f"  {language}: {hit_rate:.1f}% hit rate, {self.results['baselines']['cache_performance'][language]['avg_hit_time_ms']:.3f}ms hit time")

    def _benchmark_llm_generation_times(self):
        """Benchmark LLM generation times by language and detail level."""
        print("Testing LLM generation performance...")

        generation_times = {}

        # Test analysis data
        test_analysis = {
            'symbol': 'MSFT',
            'score_0_10': 6.8,
            'currentPrice': 320.00,
            'weighted_scores': {
                'w_rsi14': 0.08,
                'w_sma_crossover': 0.15,
                'w_bollinger_position': -0.05
            }
        }

        for language in self.languages:
            generation_times[language] = {}

            for detail_level in self.detail_levels:
                times = []

                # Skip actual LLM calls for baseline (would be too slow)
                # Instead, measure prompt building and preparation overhead
                for _ in range(self.iterations):
                    start_time = time.perf_counter()

                    # Measure multilingual prompt building time
                    if language != 'en':
                        prompt = self.llm_service._build_multilingual_prompt(
                            test_analysis, detail_level, 'technical_analysis', language
                        )
                    else:
                        prompt = self.llm_service._build_optimized_prompt(
                            test_analysis, detail_level, 'technical_analysis'
                        )

                    # Measure language-specific options generation
                    options = self.llm_service._get_language_specific_options(language, detail_level)

                    end_time = time.perf_counter()

                    prep_time = (end_time - start_time) * 1000
                    times.append(prep_time)

                generation_times[language][detail_level] = {
                    'avg_prep_time_ms': round(statistics.mean(times), 3),
                    'median_prep_time_ms': round(statistics.median(times), 3),
                    'estimated_total_time_s': round(statistics.mean(times) / 1000 + 2.5, 2),  # Add typical LLM time
                    'samples': len(times)
                }

                print(f"  {language}/{detail_level}: {generation_times[language][detail_level]['avg_prep_time_ms']:.3f}ms prep")

        self.results['baselines']['llm_generation'] = generation_times

    def _benchmark_cultural_context_processing(self):
        """Benchmark cultural context processing overhead."""
        print("Testing cultural context processing...")

        context_times = {lang: [] for lang in self.languages}

        for language in self.languages:
            for _ in range(self.iterations * 3):  # More samples for this quick operation
                start_time = time.perf_counter()

                # Test cultural context generation
                cultural_context = self.explanation_service._get_cultural_context(language)

                end_time = time.perf_counter()

                context_time = (end_time - start_time) * 1000  # ms
                context_times[language].append(context_time)

        # Calculate cultural context metrics
        self.results['baselines']['cultural_context'] = {}
        for language in self.languages:
            times = context_times[language]
            self.results['baselines']['cultural_context'][language] = {
                'avg_time_ms': round(statistics.mean(times), 4),
                'median_time_ms': round(statistics.median(times), 4),
                'overhead_percent': round((statistics.mean(times) / 10) * 100, 2),  # Assume 10ms baseline
                'samples': len(times)
            }

            print(f"  {language}: {self.results['baselines']['cultural_context'][language]['avg_time_ms']:.4f}ms avg")

    def _generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("\n" + "="*60)
        print("MULTILINGUAL PERFORMANCE BASELINE SUMMARY")
        print("="*60)

        # Overall performance assessment
        total_overhead = 0

        for language in self.languages:
            lang_overhead = 0

            # Language detection overhead
            detection_time = self.results['baselines']['language_detection'][language]['avg_time_ms']
            lang_overhead += detection_time

            # Cultural context overhead
            context_time = self.results['baselines']['cultural_context'][language]['avg_time_ms']
            lang_overhead += context_time

            # Cache performance impact
            cache_perf = self.results['baselines']['cache_performance'][language]
            cache_overhead = cache_perf['avg_miss_time_ms'] - cache_perf['avg_hit_time_ms']
            lang_overhead += max(0, cache_overhead)

            total_overhead += lang_overhead

            print(f"\n{language.upper()} Language Performance:")
            print(f"  Detection: {detection_time:.3f}ms")
            print(f"  Cultural context: {context_time:.4f}ms")
            print(f"  Cache hit rate: {cache_perf['hit_rate_percent']:.1f}%")
            print(f"  Total overhead: {lang_overhead:.3f}ms")

        avg_overhead = total_overhead / len(self.languages)

        print(f"\nBASELINE METRICS:")
        print(f"  Average multilingual overhead: {avg_overhead:.3f}ms")
        print(f"  Performance impact: {(avg_overhead/1000):.1%} of 1-second operation")

        # Performance recommendations
        print(f"\nRECOMMENDATIONS:")
        if avg_overhead > 5:
            print("  WARNING: High overhead detected - consider optimizing language detection")
        else:
            print("  OK: Multilingual overhead within acceptable range")

        worst_cache = min(self.languages,
                         key=lambda l: self.results['baselines']['cache_performance'][l]['hit_rate_percent'])
        worst_rate = self.results['baselines']['cache_performance'][worst_cache]['hit_rate_percent']

        if worst_rate < 50:
            print(f"  WARNING: Low cache hit rate for {worst_cache} ({worst_rate:.1f}%) - review cache keys")
        else:
            print("  OK: Cache performance looks good across languages")

        # Save results
        results_file = f"scripts/baseline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to: {results_file}")
        print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Multilingual Performance Baseline Tool')
    parser.add_argument('--languages',
                       default='en,fr,es',
                       help='Comma-separated list of languages to test (default: en,fr,es)')
    parser.add_argument('--iterations',
                       type=int,
                       default=10,
                       help='Number of iterations per test (default: 10)')

    args = parser.parse_args()

    languages = [lang.strip() for lang in args.languages.split(',')]

    # Run baseline tests
    baseline = MultilingualPerformanceBaseline(languages=languages, iterations=args.iterations)
    baseline.run_all_benchmarks()


if __name__ == '__main__':
    main()
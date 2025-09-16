import time
import json
import statistics
from django.core.management.base import BaseCommand
from django.conf import settings
from Analytics.services.translation_service import TranslationService
from Analytics.services.local_llm_service import LocalLLMService

class Command(BaseCommand):
    help = 'Benchmark translation service performance for multilingual support'

    def add_arguments(self, parser):
        parser.add_argument(
            '--iterations',
            type=int,
            default=10,
            help='Number of iterations for each test (default: 10)'
        )
        parser.add_argument(
            '--languages',
            nargs='+',
            default=['fr', 'es'],
            help='Languages to test (default: fr es)'
        )
        parser.add_argument(
            '--output',
            type=str,
            help='Output file path for results (JSON format)'
        )
        parser.add_argument(
            '--warmup',
            action='store_true',
            help='Perform warmup runs to initialize caches'
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting translation performance benchmarking...'))

        translation_service = TranslationService()
        iterations = options['iterations']
        languages = options['languages']
        output_file = options.get('output')
        warmup = options.get('warmup', False)

        # Test datasets categorized by complexity and use case
        test_datasets = {
            'ui_short': [
                "Home",
                "Search",
                "Settings",
                "Welcome back",
                "Quick Actions",
                "Recent Analysis",
                "Popular Stocks",
                "Sign In",
                "Sign Out",
                "Dashboard"
            ],
            'ui_medium': [
                "Welcome back to your financial dashboard",
                "Search for stocks by symbol or company name",
                "Your recent analysis results and recommendations",
                "Configure your preferences and settings",
                "Quick actions for portfolio management",
                "View detailed technical analysis reports",
                "Access premium features and tools",
                "Manage your account and subscription",
                "Export data and generate reports",
                "Get help and support resources"
            ],
            'financial_short': [
                "Stock price increased 5%",
                "Market cap: $2.5B",
                "P/E ratio: 18.5",
                "Strong buy recommendation",
                "High trading volume",
                "Bullish trend confirmed",
                "Earnings beat estimates",
                "Dividend yield: 3.2%",
                "52-week high reached",
                "Technical breakout"
            ],
            'financial_medium': [
                "The company's quarterly earnings exceeded analyst expectations by 12%.",
                "Technical analysis indicates a strong bullish trend with increasing volume.",
                "The stock shows potential for continued growth based on fundamental analysis.",
                "Market volatility may impact short-term price movements significantly.",
                "The P/E ratio suggests the stock is undervalued compared to industry peers.",
                "Recent insider trading activity shows confidence from company executives.",
                "The dividend policy provides stable income for long-term investors.",
                "Sector rotation trends favor this particular industry segment currently.",
                "Risk management suggests maintaining a diversified portfolio approach.",
                "Macroeconomic factors support the current investment thesis strongly."
            ],
            'financial_long': [
                "This comprehensive analysis examines the company's financial performance over the past five quarters, considering revenue growth, profit margins, debt levels, and cash flow generation. Key technical indicators including moving averages, RSI, and MACD suggest a bullish trend continuation. The fundamental analysis reveals strong competitive positioning within the industry, supported by innovation in product development and strategic market expansion. However, investors should consider potential risks including regulatory changes, market volatility, and sector-specific challenges that could impact future performance.",
                "The investment recommendation is based on a thorough evaluation of quantitative metrics and qualitative factors. The company demonstrates consistent revenue growth, improving operational efficiency, and strong balance sheet management. Technical analysis shows positive momentum with key support and resistance levels identified. The current valuation appears attractive relative to historical multiples and peer comparison. Long-term growth prospects remain positive despite short-term market uncertainties and geopolitical risks affecting the broader financial markets."
            ]
        }

        results = {
            'benchmark_metadata': {
                'timestamp': time.time(),
                'iterations': iterations,
                'languages': languages,
                'warmup_performed': warmup
            },
            'performance_metrics': {},
            'cache_efficiency': {},
            'error_rates': {},
            'quality_scores': {}
        }

        # Warmup phase
        if warmup:
            self.stdout.write("Performing warmup runs...")
            for language in languages:
                translation_service.translate_text("Warmup test", language, context='financial')

        # Main benchmarking loop
        for dataset_name, texts in test_datasets.items():
            self.stdout.write(f"Benchmarking dataset: {dataset_name}")
            results['performance_metrics'][dataset_name] = {}

            for language in languages:
                self.stdout.write(f"  Testing {language} translations...")

                # Metrics collection
                times = []
                cache_hits = 0
                errors = 0
                quality_scores = []

                for iteration in range(iterations):
                    for text in texts:
                        try:
                            # Clear cache for accurate timing
                            if iteration == 0:  # Only clear on first iteration for cache testing
                                cache_key = f"translation:{language}:{hash(text + 'financial')}"
                                translation_service.cache_manager.delete(cache_key)

                            start_time = time.perf_counter()

                            # Perform translation
                            result = translation_service.translate_text(
                                text, language, context='financial'
                            )

                            end_time = time.perf_counter()
                            translation_time = (end_time - start_time) * 1000  # Convert to milliseconds

                            times.append(translation_time)

                            # Check if result came from cache
                            if iteration > 0:  # Cache should hit after first iteration
                                cached_result = translation_service.cache_manager.get(cache_key)
                                if cached_result and translation_time < 50:  # Fast response indicates cache hit
                                    cache_hits += 1

                            # Assess translation quality (basic heuristics)
                            if result and result != text and len(result) > 0:
                                quality_score = self._assess_translation_quality(text, result, language)
                                quality_scores.append(quality_score)
                            else:
                                errors += 1

                        except Exception as e:
                            self.stdout.write(
                                self.style.ERROR(f"    Error translating '{text}': {str(e)}")
                            )
                            errors += 1
                            times.append(float('inf'))  # Mark as failed

                # Calculate statistics
                valid_times = [t for t in times if t != float('inf')]
                if valid_times:
                    results['performance_metrics'][dataset_name][language] = {
                        'avg_time_ms': statistics.mean(valid_times),
                        'median_time_ms': statistics.median(valid_times),
                        'min_time_ms': min(valid_times),
                        'max_time_ms': max(valid_times),
                        'std_dev_ms': statistics.stdev(valid_times) if len(valid_times) > 1 else 0,
                        'total_translations': len(texts) * iterations,
                        'successful_translations': len(valid_times),
                        'throughput_per_second': len(valid_times) / (sum(valid_times) / 1000) if sum(valid_times) > 0 else 0
                    }

                    results['cache_efficiency'][f"{dataset_name}_{language}"] = {
                        'cache_hit_rate': cache_hits / max(1, len(texts) * (iterations - 1)),
                        'cache_hits': cache_hits,
                        'total_cache_opportunities': len(texts) * (iterations - 1)
                    }

                    results['error_rates'][f"{dataset_name}_{language}"] = {
                        'error_rate': errors / (len(texts) * iterations),
                        'total_errors': errors,
                        'total_attempts': len(texts) * iterations
                    }

                    if quality_scores:
                        results['quality_scores'][f"{dataset_name}_{language}"] = {
                            'avg_quality': statistics.mean(quality_scores),
                            'min_quality': min(quality_scores),
                            'max_quality': max(quality_scores)
                        }

        # Performance analysis and recommendations
        results['analysis'] = self._analyze_performance_results(results)

        # Output results
        self._display_results(results)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.stdout.write(f"Results saved to: {output_file}")

        self.stdout.write(self.style.SUCCESS('Translation performance benchmarking completed!'))

    def _assess_translation_quality(self, original, translation, language):
        """Basic heuristic quality assessment for translations"""
        score = 0.0

        # Length similarity (translated text should be reasonably similar in length)
        length_ratio = len(translation) / max(1, len(original))
        if 0.5 <= length_ratio <= 2.0:
            score += 0.3
        elif 0.3 <= length_ratio <= 3.0:
            score += 0.1

        # Character diversity (good translations should have diverse characters)
        unique_chars = len(set(translation.lower()))
        char_diversity = unique_chars / max(1, len(translation))
        if char_diversity > 0.3:
            score += 0.2

        # Language-specific character presence
        language_chars = {
            'fr': ['é', 'è', 'à', 'ç', 'ê', 'ô', 'û', 'î', 'ï', 'ù'],
            'es': ['ñ', 'á', 'é', 'í', 'ó', 'ú', '¿', '¡']
        }

        if language in language_chars:
            has_language_chars = any(char in translation for char in language_chars[language])
            if has_language_chars:
                score += 0.2

        # Financial terminology preservation (basic check)
        financial_terms = ['%', '$', 'EUR', '€', 'USD', 'billion', 'million']
        original_financial_terms = sum(1 for term in financial_terms if term in original)
        translated_financial_terms = sum(1 for term in financial_terms if term in translation)

        if original_financial_terms > 0:
            term_preservation = translated_financial_terms / original_financial_terms
            score += min(0.3, term_preservation * 0.3)

        return min(1.0, score)  # Cap at 1.0

    def _analyze_performance_results(self, results):
        """Analyze performance results and provide recommendations"""
        analysis = {
            'summary': {},
            'bottlenecks': [],
            'recommendations': []
        }

        # Find performance patterns
        avg_times = []
        for dataset_name, dataset_results in results['performance_metrics'].items():
            for language, metrics in dataset_results.items():
                avg_times.append(metrics['avg_time_ms'])

        if avg_times:
            overall_avg = statistics.mean(avg_times)
            analysis['summary']['overall_avg_time_ms'] = overall_avg
            analysis['summary']['overall_median_time_ms'] = statistics.median(avg_times)

            # Identify bottlenecks
            if overall_avg > 2000:
                analysis['bottlenecks'].append("High average translation time indicates potential LLM performance issues")

            # Cache efficiency analysis
            cache_rates = [data['cache_hit_rate'] for data in results['cache_efficiency'].values()]
            if cache_rates:
                avg_cache_rate = statistics.mean(cache_rates)
                analysis['summary']['avg_cache_hit_rate'] = avg_cache_rate

                if avg_cache_rate < 0.8:
                    analysis['bottlenecks'].append("Low cache hit rate - consider increasing cache TTL or size")

            # Error rate analysis
            error_rates = [data['error_rate'] for data in results['error_rates'].values()]
            if error_rates:
                avg_error_rate = statistics.mean(error_rates)
                analysis['summary']['avg_error_rate'] = avg_error_rate

                if avg_error_rate > 0.05:
                    analysis['bottlenecks'].append("High error rate indicates translation service reliability issues")

            # Recommendations
            if overall_avg > 1000:
                analysis['recommendations'].append("Consider implementing asynchronous translation for better user experience")

            if any(data['cache_hit_rate'] < 0.5 for data in results['cache_efficiency'].values()):
                analysis['recommendations'].append("Optimize caching strategy to improve response times")

            analysis['recommendations'].append("Monitor translation quality scores and implement feedback mechanisms")

        return analysis

    def _display_results(self, results):
        """Display benchmark results in a formatted manner"""
        self.stdout.write("\n" + "="*80)
        self.stdout.write(self.style.SUCCESS("TRANSLATION PERFORMANCE BENCHMARK RESULTS"))
        self.stdout.write("="*80)

        # Summary
        if 'analysis' in results and 'summary' in results['analysis']:
            summary = results['analysis']['summary']
            self.stdout.write(f"\nOVERALL SUMMARY:")
            if 'overall_avg_time_ms' in summary:
                self.stdout.write(f"  Average Translation Time: {summary['overall_avg_time_ms']:.2f}ms")
            if 'avg_cache_hit_rate' in summary:
                self.stdout.write(f"  Average Cache Hit Rate: {summary['avg_cache_hit_rate']:.2%}")
            if 'avg_error_rate' in summary:
                self.stdout.write(f"  Average Error Rate: {summary['avg_error_rate']:.2%}")

        # Detailed results by dataset
        for dataset_name, dataset_results in results['performance_metrics'].items():
            self.stdout.write(f"\nDATASET: {dataset_name.upper()}")
            self.stdout.write("-" * 40)

            for language, metrics in dataset_results.items():
                self.stdout.write(f"  {language.upper()}:")
                self.stdout.write(f"    Avg Time: {metrics['avg_time_ms']:.2f}ms")
                self.stdout.write(f"    Median Time: {metrics['median_time_ms']:.2f}ms")
                self.stdout.write(f"    Throughput: {metrics['throughput_per_second']:.2f}/sec")
                self.stdout.write(f"    Success Rate: {(metrics['successful_translations']/metrics['total_translations']):.2%}")

        # Bottlenecks and recommendations
        if 'analysis' in results:
            analysis = results['analysis']
            if analysis.get('bottlenecks'):
                self.stdout.write(f"\nIDENTIFIED BOTTLENECKS:")
                for bottleneck in analysis['bottlenecks']:
                    self.stdout.write(f"  ⚠️  {bottleneck}")

            if analysis.get('recommendations'):
                self.stdout.write(f"\nRECOMMENDATIONS:")
                for recommendation in analysis['recommendations']:
                    self.stdout.write(f"  💡 {recommendation}")

        self.stdout.write("\n" + "="*80)
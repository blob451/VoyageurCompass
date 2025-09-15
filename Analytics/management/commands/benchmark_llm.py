"""
LLM Performance Benchmarking Management Command

Tests and benchmarks LLM performance including:
- Cold start vs warm start times
- Model switching performance  
- Concurrent request handling
- Response quality consistency
"""

import json
import logging
import statistics
import threading
import time
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
from scipy import stats

from django.core.cache import cache
from django.core.management.base import BaseCommand, CommandError

from Analytics.services.local_llm_service import get_local_llm_service

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Benchmark LLM performance across all models and detail levels'

    def add_arguments(self, parser):
        parser.add_argument(
            '--iterations',
            type=int,
            default=5,
            help='Number of iterations per test (default: 5)'
        )
        
        parser.add_argument(
            '--concurrent-requests',
            type=int,
            default=3,
            help='Number of concurrent requests to test (default: 3)'
        )
        
        parser.add_argument(
            '--warm-up',
            action='store_true',
            help='Run warm-up before benchmarking'
        )
        
        parser.add_argument(
            '--output-file',
            type=str,
            help='Output results to JSON file'
        )
        
        parser.add_argument(
            '--detail-levels',
            nargs='+',
            choices=['summary', 'standard', 'detailed'],
            default=['summary', 'standard', 'detailed'],
            help='Detail levels to benchmark (default: all)'
        )
        
        parser.add_argument(
            '--percentiles',
            action='store_true',
            help='Calculate detailed percentile statistics (90th, 95th, 99th)'
        )
        
        parser.add_argument(
            '--regression-test',
            type=str,
            help='Compare against baseline results file for regression testing'
        )
        
        parser.add_argument(
            '--statistical-test',
            action='store_true',
            help='Perform statistical significance testing for performance changes'
        )

    def handle(self, *args, **options):
        """Execute LLM performance benchmarking."""
        self.stdout.write(self.style.SUCCESS('Starting LLM Performance Benchmarking...'))
        
        # Clear any existing cache
        cache.clear()
        
        # Get LLM service
        llm_service = get_local_llm_service()
        if not llm_service:
            raise CommandError("LLM service not available")
        
        # Check if Ollama is available
        status = llm_service.get_service_status()
        if not status.get('available'):
            raise CommandError(f"Ollama not available: {status.get('error', 'Unknown error')}")
        
        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'iterations': options['iterations'],
                'concurrent_requests': options['concurrent_requests'],
                'detail_levels': options['detail_levels'],
                'warm_up_enabled': options['warm_up']
            },
            'system_info': status,
            'tests': {}
        }
        
        try:
            # Warm-up phase
            if options['warm_up']:
                self.stdout.write('Running warm-up...')
                self._run_warmup(llm_service)
            
            # Cold start benchmark
            self.stdout.write('Running cold start benchmark...')
            benchmark_results['tests']['cold_start'] = self._benchmark_cold_start(
                llm_service, options['detail_levels'], options['iterations']
            )
            
            # Warm start benchmark  
            self.stdout.write('Running warm start benchmark...')
            benchmark_results['tests']['warm_start'] = self._benchmark_warm_start(
                llm_service, options['detail_levels'], options['iterations']
            )
            
            # Model switching benchmark
            self.stdout.write('Running model switching benchmark...')
            benchmark_results['tests']['model_switching'] = self._benchmark_model_switching(
                llm_service, options['iterations']
            )
            
            # Concurrent requests benchmark
            self.stdout.write('Running concurrent requests benchmark...')
            benchmark_results['tests']['concurrent'] = self._benchmark_concurrent_requests(
                llm_service, options['concurrent_requests'], options['iterations']
            )
            
            # Enhance results with percentiles if requested
            if options['percentiles']:
                benchmark_results = self._enhance_results_with_percentiles(
                    benchmark_results, calculate_percentiles=True
                )
            
            # Calculate summary statistics
            benchmark_results['summary'] = self._calculate_summary_stats(benchmark_results['tests'])
            
            # Perform regression testing if requested
            if options.get('regression_test'):
                regression_results = self._perform_regression_test(
                    benchmark_results, options['regression_test']
                )
                benchmark_results['regression_analysis'] = regression_results
            
            # Output results
            self._output_results(benchmark_results, options.get('output_file'))
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {str(e)}")
            raise CommandError(f"Benchmarking failed: {str(e)}")

    def _run_warmup(self, llm_service):
        """Run warm-up to load models."""
        try:
            warm_up_result = llm_service.warm_up_models()
            self.stdout.write(f"Warm-up completed: {warm_up_result.get('models_successful', 0)} models ready")
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"Warm-up failed: {str(e)}"))

    def _create_test_analysis_data(self) -> Dict[str, Any]:
        """Create consistent test analysis data."""
        return {
            'symbol': 'BENCHMARK',
            'technical_score': 7.5,
            'recommendation': 'BUY',
            'analysis_date': datetime.now().isoformat(),
            'indicators': {
                'rsi': {'value': 45.2, 'signal': 'neutral'},
                'macd': {'value': 0.15, 'signal': 'bullish'},
                'sma_50': {'value': 98.5, 'signal': 'bullish'},
                'sma_200': {'value': 95.0, 'signal': 'bullish'},
                'bollinger_bands': {'upper': 102.5, 'lower': 96.5, 'position': 0.6},
                'volume_trend': {'value': 1.2, 'signal': 'bullish'}
            }
        }

    def _benchmark_cold_start(self, llm_service, detail_levels: List[str], iterations: int) -> Dict[str, Any]:
        """Benchmark cold start performance."""
        results = {}
        
        for detail_level in detail_levels:
            times = []
            errors = []
            
            for i in range(iterations):
                # Clear cache to ensure cold start
                cache.clear()
                
                analysis_data = self._create_test_analysis_data()
                
                start_time = time.time()
                try:
                    result = llm_service.generate_explanation(
                        analysis_data=analysis_data,
                        detail_level=detail_level
                    )
                    response_time = time.time() - start_time
                    
                    if result and 'explanation' in result:
                        times.append(response_time)
                    else:
                        errors.append(f"No valid response on iteration {i+1}")
                        
                except Exception as e:
                    response_time = time.time() - start_time
                    errors.append(f"Error on iteration {i+1}: {str(e)}")
                
                self.stdout.write(f"  {detail_level} cold start {i+1}/{iterations}: {response_time:.2f}s")
            
            results[detail_level] = {
                'times': times,
                'avg_time': statistics.mean(times) if times else None,
                'median_time': statistics.median(times) if times else None,
                'std_dev': statistics.stdev(times) if len(times) > 1 else None,
                'min_time': min(times) if times else None,
                'max_time': max(times) if times else None,
                'success_rate': len(times) / iterations,
                'errors': errors
            }
        
        return results

    def _benchmark_warm_start(self, llm_service, detail_levels: List[str], iterations: int) -> Dict[str, Any]:
        """Benchmark warm start performance (after initial warm-up)."""
        results = {}
        
        # Initial warm-up request for each detail level
        for detail_level in detail_levels:
            analysis_data = self._create_test_analysis_data()
            try:
                llm_service.generate_explanation(
                    analysis_data=analysis_data,
                    detail_level=detail_level
                )
            except Exception:
                pass  # Ignore warm-up errors
        
        # Now benchmark warm starts
        for detail_level in detail_levels:
            times = []
            errors = []
            
            for i in range(iterations):
                analysis_data = self._create_test_analysis_data()
                
                start_time = time.time()
                try:
                    result = llm_service.generate_explanation(
                        analysis_data=analysis_data,
                        detail_level=detail_level
                    )
                    response_time = time.time() - start_time
                    
                    if result and 'explanation' in result:
                        times.append(response_time)
                    else:
                        errors.append(f"No valid response on iteration {i+1}")
                        
                except Exception as e:
                    response_time = time.time() - start_time
                    errors.append(f"Error on iteration {i+1}: {str(e)}")
                
                self.stdout.write(f"  {detail_level} warm start {i+1}/{iterations}: {response_time:.2f}s")
            
            results[detail_level] = {
                'times': times,
                'avg_time': statistics.mean(times) if times else None,
                'median_time': statistics.median(times) if times else None,
                'std_dev': statistics.stdev(times) if len(times) > 1 else None,
                'min_time': min(times) if times else None,
                'max_time': max(times) if times else None,
                'success_rate': len(times) / iterations,
                'errors': errors
            }
        
        return results

    def _benchmark_model_switching(self, llm_service, iterations: int) -> Dict[str, Any]:
        """Benchmark performance when switching between models."""
        detail_levels = ['summary', 'detailed', 'standard']  # Mix different models
        times = []
        errors = []
        model_switches = []
        
        previous_model = None
        
        for i in range(iterations):
            detail_level = detail_levels[i % len(detail_levels)]
            analysis_data = self._create_test_analysis_data()
            
            start_time = time.time()
            try:
                result = llm_service.generate_explanation(
                    analysis_data=analysis_data,
                    detail_level=detail_level
                )
                response_time = time.time() - start_time
                
                if result and 'explanation' in result:
                    current_model = result.get('model_used', 'unknown')
                    model_switched = previous_model != current_model if previous_model else False
                    
                    times.append(response_time)
                    model_switches.append({
                        'iteration': i + 1,
                        'from_model': previous_model,
                        'to_model': current_model,
                        'switched': model_switched,
                        'response_time': response_time
                    })
                    
                    previous_model = current_model
                else:
                    errors.append(f"No valid response on iteration {i+1}")
                    
            except Exception as e:
                response_time = time.time() - start_time
                errors.append(f"Error on iteration {i+1}: {str(e)}")
            
            self.stdout.write(f"  Model switching {i+1}/{iterations}: {response_time:.2f}s")
        
        return {
            'times': times,
            'avg_time': statistics.mean(times) if times else None,
            'model_switches': model_switches,
            'total_switches': sum(1 for switch in model_switches if switch['switched']),
            'success_rate': len(times) / iterations,
            'errors': errors
        }

    def _benchmark_concurrent_requests(self, llm_service, concurrent_requests: int, iterations: int) -> Dict[str, Any]:
        """Benchmark concurrent request handling."""
        results = []
        
        def make_request(request_id: int, detail_level: str):
            analysis_data = self._create_test_analysis_data()
            
            start_time = time.time()
            try:
                result = llm_service.generate_explanation(
                    analysis_data=analysis_data,
                    detail_level=detail_level
                )
                response_time = time.time() - start_time
                
                results.append({
                    'request_id': request_id,
                    'detail_level': detail_level,
                    'success': bool(result and 'explanation' in result),
                    'response_time': response_time,
                    'model_used': result.get('model_used') if result else None
                })
                
            except Exception as e:
                response_time = time.time() - start_time
                results.append({
                    'request_id': request_id,
                    'detail_level': detail_level,
                    'success': False,
                    'response_time': response_time,
                    'error': str(e)
                })
        
        detail_levels = ['summary', 'standard', 'detailed']
        
        for iteration in range(iterations):
            threads = []
            iteration_start = time.time()
            
            # Launch concurrent requests
            for i in range(concurrent_requests):
                detail_level = detail_levels[i % len(detail_levels)]
                thread = threading.Thread(
                    target=make_request, 
                    args=(i, detail_level)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            iteration_time = time.time() - iteration_start
            self.stdout.write(f"  Concurrent iteration {iteration+1}/{iterations}: {iteration_time:.2f}s")
        
        # Analyse results
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        response_times = [r['response_time'] for r in successful_requests]
        
        return {
            'total_requests': len(results),
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'success_rate': len(successful_requests) / len(results) if results else 0,
            'avg_response_time': statistics.mean(response_times) if response_times else None,
            'median_response_time': statistics.median(response_times) if response_times else None,
            'max_response_time': max(response_times) if response_times else None,
            'min_response_time': min(response_times) if response_times else None,
            'concurrent_efficiency': concurrent_requests / statistics.mean(response_times) if response_times else 0,
            'detailed_results': results
        }

    def _calculate_summary_stats(self, tests: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall summary statistics."""
        summary = {
            'cold_vs_warm_improvement': {},
            'model_switching_overhead': None,
            'concurrent_efficiency': None,
            'overall_performance': {}
        }
        
        # Cold vs warm comparison
        if 'cold_start' in tests and 'warm_start' in tests:
            for detail_level in tests['cold_start']:
                if detail_level in tests['warm_start']:
                    cold_avg = tests['cold_start'][detail_level].get('avg_time')
                    warm_avg = tests['warm_start'][detail_level].get('avg_time')
                    
                    if cold_avg and warm_avg:
                        improvement = (cold_avg - warm_avg) / cold_avg * 100
                        summary['cold_vs_warm_improvement'][detail_level] = {
                            'cold_avg': cold_avg,
                            'warm_avg': warm_avg,
                            'improvement_percent': improvement
                        }
        
        # Model switching overhead
        if 'model_switching' in tests:
            switch_avg = tests['model_switching'].get('avg_time')
            if switch_avg:
                summary['model_switching_overhead'] = switch_avg
        
        # Concurrent efficiency
        if 'concurrent' in tests:
            summary['concurrent_efficiency'] = tests['concurrent'].get('concurrent_efficiency')
        
        return summary

    def _output_results(self, results: Dict[str, Any], output_file: str = None):
        """Output benchmark results."""
        # Console output
        self.stdout.write(self.style.SUCCESS('\n=== LLM BENCHMARK RESULTS ==='))
        
        # Summary statistics
        if 'summary' in results:
            summary = results['summary']
            
            self.stdout.write('\nCold vs Warm Start Improvements:')
            for detail_level, data in summary.get('cold_vs_warm_improvement', {}).items():
                improvement = data['improvement_percent']
                self.stdout.write(f"  {detail_level}: {improvement:.1f}% faster ({data['cold_avg']:.2f}s → {data['warm_avg']:.2f}s)")
            
            if summary.get('model_switching_overhead'):
                self.stdout.write(f"\nModel Switching Average: {summary['model_switching_overhead']:.2f}s")
            
            if summary.get('concurrent_efficiency'):
                self.stdout.write(f"Concurrent Efficiency: {summary['concurrent_efficiency']:.2f} requests/second")
        
        # Performance by detail level
        if 'warm_start' in results:
            self.stdout.write('\nWarm Start Performance:')
            for detail_level, data in results['warm_start'].items():
                if data.get('avg_time'):
                    output_line = f"  {detail_level}: {data['avg_time']:.2f}s avg (success rate: {data['success_rate']:.1%})"
                    
                    # Add percentile information if available
                    if 'percentiles' in data:
                        percentiles = data['percentiles']
                        output_line += f" | 95th: {percentiles.get('p95', 0):.2f}s"
                    
                    self.stdout.write(output_line)
        
        # Regression analysis results
        if 'regression_analysis' in results:
            regression = results['regression_analysis']
            if 'error' not in regression:
                self.stdout.write('\nRegression Analysis:')
                for test_type, comparison in regression.get('comparisons', {}).items():
                    if comparison.get('regression_detected'):
                        self.stdout.write(self.style.ERROR(f"  ⚠️  REGRESSION DETECTED in {test_type}"))
                    
                    for detail_level, change in comparison.get('performance_change', {}).items():
                        direction = change['change_direction']
                        change_pct = change['change_percent']
                        
                        if direction == 'regression':
                            style_func = self.style.ERROR
                            symbol = "📉"
                        elif direction == 'improvement':
                            style_func = self.style.SUCCESS
                            symbol = "📈"
                        else:
                            style_func = self.style.WARNING
                            symbol = "📊"
                        
                        self.stdout.write(style_func(
                            f"    {symbol} {test_type}.{detail_level}: {change_pct:+.1f}% "
                            f"({change['baseline_avg']:.2f}s → {change['current_avg']:.2f}s)"
                        ))
            else:
                self.stdout.write(self.style.ERROR(f"Regression analysis failed: {regression['error']}"))
        
        # File output
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                self.stdout.write(f"\nResults saved to: {output_file}")
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Failed to save results: {str(e)}"))
        
        self.stdout.write(self.style.SUCCESS('\nBenchmarking completed successfully!'))

    def _calculate_percentiles(self, times: List[float]) -> Dict[str, float]:
        """Calculate detailed percentile statistics."""
        if not times or len(times) < 3:
            return {}
        
        try:
            percentiles = {}
            for p in [50, 90, 95, 99]:
                percentiles[f'p{p}'] = np.percentile(times, p)
            
            percentiles.update({
                'mean': np.mean(times),
                'std_dev': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'coefficient_of_variation': np.std(times) / np.mean(times) if np.mean(times) > 0 else 0
            })
            
            return percentiles
        except Exception as e:
            logger.warning(f"Failed to calculate percentiles: {str(e)}")
            return {}

    def _perform_regression_test(self, current_results: Dict[str, Any], baseline_file: str) -> Dict[str, Any]:
        """Compare current results against baseline for regression testing."""
        try:
            with open(baseline_file, 'r') as f:
                baseline_results = json.load(f)
            
            regression_analysis = {
                'baseline_timestamp': baseline_results.get('timestamp'),
                'current_timestamp': current_results.get('timestamp'),
                'comparisons': {}
            }
            
            for test_type in ['cold_start', 'warm_start', 'model_switching']:
                if test_type in baseline_results.get('tests', {}) and test_type in current_results.get('tests', {}):
                    baseline_data = baseline_results['tests'][test_type]
                    current_data = current_results['tests'][test_type]
                    
                    comparison = self._compare_performance_data(baseline_data, current_data, test_type)
                    regression_analysis['comparisons'][test_type] = comparison
            
            return regression_analysis
            
        except Exception as e:
            logger.error(f"Regression test failed: {str(e)}")
            return {'error': str(e)}

    def _compare_performance_data(self, baseline: Dict[str, Any], current: Dict[str, Any], test_type: str) -> Dict[str, Any]:
        """Compare performance data between baseline and current results."""
        comparison = {
            'performance_change': {},
            'statistical_significance': {},
            'regression_detected': False
        }
        
        if test_type in ['cold_start', 'warm_start']:
            for detail_level in baseline.keys():
                if detail_level in current:
                    baseline_times = baseline[detail_level].get('times', [])
                    current_times = current[detail_level].get('times', [])
                    
                    if baseline_times and current_times:
                        baseline_avg = statistics.mean(baseline_times)
                        current_avg = statistics.mean(current_times)
                        
                        change_percent = ((current_avg - baseline_avg) / baseline_avg) * 100
                        comparison['performance_change'][detail_level] = {
                            'baseline_avg': baseline_avg,
                            'current_avg': current_avg,
                            'change_percent': change_percent,
                            'change_direction': 'regression' if change_percent > 10 else 'improvement' if change_percent < -5 else 'stable'
                        }
                        
                        # Statistical significance test
                        try:
                            t_stat, p_value = stats.ttest_ind(baseline_times, current_times)
                            comparison['statistical_significance'][detail_level] = {
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            }
                        except Exception as e:
                            logger.warning(f"Statistical test failed for {detail_level}: {str(e)}")
                        
                        # Mark regression if performance degraded significantly
                        if change_percent > 15:  # More than 15% slower
                            comparison['regression_detected'] = True
        
        return comparison

    def _enhance_results_with_percentiles(self, results: Dict[str, Any], calculate_percentiles: bool) -> Dict[str, Any]:
        """Enhance results with detailed percentile statistics."""
        if not calculate_percentiles:
            return results
        
        enhanced_results = results.copy()
        
        for test_type in ['cold_start', 'warm_start']:
            if test_type in enhanced_results.get('tests', {}):
                test_data = enhanced_results['tests'][test_type]
                for detail_level, data in test_data.items():
                    if 'times' in data and data['times']:
                        percentiles = self._calculate_percentiles(data['times'])
                        data['percentiles'] = percentiles
        
        return enhanced_results
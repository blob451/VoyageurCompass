"""
Performance testing command for optimization validation.
Tests various system components and measures performance improvements.
"""

import logging
import statistics
import time
from datetime import datetime

from django.core.management.base import BaseCommand
from django.db import transaction

from Analytics.engine.ta_engine import TechnicalAnalysisEngine
from Analytics.services.batch_analysis_service import get_batch_analysis_service
from Analytics.services.sentiment_analyzer import get_sentiment_analyzer
from Core.caching import multi_cache
from Data.models import Stock

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Run performance tests to validate optimization improvements"
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--test-type',
            type=str,
            choices=['all', 'single', 'batch', 'cache', 'database'],
            default='all',
            help='Type of performance test to run'
        )
        
        parser.add_argument(
            '--symbols',
            type=str,
            default='AAPL,MSFT,GOOGL,AMZN,TSLA',
            help='Comma-separated list of symbols to test'
        )
        
        parser.add_argument(
            '--iterations',
            type=int,
            default=5,
            help='Number of test iterations'
        )
        
        parser.add_argument(
            '--warmup',
            action='store_true',
            help='Run warmup iterations before testing'
        )
    
    def handle(self, *args, **options):
        """Execute performance tests based on options."""
        
        self.stdout.write(
            self.style.SUCCESS(
                f"Starting performance tests at {datetime.now()}"
            )
        )
        
        symbols = [s.strip() for s in options['symbols'].split(',')]
        iterations = options['iterations']
        test_type = options['test_type']
        
        # Validate symbols exist
        valid_symbols = list(
            Stock.objects.filter(
                symbol__in=symbols, 
                is_active=True
            ).values_list('symbol', flat=True)
        )
        
        if not valid_symbols:
            self.stdout.write(
                self.style.ERROR("No valid symbols found for testing")
            )
            return
        
        self.stdout.write(f"Testing with symbols: {valid_symbols}")
        
        # Run warmup if requested
        if options['warmup']:
            self.stdout.write("Running warmup iterations...")
            self._run_warmup(valid_symbols[:2])  # Use first 2 symbols for warmup
        
        results = {}
        
        # Run tests based on type
        if test_type in ['all', 'single']:
            results['single_analysis'] = self._test_single_analysis(valid_symbols, iterations)
        
        if test_type in ['all', 'batch']:
            results['batch_analysis'] = self._test_batch_analysis(valid_symbols, iterations)
        
        if test_type in ['all', 'cache']:
            results['cache_performance'] = self._test_cache_performance(valid_symbols, iterations)
        
        if test_type in ['all', 'database']:
            results['database_queries'] = self._test_database_performance(valid_symbols, iterations)
        
        # Display results
        self._display_results(results)
        
        self.stdout.write(
            self.style.SUCCESS(
                f"Performance tests completed at {datetime.now()}"
            )
        )
    
    def _run_warmup(self, symbols):
        """Run warmup iterations to stabilize performance."""
        ta_engine = TechnicalAnalysisEngine()
        
        for symbol in symbols:
            try:
                ta_engine.analyze_stock(symbol)
                time.sleep(0.5)  # Brief pause between warmup runs
            except Exception as e:
                logger.warning(f"Warmup failed for {symbol}: {str(e)}")
    
    def _test_single_analysis(self, symbols, iterations):
        """Test single stock analysis performance."""
        self.stdout.write("Testing single stock analysis...")
        
        ta_engine = TechnicalAnalysisEngine()
        results = []
        
        for i in range(iterations):
            iteration_times = []
            
            for symbol in symbols:
                start_time = time.time()
                
                try:
                    result = ta_engine.analyze_stock(symbol)
                    execution_time = time.time() - start_time
                    iteration_times.append(execution_time)
                    
                    self.stdout.write(
                        f"  Iteration {i+1}, {symbol}: {execution_time:.2f}s, "
                        f"Score: {result.get('composite_score', 'N/A')}/10"
                    )
                    
                except Exception as e:
                    self.stdout.write(
                        self.style.WARNING(
                            f"  Iteration {i+1}, {symbol}: FAILED - {str(e)}"
                        )
                    )
                    continue
            
            if iteration_times:
                results.append({
                    'iteration': i + 1,
                    'average_time': statistics.mean(iteration_times),
                    'min_time': min(iteration_times),
                    'max_time': max(iteration_times),
                    'success_count': len(iteration_times),
                    'total_symbols': len(symbols)
                })
        
        if results:
            overall_avg = statistics.mean([r['average_time'] for r in results])
            overall_min = min([r['min_time'] for r in results])
            overall_max = max([r['max_time'] for r in results])
            
            return {
                'overall_average': overall_avg,
                'overall_min': overall_min,
                'overall_max': overall_max,
                'iterations': results,
                'total_tests': len(results) * len(symbols)
            }
        
        return {'error': 'No successful tests completed'}
    
    def _test_batch_analysis(self, symbols, iterations):
        """Test batch analysis performance."""
        self.stdout.write("Testing batch analysis...")
        
        batch_service = get_batch_analysis_service()
        results = []
        
        for i in range(iterations):
            start_time = time.time()
            
            try:
                result = batch_service.analyze_stock_batch(
                    symbols=symbols,
                    use_cache=(i > 0),  # Use cache after first iteration
                )
                
                execution_time = time.time() - start_time
                
                results.append({
                    'iteration': i + 1,
                    'execution_time': execution_time,
                    'symbols_processed': result.get('stats', {}).get('new_analyses', 0),
                    'cache_hits': result.get('stats', {}).get('cache_hits', 0),
                    'success_rate': result.get('stats', {}).get('success_rate', 0.0),
                    'avg_time_per_stock': result.get('stats', {}).get('average_time_per_stock', 0.0)
                })
                
                self.stdout.write(
                    f"  Batch iteration {i+1}: {execution_time:.2f}s, "
                    f"Success rate: {result.get('stats', {}).get('success_rate', 0.0):.1%}, "
                    f"Cache hits: {result.get('stats', {}).get('cache_hits', 0)}"
                )
                
            except Exception as e:
                self.stdout.write(
                    self.style.WARNING(
                        f"  Batch iteration {i+1}: FAILED - {str(e)}"
                    )
                )
        
        if results:
            return {
                'average_batch_time': statistics.mean([r['execution_time'] for r in results]),
                'average_success_rate': statistics.mean([r['success_rate'] for r in results]),
                'total_cache_hits': sum([r['cache_hits'] for r in results]),
                'iterations': results
            }
        
        return {'error': 'No successful batch tests completed'}
    
    def _test_cache_performance(self, symbols, iterations):
        """Test cache performance."""
        self.stdout.write("Testing cache performance...")
        
        # Test cache hit/miss performance
        cache_results = {
            'set_times': [],
            'get_times_hit': [],
            'get_times_miss': [],
            'multi_get_times': []
        }
        
        test_data = {f"perf_test_{symbol}": f"test_data_for_{symbol}" for symbol in symbols}
        
        for i in range(iterations):
            # Test cache set operations
            start_time = time.time()
            multi_cache.set(f"test_key_{i}", f"test_value_{i}")
            cache_results['set_times'].append(time.time() - start_time)
            
            # Test cache hit
            start_time = time.time()
            multi_cache.get(f"test_key_{i}")
            cache_results['get_times_hit'].append(time.time() - start_time)
            
            # Test cache miss
            start_time = time.time()
            multi_cache.get(f"nonexistent_key_{i}")
            cache_results['get_times_miss'].append(time.time() - start_time)
            
            # Test multi-operations
            keys_to_set = {f"multi_key_{j}": f"multi_value_{j}" for j in range(5)}
            start_time = time.time()
            for key, value in keys_to_set.items():
                multi_cache.set(key, value)
            cache_results['multi_get_times'].append(time.time() - start_time)
        
        # Get cache statistics
        cache_stats = multi_cache.get_stats()
        
        return {
            'average_set_time': statistics.mean(cache_results['set_times']),
            'average_get_hit_time': statistics.mean(cache_results['get_times_hit']),
            'average_get_miss_time': statistics.mean(cache_results['get_times_miss']),
            'average_multi_set_time': statistics.mean(cache_results['multi_get_times']),
            'cache_stats': cache_stats,
            'l1_size': len(multi_cache.l1_cache)
        }
    
    def _test_database_performance(self, symbols, iterations):
        """Test database query performance."""
        self.stdout.write("Testing database performance...")
        
        results = {
            'stock_queries': [],
            'bulk_queries': [],
            'index_usage': []
        }
        
        for i in range(iterations):
            # Test individual stock queries
            start_time = time.time()
            for symbol in symbols:
                Stock.objects.filter(symbol=symbol).first()
            results['stock_queries'].append(time.time() - start_time)
            
            # Test bulk queries
            start_time = time.time()
            stocks = Stock.objects.filter(symbol__in=symbols).select_related()
            list(stocks)  # Force query execution
            results['bulk_queries'].append(time.time() - start_time)
            
            # Test index usage (price queries)
            start_time = time.time()
            for stock in stocks:
                # This should use the new index
                recent_prices = stock.prices.all()[:5]
                list(recent_prices)
            results['index_usage'].append(time.time() - start_time)
        
        return {
            'average_individual_queries': statistics.mean(results['stock_queries']),
            'average_bulk_queries': statistics.mean(results['bulk_queries']),
            'average_index_queries': statistics.mean(results['index_usage']),
            'query_efficiency_ratio': (
                statistics.mean(results['stock_queries']) / 
                statistics.mean(results['bulk_queries'])
            )
        }
    
    def _display_results(self, results):
        """Display performance test results in formatted output."""
        
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write(self.style.SUCCESS("PERFORMANCE TEST RESULTS"))
        self.stdout.write("=" * 80)
        
        for test_name, test_results in results.items():
            if 'error' in test_results:
                self.stdout.write(
                    self.style.ERROR(f"\n{test_name.upper()}: {test_results['error']}")
                )
                continue
            
            self.stdout.write(f"\n{test_name.upper()}:")
            self.stdout.write("-" * 40)
            
            if test_name == 'single_analysis':
                self.stdout.write(f"  Overall Average Time: {test_results['overall_average']:.2f}s")
                self.stdout.write(f"  Best Time: {test_results['overall_min']:.2f}s")
                self.stdout.write(f"  Worst Time: {test_results['overall_max']:.2f}s")
                self.stdout.write(f"  Total Tests: {test_results['total_tests']}")
            
            elif test_name == 'batch_analysis':
                self.stdout.write(f"  Average Batch Time: {test_results['average_batch_time']:.2f}s")
                self.stdout.write(f"  Average Success Rate: {test_results['average_success_rate']:.1%}")
                self.stdout.write(f"  Total Cache Hits: {test_results['total_cache_hits']}")
            
            elif test_name == 'cache_performance':
                self.stdout.write(f"  Average Set Time: {test_results['average_set_time']*1000:.2f}ms")
                self.stdout.write(f"  Average Get (Hit) Time: {test_results['average_get_hit_time']*1000:.2f}ms")
                self.stdout.write(f"  Average Get (Miss) Time: {test_results['average_get_miss_time']*1000:.2f}ms")
                self.stdout.write(f"  L1 Cache Hit Rate: {test_results['cache_stats']['l1_hit_rate']:.1%}")
                self.stdout.write(f"  L1 Cache Size: {test_results['l1_size']} items")
            
            elif test_name == 'database_queries':
                self.stdout.write(f"  Individual Queries: {test_results['average_individual_queries']:.3f}s")
                self.stdout.write(f"  Bulk Queries: {test_results['average_bulk_queries']:.3f}s")
                self.stdout.write(f"  Index Queries: {test_results['average_index_queries']:.3f}s")
                self.stdout.write(f"  Efficiency Ratio: {test_results['query_efficiency_ratio']:.1f}x")
        
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write("Test completed successfully!")
        self.stdout.write("=" * 80)
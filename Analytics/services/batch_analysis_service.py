"""
Batch analysis service implementing concurrent stock analysis processing.
Leverages async processing pipeline for high-performance batch operations.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional

from django.core.cache import cache
from django.db import transaction

from Analytics.engine.ta_engine import TechnicalAnalysisEngine
from Analytics.services.async_processing_pipeline import AsyncProcessingPipeline
from Data.models import AnalyticsResults, Stock
from Data.repo.analytics_writer import AnalyticsWriter

logger = logging.getLogger(__name__)


class BatchAnalysisService:
    """High-performance batch analysis service with async processing."""
    
    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self.ta_engine = TechnicalAnalysisEngine()
        self.analytics_writer = AnalyticsWriter()
        self.async_pipeline = AsyncProcessingPipeline(max_workers=max_workers)
        
        # Performance tracking
        self.batch_stats = {
            'total_batches': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'average_time_per_stock': 0.0,
            'cache_hits': 0,
        }
        
        logger.info(f"BatchAnalysisService initialised with {max_workers} workers")
    
    def analyze_stock_batch(self, 
                           symbols: List[str], 
                           user=None,
                           use_cache: bool = True,
                           cache_ttl: int = 1800) -> Dict[str, Any]:
        """
        Analyse multiple stocks concurrently with optimised performance.
        
        Args:
            symbols: Stock symbols to analyse
            user: User requesting the analysis
            use_cache: Whether to use cached results
            cache_ttl: Cache time-to-live in seconds
            
        Returns:
            Batch analysis results with performance metrics
        """
        batch_id = f"batch_{int(time.time())}_{len(symbols)}"
        start_time = time.time()
        
        logger.info(f"Starting batch analysis for {len(symbols)} stocks: {batch_id}")
        
        # Filter valid symbols and check cache
        valid_symbols, cached_results = self._prepare_batch_symbols(symbols, use_cache)
        
        if not valid_symbols:
            logger.warning(f"No valid symbols to analyse in batch {batch_id}")
            return {
                'batch_id': batch_id,
                'results': cached_results,
                'stats': {'cache_hits': len(cached_results), 'new_analyses': 0}
            }
        
        # Create analysis requests
        analysis_requests = [
            {
                'symbol': symbol,
                'user': user,
                'batch_id': batch_id,
                'analysis_date': datetime.now()
            }
            for symbol in valid_symbols
        ]
        
        # Execute concurrent analysis
        batch_results = self.async_pipeline.process_batch_analysis(
            analysis_requests=analysis_requests,
            processor_func=self._analyze_single_stock,
            batch_id=batch_id
        )
        
        # Combine cached and new results
        all_results = {**cached_results, **batch_results.get('results', {})}
        
        # Cache new results
        if use_cache:
            self._cache_batch_results(batch_results.get('results', {}), cache_ttl)
        
        # Update performance statistics
        processing_time = time.time() - start_time
        self._update_batch_stats(len(symbols), len(valid_symbols), processing_time)
        
        logger.info(f"Batch analysis completed: {batch_id} in {processing_time:.2f}s")
        
        return {
            'batch_id': batch_id,
            'results': all_results,
            'stats': {
                'total_symbols': len(symbols),
                'new_analyses': len(valid_symbols),
                'cache_hits': len(cached_results),
                'processing_time': processing_time,
                'success_rate': batch_results.get('success_rate', 0.0),
                'average_time_per_stock': processing_time / max(len(valid_symbols), 1)
            },
            'errors': batch_results.get('errors', [])
        }
    
    def _prepare_batch_symbols(self, symbols: List[str], use_cache: bool) -> tuple:
        """Prepare symbols for batch processing and check cache."""
        
        # Validate symbols against database
        valid_stocks = Stock.objects.filter(
            symbol__in=symbols, 
            is_active=True
        ).values_list('symbol', flat=True)
        
        valid_symbols = list(valid_stocks)
        invalid_symbols = set(symbols) - set(valid_symbols)
        
        if invalid_symbols:
            logger.warning(f"Invalid symbols filtered out: {invalid_symbols}")
        
        cached_results = {}
        symbols_to_analyze = valid_symbols.copy()
        
        # Check cache for existing results
        if use_cache:
            cache_keys = [f"analysis_result:{symbol}" for symbol in valid_symbols]
            cached_data = cache.get_many(cache_keys)
            
            for cache_key, result in cached_data.items():
                symbol = cache_key.split(':')[1]
                cached_results[symbol] = result
                symbols_to_analyze.remove(symbol)
                self.batch_stats['cache_hits'] += 1
        
        return symbols_to_analyze, cached_results
    
    def _analyze_single_stock(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse individual stock with error handling."""
        
        symbol = request_data['symbol']
        user = request_data.get('user')
        
        try:
            # Execute technical analysis
            analysis_result = self.ta_engine.analyze_stock(
                symbol=symbol,
                analysis_date=request_data['analysis_date'],
                user=user
            )
            
            return {
                'symbol': symbol,
                'success': True,
                'result': analysis_result,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'success': False,
                'result': None,
                'error': str(e)
            }
    
    def _cache_batch_results(self, results: Dict[str, Any], cache_ttl: int):
        """Cache successful analysis results."""
        
        cache_data = {}
        for symbol, result_data in results.items():
            if result_data.get('success'):
                cache_key = f"analysis_result:{symbol}"
                cache_data[cache_key] = result_data['result']
        
        if cache_data:
            cache.set_many(cache_data, cache_ttl)
            logger.debug(f"Cached {len(cache_data)} analysis results")
    
    def _update_batch_stats(self, total_requested: int, analyzed: int, processing_time: float):
        """Update batch processing statistics."""
        
        self.batch_stats['total_batches'] += 1
        self.batch_stats['successful_analyses'] += analyzed
        
        # Update average time per stock (exponential moving average)
        time_per_stock = processing_time / max(analyzed, 1)
        current_avg = self.batch_stats['average_time_per_stock']
        self.batch_stats['average_time_per_stock'] = (
            0.7 * current_avg + 0.3 * time_per_stock if current_avg else time_per_stock
        )
    
    def get_batch_performance_stats(self) -> Dict[str, Any]:
        """Retrieve batch processing performance statistics."""
        
        return {
            **self.batch_stats,
            'cache_hit_rate': (
                self.batch_stats['cache_hits'] / 
                max(self.batch_stats['successful_analyses'], 1)
            ) if self.batch_stats['successful_analyses'] else 0,
            'throughput_stocks_per_second': (
                1.0 / max(self.batch_stats['average_time_per_stock'], 0.001)
            )
        }
    
    async def analyze_portfolio_async(self, portfolio_symbols: List[str], user=None) -> Dict[str, Any]:
        """Asynchronous portfolio analysis with optimised concurrency."""
        
        loop = asyncio.get_event_loop()
        
        # Execute batch analysis in thread pool
        result = await loop.run_in_executor(
            None,
            self.analyze_stock_batch,
            portfolio_symbols,
            user,
            True,  # use_cache
            3600   # cache_ttl (1 hour for portfolios)
        )
        
        return result
    
    def warm_cache_for_symbols(self, symbols: List[str]):
        """Pre-warm cache for frequently accessed symbols."""
        
        logger.info(f"Cache warming initiated for {len(symbols)} symbols")
        
        # Analyse in small batches to avoid overwhelming the system
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i + batch_size]
            try:
                self.analyze_stock_batch(
                    symbols=batch_symbols,
                    use_cache=False,  # Force fresh analysis for cache warming
                    cache_ttl=7200    # 2 hours for warmed cache
                )
                time.sleep(1)  # Brief pause between batches
            except Exception as e:
                logger.warning(f"Cache warming failed for batch {batch_symbols}: {str(e)}")
        
        logger.info("Cache warming completed")


# Global service instance
_batch_analysis_service = None


def get_batch_analysis_service() -> BatchAnalysisService:
    """Retrieve singleton batch analysis service instance."""
    global _batch_analysis_service
    if _batch_analysis_service is None:
        _batch_analysis_service = BatchAnalysisService()
    return _batch_analysis_service
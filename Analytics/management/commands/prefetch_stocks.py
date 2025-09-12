"""
Management command to pre-fetch and cache data for popular stocks.
This ensures faster analysis times for commonly requested stocks.
"""

import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from django.core.management.base import BaseCommand
from django.core.cache import cache
from django.db.models import Count, Q

from Data.models import Stock, StockPrice, AnalyticsResults
from Data.services.yahoo_finance import yahoo_finance_service
from Analytics.engine.ta_engine import TechnicalAnalysisEngine

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Pre-fetch and cache data for popular stocks to improve analysis speed'

    def add_arguments(self, parser):
        parser.add_argument(
            '--top-n',
            type=int,
            default=50,
            help='Number of top stocks to prefetch (default: 50)'
        )
        
        parser.add_argument(
            '--symbols',
            nargs='+',
            help='Specific symbols to prefetch'
        )
        
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force refresh even if data exists'
        )
        
        parser.add_argument(
            '--include-analysis',
            action='store_true',
            help='Also run and cache technical analysis'
        )
        
        parser.add_argument(
            '--parallel',
            type=int,
            default=3,
            help='Number of parallel workers (default: 3)'
        )

    def handle(self, *args, **options):
        """Execute stock pre-fetching."""
        self.stdout.write(self.style.SUCCESS('=== Stock Data Pre-fetching ==='))
        
        top_n = options.get('top_n', 50)
        symbols = options.get('symbols')
        force_refresh = options.get('force', False)
        include_analysis = options.get('include_analysis', False)
        parallel_workers = options.get('parallel', 3)
        
        # Get stocks to prefetch
        if symbols:
            stocks_to_fetch = symbols
            self.stdout.write(f'Pre-fetching specified symbols: {", ".join(symbols)}')
        else:
            # Get most analyzed stocks or popular market stocks
            stocks_to_fetch = self._get_popular_stocks(top_n)
            self.stdout.write(f'Pre-fetching top {len(stocks_to_fetch)} stocks')
        
        if not stocks_to_fetch:
            self.stdout.write(self.style.WARNING('No stocks to prefetch'))
            return
        
        # Track results
        success_count = 0
        failure_count = 0
        cached_count = 0
        
        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(
                    self._prefetch_stock, 
                    symbol, 
                    force_refresh,
                    include_analysis
                ): symbol 
                for symbol in stocks_to_fetch
            }
            
            # Process completed tasks
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result(timeout=120)  # 2 minute timeout
                    
                    if result['status'] == 'success':
                        success_count += 1
                        self.stdout.write(
                            self.style.SUCCESS(
                                f'{symbol}: Pre-fetched successfully '
                                f'({result["records"]} records, '
                                f'{result["time"]:.1f}s)'
                            )
                        )
                    elif result['status'] == 'cached':
                        cached_count += 1
                        self.stdout.write(f'{symbol}: Already cached (skipped)')
                    else:
                        failure_count += 1
                        self.stdout.write(
                            self.style.ERROR(
                                f'{symbol}: Failed - {result.get("error", "Unknown error")}'
                            )
                        )
                        
                except Exception as e:
                    failure_count += 1
                    self.stdout.write(
                        self.style.ERROR(f'{symbol}: Error - {str(e)}')
                    )
                    logger.exception(f'Error prefetching {symbol}')
        
        # Summary
        self.stdout.write('\n' + '='*60)
        self.stdout.write(self.style.SUCCESS(f'Successfully prefetched: {success_count}'))
        if cached_count > 0:
            self.stdout.write(f'Already cached: {cached_count}')
        if failure_count > 0:
            self.stdout.write(self.style.WARNING(f'Failed: {failure_count}'))
        
        self.stdout.write(self.style.SUCCESS('\nPre-fetching complete!'))

    def _get_popular_stocks(self, limit: int) -> List[str]:
        """Get list of popular stocks to prefetch."""
        popular_stocks = []
        
        # 1. Get frequently analyzed stocks from our database
        recent_cutoff = datetime.now() - timedelta(days=30)
        frequently_analyzed = AnalyticsResults.objects.filter(
            created_at__gte=recent_cutoff
        ).values('stock__symbol').annotate(
            count=Count('stock__symbol')
        ).order_by('-count')[:limit//2]
        
        for item in frequently_analyzed:
            popular_stocks.append(item['stock__symbol'])
        
        # 2. Add major market stocks if not already included
        major_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
            'JPM', 'JNJ', 'V', 'UNH', 'HD', 'PG', 'MA', 'DIS', 'BAC', 'XOM',
            'ABBV', 'KO', 'PFE', 'AVGO', 'NKE', 'TMO', 'CSCO', 'PEP', 'CVX',
            'WMT', 'ABT', 'ADBE', 'CRM', 'MRK', 'ACN', 'LLY', 'COST', 'TXN',
            'NFLX', 'AMD', 'WFC', 'MDT', 'UPS', 'INTC', 'ORCL', 'QCOM', 'VZ',
            'T', 'MS', 'HON', 'PM', 'IBM', 'GE'
        ]
        
        # Add major stocks not already in the list
        for stock in major_stocks:
            if stock not in popular_stocks and len(popular_stocks) < limit:
                popular_stocks.append(stock)
        
        return popular_stocks[:limit]

    def _prefetch_stock(
        self, 
        symbol: str, 
        force_refresh: bool = False,
        include_analysis: bool = False
    ) -> Dict[str, Any]:
        """Prefetch data for a single stock."""
        start_time = datetime.now()
        
        try:
            # Check if already cached (24-hour cache)
            cache_key = f'prefetch_stock_{symbol}_24h'
            if not force_refresh and cache.get(cache_key):
                return {
                    'status': 'cached',
                    'symbol': symbol,
                    'time': 0
                }
            
            # Check if stock exists in database
            try:
                stock = Stock.objects.get(symbol=symbol)
                
                # Check data coverage
                one_day_ago = datetime.now() - timedelta(days=1)
                recent_prices = StockPrice.objects.filter(
                    stock=stock,
                    date__gte=one_day_ago
                ).count()
                
                if recent_prices > 0 and not force_refresh:
                    # Data is recent enough
                    cache.set(cache_key, True, 24 * 60 * 60)
                    return {
                        'status': 'cached',
                        'symbol': symbol,
                        'time': 0
                    }
                    
            except Stock.DoesNotExist:
                # Stock doesn't exist, will be created during backfill
                pass
            
            # Perform backfill
            logger.info(f'Pre-fetching data for {symbol}')
            
            # Use concurrent backfill for efficiency
            backfill_result = yahoo_finance_service.backfill_eod_gaps_concurrent(
                symbol=symbol,
                required_years=2,
                max_attempts=1  # Single attempt for prefetch
            )
            
            if not backfill_result.get('success'):
                return {
                    'status': 'failed',
                    'symbol': symbol,
                    'error': backfill_result.get('errors', ['Unknown error'])[0],
                    'time': (datetime.now() - start_time).total_seconds()
                }
            
            records_fetched = backfill_result.get('stock_backfilled', 0)
            
            # Optionally run analysis to cache results
            if include_analysis:
                try:
                    engine = TechnicalAnalysisEngine()
                    
                    # Get fresh stock object
                    stock = Stock.objects.get(symbol=symbol)
                    
                    # Run analysis (this will cache sentiment/prediction)
                    analysis_result = engine.analyze_stock(
                        symbol=symbol,
                        horizon='blend',
                        fast_mode=False  # Full analysis to populate cache
                    )
                    
                    if analysis_result:
                        logger.info(f'Analysis cached for {symbol}: Score {analysis_result.composite_score}/10')
                        
                except Exception as e:
                    logger.warning(f'Could not cache analysis for {symbol}: {str(e)}')
            
            # Mark as cached
            cache.set(cache_key, True, 24 * 60 * 60)
            
            # Also fit scalers for this stock
            self._fit_scalers_for_stock(symbol)
            
            return {
                'status': 'success',
                'symbol': symbol,
                'records': records_fetched,
                'time': (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            logger.error(f'Failed to prefetch {symbol}: {str(e)}')
            return {
                'status': 'failed',
                'symbol': symbol,
                'error': str(e),
                'time': (datetime.now() - start_time).total_seconds()
            }

    def _fit_scalers_for_stock(self, symbol: str):
        """Fit Universal LSTM scalers for the stock."""
        try:
            from Analytics.management.commands.fit_universal_scalers import Command as ScalerCommand
            
            # Create instance of scaler fitting command
            scaler_cmd = ScalerCommand()
            
            # Get stock prices
            stock = Stock.objects.get(symbol=symbol)
            prices = StockPrice.objects.filter(
                stock=stock
            ).order_by('-date')[:252]  # 1 year of data
            
            if prices.count() < 60:
                return  # Not enough data
            
            # Build features
            features = []
            for i in range(len(prices) - 1):
                price_data = prices[i]
                prev_price = prices[i + 1] if i + 1 < len(prices) else None
                
                if prev_price:
                    feature_dict = {
                        'price': float(price_data.close),
                        'volume': float(price_data.volume),
                        'price_change': float(price_data.close - prev_price.close),
                        'volume_ratio': float(price_data.volume / prev_price.volume) if prev_price.volume > 0 else 1.0,
                        'high_low_ratio': float(price_data.high / price_data.low) if price_data.low > 0 else 1.0,
                        'close_open_ratio': float(price_data.close / price_data.open) if price_data.open > 0 else 1.0,
                    }
                    features.append(feature_dict)
            
            if len(features) >= 30:
                # Fit scalers
                scaler_data = scaler_cmd._fit_scalers_for_stock(symbol, features)
                if scaler_data:
                    scaler_cmd._save_scalers_to_file(symbol, scaler_data)
                    logger.info(f'Fitted scalers for {symbol}')
                    
        except Exception as e:
            logger.debug(f'Could not fit scalers for {symbol}: {str(e)}')
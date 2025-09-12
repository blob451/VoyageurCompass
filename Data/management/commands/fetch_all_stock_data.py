"""
Bulk Stock Data Fetching Command

Fetches historical price data for all stocks missing data from Yahoo Finance.
Processes stocks in batches to avoid API rate limits and handles errors gracefully.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from django.core.management.base import BaseCommand
from django.db.models import Q
from django.utils import timezone

from Data.models import Stock
from Data.services.yahoo_finance import yahoo_finance_service

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Fetch historical price data for all stocks missing data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--batch-size',
            type=int,
            default=10,
            help='Number of stocks to process in parallel (default: 10)'
        )
        
        parser.add_argument(
            '--years',
            type=int,
            default=2,
            help='Number of years of history to fetch (default: 2)'
        )
        
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be processed without fetching data'
        )
        
        parser.add_argument(
            '--force',
            action='store_true',
            help='Fetch data for all stocks, even those with existing data'
        )

    def handle(self, *args, **options):
        """Execute bulk stock data fetching."""
        batch_size = options['batch_size']
        years = options['years']
        dry_run = options['dry_run']
        force = options['force']
        
        self.stdout.write(self.style.SUCCESS('=== BULK STOCK DATA FETCHING ==='))
        self.stdout.write(f'Batch size: {batch_size}')
        self.stdout.write(f'Years of history: {years}')
        self.stdout.write(f'Dry run: {dry_run}')
        self.stdout.write(f'Force update: {force}')
        
        # Get real stocks (exclude test stocks)
        test_patterns = [
            'AAPL_', 'CASCADE_TEST', 'CONSTRAINT_TEST', 'NO_PRICES', 
            'PORTFOLIO_TEST', 'TSLA_MOCK', 'TEST_STOCK'
        ]
        
        real_stocks_query = Stock.objects.filter(is_active=True)
        for pattern in test_patterns:
            real_stocks_query = real_stocks_query.exclude(symbol__icontains=pattern)
        
        if not force:
            # Only get stocks without price data
            stocks_to_fetch = real_stocks_query.filter(prices__isnull=True).distinct()
        else:
            stocks_to_fetch = real_stocks_query
        
        stocks_list = list(stocks_to_fetch.values_list('symbol', flat=True))
        total_stocks = len(stocks_list)
        
        self.stdout.write(f'\nFound {total_stocks} stocks to process')
        
        if dry_run:
            self.stdout.write('\nDRY RUN - Stocks that would be processed:')
            for i, symbol in enumerate(stocks_list, 1):
                self.stdout.write(f'  {i:3d}. {symbol}')
            return
        
        if total_stocks == 0:
            self.stdout.write(self.style.SUCCESS('All stocks already have data'))
            return
        
        # Process stocks in batches
        self.stdout.write(f'\nStarting data fetching for {total_stocks} stocks...')
        start_time = timezone.now()
        
        total_success = 0
        total_failed = 0
        
        for i in range(0, total_stocks, batch_size):
            batch_symbols = stocks_list[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_stocks + batch_size - 1) // batch_size
            
            self.stdout.write(f'\nProcessing batch {batch_num}/{total_batches}: {len(batch_symbols)} stocks')
            
            batch_success, batch_failed = self._process_batch(batch_symbols, years)
            
            total_success += batch_success
            total_failed += batch_failed
            
            # Progress update
            completed = i + len(batch_symbols)
            progress_pct = (completed / total_stocks) * 100
            self.stdout.write(f'Batch completed: {batch_success} success, {batch_failed} failed')
            self.stdout.write(f'Progress: {completed}/{total_stocks} ({progress_pct:.1f}%)')
            
            # Rate limiting between batches (avoid overwhelming Yahoo Finance API)
            if i + batch_size < total_stocks:
                self.stdout.write('Waiting 30 seconds before next batch...')
                time.sleep(30)
        
        # Final summary
        end_time = timezone.now()
        duration = end_time - start_time
        
        self.stdout.write('\n' + '='*60)
        self.stdout.write(self.style.SUCCESS('BULK FETCH COMPLETED'))
        self.stdout.write(f'Total processed: {total_stocks} stocks')
        self.stdout.write(f'Successful: {total_success}')
        self.stdout.write(f'Failed: {total_failed}')
        self.stdout.write(f'Duration: {duration}')
        
        if total_failed > 0:
            success_rate = (total_success / total_stocks) * 100
            self.stdout.write(f'Success rate: {success_rate:.1f}%')
        
        self.stdout.write('='*60)

    def _process_batch(self, symbols, years):
        """Process a batch of stock symbols concurrently."""
        success_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._fetch_stock_data, symbol, years): symbol 
                for symbol in symbols
            }
            
            # Process completed tasks
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                        self.stdout.write(f'    SUCCESS {symbol}')
                    else:
                        failed_count += 1
                        self.stdout.write(f'    FAILED {symbol} (fetch failed)')
                except Exception as e:
                    failed_count += 1
                    self.stdout.write(f'    ERROR {symbol} (error: {str(e)})')
        
        return success_count, failed_count

    def _fetch_stock_data(self, symbol, years):
        """Fetch historical data for a single stock."""
        try:
            # Use the existing backfill service
            result = yahoo_finance_service.backfill_eod_gaps_concurrent(
                symbol=symbol,
                required_years=years,
                max_attempts=3
            )
            
            if result.get('success', False):
                prices_added = result.get('gaps_filled', 0)
                logger.info(f'Successfully fetched {prices_added} price records for {symbol}')
                return True
            else:
                error = result.get('error', 'Unknown error')
                logger.warning(f'Failed to fetch data for {symbol}: {error}')
                return False
                
        except Exception as e:
            logger.error(f'Exception fetching data for {symbol}: {str(e)}')
            return False
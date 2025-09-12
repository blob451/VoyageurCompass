"""
Management command to sync historical price data for benchmark stocks.
"""

import logging
import time
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from Data.models import Stock, StockPrice
from Data.services.yahoo_finance import yahoo_finance_service

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Sync historical price data for benchmark stocks'

    def add_arguments(self, parser):
        parser.add_argument(
            '--symbols',
            nargs='+',
            default=['SPY', 'QQQ', 'DIA', 'VTI', 'VXUS', 'VEA', 'VWO', 'BND', 'VNQ', 'GLD'],
            help='Benchmark symbols to sync (default: all major benchmarks)'
        )
        
        parser.add_argument(
            '--years-back',
            type=int,
            default=3,
            help='Number of years of historical data to fetch (default: 3)'
        )
        
        parser.add_argument(
            '--force-refresh',
            action='store_true',
            help='Force refresh all data even if already exists'
        )
        
        parser.add_argument(
            '--batch-size',
            type=int,
            default=100,
            help='Batch size for database inserts (default: 100)'
        )

    def handle(self, *args, **options):
        """Execute benchmark price synchronization."""
        start_time = time.time()
        
        self.stdout.write(self.style.SUCCESS('Starting benchmark price synchronization...'))
        
        symbols = options['symbols']
        years_back = options['years_back']
        force_refresh = options['force_refresh']
        batch_size = options['batch_size']
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=years_back * 365)
        
        total_synced = 0
        failed_symbols = []
        
        try:
            for i, symbol in enumerate(symbols, 1):
                self.stdout.write(f'[{i}/{len(symbols)}] Syncing {symbol}...')
                
                try:
                    synced_count = self._sync_symbol_data(
                        symbol, start_date, end_date, force_refresh, batch_size
                    )
                    total_synced += synced_count
                    self.stdout.write(
                        self.style.SUCCESS(f'  SUCCESS {symbol}: {synced_count} records synced')
                    )
                    
                    # Brief pause to avoid rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    failed_symbols.append((symbol, str(e)))
                    self.stdout.write(
                        self.style.ERROR(f'  FAILED {symbol}: {str(e)}')
                    )
                    logger.error(f"Failed to sync {symbol}: {str(e)}")
            
            duration = time.time() - start_time
            
            # Summary
            self.stdout.write('\n' + '='*60)
            self.stdout.write(self.style.SUCCESS(f'Benchmark sync completed in {duration:.2f}s'))
            self.stdout.write(f'Total records synced: {total_synced}')
            self.stdout.write(f'Successful symbols: {len(symbols) - len(failed_symbols)}/{len(symbols)}')
            
            if failed_symbols:
                self.stdout.write(self.style.WARNING('\nFailed symbols:'))
                for symbol, error in failed_symbols:
                    self.stdout.write(f'  {symbol}: {error}')
            
            # Validate completeness
            self.stdout.write('\nValidating data completeness...')
            self._validate_completeness(symbols, start_date, end_date)
            
        except Exception as e:
            raise CommandError(f'Benchmark sync failed: {str(e)}')

    def _sync_symbol_data(self, symbol: str, start_date, end_date, force_refresh: bool, batch_size: int) -> int:
        """Sync price data for a single symbol."""
        # Get or create stock record
        try:
            stock = Stock.objects.get(symbol=symbol)
        except Stock.DoesNotExist:
            self.stdout.write(
                self.style.WARNING(f'  Warning: {symbol} not found in database, skipping')
            )
            return 0
        
        # Check existing data unless force refresh
        existing_count = 0
        if not force_refresh:
            existing_count = StockPrice.objects.filter(
                stock=stock,
                date__gte=start_date,
                date__lte=end_date
            ).count()
            
            if existing_count > 0:
                expected_days = (end_date - start_date).days
                coverage_pct = (existing_count / expected_days) * 100
                
                # Skip if we have good coverage (>80%)
                if coverage_pct > 80:
                    return existing_count
        
        # Fetch data from Yahoo Finance with retry logic
        price_data = None
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    self.stdout.write(f'    Retry {attempt}/{max_retries}...')
                    time.sleep(2 ** attempt)  # Exponential backoff
                
                # Convert date objects to datetime objects
                start_datetime = datetime.combine(start_date, datetime.min.time())
                end_datetime = datetime.combine(end_date, datetime.min.time())
                
                price_data = yahoo_finance_service.fetchStockEodHistory(
                    symbol, start_datetime, end_datetime
                )
                break
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
        
        if not price_data:
            raise Exception("No price data returned from Yahoo Finance")
        
        # Delete existing data if force refresh
        if force_refresh and existing_count > 0:
            StockPrice.objects.filter(
                stock=stock,
                date__gte=start_date,
                date__lte=end_date
            ).delete()
        
        # Batch insert new data
        new_prices = []
        synced_count = 0
        
        for price_record in price_data:
            # Skip if record already exists (unless force refresh)
            if not force_refresh:
                if StockPrice.objects.filter(
                    stock=stock,
                    date=price_record['date']
                ).exists():
                    continue
            
            new_prices.append(StockPrice(
                stock=stock,
                date=price_record['date'],
                open=price_record['open'],
                high=price_record['high'],
                low=price_record['low'],
                close=price_record['close'],
                volume=price_record['volume'],
                adjusted_close=price_record.get('adjusted_close', price_record['close'])
            ))
            
            # Batch insert when we reach batch size
            if len(new_prices) >= batch_size:
                with transaction.atomic():
                    StockPrice.objects.bulk_create(new_prices, ignore_conflicts=True)
                synced_count += len(new_prices)
                new_prices = []
        
        # Insert remaining records
        if new_prices:
            with transaction.atomic():
                StockPrice.objects.bulk_create(new_prices, ignore_conflicts=True)
            synced_count += len(new_prices)
        
        return synced_count

    def _validate_completeness(self, symbols: list, start_date, end_date):
        """Validate data completeness for all symbols."""
        expected_days = (end_date - start_date).days
        
        for symbol in symbols:
            try:
                stock = Stock.objects.get(symbol=symbol)
                actual_count = StockPrice.objects.filter(
                    stock=stock,
                    date__gte=start_date,
                    date__lte=end_date
                ).count()
                
                coverage_pct = (actual_count / expected_days) * 100
                
                if coverage_pct >= 90:
                    status = self.style.SUCCESS('OK')
                elif coverage_pct >= 70:
                    status = self.style.WARNING('WARN')
                else:
                    status = self.style.ERROR('FAIL')
                
                self.stdout.write(
                    f'  {status} {symbol}: {actual_count}/{expected_days} days ({coverage_pct:.1f}%)'
                )
                
            except Stock.DoesNotExist:
                self.stdout.write(
                    self.style.ERROR(f'  FAIL {symbol}: Stock not found in database')
                )
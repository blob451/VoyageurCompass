"""
Update Sector Mappings Command

Updates sector and industry mappings for all stocks by fetching metadata from Yahoo Finance.
Creates proper foreign key relationships to DataSector and DataIndustry models.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from Data.models import Stock, DataSector, DataIndustry
from Data.services.yahoo_finance import yahoo_finance_service

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Update sector and industry mappings for all stocks'

    def add_arguments(self, parser):
        parser.add_argument(
            '--batch-size',
            type=int,
            default=10,
            help='Number of stocks to process in parallel (default: 10)'
        )
        
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be updated without making changes'
        )
        
        parser.add_argument(
            '--force',
            action='store_true',
            help='Update all stocks, even those with existing mappings'
        )
        
        parser.add_argument(
            '--symbols',
            nargs='+',
            help='Update specific symbols only'
        )

    def handle(self, *args, **options):
        """Execute sector mapping updates."""
        batch_size = options['batch_size']
        dry_run = options['dry_run']
        force = options['force']
        specific_symbols = options.get('symbols', [])
        
        self.stdout.write(self.style.SUCCESS('=== SECTOR MAPPING UPDATE ==='))
        self.stdout.write(f'Batch size: {batch_size}')
        self.stdout.write(f'Dry run: {dry_run}')
        self.stdout.write(f'Force update: {force}')
        
        if specific_symbols:
            self.stdout.write(f'Specific symbols: {", ".join(specific_symbols)}')
        
        # Get stocks to update
        if specific_symbols:
            stocks_to_update = Stock.objects.filter(
                symbol__in=[s.upper() for s in specific_symbols],
                is_active=True
            )
        else:
            # Exclude test stocks
            test_patterns = [
                'AAPL_', 'CASCADE_TEST', 'CONSTRAINT_TEST', 'NO_PRICES', 
                'PORTFOLIO_TEST', 'TSLA_MOCK', 'TEST_STOCK', 'PERF_', 'ROLLBACK_TEST'
            ]
            
            stocks_query = Stock.objects.filter(is_active=True)
            for pattern in test_patterns:
                stocks_query = stocks_query.exclude(symbol__icontains=pattern)
            
            if not force:
                # Only update stocks missing mappings
                stocks_to_update = stocks_query.filter(
                    sector_id__isnull=True
                ).distinct()
            else:
                stocks_to_update = stocks_query
        
        stocks_list = list(stocks_to_update.values_list('symbol', flat=True))
        total_stocks = len(stocks_list)
        
        self.stdout.write(f'\nFound {total_stocks} stocks to update')
        
        if dry_run:
            self.stdout.write('\nDRY RUN - Stocks that would be updated:')
            for i, symbol in enumerate(stocks_list, 1):
                try:
                    stock = Stock.objects.get(symbol=symbol)
                    current_sector = stock.sector or 'None'
                    current_industry = stock.industry or 'None'
                    self.stdout.write(f'  {i:3d}. {symbol} (sector: {current_sector}, industry: {current_industry})')
                except Stock.DoesNotExist:
                    pass
            return
        
        if total_stocks == 0:
            self.stdout.write(self.style.SUCCESS('All stocks already have sector mappings'))
            return
        
        # Process stocks in batches
        self.stdout.write(f'\nStarting sector mapping for {total_stocks} stocks...')
        start_time = timezone.now()
        
        total_success = 0
        total_failed = 0
        total_unchanged = 0
        
        for i in range(0, total_stocks, batch_size):
            batch_symbols = stocks_list[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_stocks + batch_size - 1) // batch_size
            
            self.stdout.write(f'\nProcessing batch {batch_num}/{total_batches}: {len(batch_symbols)} stocks')
            
            batch_success, batch_failed, batch_unchanged = self._process_batch(batch_symbols)
            
            total_success += batch_success
            total_failed += batch_failed
            total_unchanged += batch_unchanged
            
            # Progress update
            completed = i + len(batch_symbols)
            progress_pct = (completed / total_stocks) * 100
            self.stdout.write(f'Batch completed: {batch_success} updated, {batch_unchanged} unchanged, {batch_failed} failed')
            self.stdout.write(f'Progress: {completed}/{total_stocks} ({progress_pct:.1f}%)')
            
            # Rate limiting between batches
            if i + batch_size < total_stocks:
                self.stdout.write('Waiting 10 seconds before next batch...')
                time.sleep(10)
        
        # Final summary
        end_time = timezone.now()
        duration = end_time - start_time
        
        self.stdout.write('\n' + '='*60)
        self.stdout.write(self.style.SUCCESS('SECTOR MAPPING UPDATE COMPLETED'))
        self.stdout.write(f'Total processed: {total_stocks} stocks')
        self.stdout.write(f'Updated: {total_success}')
        self.stdout.write(f'Unchanged: {total_unchanged}')
        self.stdout.write(f'Failed: {total_failed}')
        self.stdout.write(f'Duration: {duration}')
        
        if total_failed > 0:
            success_rate = (total_success / total_stocks) * 100
            self.stdout.write(f'Success rate: {success_rate:.1f}%')
        
        self.stdout.write('='*60)

    def _process_batch(self, symbols):
        """Process a batch of stock symbols concurrently."""
        success_count = 0
        failed_count = 0
        unchanged_count = 0
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._update_stock_mapping, symbol): symbol 
                for symbol in symbols
            }
            
            # Process completed tasks
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result == 'updated':
                        success_count += 1
                        self.stdout.write(f'    UPDATED {symbol}')
                    elif result == 'unchanged':
                        unchanged_count += 1
                        self.stdout.write(f'    UNCHANGED {symbol}')
                    else:
                        failed_count += 1
                        self.stdout.write(f'    FAILED {symbol} ({result})')
                except Exception as e:
                    failed_count += 1
                    self.stdout.write(f'    ERROR {symbol} (error: {str(e)})')
        
        return success_count, failed_count, unchanged_count

    def _update_stock_mapping(self, symbol):
        """Update sector and industry mapping for a single stock."""
        try:
            with transaction.atomic():
                # Get stock record
                stock = Stock.objects.get(symbol=symbol)
                
                # Fetch info from Yahoo Finance
                stock_info = yahoo_finance_service.get_stock_info(symbol)
                
                if 'error' in stock_info:
                    return f"API error: {stock_info['error']}"
                
                # Extract sector and industry
                sector_name = (stock_info.get('sector') or '').strip()
                industry_name = (stock_info.get('industry') or '').strip()
                
                # Handle special cases
                if not sector_name:
                    # Check if it's an ETF
                    if any(keyword in stock_info.get('longName', '').lower() for keyword in ['etf', 'fund', 'trust']):
                        sector_name = 'ETF/Index'
                        industry_name = 'Exchange Traded Fund'
                    else:
                        sector_name = 'Unknown'
                        industry_name = 'Unknown'
                
                # Check if already up to date
                existing_sector = stock.sector or ''
                if existing_sector == sector_name and stock.sector_id is not None:
                    return 'unchanged'
                
                # Create or get sector
                sector_obj = None
                if sector_name and sector_name != 'Unknown':
                    sector_obj = self._get_or_create_sector(sector_name)
                
                # Create or get industry
                industry_obj = None
                if industry_name and industry_name != 'Unknown' and sector_obj:
                    industry_obj = self._get_or_create_industry(industry_name, sector_obj)
                
                # Update stock record
                stock.sector = sector_name
                stock.industry = industry_name
                stock.sector_id = sector_obj
                stock.industry_id = industry_obj
                stock.sectorUpdatedAt = timezone.now()
                stock.save()
                
                logger.info(f'Updated {symbol}: sector={sector_name}, industry={industry_name}')
                return 'updated'
                
        except Stock.DoesNotExist:
            return 'stock not found'
        except Exception as e:
            logger.error(f'Error updating {symbol}: {str(e)}')
            return str(e)

    def _get_or_create_sector(self, sector_name):
        """Get or create a DataSector object."""
        sector_key = self._normalize_key(sector_name)
        
        sector_obj, created = DataSector.objects.get_or_create(
            sectorKey=sector_key,
            defaults={
                'sectorName': sector_name,
                'isActive': True,
                'data_source': 'yahoo'
            }
        )
        
        if created:
            logger.info(f'Created new sector: {sector_name} ({sector_key})')
        
        return sector_obj

    def _get_or_create_industry(self, industry_name, sector_obj):
        """Get or create a DataIndustry object."""
        industry_key = self._normalize_key(industry_name)
        
        industry_obj, created = DataIndustry.objects.get_or_create(
            industryKey=industry_key,
            defaults={
                'industryName': industry_name,
                'sector': sector_obj,
                'isActive': True,
                'data_source': 'yahoo'
            }
        )
        
        if created:
            logger.info(f'Created new industry: {industry_name} ({industry_key}) under {sector_obj.sectorName}')
        
        return industry_obj

    def _normalize_key(self, name):
        """Normalize a name to create a consistent key."""
        if not name:
            return 'unknown'
        
        # Convert to lowercase, replace spaces and special chars with underscores
        import re
        normalized = re.sub(r'[^a-zA-Z0-9]+', '_', name.lower()).strip('_')
        
        # Limit length
        if len(normalized) > 50:
            normalized = normalized[:50]
        
        return normalized or 'unknown'
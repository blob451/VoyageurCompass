"""
Update Industry Mappings Command

Updates industry mappings for all stocks and ensures proper sector-industry relationships.
This command should be run after sector mappings are updated.
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
    help = 'Update industry mappings for all stocks and fix sector-industry relationships'

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
            help='Update all stocks, even those with existing industry mappings'
        )
        
        parser.add_argument(
            '--symbols',
            nargs='+',
            help='Update specific symbols only'
        )
        
        parser.add_argument(
            '--fix-relationships',
            action='store_true',
            help='Fix existing sector-industry relationships for consistency'
        )

    def handle(self, *args, **options):
        """Execute industry mapping updates."""
        batch_size = options['batch_size']
        dry_run = options['dry_run']
        force = options['force']
        specific_symbols = options.get('symbols', [])
        fix_relationships = options['fix_relationships']
        
        self.stdout.write(self.style.SUCCESS('=== INDUSTRY MAPPING UPDATE ==='))
        self.stdout.write(f'Batch size: {batch_size}')
        self.stdout.write(f'Dry run: {dry_run}')
        self.stdout.write(f'Force update: {force}')
        self.stdout.write(f'Fix relationships: {fix_relationships}')
        
        if specific_symbols:
            self.stdout.write(f'Specific symbols: {", ".join(specific_symbols)}')
        
        if fix_relationships:
            self.stdout.write('\nFixing existing sector-industry relationships...')
            self._fix_sector_industry_relationships(dry_run)
        
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
                # Only update stocks missing industry mappings
                stocks_to_update = stocks_query.filter(
                    industry_id__isnull=True
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
                    sector_obj = 'Yes' if stock.sector_id else 'No'
                    industry_obj = 'Yes' if stock.industry_id else 'No'
                    self.stdout.write(f'  {i:3d}. {symbol} | sector: {current_sector} ({sector_obj}) | industry: {current_industry} ({industry_obj})')
                except Stock.DoesNotExist:
                    pass
            return
        
        if total_stocks == 0:
            self.stdout.write(self.style.SUCCESS('All stocks already have industry mappings'))
            return
        
        # Process stocks in batches
        self.stdout.write(f'\nStarting industry mapping for {total_stocks} stocks...')
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
                self.stdout.write('Waiting 5 seconds before next batch...')
                time.sleep(5)
        
        # Final summary
        end_time = timezone.now()
        duration = end_time - start_time
        
        self.stdout.write('\n' + '='*60)
        self.stdout.write(self.style.SUCCESS('INDUSTRY MAPPING UPDATE COMPLETED'))
        self.stdout.write(f'Total processed: {total_stocks} stocks')
        self.stdout.write(f'Updated: {total_success}')
        self.stdout.write(f'Unchanged: {total_unchanged}')
        self.stdout.write(f'Failed: {total_failed}')
        self.stdout.write(f'Duration: {duration}')
        
        if total_failed > 0:
            success_rate = (total_success / total_stocks) * 100
            self.stdout.write(f'Success rate: {success_rate:.1f}%')
        
        self.stdout.write('='*60)

    def _fix_sector_industry_relationships(self, dry_run):
        """Fix existing industries that have incorrect sector relationships."""
        self.stdout.write('Analyzing sector-industry relationships...')
        
        # Find industries with potentially incorrect sector mappings
        problematic_industries = []
        
        for industry in DataIndustry.objects.all():
            # Get stocks in this industry
            stocks_in_industry = Stock.objects.filter(industry_id=industry)
            if not stocks_in_industry.exists():
                continue
            
            # Check if most stocks have a different sector than the industry's sector
            sector_counts = {}
            for stock in stocks_in_industry:
                if stock.sector_id:
                    sector_id = stock.sector_id.id
                    sector_counts[sector_id] = sector_counts.get(sector_id, 0) + 1
            
            if sector_counts:
                most_common_sector_id = max(sector_counts, key=sector_counts.get)
                if most_common_sector_id != industry.sector.id:
                    problematic_industries.append({
                        'industry': industry,
                        'current_sector': industry.sector,
                        'should_be_sector': DataSector.objects.get(id=most_common_sector_id),
                        'stock_count': sector_counts[most_common_sector_id]
                    })
        
        if problematic_industries:
            self.stdout.write(f'Found {len(problematic_industries)} industries with incorrect sector mappings:')
            for problem in problematic_industries:
                self.stdout.write(
                    f'  - {problem["industry"].industryName}: '
                    f'{problem["current_sector"].sectorName} -> {problem["should_be_sector"].sectorName} '
                    f'({problem["stock_count"]} stocks)'
                )
            
            if not dry_run:
                self.stdout.write('Fixing relationships...')
                for problem in problematic_industries:
                    problem['industry'].sector = problem['should_be_sector']
                    problem['industry'].save()
                    self.stdout.write(f'  Fixed: {problem["industry"].industryName}')
        else:
            self.stdout.write('All sector-industry relationships are correct.')

    def _process_batch(self, symbols):
        """Process a batch of stock symbols concurrently."""
        success_count = 0
        failed_count = 0
        unchanged_count = 0
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._update_industry_mapping, symbol): symbol 
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

    def _update_industry_mapping(self, symbol):
        """Update industry mapping for a single stock."""
        try:
            with transaction.atomic():
                # Get stock record
                stock = Stock.objects.get(symbol=symbol)
                
                # Check if industry mapping already exists and is reasonable
                if stock.industry_id and stock.industry:
                    return 'unchanged'
                
                # Fetch info from Yahoo Finance (use cached data if available)
                stock_info = yahoo_finance_service.get_stock_info(symbol)
                
                if 'error' in stock_info:
                    return f"API error: {stock_info['error']}"
                
                # Extract sector and industry
                sector_name = (stock_info.get('sector') or '').strip()
                industry_name = (stock_info.get('industry') or '').strip()
                
                # Handle special cases
                if not sector_name or not industry_name:
                    # Check if it's an ETF
                    long_name = stock_info.get('longName', '').lower()
                    if any(keyword in long_name for keyword in ['etf', 'fund', 'trust']):
                        sector_name = 'ETF/Index'
                        industry_name = 'Exchange Traded Fund'
                    else:
                        sector_name = sector_name or 'Unknown'
                        industry_name = industry_name or 'Unknown'
                
                # Skip if we don't have meaningful data
                if industry_name in ['Unknown', '']:
                    return 'no industry data'
                
                # Get or create sector first
                sector_obj = self._get_or_create_sector(sector_name)
                
                # Get or create industry linked to the correct sector
                industry_obj = self._get_or_create_industry(industry_name, sector_obj)
                
                # Check if this is actually an update
                if (stock.industry == industry_name and 
                    stock.industry_id == industry_obj and
                    stock.sector_id == sector_obj):
                    return 'unchanged'
                
                # Update stock record
                stock.industry = industry_name
                stock.industry_id = industry_obj
                
                # Also update sector if it's not set or different
                if not stock.sector_id or stock.sector_id != sector_obj:
                    stock.sector = sector_name
                    stock.sector_id = sector_obj
                
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
        """Get or create a DataIndustry object linked to the correct sector."""
        industry_key = self._normalize_key(industry_name)
        
        # First try to find existing industry
        try:
            industry_obj = DataIndustry.objects.get(industryKey=industry_key)
            
            # Check if it's linked to the correct sector
            if industry_obj.sector != sector_obj:
                # Update the sector relationship
                industry_obj.sector = sector_obj
                industry_obj.save()
                logger.info(f'Updated industry {industry_name} to be under sector {sector_obj.sectorName}')
            
            return industry_obj
            
        except DataIndustry.DoesNotExist:
            # Create new industry
            industry_obj = DataIndustry.objects.create(
                industryKey=industry_key,
                industryName=industry_name,
                sector=sector_obj,
                isActive=True,
                data_source='yahoo'
            )
            
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
        if len(normalized) > 100:  # DataIndustry.industryKey has max_length=100
            normalized = normalized[:100]
        
        return normalized or 'unknown'
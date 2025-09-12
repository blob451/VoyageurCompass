"""
Management command to populate essential market benchmark stocks.
"""

import logging
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from Data.models import Stock
from Data.services.yahoo_finance import yahoo_finance_service

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Populate essential market benchmark stocks (SPY, QQQ, etc.)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--sync-data',
            action='store_true',
            help='Also sync price data after creating stock records'
        )
        
        parser.add_argument(
            '--custom-symbols',
            nargs='+',
            help='Custom list of symbols to populate'
        )

    def handle(self, *args, **options):
        """Execute benchmark population."""
        self.stdout.write(self.style.SUCCESS('=== Populating Market Benchmarks ===\n'))
        
        # Default benchmark symbols
        default_benchmarks = [
            ('SPY', 'SPDR S&P 500 ETF Trust'),
            ('QQQ', 'Invesco QQQ Trust'),
            ('DIA', 'SPDR Dow Jones Industrial Average ETF'),
            ('IWM', 'iShares Russell 2000 ETF'),
            ('VTI', 'Vanguard Total Stock Market ETF'),
            ('VEA', 'Vanguard FTSE Developed Markets ETF'),
            ('VWO', 'Vanguard FTSE Emerging Markets ETF'),
            ('AGG', 'iShares Core U.S. Aggregate Bond ETF'),
            ('GLD', 'SPDR Gold Shares'),
            ('TLT', 'iShares 20+ Year Treasury Bond ETF')
        ]
        
        # Use custom symbols if provided
        if options.get('custom_symbols'):
            benchmarks = [(symbol, symbol) for symbol in options['custom_symbols']]
        else:
            benchmarks = default_benchmarks
        
        created_count = 0
        updated_count = 0
        synced_count = 0
        
        try:
            with transaction.atomic():
                self.stdout.write(f"Processing {len(benchmarks)} benchmark symbols...")
                
                for symbol, name in benchmarks:
                    try:
                        stock, created = Stock.objects.get_or_create(
                            symbol=symbol.upper(),
                            defaults={
                                'short_name': name,
                                'long_name': name,
                                'data_source': 'yahoo',
                                'sector': 'ETF/Index',
                                'industry': 'Market Benchmark',
                                'is_active': True
                            }
                        )
                        
                        if created:
                            created_count += 1
                            self.stdout.write(f"  + Created: {symbol} - {name}")
                        else:
                            # Update existing record
                            if not stock.short_name or stock.short_name == symbol:
                                stock.short_name = name
                                stock.long_name = name
                                stock.save(update_fields=['short_name', 'long_name'])
                                updated_count += 1
                                self.stdout.write(f"  ~ Updated: {symbol} - {name}")
                            else:
                                self.stdout.write(f"  = Exists: {symbol} - {stock.short_name}")
                        
                    except Exception as e:
                        self.stdout.write(
                            self.style.ERROR(f"  X Error creating {symbol}: {str(e)}")
                        )
                        continue
                
                self.stdout.write(f"\nStock creation complete:")
                self.stdout.write(f"  Created: {created_count}")
                self.stdout.write(f"  Updated: {updated_count}")
                self.stdout.write(f"  Total processed: {len(benchmarks)}")
                
                # Sync price data if requested
                if options.get('sync_data'):
                    self.stdout.write(f"\nSyncing price data for benchmarks...")
                    
                    for symbol, _ in benchmarks:
                        try:
                            self.stdout.write(f"  Syncing {symbol}...")
                            result = yahoo_finance_service.get_stock_data(
                                symbol, 
                                period="2y", 
                                sync_db=True
                            )
                            
                            if result and "error" not in result:
                                synced_count += 1
                                self.stdout.write(f"    + Synced {symbol}")
                            else:
                                error_msg = result.get("error", "Unknown error") if result else "No data returned"
                                self.stdout.write(
                                    self.style.WARNING(f"    ! Sync failed for {symbol}: {error_msg}")
                                )
                                
                        except Exception as e:
                            self.stdout.write(
                                self.style.ERROR(f"    X Sync error for {symbol}: {str(e)}")
                            )
                            continue
                    
                    self.stdout.write(f"\nData sync complete: {synced_count}/{len(benchmarks)} successful")
                
                self.stdout.write(self.style.SUCCESS('\nBenchmark population completed'))
                
        except Exception as e:
            raise CommandError(f'Benchmark population failed: {str(e)}')

    def _get_benchmark_info(self, symbol: str) -> dict:
        """Get benchmark information from predefined list."""
        benchmark_info = {
            'SPY': 'SPDR S&P 500 ETF Trust',
            'QQQ': 'Invesco QQQ Trust',
            'DIA': 'SPDR Dow Jones Industrial Average ETF',
            'IWM': 'iShares Russell 2000 ETF',
            'VTI': 'Vanguard Total Stock Market ETF',
            'VEA': 'Vanguard FTSE Developed Markets ETF',
            'VWO': 'Vanguard FTSE Emerging Markets ETF',
            'AGG': 'iShares Core U.S. Aggregate Bond ETF',
            'GLD': 'SPDR Gold Shares',
            'TLT': 'iShares 20+ Year Treasury Bond ETF'
        }
        
        return {
            'name': benchmark_info.get(symbol, symbol),
            'sector': 'ETF/Index',
            'industry': 'Market Benchmark'
        }
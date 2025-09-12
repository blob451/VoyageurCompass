"""
Django management command implementing bulk S&P 500 stock data loading.
Executes comprehensive data retrieval from Yahoo Finance API with database population.
"""

import logging
import time
from datetime import datetime, timedelta

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Bulk load S&P 500 stocks from Yahoo Finance with comprehensive data population'

    def add_arguments(self, parser):
        parser.add_argument(
            '--batch-size',
            type=int,
            default=10,
            help='Number of stocks to process in each batch (default: 10)'
        )
        parser.add_argument(
            '--delay',
            type=float,
            default=2.0,
            help='Delay between batches in seconds (default: 2.0)'
        )
        parser.add_argument(
            '--warm-cache',
            action='store_true',
            help='Also warm the Yahoo Finance cache after loading stocks'
        )
        parser.add_argument(
            '--force-update',
            action='store_true',
            help='Update existing stocks even if they already exist'
        )

    def handle(self, *args, **options):
        """Main command handler."""
        batch_size = options['batch_size']
        delay = options['delay']
        warm_cache = options['warm_cache']
        force_update = options['force_update']

        self.stdout.write(
            self.style.SUCCESS(f"Starting S&P 500 stock loading with batch size {batch_size}")
        )

        # S&P 500 symbols (comprehensive list)
        sp500_symbols = [
            # Technology
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "TSLA", "META", "NVDA", "CRM", "ORCL",
            "ADBE", "NFLX", "INTC", "CSCO", "AMD", "QCOM", "AVGO", "TXN", "INTU", "IBM",
            "NOW", "AMAT", "MU", "ADI", "LRCX", "KLAC", "MCHP", "SNPS", "CDNS", "FTNT",
            "MPWR", "ENPH", "ON", "MRVL", "TEAM", "DDOG", "CRWD", "ZS", "OKTA", "SPLK",
            
            # Healthcare & Biotech
            "JNJ", "PFE", "ABBV", "UNH", "TMO", "ABT", "DHR", "BMY", "MRK", "LLY",
            "AMGN", "GILD", "VRTX", "REGN", "ISRG", "ZTS", "BSX", "SYK", "EW", "DXCM",
            "ALGN", "IDXX", "IQV", "A", "RMD", "TECH", "BDX", "BAX", "WAT", "MTD",
            
            # Financial Services
            "V", "MA", "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK",
            "SPGI", "ICE", "CME", "MCO", "MSCI", "COF", "SCHW", "CB", "MMC", "AON",
            "PGR", "TRV", "ALL", "AIG", "MET", "PRU", "AFL", "CINF", "L", "RE",
            
            # Consumer Discretionary
            "HD", "MCD", "NKE", "SBUX", "TJX", "LOW", "BKNG", "DIS", "GM", "F",
            "CMG", "MAR", "HLT", "RCL", "NCLH", "CCL", "LVS", "WYNN", "MGM", "PENN",
            "ETSY", "EBAY", "PYPL", "SQ", "SHOP", "W", "CHWY", "CVNA", "ROKU", "NFLX",
            
            # Consumer Staples
            "PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "MDLZ", "CL", "KMB",
            "GIS", "K", "HSY", "SJM", "CAG", "CPB", "HRL", "MKC", "CLX", "CHD",
            "COKE", "KDP", "STZ", "TAP", "BF.B", "TSN", "CALM", "JBHT", "SYY", "UNFI",
            
            # Energy
            "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC", "HES", "DVN",
            "FANG", "EQT", "AR", "MRO", "APA", "OXY", "PXD", "BKR", "HAL", "OIH",
            
            # Industrial
            "UNP", "CAT", "HON", "RTX", "UPS", "BA", "GE", "MMM", "DE", "LMT",
            "NOC", "GD", "ITW", "EMR", "ETN", "PH", "CMI", "ROK", "DOV", "XYL",
            "IEX", "FTV", "AME", "ROP", "SWK", "FAST", "PCAR", "CSX", "NSC", "FDX",
            
            # Materials
            "LIN", "APD", "SHW", "FCX", "NEM", "CTVA", "DD", "DOW", "ALB", "CE",
            "ECL", "PPG", "IFF", "FMC", "LYB", "VMC", "MLM", "NUE", "STLD", "CF",
            
            # Utilities
            "NEE", "DUK", "SO", "D", "EXC", "AEP", "XEL", "SRE", "PCG", "PEG",
            "ED", "EIX", "WEC", "ES", "DTE", "PPL", "CMS", "NI", "LNT", "AES",
            
            # Real Estate
            "PLD", "AMT", "CCI", "EQIX", "PSA", "EXR", "WELL", "DLR", "BXP", "VTR",
            "SBAC", "ARE", "EQR", "AVB", "ESS", "MAA", "UDR", "CPT", "HST", "REG",
            
            # Communication Services  
            "VZ", "T", "CMCSA", "CHTR", "TMUS", "DIS", "NFLX", "FB", "TWTR", "SNAP",
            "PINS", "SPOT", "ZM", "DOCU", "UBER", "LYFT", "DASH", "ABNB", "RBLX", "U"
        ]

        # Remove duplicates and sort
        sp500_symbols = sorted(list(set(sp500_symbols)))
        
        self.stdout.write(f"Processing {len(sp500_symbols)} S&P 500 stocks...")

        # Process stocks in batches
        total_processed = 0
        total_created = 0
        total_updated = 0
        total_errors = 0
        
        for i in range(0, len(sp500_symbols), batch_size):
            batch = sp500_symbols[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            self.stdout.write(
                self.style.WARNING(f"Processing batch {batch_num}: {', '.join(batch)}")
            )
            
            batch_created, batch_updated, batch_errors = self._process_batch(
                batch, force_update
            )
            
            total_processed += len(batch)
            total_created += batch_created
            total_updated += batch_updated
            total_errors += batch_errors
            
            # Progress report
            self.stdout.write(
                f"Batch {batch_num} completed: {batch_created} created, "
                f"{batch_updated} updated, {batch_errors} errors"
            )
            
            # Delay between batches to be respectful to Yahoo Finance
            if i + batch_size < len(sp500_symbols):
                self.stdout.write(f"Waiting {delay} seconds before next batch...")
                time.sleep(delay)

        # Final summary
        self.stdout.write(
            self.style.SUCCESS(
                f"\nS&P 500 loading completed:\n"
                f"  Total processed: {total_processed}\n"
                f"  Created: {total_created}\n"
                f"  Updated: {total_updated}\n"
                f"  Errors: {total_errors}"
            )
        )

        # Warm cache if requested
        if warm_cache:
            self._warm_cache(sp500_symbols)

    def _process_batch(self, symbols, force_update):
        """Process a batch of stock symbols."""
        from Data.services.yahoo_cache import yahoo_cache
        from Data.models import Stock, DataSector, DataIndustry
        from decimal import Decimal
        
        created_count = 0
        updated_count = 0
        error_count = 0
        
        for symbol in symbols:
            try:
                # Check if stock already exists
                existing_stock = None
                try:
                    existing_stock = Stock.objects.get(symbol=symbol.upper())
                    if existing_stock and not force_update:
                        self.stdout.write(
                            self.style.WARNING(f"  {symbol}: Already exists, skipping")
                        )
                        continue
                except Stock.DoesNotExist:
                    pass

                # Fetch stock info from Yahoo Finance
                self.stdout.write(f"  {symbol}: Fetching from Yahoo Finance...")
                stock_info = yahoo_cache.get_stock_info(symbol, use_cache=False)
                
                if not stock_info or "error" in stock_info:
                    self.stdout.write(
                        self.style.ERROR(f"  {symbol}: Failed to fetch data")
                    )
                    error_count += 1
                    continue

                # Extract and normalize data
                sector_name = stock_info.get("sector", "Unknown")
                industry_name = stock_info.get("industry", "Unknown")
                
                # Get or create sector
                sector, _ = DataSector.objects.get_or_create(
                    sectorKey=self._normalize_key(sector_name),
                    defaults={
                        'sectorName': sector_name,
                        'isActive': True,
                        'last_sync': timezone.now()
                    }
                )
                
                # Get or create industry
                industry, _ = DataIndustry.objects.get_or_create(
                    industryKey=self._normalize_key(industry_name),
                    defaults={
                        'industryName': industry_name,
                        'sector': sector,
                        'isActive': True,
                        'last_sync': timezone.now()
                    }
                )
                
                # Prepare stock data
                stock_data = {
                    'symbol': symbol.upper(),
                    'short_name': stock_info.get('shortName', symbol),
                    'long_name': stock_info.get('longName', stock_info.get('shortName', symbol)),
                    'currency': stock_info.get('currency', 'USD'),
                    'exchange': stock_info.get('exchange', 'NASDAQ'),
                    'sector': sector_name,
                    'industry': industry_name,
                    'sector_id': sector,
                    'industry_id': industry,
                    'last_sync': timezone.now(),
                    'is_active': True,
                }
                
                # Handle market cap
                market_cap = stock_info.get('marketCap')
                if market_cap:
                    try:
                        stock_data['market_cap'] = Decimal(str(market_cap))
                    except (ValueError, TypeError):
                        stock_data['market_cap'] = None

                # Create or update stock
                if existing_stock:
                    # Update existing stock
                    for key, value in stock_data.items():
                        setattr(existing_stock, key, value)
                    existing_stock.save()
                    updated_count += 1
                    self.stdout.write(
                        self.style.SUCCESS(f"  {symbol}: Updated successfully")
                    )
                else:
                    # Create new stock
                    Stock.objects.create(**stock_data)
                    created_count += 1
                    self.stdout.write(
                        self.style.SUCCESS(f"  {symbol}: Created successfully")
                    )

            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"  {symbol}: Error - {str(e)}")
                )
                error_count += 1
                logger.error(f"Error processing {symbol}: {e}", exc_info=True)

        return created_count, updated_count, error_count

    def _normalize_key(self, name):
        """Normalize sector/industry names to create consistent keys."""
        import re
        if not name or name == "Unknown":
            return "unknown"
        # Convert to lowercase, remove special chars, replace spaces with underscores
        return re.sub(r'[^\w\s-]', '', name.lower().strip()).replace(' ', '_')[:50]

    def _warm_cache(self, symbols):
        """Trigger cache warming for the loaded stocks."""
        self.stdout.write(
            self.style.SUCCESS("\nTriggering cache warming for loaded stocks...")
        )
        
        try:
            from Data.services.tasks import warm_yahoo_cache_batch
            
            # Trigger async cache warming
            task = warm_yahoo_cache_batch.delay(symbols, batch_size=10)
            
            self.stdout.write(
                self.style.SUCCESS(
                    f"Cache warming task started: {task.id}\n"
                    f"This will run in the background to warm cache for {len(symbols)} stocks."
                )
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Failed to start cache warming: {e}")
            )
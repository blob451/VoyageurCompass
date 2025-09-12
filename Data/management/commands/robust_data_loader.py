"""
Django management command for robust stock data loading with timeout handling.
Uses the enhanced auto-sync method with direct yfinance fallback for maximum reliability.
"""

import logging
import time
from datetime import datetime, timedelta

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Robust stock data loader with timeout handling and direct yfinance fallback'

    def add_arguments(self, parser):
        parser.add_argument(
            'symbols',
            nargs='*',
            help='Stock symbols to load (space-separated). If not provided, loads from common lists.'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=5,
            help='Number of stocks to process in each batch (default: 5)'
        )
        parser.add_argument(
            '--delay',
            type=float,
            default=3.0,
            help='Delay between batches in seconds (default: 3.0)'
        )
        parser.add_argument(
            '--load-prices',
            action='store_true',
            help='Load historical price data (uses enhanced auto-sync method)'
        )
        parser.add_argument(
            '--force-reload',
            action='store_true',
            help='Force reload even if stock already has recent data'
        )
        parser.add_argument(
            '--max-retries',
            type=int,
            default=3,
            help='Maximum number of retry attempts per stock (default: 3)'
        )
        parser.add_argument(
            '--load-sp500',
            action='store_true',
            help='Load S&P 500 stocks'
        )
        parser.add_argument(
            '--load-popular',
            action='store_true',
            help='Load popular/commonly requested stocks'
        )

    def handle(self, *args, **options):
        """Main command handler."""
        symbols = options['symbols']
        batch_size = options['batch_size']
        delay = options['delay']
        load_prices = options['load_prices']
        force_reload = options['force_reload']
        max_retries = options['max_retries']
        load_sp500 = options['load_sp500']
        load_popular = options['load_popular']

        # Determine which symbols to load
        if load_sp500:
            symbols = self._get_sp500_symbols()
        elif load_popular:
            symbols = self._get_popular_symbols()
        elif not symbols:
            # Default to popular stocks if no symbols specified
            symbols = self._get_popular_symbols()
            self.stdout.write("No symbols specified, loading popular stocks...")

        # Remove duplicates and convert to uppercase
        symbols = list(set([s.upper() for s in symbols]))
        
        self.stdout.write(
            self.style.SUCCESS(
                f"Starting robust data loading for {len(symbols)} symbols\n"
                f"Batch size: {batch_size}, Delay: {delay}s, Load prices: {load_prices}"
            )
        )

        # Import the enhanced TA engine for auto-sync
        if load_prices:
            from Analytics.engine.ta_engine import TechnicalAnalysisEngine
            ta_engine = TechnicalAnalysisEngine()

        # Track statistics
        total_processed = 0
        total_success = 0
        total_failed = 0
        failed_symbols = []
        
        # Process stocks in batches
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            self.stdout.write(
                self.style.WARNING(
                    f"\nProcessing batch {batch_num}/{(len(symbols) + batch_size - 1) // batch_size}: {', '.join(batch)}"
                )
            )
            
            for symbol in batch:
                total_processed += 1
                success = False
                
                # Try multiple times for robustness
                for attempt in range(max_retries):
                    try:
                        if load_prices:
                            # Use enhanced auto-sync method with timeout handling
                            self.stdout.write(f"  {symbol}: Loading with price data (attempt {attempt + 1})...")
                            
                            # Check if we should skip (already has recent data)
                            if not force_reload and self._has_recent_data(symbol):
                                self.stdout.write(
                                    self.style.SUCCESS(f"  {symbol}: Already has recent data, skipping")
                                )
                                success = True
                                break
                            
                            # Use the robust auto-sync method
                            success = ta_engine._auto_sync_stock_data(symbol)
                            
                            if success:
                                self.stdout.write(
                                    self.style.SUCCESS(f"  {symbol}: Successfully loaded with prices")
                                )
                                break
                            else:
                                self.stdout.write(
                                    self.style.ERROR(f"  {symbol}: Auto-sync failed (attempt {attempt + 1})")
                                )
                        else:
                            # Load just basic stock info (faster)
                            self.stdout.write(f"  {symbol}: Loading basic info (attempt {attempt + 1})...")
                            success = self._load_basic_stock_info(symbol, force_reload)
                            
                            if success:
                                self.stdout.write(
                                    self.style.SUCCESS(f"  {symbol}: Successfully loaded basic info")
                                )
                                break
                            else:
                                self.stdout.write(
                                    self.style.ERROR(f"  {symbol}: Basic info load failed (attempt {attempt + 1})")
                                )
                        
                        # Small delay between retry attempts
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            
                    except Exception as e:
                        self.stdout.write(
                            self.style.ERROR(f"  {symbol}: Error on attempt {attempt + 1}: {str(e)}")
                        )
                        logger.error(f"Error loading {symbol} on attempt {attempt + 1}: {e}", exc_info=True)
                        
                        if attempt < max_retries - 1:
                            time.sleep(2)  # Longer delay after errors
                
                # Track results
                if success:
                    total_success += 1
                else:
                    total_failed += 1
                    failed_symbols.append(symbol)
                    self.stdout.write(
                        self.style.ERROR(f"  {symbol}: FAILED after {max_retries} attempts")
                    )
            
            # Progress report
            self.stdout.write(
                f"Batch {batch_num} completed. Total so far: {total_success} success, {total_failed} failed"
            )
            
            # Delay between batches
            if i + batch_size < len(symbols):
                self.stdout.write(f"Waiting {delay} seconds before next batch...")
                time.sleep(delay)

        # Final summary
        success_rate = (total_success / total_processed * 100) if total_processed > 0 else 0
        
        self.stdout.write(
            self.style.SUCCESS(
                f"\n{'='*50}\n"
                f"ROBUST DATA LOADING COMPLETED\n"
                f"{'='*50}\n"
                f"  Total processed: {total_processed}\n"
                f"  Successful: {total_success}\n"
                f"  Failed: {total_failed}\n"
                f"  Success rate: {success_rate:.1f}%"
            )
        )
        
        if failed_symbols:
            self.stdout.write(
                self.style.ERROR(
                    f"\nFailed symbols ({len(failed_symbols)}):\n  {', '.join(failed_symbols)}"
                )
            )
        
        self.stdout.write(self.style.SUCCESS("\nRobust data loading complete!"))

    def _get_sp500_symbols(self):
        """Get S&P 500 symbols (subset for testing)."""
        return [
            # Major tech
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "ADBE", "CRM",
            # Major financial
            "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "BLK", "SPGI", "AXP",
            # Major healthcare
            "JNJ", "PFE", "UNH", "ABBV", "TMO", "ABT", "DHR", "MRK", "BMY", "LLY",
            # Major consumer
            "PG", "KO", "PEP", "WMT", "COST", "HD", "MCD", "NKE", "SBUX", "DIS",
            # Major industrial/energy
            "XOM", "CVX", "COP", "UNP", "CAT", "HON", "RTX", "UPS", "BA", "GE"
        ]

    def _get_popular_symbols(self):
        """Get commonly requested/popular symbols."""
        return [
            # FAANG + major tech
            "AAPL", "AMZN", "GOOGL", "META", "NFLX", "MSFT", "TSLA", "NVDA", "ADBE",
            # Meme/popular stocks
            "GME", "AMC", "BB", "NOK", "PLTR", "SPCE", "COIN", "ROKU", "ZM", "PELOTON",
            # Major financial
            "JPM", "BAC", "V", "MA", "BRK.A", "BRK.B",
            # Popular ETFs (if supported)
            "SPY", "QQQ", "IWM", "VTI", "VOO",
            # Crypto-related
            "MSTR", "SQ", "PYPL", "COIN",
            # Popular recent IPOs
            "RBLX", "ABNB", "DASH", "SNOW", "PLTR"
        ]

    def _has_recent_data(self, symbol):
        """Check if stock has recent price data."""
        try:
            from Data.models import Stock, StockPrice
            
            stock = Stock.objects.get(symbol=symbol.upper())
            
            # Check if we have price data from the last 7 days
            recent_cutoff = timezone.now().date() - timedelta(days=7)
            recent_prices = StockPrice.objects.filter(
                stock=stock,
                date__gte=recent_cutoff
            ).exists()
            
            return recent_prices
            
        except Stock.DoesNotExist:
            return False
        except Exception:
            return False

    def _load_basic_stock_info(self, symbol, force_reload=False):
        """Load basic stock information without prices."""
        try:
            from Data.models import Stock, DataSector, DataIndustry
            from Data.services.yahoo_cache import yahoo_cache
            from decimal import Decimal
            
            # Check if already exists
            if not force_reload:
                try:
                    existing_stock = Stock.objects.get(symbol=symbol.upper())
                    if existing_stock.last_sync and existing_stock.last_sync > timezone.now() - timedelta(days=1):
                        return True  # Recent sync, skip
                except Stock.DoesNotExist:
                    pass
            
            # Fetch from Yahoo Finance
            stock_info = yahoo_cache.get_stock_info(symbol, use_cache=False)
            
            if not stock_info or "error" in stock_info:
                return False
            
            # Create sector and industry
            sector_name = stock_info.get("sector", "Unknown")
            industry_name = stock_info.get("industry", "Unknown")
            
            sector, _ = DataSector.objects.get_or_create(
                sectorKey=self._normalize_key(sector_name),
                defaults={
                    'sectorName': sector_name,
                    'isActive': True,
                    'last_sync': timezone.now()
                }
            )
            
            industry, _ = DataIndustry.objects.get_or_create(
                industryKey=self._normalize_key(industry_name),
                defaults={
                    'industryName': industry_name,
                    'sector': sector,
                    'isActive': True,
                    'last_sync': timezone.now()
                }
            )
            
            # Create or update stock
            stock_data = {
                'short_name': stock_info.get('shortName', symbol)[:100],
                'long_name': stock_info.get('longName', stock_info.get('shortName', symbol))[:255],
                'currency': stock_info.get('currency', 'USD')[:10],
                'exchange': stock_info.get('exchange', 'NASDAQ')[:50],
                'sector': sector_name[:100],
                'industry': industry_name[:100],
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
            
            # Create or update
            Stock.objects.update_or_create(
                symbol=symbol.upper(),
                defaults=stock_data
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading basic info for {symbol}: {e}", exc_info=True)
            return False

    def _normalize_key(self, name):
        """Normalize sector/industry names to create consistent keys."""
        import re
        if not name or name == "Unknown":
            return "unknown"
        return re.sub(r'[^\w\s-]', '', name.lower().strip()).replace(' ', '_')[:50]
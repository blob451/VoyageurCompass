"""
Management command to pull AAPL 3-year data with schema verification and database guard.
"""

import time
import logging
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.db import connection
from django.utils import timezone
from django.conf import settings

from Data.services.yahoo_finance import yahoo_finance_service

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Pull AAPL 3-year daily history and complete Stocks field set plus industry & sector from real YahooFinance'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force refresh even if recent data exists'
        )

    def handle(self, *args, **options):
        """Execute the AAPL 3-year data pull with all validations."""
        
        self.stdout.write("=== AAPL 3-Year Data Pull Starting ===")
        
        # 1. Database Engine Guard
        if not self.validate_db_engine():
            return
        
        # 2. Schema Verification
        if not self.verify_schema():
            return
        
        # 3. Apply Recency Policy 
        if not options.get('force') and not self.check_recency_policy():
            self.stdout.write(self.style.SUCCESS("Recent data exists, skipping fetch (use --force to override)"))
            return
        
        # 4. Pull AAPL Data
        self.pull_aapl_data()
        
        self.stdout.write(self.style.SUCCESS("=== AAPL 3-Year Data Pull Complete ==="))

    def validate_db_engine(self) -> bool:
        """Validate database engine is not SQLite."""
        try:
            db_engine = settings.DATABASES['default']['ENGINE']
            if 'sqlite' in db_engine.lower():
                self.stdout.write(
                    self.style.ERROR("ABORT: SQLite database detected. This command requires PostgreSQL.")
                )
                return False
            
            self.stdout.write(f"✓ Database engine validated: {db_engine}")
            return True
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Database validation failed: {str(e)}"))
            return False

    def verify_schema(self) -> bool:
        """Verify required fields exist in DATA tables."""
        try:
            from Data.models import Stock, DataSectorPrice, DataIndustryPrice
            
            # Check Stock model for required Stocks fields
            stock_fields = [f.name for f in Stock._meta.get_fields()]
            required_stock_fields = [
                'currentPrice', 'previousClose', 'dayLow', 'dayHigh',
                'regularMarketPrice', 'regularMarketOpen', 'regularMarketDayLow',
                'regularMarketDayHigh', 'regularMarketPreviousClose', 'fiftyTwoWeekLow',
                'fiftyTwoWeekHigh', 'fiftyTwoWeekChange', 'fiftyDayAverage',
                'twoHundredDayAverage', 'beta', 'impliedVolatility',
                'regularMarketVolume', 'averageVolume', 'averageVolume10days',
                'averageVolume3months'
            ]
            
            missing_stock_fields = [f for f in required_stock_fields if f not in stock_fields]
            if missing_stock_fields:
                self.stdout.write(
                    self.style.ERROR(f"ABORT: Missing Stock fields: {missing_stock_fields}")
                )
                return False
            
            # Check DataSectorPrice model for required Industry & Sector fields
            sector_fields = [f.name for f in DataSectorPrice._meta.get_fields()]
            required_sector_fields = [
                'fiftyTwoWeekChange', 'fiftyDayAverage', 'twoHundredDayAverage',
                'averageVolume', 'averageVolume3months'
            ]
            
            missing_sector_fields = [f for f in required_sector_fields if f not in sector_fields]
            if missing_sector_fields:
                self.stdout.write(
                    self.style.ERROR(f"ABORT: Missing DataSectorPrice fields: {missing_sector_fields}")
                )
                return False
            
            # Check DataIndustryPrice model  
            industry_fields = [f.name for f in DataIndustryPrice._meta.get_fields()]
            missing_industry_fields = [f for f in required_sector_fields if f not in industry_fields]
            if missing_industry_fields:
                self.stdout.write(
                    self.style.ERROR(f"ABORT: Missing DataIndustryPrice fields: {missing_industry_fields}")
                )
                return False
            
            self.stdout.write("✓ Schema verification passed - all required fields present")
            return True
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Schema verification failed: {str(e)}"))
            return False

    def check_recency_policy(self) -> bool:
        """Check if industry & sector data needs updating (3-year policy)."""
        try:
            from Data.models import Stock
            
            threshold_date = timezone.now() - timedelta(days=3*365)
            
            try:
                stock = Stock.objects.get(symbol='AAPL')
                if stock.sectorUpdatedAt and stock.sectorUpdatedAt >= threshold_date:
                    self.stdout.write("Recent sector/industry data exists (< 3 years old)")
                    return False
            except Stock.DoesNotExist:
                pass
            
            self.stdout.write("✓ Recency policy check passed - data needs updating")
            return True
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Recency check failed: {str(e)}"))
            return True  # Proceed on error

    def pull_aapl_data(self):
        """Pull AAPL 3-year data from Yahoo Finance."""
        try:
            symbol = 'AAPL'
            end_date = datetime.now()
            start_date = end_date - timedelta(days=3*365)  # 3 years
            
            self.stdout.write(f"Fetching {symbol} data from {start_date.date()} to {end_date.date()}")
            
            # 1. Fetch quote data (Stocks fields)
            quote_data = yahoo_finance_service.fetchQuoteSingle(symbol)
            if 'error' not in quote_data:
                success = yahoo_finance_service.upsertStocksMetrics(quote_data)
                self.stdout.write(f"✓ Quote data: {'success' if success else 'failed'}")
            
            # 2. Fetch sector/industry data
            sector_data = yahoo_finance_service.fetchIndustrySectorSingle(symbol)
            if 'error' not in sector_data:
                success = yahoo_finance_service.upsertIndustrySector(sector_data)
                self.stdout.write(f"✓ Sector/industry data: {'success' if success else 'failed'}")
            
            # 3. Fetch 3-year history
            history_data = yahoo_finance_service.fetchHistory(symbol, start_date, end_date)
            if history_data:
                upserted_count = yahoo_finance_service.upsertHistoryBars(history_data)
                self.stdout.write(f"✓ History data: {upserted_count} bars upserted")
            
            self.stdout.write(f"AAPL data pull completed successfully")
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error pulling AAPL data: {str(e)}"))
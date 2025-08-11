"""
Technical Analysis Bootstrap Management Command
Initializes technical analysis data infrastructure for a single stock.
Creates normalized sector/industry tables and populates EOD data.
"""

from django.core.management.base import BaseCommand, CommandError
from django.db import connection, transaction, models
from django.utils import timezone
from datetime import datetime, timedelta
from Data.services.yahoo_finance import create_yahoo_finance_service
from Data.models import (
    Stock, StockPrice, DataSector, DataIndustry, 
    DataSectorPrice, DataIndustryPrice
)
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Bootstrap technical analysis data for a single stock with sector/industry composites'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--symbol',
            type=str,
            required=True,
            help='Stock ticker symbol (e.g., AAPL)'
        )
        parser.add_argument(
            '--start',
            type=str,
            default=(timezone.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
            help='Start date in YYYY-MM-DD format (default: 1 year ago)'
        )
        parser.add_argument(
            '--end',
            type=str,
            default=timezone.now().strftime('%Y-%m-%d'),
            help='End date in YYYY-MM-DD format (default: today)'
        )
    
    def handle(self, *args, **options):
        """Execute the technical analysis bootstrap process."""
        try:
            # 1. Engine guard - abort if SQLite detected
            self.validateDatabaseEngine()
            
            # 2. Parse and validate arguments
            symbol = options['symbol'].upper().strip()
            start_date, end_date = self.parseDateRange(options['start'], options['end'])
            
            self.stdout.write(
                f"Bootstrapping technical analysis for {symbol} "
                f"from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            )
            
            # 3. Ensure migrations are applied
            self.checkMigrations()
            
            with create_yahoo_finance_service() as service:
                # 4. Fetch and upsert classification data
                self.stdout.write("Step 1: Fetching and upserting sector/industry classification...")
                classification_result = self.fetchAndUpsertClassification(service, symbol)
                
                # 5. Pull 1-year EOD data for the symbol
                self.stdout.write("Step 2: Fetching EOD stock price data...")
                eod_result = self.fetchAndPopulateEodData(service, symbol, start_date, end_date)
                
                # 6. Build and persist sector/industry composites
                self.stdout.write("Step 3: Computing sector/industry composites...")
                composite_result = self.buildSectorIndustryComposites(service, symbol, start_date, end_date)
                
                # 7. Verify data integrity and continuity
                self.stdout.write("Step 4: Verifying data integrity...")
                verification_result = self.verifyDataIntegrity(symbol, start_date, end_date)
                
                # 8. Print verification output
                self.stdout.write("Step 5: Printing verification output...")
                self.printVerificationOutput(symbol)
            
            # Summary
            self.stdout.write(
                self.style.SUCCESS(
                    f'Technical analysis bootstrap complete for {symbol}:\n'
                    f'  - Classification: {classification_result}\n'
                    f'  - EOD records: {eod_result}\n'
                    f'  - Sector composites: {composite_result.get("sector_prices_created", 0)}\n'
                    f'  - Industry composites: {composite_result.get("industry_prices_created", 0)}\n'
                    f'  - Data integrity: {verification_result}'
                )
            )
            
        except Exception as e:
            logger.error(f'Technical analysis bootstrap failed: {str(e)}')
            raise CommandError(f'Bootstrap failed: {str(e)}')
    
    def validateDatabaseEngine(self):
        """Validate database engine and abort if SQLite is detected."""
        engine_name = connection.vendor
        
        if engine_name == 'sqlite':
            raise CommandError(
                'SQLite database detected. This command requires PostgreSQL or another '
                'production-grade database. SQLite is not supported for technical analysis '
                'operations due to performance and data integrity requirements.'
            )
        
        self.stdout.write(f"Database engine validated: {engine_name}")
        logger.info(f"Database engine validation passed: {engine_name}")
    
    def parseDateRange(self, start_str: str, end_str: str):
        """Parse and validate date range parameters."""
        try:
            start_date = datetime.strptime(start_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_str, '%Y-%m-%d')
            
            if end_date < start_date:
                raise CommandError("End date must be after start date")
            
            # Reasonable limit: no more than 5 years
            if (end_date - start_date).days > 1825:
                raise CommandError("Date range cannot exceed 5 years")
            
            return start_date, end_date
            
        except ValueError as e:
            raise CommandError(f"Invalid date format. Use YYYY-MM-DD: {str(e)}")
    
    def checkMigrations(self):
        """Check that required migrations are applied."""
        from django.db import connection
        
        with connection.cursor() as cursor:
            # Check if new tables exist
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('data_sector', 'data_industry', 'data_sectorprice', 'data_industryprice')
            """)
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['data_sector', 'data_industry', 'data_sectorprice', 'data_industryprice']
            missing_tables = [table for table in required_tables if table not in existing_tables]
            
            if missing_tables:
                raise CommandError(
                    f'Required database tables missing: {missing_tables}. '
                    'Run "python manage.py makemigrations" and "python manage.py migrate" first.'
                )
        
        self.stdout.write("Database migrations verified")
    
    def fetchAndUpsertClassification(self, service, symbol: str) -> str:
        """Fetch sector/industry classification and upsert to normalized tables."""
        try:
            # Fetch classification data
            classification_data = service.fetchSectorIndustrySingle(symbol)
            
            if 'error' in classification_data:
                raise CommandError(f"Failed to fetch classification for {symbol}: {classification_data['error']}")
            
            # Prepare for upsert
            classification_rows = [classification_data]
            
            # Upsert to normalized tables and establish FK relationships
            created, updated, skipped = service.upsertClassification(classification_rows)
            
            result = f"{created} created, {updated} updated, {skipped} skipped"
            self.stdout.write(f"  Classification upsert: {result}")
            
            return result
            
        except Exception as e:
            raise CommandError(f"Classification upsert failed: {str(e)}")
    
    def fetchAndPopulateEodData(self, service, symbol: str, start_date: datetime, end_date: datetime) -> int:
        """Fetch and populate EOD stock price data."""
        try:
            # Fetch EOD history
            eod_data = service.fetchStockEodHistory(symbol, start_date, end_date)
            
            if not eod_data:
                raise CommandError(f"No EOD data returned for {symbol}")
            
            # Get or create stock record
            try:
                stock = Stock.objects.get(symbol=symbol)
            except Stock.DoesNotExist:
                raise CommandError(f"Stock record for {symbol} not found. Classification step may have failed.")
            
            # Bulk create/update price records
            records_created = 0
            records_updated = 0
            
            with transaction.atomic():
                for price_data in eod_data:
                    stock_price, created = StockPrice.objects.get_or_create(
                        stock=stock,
                        date=price_data['date'],
                        defaults={
                            'open': price_data['open'],
                            'high': price_data['high'],
                            'low': price_data['low'],
                            'close': price_data['close'],
                            'adjusted_close': price_data['adjusted_close'],
                            'volume': price_data['volume'],
                            'data_source': price_data['data_source']
                        }
                    )
                    
                    if created:
                        records_created += 1
                    else:
                        # Update if adjusted_close differs
                        if stock_price.adjusted_close != price_data['adjusted_close']:
                            stock_price.adjusted_close = price_data['adjusted_close']
                            stock_price.save(update_fields=['adjusted_close'])
                            records_updated += 1
            
            total_records = records_created + records_updated
            self.stdout.write(f"  EOD data: {records_created} created, {records_updated} updated")
            
            return total_records
            
        except Exception as e:
            raise CommandError(f"EOD data population failed: {str(e)}")
    
    def buildSectorIndustryComposites(self, service, symbol: str, start_date: datetime, end_date: datetime) -> dict:
        """Build and persist sector/industry composite price data for full available stock price range."""
        try:
            # Get stock's sector and industry
            stock = Stock.objects.get(symbol=symbol)
            
            if not stock.sector_id or not stock.industry_id:
                raise CommandError(f"Stock {symbol} missing sector/industry classification")
            
            # Get the actual date range of available stock price data
            price_range = StockPrice.objects.filter(stock=stock).aggregate(
                min_date=models.Min('date'),
                max_date=models.Max('date')
            )
            
            if not price_range['min_date'] or not price_range['max_date']:
                raise CommandError(f"No stock price data found for {symbol}")
            
            actual_start = datetime.combine(price_range['min_date'], datetime.min.time())
            actual_end = datetime.combine(price_range['max_date'], datetime.min.time())
            
            self.stdout.write(
                f"  Using full stock price date range: {actual_start.date()} to {actual_end.date()} "
                f"(requested: {start_date.date()} to {end_date.date()})"
            )
            
            # Create composites for the full available date range
            composite_result = service.composeSectorIndustryEod((actual_start, actual_end))
            
            self.stdout.write(
                f"  Composites: {composite_result.get('sector_prices_created', 0)} sector, "
                f"{composite_result.get('industry_prices_created', 0)} industry"
            )
            
            return composite_result
            
        except Exception as e:
            raise CommandError(f"Composite creation failed: {str(e)}")
    
    def verifyDataIntegrity(self, symbol: str, start_date: datetime, end_date: datetime) -> str:
        """Verify data integrity and continuity."""
        try:
            stock = Stock.objects.get(symbol=symbol)
            
            # Check total price data available
            total_price_count = StockPrice.objects.filter(stock=stock).count()
            
            # Check price data in requested range
            requested_price_count = StockPrice.objects.filter(
                stock=stock,
                date__gte=start_date.date(),
                date__lte=end_date.date()
            ).count()
            
            # Get actual stock price date range
            price_range = StockPrice.objects.filter(stock=stock).aggregate(
                min_date=models.Min('date'),
                max_date=models.Max('date')
            )
            
            # Check sector/industry composites (should match total stock price range)
            sector_count = DataSectorPrice.objects.filter(sector=stock.sector_id).count()
            industry_count = DataIndustryPrice.objects.filter(industry=stock.industry_id).count()
            
            # Get composite date range for verification
            sector_range = DataSectorPrice.objects.filter(sector=stock.sector_id).aggregate(
                min_date=models.Min('date'),
                max_date=models.Max('date')
            )
            
            result = (f"Stock: {total_price_count} total records ({price_range['min_date']} to {price_range['max_date']}), "
                     f"{requested_price_count} in requested range. "
                     f"Sector: {sector_count} composites, Industry: {industry_count} composites "
                     f"({sector_range['min_date']} to {sector_range['max_date']})")
            
            self.stdout.write(f"  Data integrity: {result}")
            
            # Verify date range alignment
            if (sector_range['min_date'] == price_range['min_date'] and 
                sector_range['max_date'] == price_range['max_date'] and
                sector_count == total_price_count):
                self.stdout.write("  ✓ Composite data perfectly aligned with stock price data")
            else:
                self.stdout.write("  ⚠ Composite data range differs from stock price range")
            
            return result
            
        except Exception as e:
            logger.error(f"Data integrity verification failed: {str(e)}")
            return f"Verification failed: {str(e)}"
    
    def printVerificationOutput(self, symbol: str):
        """Print verification output showing first and last 5 records of each table."""
        try:
            stock = Stock.objects.get(symbol=symbol)
            
            self.stdout.write(self.style.SUCCESS("\\n=== VERIFICATION OUTPUT ===\\n"))
            
            # DATA_STOCK - first & last 5 by symbol
            self.stdout.write("DATA_STOCK (first & last 5 by symbol):")
            stocks_first = Stock.objects.order_by('symbol')[:5]
            stocks_last = Stock.objects.order_by('symbol').reverse()[:5]
            
            for s in stocks_first:
                self.stdout.write(f"  {s.symbol} | {s.short_name} | {s.sector_id_id if s.sector_id else 'None'} | {s.industry_id_id if s.industry_id else 'None'}")
            
            self.stdout.write("  ...")
            for s in reversed(stocks_last):
                self.stdout.write(f"  {s.symbol} | {s.short_name} | {s.sector_id_id if s.sector_id else 'None'} | {s.industry_id_id if s.industry_id else 'None'}")
            
            # DATA_STOCKPRICE - first & last 5 by date for the symbol
            self.stdout.write(f"\\nDATA_STOCKPRICE for {symbol} (first & last 5 by date):")
            prices_first = StockPrice.objects.filter(stock=stock).order_by('date')[:5]
            prices_last = StockPrice.objects.filter(stock=stock).order_by('-date')[:5]
            
            for p in prices_first:
                self.stdout.write(f"  {p.date} | {p.open} | {p.high} | {p.low} | {p.close} | {p.adjusted_close} | {p.volume}")
            
            self.stdout.write("  ...")
            for p in prices_last:
                self.stdout.write(f"  {p.date} | {p.open} | {p.high} | {p.low} | {p.close} | {p.adjusted_close} | {p.volume}")
            
            # DATA_SECTOR - first & last 5 by primary key
            self.stdout.write("\\nDATA_SECTOR (first & last 5 by ID):")
            sectors_first = DataSector.objects.order_by('id')[:5]
            sectors_last = DataSector.objects.order_by('-id')[:5]
            
            for s in sectors_first:
                self.stdout.write(f"  {s.id} | {s.sectorKey} | {s.sectorName} | {s.isActive}")
            
            if DataSector.objects.count() > 5:
                self.stdout.write("  ...")
                for s in sectors_last:
                    self.stdout.write(f"  {s.id} | {s.sectorKey} | {s.sectorName} | {s.isActive}")
            
            # DATA_INDUSTRY - first & last 5 by primary key
            self.stdout.write("\\nDATA_INDUSTRY (first & last 5 by ID):")
            industries_first = DataIndustry.objects.order_by('id')[:5]
            industries_last = DataIndustry.objects.order_by('-id')[:5]
            
            for i in industries_first:
                self.stdout.write(f"  {i.id} | {i.industryKey} | {i.industryName} | {i.sector.sectorName} | {i.isActive}")
            
            if DataIndustry.objects.count() > 5:
                self.stdout.write("  ...")
                for i in industries_last:
                    self.stdout.write(f"  {i.id} | {i.industryKey} | {i.industryName} | {i.sector.sectorName} | {i.isActive}")
            
            # DATA_SECTORPRICE - first & last 5 by date for the sector
            if stock.sector_id:
                self.stdout.write(f"\\nDATA_SECTORPRICE for {stock.sector_id.sectorName} (first & last 5 by date):")
                sector_prices_first = DataSectorPrice.objects.filter(sector=stock.sector_id).order_by('date')[:5]
                sector_prices_last = DataSectorPrice.objects.filter(sector=stock.sector_id).order_by('-date')[:5]
                
                for sp in sector_prices_first:
                    self.stdout.write(f"  {sp.date} | {sp.close_index} | {sp.volume_agg} | {sp.constituents_count} | {sp.method}")
                
                if DataSectorPrice.objects.filter(sector=stock.sector_id).count() > 5:
                    self.stdout.write("  ...")
                    for sp in sector_prices_last:
                        self.stdout.write(f"  {sp.date} | {sp.close_index} | {sp.volume_agg} | {sp.constituents_count} | {sp.method}")
            
            # DATA_INDUSTRYPRICE - first & last 5 by date for the industry
            if stock.industry_id:
                self.stdout.write(f"\\nDATA_INDUSTRYPRICE for {stock.industry_id.industryName} (first & last 5 by date):")
                industry_prices_first = DataIndustryPrice.objects.filter(industry=stock.industry_id).order_by('date')[:5]
                industry_prices_last = DataIndustryPrice.objects.filter(industry=stock.industry_id).order_by('-date')[:5]
                
                for ip in industry_prices_first:
                    self.stdout.write(f"  {ip.date} | {ip.close_index} | {ip.volume_agg} | {ip.constituents_count} | {ip.method}")
                
                if DataIndustryPrice.objects.filter(industry=stock.industry_id).count() > 5:
                    self.stdout.write("  ...")
                    for ip in industry_prices_last:
                        self.stdout.write(f"  {ip.date} | {ip.close_index} | {ip.volume_agg} | {ip.constituents_count} | {ip.method}")
            
            self.stdout.write(self.style.SUCCESS("\\n=== END VERIFICATION OUTPUT ==="))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Verification output failed: {str(e)}"))
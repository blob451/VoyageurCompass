"""
Database verification command for VoyageurCompass.
Validates schema integrity and data separation.
"""

from django.core.management.base import BaseCommand
from django.db import connection
from Data.models import Stock, StockPrice, PriceBar, Portfolio, PortfolioHolding


class Command(BaseCommand):
    help = 'Verify database schema and data integrity'

    def handle(self, *args, **options):
        """Run comprehensive database verification."""
        self.stdout.write(self.style.SUCCESS('VoyageurCompass Database Verification'))
        
        try:
            self.checkSchema()
            self.checkDataSeparation()
            self.checkIndexes()
            self.stdout.write(self.style.SUCCESS('[OK] Database verification completed successfully'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'[ERROR] Verification failed: {str(e)}'))
            raise
    
    def checkSchema(self):
        """Verify all required tables exist."""
        self.stdout.write('[INFO] Checking schema...')
        
        # Check model tables exist
        models = [Stock, StockPrice, PriceBar, Portfolio, PortfolioHolding]
        for model in models:
            count = model.objects.count()
            self.stdout.write(f'  [OK] {model._meta.db_table}: {count} records')
        
        # Check critical fields exist
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'Data_stock' AND column_name = 'dataSource'
            """)
            if cursor.fetchone():
                self.stdout.write('  [OK] dataSource field exists in Stock table')
            else:
                raise Exception('dataSource field missing in Stock table')
    
    def checkDataSeparation(self):
        """Verify mock vs real data separation."""
        self.stdout.write('[INFO] Checking data separation...')
        
        # Check Stock data sources
        yahooCount = Stock.objects.filter(dataSource='yahoo').count()
        mockCount = Stock.objects.filter(dataSource='mock').count()
        self.stdout.write(f'  [DATA] Stocks - Yahoo: {yahooCount}, Mock: {mockCount}')
        
        # Check StockPrice data sources
        yahooPrice = StockPrice.objects.filter(dataSource='yahoo').count()
        mockPrice = StockPrice.objects.filter(dataSource='mock').count()
        self.stdout.write(f'  [DATA] Prices - Yahoo: {yahooPrice}, Mock: {mockPrice}')
        
        # Check PriceBar data sources
        yahooBars = PriceBar.objects.filter(dataSource='yahoo').count()
        mockBars = PriceBar.objects.filter(dataSource='mock').count()
        self.stdout.write(f'  [DATA] PriceBars - Yahoo: {yahooBars}, Mock: {mockBars}')
    
    def checkIndexes(self):
        """Verify critical indexes exist."""
        self.stdout.write('[INFO] Checking indexes...')
        
        with connection.cursor() as cursor:
            # Check for time-series indexes
            cursor.execute("""
                SELECT indexname FROM pg_indexes 
                WHERE tablename IN ('Data_stock', 'Data_stockprice', 'Data_pricebar')
                ORDER BY indexname
            """)
            indexes = [row[0] for row in cursor.fetchall()]
            
            expectedIndexes = [
                'Data_stock_dataSou',  # Django truncates long index names
                'Data_stockp_dataSou', 
                'Data_priceb_stock_i'
            ]
            
            for expected in expectedIndexes:
                if any(expected in idx for idx in indexes):
                    self.stdout.write(f'  [OK] Index found: {expected}')
                else:
                    self.stdout.write(f'  [WARN] Missing index: {expected}')
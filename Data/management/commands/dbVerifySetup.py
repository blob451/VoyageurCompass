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
        
        # Check critical fields exist using Django's introspection API
        with connection.cursor() as cursor:
            # Get table description for Data_stock table
            table_description = connection.introspection.get_table_description(cursor, 'Data_stock')
            column_names = [col.name for col in table_description]
            
            if 'data_source' in column_names:
                self.stdout.write('  [OK] data_source field exists in Stock table')
            else:
                raise Exception('data_source field missing in Stock table')
    
    def checkDataSeparation(self):
        """Verify mock vs real data separation."""
        self.stdout.write('[INFO] Checking data separation...')
        
        # Check Stock data sources
        yahooCount = Stock.objects.filter(data_source='yahoo').count()
        mockCount = Stock.objects.filter(data_source='mock').count()
        self.stdout.write(f'  [DATA] Stocks - Yahoo: {yahooCount}, Mock: {mockCount}')
        
        # Check StockPrice data sources
        yahooPrice = StockPrice.objects.filter(data_source='yahoo').count()
        mockPrice = StockPrice.objects.filter(data_source='mock').count()
        self.stdout.write(f'  [DATA] Prices - Yahoo: {yahooPrice}, Mock: {mockPrice}')
        
        # Check PriceBar data sources
        yahooBars = PriceBar.objects.filter(data_source='yahoo').count()
        mockBars = PriceBar.objects.filter(data_source='mock').count()
        self.stdout.write(f'  [DATA] PriceBars - Yahoo: {yahooBars}, Mock: {mockBars}')
    
    def checkIndexes(self):
        """Verify critical indexes exist."""
        self.stdout.write('[INFO] Checking indexes...')
        
        with connection.cursor() as cursor:
            # Check for time-series indexes using Django's introspection API
            table_names = ['Data_stock', 'Data_stockprice', 'Data_pricebar']
            all_indexes = []
            
            for table_name in table_names:
                try:
                    # Get indexes for each table
                    table_indexes = connection.introspection.get_indexes(cursor, table_name)
                    # Extract index names (keys from the dictionary)
                    all_indexes.extend(table_indexes.keys())
                except Exception as e:
                    self.stdout.write(f'  [WARN] Could not get indexes for {table_name}: {str(e)}')
            
            # Expected index prefixes for robust verification
            expected_index_prefixes = [
                ('Data_stock_dataSou', 'Stock dataSource index'),
                ('Data_stockp_dataSou', 'StockPrice dataSource index'), 
                ('Data_priceb_stock_i', 'PriceBar stock/interval index'),
                ('Data_priceb_dataSou', 'PriceBar dataSource index')
            ]
            
            for prefix, description in expected_index_prefixes:
                found_indexes = [idx for idx in all_indexes if idx.startswith(prefix)]
                if found_indexes:
                    # Show the actual index name found
                    actual_name = found_indexes[0]  # Take first match
                    self.stdout.write(f'  [OK] {description}: {actual_name}')
                else:
                    self.stdout.write(f'  [WARN] Missing {description} (expected prefix: {prefix})')
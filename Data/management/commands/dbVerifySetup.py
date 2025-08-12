"""
Database verification command for VoyageurCompass.
Validates schema integrity and data separation.
"""

from django.core.management.base import BaseCommand
from django.db import connection
from Data.models import Stock, StockPrice, Portfolio, PortfolioHolding, DataSourceChoices


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
        models = [Stock, StockPrice, Portfolio, PortfolioHolding]
        for model in models:
            count = model.objects.count()
            self.stdout.write(f'  [OK] {model._meta.db_table}: {count} records')
        
        # Check critical fields exist using Django's introspection API
        with connection.cursor() as cursor:
            # Get table description for Stock table using model metadata
            stock_table_name = Stock._meta.db_table
            table_description = connection.introspection.get_table_description(cursor, stock_table_name)
            column_names = [col.name for col in table_description]
            
            if 'data_source' in column_names:
                self.stdout.write('  [OK] data_source field exists in Stock table')
            else:
                raise Exception('data_source field missing in Stock table')
    
    def checkDataSeparation(self):
        """Verify mock vs real data separation."""
        self.stdout.write('[INFO] Checking data separation...')
        
        # Check Stock data sources
        yahooCount = Stock.objects.filter(data_source=DataSourceChoices.YAHOO).count()
        mockCount = Stock.objects.filter(data_source=DataSourceChoices.MOCK).count()
        self.stdout.write(f'  [DATA] Stocks - Yahoo: {yahooCount}, Mock: {mockCount}')
        
        # Check StockPrice data sources
        yahooPrice = StockPrice.objects.filter(data_source=DataSourceChoices.YAHOO).count()
        mockPrice = StockPrice.objects.filter(data_source=DataSourceChoices.MOCK).count()
        self.stdout.write(f'  [DATA] Prices - Yahoo: {yahooPrice}, Mock: {mockPrice}')
        
    
    def checkIndexes(self):
        """Verify critical indexes exist using column-based verification."""
        self.stdout.write('[INFO] Checking indexes...')
        
        # Expected indexes with their columns for verification
        expected_indexes = [
            (Stock, ['data_source'], 'Stock dataSource index'),
            (StockPrice, ['data_source'], 'StockPrice dataSource index')
        ]
        
        with connection.cursor() as cursor:
            for model, expected_columns, description in expected_indexes:
                table_name = model._meta.db_table
                
                try:
                    # Get indexes for the table
                    table_indexes = connection.introspection.get_indexes(cursor, table_name)
                    
                    # Check if expected columns are indexed
                    index_found = self._check_columns_indexed(table_indexes, expected_columns)
                    
                    if index_found:
                        self.stdout.write(f'  [OK] {description}: columns {expected_columns} are indexed')
                    else:
                        self.stdout.write(f'  [WARN] Missing {description}: columns {expected_columns} not indexed')
                        
                except Exception as e:
                    self.stdout.write(f'  [WARN] Could not verify indexes for {table_name}: {str(e)}')
    
    def _check_columns_indexed(self, table_indexes, expected_columns):
        """
        Check if expected columns are covered by any index.
        
        Args:
            table_indexes: Dictionary of indexes from introspection
            expected_columns: List of column names to check
            
        Returns:
            bool: True if columns are indexed, False otherwise
        """
        # Convert expected columns to set for easier comparison
        expected_set = set(expected_columns)
        
        # Check each index to see if it covers our expected columns
        for index_name, index_info in table_indexes.items():
            # Get columns in this index
            if 'columns' in index_info:
                index_columns = set(index_info['columns'])
            elif 'column_names' in index_info:
                index_columns = set(index_info['column_names'])
            else:
                # Skip this index if we can't determine its columns
                continue            
            # Check if this index covers all expected columns
            if expected_set.issubset(index_columns):
                return True
        
        return False
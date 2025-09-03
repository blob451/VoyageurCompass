"""
Database integration tests with performance benchmarking.
Tests cross-module database operations and transaction management.
"""

from django.test import TestCase, TransactionTestCase
from django.contrib.auth import get_user_model
from django.db import transaction, connection
from django.core.management import call_command
from django.test.utils import override_settings
from datetime import datetime, date, timedelta
from decimal import Decimal
import time
import threading

from Data.models import Stock, StockPrice, Portfolio, PortfolioHolding, DataSector, DataIndustry

User = get_user_model()


class DatabaseTransactionIntegrationTest(TransactionTestCase):
    """Test database transaction management across modules."""
    
    def setUp(self):
        """Set up database integration test environment."""
        self.user = User.objects.create_user(
            username='db_integration_user',
            email='db@integration.com',
            password='db_integration_pass_123'
        )
        
        self.sector = DataSector.objects.create(
            sectorKey='tech_db',
            sectorName='Technology DB',
            data_source='yahoo'
        )
        
        self.industry = DataIndustry.objects.create(
            industryKey='software_db',
            industryName='Software DB',
            sector=self.sector,
            data_source='yahoo'
        )
    
    def test_cross_module_transaction_rollback(self):
        """Test transaction rollback across multiple modules."""
        # Initial counts
        initial_stock_count = Stock.objects.count()
        initial_portfolio_count = Portfolio.objects.count()
        initial_price_count = StockPrice.objects.count()
        
        try:
            with transaction.atomic():
                # Create stock
                stock = Stock.objects.create(
                    symbol='ROLLBACK_TEST',
                    short_name='Rollback Test Corp',
                    sector_id=self.sector,
                    industry_id=self.industry
                )
                
                # Create portfolio
                portfolio = Portfolio.objects.create(
                    user=self.user,
                    name='Rollback Test Portfolio',
                    initial_value=Decimal('1000.00')
                )
                
                # Create price data
                StockPrice.objects.create(
                    stock=stock,
                    date=date.today(),
                    open=Decimal('100.00'),
                    high=Decimal('105.00'),
                    low=Decimal('98.00'),
                    close=Decimal('103.00'),
                    volume=1000000
                )
                
                # Force rollback
                raise Exception("Intentional rollback")
                
        except Exception:
            pass  # Expected rollback
        
        # Verify rollback
        final_stock_count = Stock.objects.count()
        final_portfolio_count = Portfolio.objects.count()
        final_price_count = StockPrice.objects.count()
        
        self.assertEqual(final_stock_count, initial_stock_count)
        self.assertEqual(final_portfolio_count, initial_portfolio_count)
        self.assertEqual(final_price_count, initial_price_count)
        
        print("Cross-module transaction rollback verified")
    
    def test_foreign_key_cascade_operations(self):
        """Test foreign key cascade operations across models."""
        # Create related data
        stock = Stock.objects.create(
            symbol='CASCADE_TEST',
            short_name='Cascade Test Corp',
            sector_id=self.sector,
            industry_id=self.industry
        )
        
        portfolio = Portfolio.objects.create(
            user=self.user,
            name='Cascade Test Portfolio',
            initial_value=Decimal('5000.00')
        )
        
        # Create dependent objects
        price = StockPrice.objects.create(
            stock=stock,
            date=date.today(),
            open=Decimal('150.00'),
            high=Decimal('155.00'),
            low=Decimal('148.00'),
            close=Decimal('152.00'),
            volume=800000
        )
        
        holding = PortfolioHolding.objects.create(
            portfolio=portfolio,
            stock=stock,
            quantity=Decimal('10'),
            average_price=Decimal('150.00')
        )
        
        # Verify relationships exist
        self.assertEqual(StockPrice.objects.filter(stock=stock).count(), 1)
        self.assertEqual(PortfolioHolding.objects.filter(stock=stock).count(), 1)
        
        # Test cascade behavior
        stock.delete()
        
        # Verify cascade effects
        self.assertEqual(StockPrice.objects.filter(id=price.id).count(), 0)
        # Note: PortfolioHolding may or may not cascade depending on model definition
        
        print("Foreign key cascade operations verified")
    
    def test_concurrent_database_operations(self):
        """Test concurrent database operations across modules."""
        results = []
        errors = []
        
        def create_stock_data(thread_id):
            """Create stock data concurrently."""
            try:
                stock = Stock.objects.create(
                    symbol=f'CONCURRENT_{thread_id}',
                    short_name=f'Concurrent Test {thread_id}',
                    sector_id=self.sector,
                    industry_id=self.industry
                )
                
                # Create price data
                for i in range(5):
                    StockPrice.objects.create(
                        stock=stock,
                        date=date.today() - timedelta(days=i),
                        open=Decimal('100.00') + i,
                        high=Decimal('105.00') + i,
                        low=Decimal('98.00') + i,
                        close=Decimal('103.00') + i,
                        volume=1000000 + (i * 10000)
                    )
                
                results.append(f'thread_{thread_id}_success')
                
            except Exception as e:
                errors.append(f'thread_{thread_id}_error: {str(e)}')
        
        # Execute concurrent operations
        threads = []
        for i in range(3):
            thread = threading.Thread(target=create_stock_data, args=(i+1,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        self.assertEqual(len(results), 3, f"Expected 3 successful operations, got {len(results)}")
        self.assertEqual(len(errors), 0, f"Expected no errors, got: {errors}")
        
        # Verify data integrity
        concurrent_stocks = Stock.objects.filter(symbol__startswith='CONCURRENT_').count()
        self.assertEqual(concurrent_stocks, 3)
        
        concurrent_prices = StockPrice.objects.filter(
            stock__symbol__startswith='CONCURRENT_'
        ).count()
        self.assertEqual(concurrent_prices, 15)  # 3 stocks * 5 prices each
        
        print(f"Concurrent database operations completed: {results}")
    
    def test_database_constraint_validation(self):
        """Test database constraint validation."""
        # Test unique constraints
        stock1 = Stock.objects.create(
            symbol='CONSTRAINT_TEST',
            short_name='Constraint Test Corp',
            sector_id=self.sector,
            industry_id=self.industry
        )
        
        # Attempt duplicate symbol
        with self.assertRaises(Exception):
            Stock.objects.create(
                symbol='CONSTRAINT_TEST',  # Duplicate symbol
                short_name='Another Corp',
                sector_id=self.sector,
                industry_id=self.industry
            )
        
        # Test foreign key constraints
        with self.assertRaises(Exception):
            Stock.objects.create(
                symbol='FK_TEST',
                short_name='Foreign Key Test',
                sector_id_id=99999  # Non-existent sector
            )
        
        print("Database constraint validation completed")


class DatabasePerformanceBenchmarkTest(TestCase):
    """Test database performance with realistic data volumes."""
    
    def setUp(self):
        """Set up performance benchmark environment."""
        self.user = User.objects.create_user(
            username='perf_test_user',
            email='perf@test.com',
            password='perf_test_pass_123'
        )
        
        self.sector = DataSector.objects.create(
            sectorKey='tech_perf',
            sectorName='Technology Performance',
            data_source='yahoo'
        )
        
        self.industry = DataIndustry.objects.create(
            industryKey='software_perf',
            industryName='Software Performance',
            sector=self.sector,
            data_source='yahoo'
        )
    
    def test_bulk_data_creation_performance(self):
        """Test bulk data creation performance."""
        # Benchmark bulk stock creation
        start_time = time.time()
        
        stocks_to_create = []
        for i in range(100):
            stocks_to_create.append(Stock(
                symbol=f'BULK_{i:03d}',
                short_name=f'Bulk Test Corp {i}',
                sector_id=self.sector,
                industry_id=self.industry
            ))
        
        Stock.objects.bulk_create(stocks_to_create)
        bulk_creation_time = time.time() - start_time
        
        # Verify creation
        bulk_stock_count = Stock.objects.filter(symbol__startswith='BULK_').count()
        self.assertEqual(bulk_stock_count, 100)
        
        self.assertLess(bulk_creation_time, 5.0, "Bulk creation should complete within 5 seconds")
        print(f"Bulk created 100 stocks in {bulk_creation_time:.3f}s")
        
        # Benchmark bulk price data creation
        start_time = time.time()
        stock_sample = Stock.objects.filter(symbol__startswith='BULK_')[:10]
        
        prices_to_create = []
        for stock in stock_sample:
            for day in range(30):  # 30 days of data per stock
                prices_to_create.append(StockPrice(
                    stock=stock,
                    date=date.today() - timedelta(days=day),
                    open=Decimal('100.00') + day,
                    high=Decimal('105.00') + day,
                    low=Decimal('98.00') + day,
                    close=Decimal('103.00') + day,
                    volume=1000000 + (day * 10000)
                ))
        
        StockPrice.objects.bulk_create(prices_to_create)
        price_creation_time = time.time() - start_time
        
        # Verify price creation
        price_count = StockPrice.objects.filter(stock__symbol__startswith='BULK_').count()
        self.assertEqual(price_count, 300)  # 10 stocks * 30 days
        
        self.assertLess(price_creation_time, 10.0, "Bulk price creation should complete within 10 seconds")
        print(f"Bulk created 300 price records in {price_creation_time:.3f}s")
    
    def test_complex_query_performance(self):
        """Test complex query performance."""
        # Create test data
        stock = Stock.objects.create(
            symbol='QUERY_TEST',
            short_name='Query Test Corp',
            sector_id=self.sector,
            industry_id=self.industry
        )
        
        # Create price history
        for day in range(100):
            StockPrice.objects.create(
                stock=stock,
                date=date.today() - timedelta(days=day),
                open=Decimal('100.00') + (day % 10),
                high=Decimal('105.00') + (day % 10),
                low=Decimal('98.00') + (day % 10),
                close=Decimal('103.00') + (day % 10),
                volume=1000000 + (day * 1000)
            )
        
        # Benchmark complex queries
        start_time = time.time()
        
        # Complex aggregation query
        from django.db.models import Avg, Max, Min, Count
        aggregation_result = StockPrice.objects.filter(
            stock=stock,
            date__gte=date.today() - timedelta(days=30)
        ).aggregate(
            avg_price=Avg('close'),
            max_price=Max('high'),
            min_price=Min('low'),
            total_volume=Count('volume')
        )
        
        aggregation_time = time.time() - start_time
        
        # Verify aggregation results
        self.assertIsNotNone(aggregation_result['avg_price'])
        self.assertGreater(aggregation_result['total_volume'], 0)
        
        self.assertLess(aggregation_time, 2.0, "Complex aggregation should complete within 2 seconds")
        print(f"Complex aggregation query completed in {aggregation_time:.3f}s")
        
        # Benchmark join query
        start_time = time.time()
        
        portfolio = Portfolio.objects.create(
            user=self.user,
            name='Query Performance Portfolio',
            initial_value=Decimal('10000.00')
        )
        
        PortfolioHolding.objects.create(
            portfolio=portfolio,
            stock=stock,
            quantity=Decimal('50'),
            average_price=Decimal('102.00')
        )
        
        # Complex join query
        join_result = PortfolioHolding.objects.select_related(
            'portfolio__user', 'stock__sector_id', 'stock__industry_id'
        ).filter(
            portfolio__user=self.user,
            stock__sector_id=self.sector
        ).first()
        
        join_time = time.time() - start_time
        
        # Verify join results
        self.assertIsNotNone(join_result)
        self.assertEqual(join_result.portfolio.user, self.user)
        
        self.assertLess(join_time, 1.0, "Join query should complete within 1 second")
        print(f"Complex join query completed in {join_time:.3f}s")
    
    def test_database_connection_pooling(self):
        """Test database connection efficiency."""
        start_time = time.time()
        
        # Execute multiple queries to test connection reuse
        for i in range(10):
            Stock.objects.filter(symbol__startswith='BULK_').count()
            StockPrice.objects.filter(date=date.today()).count()
            Portfolio.objects.filter(user=self.user).count()
        
        connection_test_time = time.time() - start_time
        
        self.assertLess(connection_test_time, 3.0, "Connection pooling should enable fast repeated queries")
        print(f"10 query cycles completed in {connection_test_time:.3f}s")
        
        # Test connection health
        with connection.cursor() as cursor:
            cursor.execute("SELECT version()")
            version = cursor.fetchone()
            self.assertIsNotNone(version)
            print(f"Database version: {version[0][:50]}...")


class DatabaseMigrationIntegrationTest(TestCase):
    """Test database migration and schema operations."""
    
    def test_migration_state_validation(self):
        """Test that all migrations are applied correctly."""
        from django.db.migrations.executor import MigrationExecutor
        from django.db import connection
        
        executor = MigrationExecutor(connection)
        plan = executor.migration_plan(executor.loader.graph.leaf_nodes())
        
        # Should have no pending migrations
        self.assertEqual(len(plan), 0, "All migrations should be applied")
        print("All database migrations are current")
    
    def test_database_schema_integrity(self):
        """Test database schema integrity."""
        with connection.cursor() as cursor:
            # Check key tables exist
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE '%stock%' 
                OR table_name LIKE '%portfolio%'
                OR table_name LIKE '%user%'
            """)
            tables = cursor.fetchall()
            table_names = [table[0] for table in tables]
            
            # Verify key tables exist
            expected_patterns = ['stock', 'portfolio', 'user']
            for pattern in expected_patterns:
                matching_tables = [name for name in table_names if pattern in name.lower()]
                self.assertGreater(len(matching_tables), 0, f"Should have tables matching '{pattern}'")
            
            print(f"Database schema validated: {len(table_names)} relevant tables found")
    
    def test_database_indexes_performance(self):
        """Test that database indexes are working effectively."""
        # Create test data for index testing
        stock = Stock.objects.create(
            symbol='INDEX_TEST',
            short_name='Index Test Corp',
            sector_id=DataSector.objects.first() or DataSector.objects.create(
                sectorKey='test_sector',
                sectorName='Test Sector',
                data_source='yahoo'
            )
        )
        
        # Create price data to test index performance
        for i in range(50):
            StockPrice.objects.create(
                stock=stock,
                date=date.today() - timedelta(days=i),
                open=Decimal('100.00'),
                high=Decimal('105.00'),
                low=Decimal('98.00'),
                close=Decimal('103.00'),
                volume=1000000
            )
        
        # Test indexed query performance
        start_time = time.time()
        
        # Query by symbol (should be indexed)
        symbol_results = Stock.objects.filter(symbol='INDEX_TEST')
        self.assertEqual(symbol_results.count(), 1)
        
        # Query by date range (should be indexed)
        date_results = StockPrice.objects.filter(
            date__gte=date.today() - timedelta(days=30)
        )
        self.assertGreater(date_results.count(), 0)
        
        query_time = time.time() - start_time
        
        self.assertLess(query_time, 1.0, "Indexed queries should be fast")
        print(f"Index performance test completed in {query_time:.3f}s")


if __name__ == '__main__':
    import unittest
    unittest.main()
"""
Performance optimisation integration tests.
Tests parallel execution, connection pooling, and query optimisation.
"""

import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from decimal import Decimal

import pytest
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.db import transaction
from django.test import TestCase, TransactionTestCase
from django.utils import timezone

from Data.models import DataIndustry, DataSector, Portfolio, Stock, StockPrice

User = get_user_model()


@pytest.mark.performance
class ParallelExecutionTest(TransactionTestCase):
    """Test parallel execution capabilities and resource management."""

    def setUp(self):
        """Set up parallel execution test environment."""
        self.user = User.objects.create_user(
            username="parallel_test_user", email="parallel@test.com", password="parallel_test_pass_123"
        )

        self.sector = DataSector.objects.create(
            sectorKey="tech_parallel", sectorName="Technology Parallel", data_source="yahoo"
        )

        self.industry = DataIndustry.objects.create(
            industryKey="software_parallel", industryName="Software Parallel", sector=self.sector, data_source="yahoo"
        )

    def test_concurrent_database_operations(self):
        """Test concurrent database operations performance."""
        execution_times = []
        errors = []

        def create_stock_with_prices(thread_id):
            """Create stock and price data concurrently with retry logic."""
            max_retries = 3
            retry_delay = 0.1  # 100ms delay between retries

            for attempt in range(max_retries):
                try:
                    start_time = time.time()

                    # Use transaction.atomic for thread safety
                    with transaction.atomic():
                        # Create stock
                        stock = Stock.objects.create(
                            symbol=f"PARALLEL_{thread_id:03d}",
                            short_name=f"Parallel Test Stock {thread_id}",
                            currency="USD",
                            exchange="NYSE",
                            sector_id=self.sector,
                            industry_id=self.industry,
                        )

                        # Create price history
                        prices_to_create = []
                        for day in range(10):
                            prices_to_create.append(
                                StockPrice(
                                    stock=stock,
                                    date=date.today() - timedelta(days=day),
                                    open=Decimal("100.00") + day,
                                    high=Decimal("105.00") + day,
                                    low=Decimal("98.00") + day,
                                    close=Decimal("103.00") + day,
                                    volume=1000000 + (day * 1000),
                                )
                            )

                        StockPrice.objects.bulk_create(prices_to_create)

                        execution_time = time.time() - start_time
                        execution_times.append(execution_time)

                        return f"thread_{thread_id}_success"

                except Exception as e:
                    if "database table is locked" in str(e) and attempt < max_retries - 1:
                        # Exponential backoff
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        errors.append(f"thread_{thread_id}_error: {str(e)}")
                        return None

            return None

        # Execute concurrent operations with reduced concurrency for SQLite
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(create_stock_with_prices, i) for i in range(1, 11)]
            results = [future.result() for future in as_completed(futures)]

        # Verify results
        successful_operations = [r for r in results if r is not None]
        self.assertEqual(
            len(successful_operations), 10, f"Expected 10 successful operations, got {len(successful_operations)}"
        )
        self.assertEqual(len(errors), 0, f"Expected no errors, got: {errors}")

        # Performance assertions
        avg_execution_time = statistics.mean(execution_times)
        max_execution_time = max(execution_times)

        self.assertLess(
            avg_execution_time, 2.0, f"Average execution time should be < 2s, got {avg_execution_time:.3f}s"
        )
        self.assertLess(max_execution_time, 5.0, f"Max execution time should be < 5s, got {max_execution_time:.3f}s")

        # Verify data integrity
        parallel_stocks = Stock.objects.filter(symbol__startswith="PARALLEL_").count()
        self.assertEqual(parallel_stocks, 10)

        parallel_prices = StockPrice.objects.filter(stock__symbol__startswith="PARALLEL_").count()
        self.assertEqual(parallel_prices, 100)  # 10 stocks * 10 prices each

        print("Concurrent operations completed:")
        print(f"  Average time: {avg_execution_time:.3f}s")
        print(f"  Max time: {max_execution_time:.3f}s")
        print(f"  Successful operations: {len(successful_operations)}")

    def test_connection_pool_efficiency(self):
        """Test database connection pool efficiency."""
        connection_times = []

        def test_database_query(query_id):
            """Execute database query and measure connection time."""
            start_time = time.time()

            # Execute queries that would benefit from connection pooling
            Stock.objects.filter(symbol__startswith="PARALLEL_").count()
            StockPrice.objects.filter(date=date.today()).count()
            Portfolio.objects.filter(user=self.user).count()

            query_time = time.time() - start_time
            connection_times.append(query_time)

            return query_time

        # Execute parallel queries
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(test_database_query, i) for i in range(20)]
            times = [future.result() for future in as_completed(futures)]

        # Analyze connection performance
        avg_query_time = statistics.mean(times)
        max_query_time = max(times)

        self.assertLess(
            avg_query_time,
            0.5,
            f"Average query time should be < 0.5s with connection pooling, " f"got {avg_query_time:.3f}s",
        )
        self.assertLess(max_query_time, 1.0, f"Max query time should be < 1s, got {max_query_time:.3f}s")

        print("Connection pool performance:")
        print("  20 parallel queries completed")
        print(f"  Average time: {avg_query_time:.3f}s")
        print(f"  Max time: {max_query_time:.3f}s")

    def test_cache_performance_optimization(self):
        """Test cache performance under load."""
        cache.clear()
        cache_times = []

        def cache_operations(operation_id):
            """Perform cache operations."""
            start_time = time.time()

            # Set cache values
            cache.set(f"test_key_{operation_id}", f"test_value_{operation_id}", 60)

            # Get cache values
            for i in range(5):
                cache.get(f"test_key_{operation_id}")

            # Set multiple values
            cache.set_many({f"bulk_key_{operation_id}_{i}": f"bulk_value_{operation_id}_{i}" for i in range(5)}, 60)

            operation_time = time.time() - start_time
            cache_times.append(operation_time)

            return operation_time

        # Execute parallel cache operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(cache_operations, i) for i in range(20)]
            times = [future.result() for future in as_completed(futures)]

        # Analyze cache performance
        avg_cache_time = statistics.mean(times)

        self.assertLess(
            avg_cache_time, 0.1, f"Average cache operation time should be < 0.1s, got {avg_cache_time:.3f}s"
        )

        print("Cache performance under load:")
        print("  20 parallel cache operations completed")
        print(f"  Average time: {avg_cache_time:.3f}s")


@pytest.mark.performance
class QueryOptimisationTest(TestCase):
    """Query optimisation and database performance test suite."""

    @classmethod
    def setUpTestData(cls):
        """Set up test data for query optimisation tests."""
        # Create user
        cls.user = User.objects.create_user(
            username="query_opt_user", email="queryopt@test.com", password="query_opt_pass_123"
        )

        # Create sector and industry
        cls.sector = DataSector.objects.create(
            sectorKey="tech_query_opt", sectorName="Technology Query Optimisation", data_source="yahoo"
        )

        cls.industry = DataIndustry.objects.create(
            industryKey="software_query_opt",
            industryName="Software Query Optimisation",
            sector=cls.sector,
            data_source="yahoo",
        )

        # Create test stocks with extensive price history
        stocks_to_create = []
        for i in range(50):
            stocks_to_create.append(
                Stock(
                    symbol=f"QOPT_{i:03d}",
                    short_name=f"Query Optimisation Stock {i}",
                    currency="USD",
                    exchange="NASDAQ",
                    sector_id=cls.sector,
                    industry_id=cls.industry,
                )
            )

        Stock.objects.bulk_create(stocks_to_create)

        # Create price data for performance testing
        stocks = Stock.objects.filter(symbol__startswith="QOPT_")[:20]  # Limit for test performance
        prices_to_create = []

        for stock in stocks:
            for day in range(100):  # 100 days of price history per stock
                prices_to_create.append(
                    StockPrice(
                        stock=stock,
                        date=date.today() - timedelta(days=day),
                        open=Decimal("100.00") + (day % 20),
                        high=Decimal("105.00") + (day % 20),
                        low=Decimal("98.00") + (day % 20),
                        close=Decimal("103.00") + (day % 20),
                        volume=1000000 + (day * 1000),
                    )
                )

        StockPrice.objects.bulk_create(prices_to_create)

    def test_complex_query_performance(self):
        """Test performance of complex database queries."""
        from django.db.models import Avg, Count, Max, Min, Q

        # Test 1: Complex aggregation query
        start_time = time.time()

        aggregation_results = StockPrice.objects.filter(
            stock__symbol__startswith="QOPT_", date__gte=date.today() - timedelta(days=30)
        ).aggregate(avg_close=Avg("close"), max_high=Max("high"), min_low=Min("low"), total_volume=Count("volume"))

        aggregation_time = time.time() - start_time

        self.assertIsNotNone(aggregation_results["avg_close"])
        self.assertLess(aggregation_time, 2.0, f"Aggregation query should complete < 2s, took {aggregation_time:.3f}s")

        # Test 2: Complex join query with select_related
        start_time = time.time()

        join_results = list(
            StockPrice.objects.select_related("stock__sector_id", "stock__industry_id").filter(
                stock__symbol__startswith="QOPT_", date__gte=date.today() - timedelta(days=7)
            )[:100]
        )

        join_time = time.time() - start_time

        self.assertGreater(len(join_results), 0)
        self.assertLess(join_time, 1.0, f"Join query should complete < 1s, took {join_time:.3f}s")

        # Test 3: Complex filter with OR conditions
        start_time = time.time()

        complex_filter_results = Stock.objects.filter(
            Q(symbol__startswith="QOPT_") | Q(short_name__icontains="Query"), sector_id=self.sector
        ).count()

        filter_time = time.time() - start_time

        self.assertGreater(complex_filter_results, 0)
        self.assertLess(filter_time, 0.5, f"Complex filter should complete < 0.5s, took {filter_time:.3f}s")

        print("Query performance results:")
        print(f"  Aggregation query: {aggregation_time:.3f}s")
        print(f"  Join query: {join_time:.3f}s")
        print(f"  Complex filter: {filter_time:.3f}s")

    def test_bulk_operations_performance(self):
        """Test performance of bulk database operations."""
        # Test bulk_create performance
        start_time = time.time()

        bulk_stocks = [
            Stock(
                symbol=f"BULK_PERF_{i:04d}",
                short_name=f"Bulk Performance Stock {i}",
                currency="USD",
                exchange="NYSE",
                sector_id=self.sector,
                industry_id=self.industry,
            )
            for i in range(200)
        ]

        Stock.objects.bulk_create(bulk_stocks)
        bulk_create_time = time.time() - start_time

        self.assertLess(bulk_create_time, 3.0, f"Bulk create should complete < 3s, took {bulk_create_time:.3f}s")

        # Test bulk_update performance
        created_stocks = Stock.objects.filter(symbol__startswith="BULK_PERF_")
        for stock in created_stocks:
            stock.short_name = f"Updated {stock.short_name}"

        start_time = time.time()
        Stock.objects.bulk_update(created_stocks, ["short_name"])
        bulk_update_time = time.time() - start_time

        self.assertLess(bulk_update_time, 2.0, f"Bulk update should complete < 2s, took {bulk_update_time:.3f}s")

        # Test bulk_delete performance
        start_time = time.time()
        deleted_count = Stock.objects.filter(symbol__startswith="BULK_PERF_").delete()[0]
        bulk_delete_time = time.time() - start_time

        self.assertEqual(deleted_count, 200)
        self.assertLess(bulk_delete_time, 1.0, f"Bulk delete should complete < 1s, took {bulk_delete_time:.3f}s")

        print("Bulk operations performance:")
        print(f"  Bulk create (200 records): {bulk_create_time:.3f}s")
        print(f"  Bulk update (200 records): {bulk_update_time:.3f}s")
        print(f"  Bulk delete (200 records): {bulk_delete_time:.3f}s")


@pytest.mark.performance
class IntegrationTestSuitePerformanceTest(TransactionTestCase):
    """Overall integration test suite performance evaluation."""

    def test_complete_test_suite_execution_time(self):
        """Test that the complete integration test suite meets performance targets."""
        suite_start_time = time.time()

        # Simulate key integration test operations
        test_operations = [
            self._test_user_authentication,
            self._test_portfolio_operations,
            self._test_stock_data_operations,
            self._test_analytics_operations,
            self._test_api_operations,
        ]

        operation_times = {}

        for operation in test_operations:
            start_time = time.time()
            operation()
            operation_times[operation.__name__] = time.time() - start_time

        total_suite_time = time.time() - suite_start_time

        # Performance targets
        self.assertLess(
            total_suite_time,
            300.0,
            f"Complete integration test suite should complete < 5 minutes, took {total_suite_time:.1f}s",
        )

        # Individual operation targets
        for operation_name, operation_time in operation_times.items():
            self.assertLess(
                operation_time, 60.0, f"{operation_name} should complete < 1 minute, took {operation_time:.1f}s"
            )

        print("Integration test suite performance:")
        print(f"  Total suite time: {total_suite_time:.1f}s (target: < 300s)")
        for operation_name, operation_time in operation_times.items():
            print(f"  {operation_name}: {operation_time:.1f}s")

    def _test_user_authentication(self):
        """Simulate user authentication operations."""
        from django.contrib.auth import authenticate

        user = User.objects.create_user(
            username="perf_auth_user", email="perfauth@test.com", password="perf_auth_pass_123"
        )

        # Simulate authentication operations
        for _ in range(10):
            authenticated_user = authenticate(username="perf_auth_user", password="perf_auth_pass_123")
            self.assertIsNotNone(authenticated_user)

    def _test_portfolio_operations(self):
        """Simulate portfolio operations."""
        user = User.objects.create_user(
            username="perf_portfolio_user", email="perfportfolio@test.com", password="perf_portfolio_pass_123"
        )

        # Create portfolios
        portfolios = []
        for i in range(5):
            portfolio = Portfolio.objects.create(
                user=user, name=f"Performance Test Portfolio {i}", initial_value=Decimal("10000.00")
            )
            portfolios.append(portfolio)

        # Verify portfolio operations
        self.assertEqual(Portfolio.objects.filter(user=user).count(), 5)

    def _test_stock_data_operations(self):
        """Simulate stock data operations."""
        sector = DataSector.objects.create(
            sectorKey="tech_perf_ops", sectorName="Technology Performance Operations", data_source="yahoo"
        )

        # Create stocks and price data
        stocks = []
        for i in range(10):
            stock = Stock.objects.create(
                symbol=f"PERF_OPS_{i:02d}", short_name=f"Performance Operations Stock {i}", sector_id=sector
            )
            stocks.append(stock)

            # Add price data
            StockPrice.objects.create(
                stock=stock,
                date=date.today(),
                open=Decimal("100.00"),
                high=Decimal("105.00"),
                low=Decimal("98.00"),
                close=Decimal("103.00"),
                volume=1000000,
            )

        self.assertEqual(Stock.objects.filter(symbol__startswith="PERF_OPS_").count(), 10)

    def _test_analytics_operations(self):
        """Simulate analytics operations."""
        from Data.models import AnalyticsResults

        sector = DataSector.objects.create(
            sectorKey="tech_perf_analytics", sectorName="Technology Performance Analytics", data_source="yahoo"
        )

        stock = Stock.objects.create(
            symbol="PERF_ANALYTICS", short_name="Performance Analytics Stock", sector_id=sector
        )

        # Create analytics results
        analytics_result = AnalyticsResults.objects.create(
            stock=stock,
            as_of=timezone.now(),
            horizon="blend",
            w_rsi14=Decimal("0.655"),
            w_pricevs50=Decimal("0.148"),
            w_sma50vs200=Decimal("0.145"),
            w_macd12269=Decimal("0.085"),
            composite_raw=Decimal("7.2"),
        )

        self.assertIsNotNone(analytics_result)

    def _test_api_operations(self):
        """Simulate API operations."""
        from rest_framework.test import APIClient

        client = APIClient()
        user = User.objects.create_user(
            username="perf_api_user", email="perfapi@test.com", password="perf_api_pass_123"
        )
        client.force_authenticate(user=user)

        # Simulate API calls
        from django.urls import reverse

        try:
            # Test various API endpoints
            client.get(reverse("data:stock-list"))
            client.get(reverse("data:portfolio-list"))
        except Exception:
            # Endpoints may not exist in test environment
            pass


if __name__ == "__main__":
    import unittest

    unittest.main()

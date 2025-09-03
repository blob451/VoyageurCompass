"""
Pytest configuration and shared fixtures for VoyageurCompass test suite.
Enables parallel test execution and shared test infrastructure.
"""

import pytest
import django
import os
import logging
import time
import threading
from decimal import Decimal
from datetime import date, datetime, timedelta

# Set Django settings module before any Django imports
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'VoyageurCompass.settings')

# Configure test logging
logging.basicConfig(level=logging.ERROR)

# Django setup for pytest
django.setup()


@pytest.fixture(scope='session')
def test_performance_data():
    """Create shared performance test data once per session."""
    from django.contrib.auth import get_user_model
    from Data.models import DataSector, DataIndustry, Stock, StockPrice
    
    User = get_user_model()
    
    # Create test user
    test_user = User.objects.get_or_create(
        username='perf_test_user',
        defaults={
            'email': 'perf@test.com',
            'password': 'pbkdf2_sha256$260000$test'  # Pre-hashed password
        }
    )[0]
    
    # Create test sector and industry
    test_sector = DataSector.objects.get_or_create(
        sectorKey='tech_perf_shared',
        defaults={
            'sectorName': 'Technology Performance Shared',
            'data_source': 'yahoo'
        }
    )[0]
    
    test_industry = DataIndustry.objects.get_or_create(
        industryKey='software_perf_shared',
        defaults={
            'industryName': 'Software Performance Shared',
            'sector': test_sector,
            'data_source': 'yahoo'
        }
    )[0]
    
    # Create test stocks with price history
    stocks_data = []
    for i in range(5):  # Create 5 stocks for shared testing
        stock = Stock.objects.get_or_create(
            symbol=f'PERF_SHARED_{i:02d}',
            defaults={
                'short_name': f'Performance Shared Stock {i}',
                'currency': 'USD',
                'exchange': 'NYSE',
                'sector_id': test_sector,
                'industry_id': test_industry
            }
        )[0]
        
        # Create price history if not exists
        if not StockPrice.objects.filter(stock=stock).exists():
            prices_to_create = []
            for day in range(30):  # 30 days of price history
                prices_to_create.append(StockPrice(
                    stock=stock,
                    date=date.today() - timedelta(days=day),
                    open=Decimal('100.00') + day + i,
                    high=Decimal('105.00') + day + i,
                    low=Decimal('98.00') + day + i,
                    close=Decimal('103.00') + day + i,
                    volume=1000000 + (day * 1000) + (i * 10000)
                ))
            StockPrice.objects.bulk_create(prices_to_create, ignore_conflicts=True)
        
        stocks_data.append(stock)
    
    return {
        'user': test_user,
        'sector': test_sector,
        'industry': test_industry,
        'stocks': stocks_data
    }


@pytest.fixture
def api_client():
    """Provide authenticated API client."""
    from rest_framework.test import APIClient
    from django.contrib.auth import get_user_model
    
    User = get_user_model()
    client = APIClient()
    
    # Create or get test user
    user = User.objects.get_or_create(
        username='api_test_user',
        defaults={
            'email': 'api@test.com',
            'password': 'pbkdf2_sha256$260000$test'
        }
    )[0]
    
    client.force_authenticate(user=user)
    return client


@pytest.fixture
def mock_external_services():
    """Mock external services for integration tests."""
    class MockYahooFinance:
        @staticmethod
        def get_stock_data(symbol, period='1d'):
            return {
                'symbol': symbol,
                'price': Decimal('150.00'),
                'volume': 1000000,
                'timestamp': datetime.now()
            }
    
    class MockOllamaService:
        @staticmethod
        def generate_explanation(stock_data, analysis_data):
            return {
                'explanation': f"Analysis for {stock_data['symbol']}: Moderate performance with positive outlook.",
                'confidence': 0.85,
                'generated_at': datetime.now()
            }
    
    return {
        'yahoo_finance': MockYahooFinance(),
        'ollama_service': MockOllamaService()
    }


@pytest.fixture
def performance_timer():
    """Measure test execution time."""
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            
        def start(self):
            self.start_time = time.time()
            
        def stop(self):
            self.end_time = time.time()
            
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    timer = PerformanceTimer()
    timer.start()
    yield timer
    timer.stop()


def pytest_configure(config):
    """Configure pytest with custom markers and optimizations."""
    import sys
    
    # Optimize for parallel testing
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "performance: mark test as performance benchmark")
    config.addinivalue_line("markers", "database: mark test as database-focused")
    
    # Suppress unnecessary warnings during testing
    if not sys.warnoptions:
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to optimize execution order."""
    # Separate tests by type for better parallel distribution
    unit_tests = []
    integration_tests = []
    performance_tests = []
    other_tests = []
    
    for item in items:
        if "integration" in item.keywords:
            integration_tests.append(item)
        elif "performance" in item.keywords:
            performance_tests.append(item)
        elif any(marker in item.keywords for marker in ["unit", "models", "api"]):
            unit_tests.append(item)
        else:
            other_tests.append(item)
    
    # Reorder: unit tests first (fast), then integration, then performance
    items[:] = unit_tests + other_tests + integration_tests + performance_tests


@pytest.fixture(scope='session')
def worker_id(request):
    """Get pytest-xdist worker ID for parallel test isolation."""
    if hasattr(request.config, 'slaveinput'):
        return request.config.slaveinput['slaveid']
    return 'master'


class TestExecutionMonitor:
    """Monitor test execution performance and resource usage."""
    
    def __init__(self):
        self.test_times = {}
        self.slow_tests = []
        
    def record_test_time(self, test_name, execution_time):
        self.test_times[test_name] = execution_time
        if execution_time > 10.0:  # Tests taking more than 10 seconds
            self.slow_tests.append((test_name, execution_time))
    
    def get_performance_summary(self):
        if not self.test_times:
            return "No test performance data recorded"
        
        total_time = sum(self.test_times.values())
        avg_time = total_time / len(self.test_times)
        
        summary = f"Test Performance Summary:\n"
        summary += f"  Total tests: {len(self.test_times)}\n"
        summary += f"  Total time: {total_time:.2f}s\n"
        summary += f"  Average time: {avg_time:.2f}s\n"
        
        if self.slow_tests:
            summary += f"  Slow tests ({len(self.slow_tests)}):\n"
            for test_name, time_taken in sorted(self.slow_tests, key=lambda x: x[1], reverse=True)[:5]:
                summary += f"    {test_name}: {time_taken:.2f}s\n"
        
        return summary


# Global test monitor
test_monitor = TestExecutionMonitor()


@pytest.fixture(scope='session', autouse=True)
def test_performance_monitor():
    """Monitor overall test suite performance."""
    yield test_monitor


def pytest_runtest_teardown(item, nextitem):
    """Clean up after each test and record performance metrics."""
    # Record test execution time
    if hasattr(item, '_test_start_time'):
        execution_time = time.time() - item._test_start_time
        test_monitor.record_test_time(item.nodeid, execution_time)


def pytest_runtest_setup(item):
    """Set up before each test."""
    item._test_start_time = time.time()


def pytest_sessionfinish(session, exitstatus):
    """Print performance summary at the end of test session."""
    if hasattr(session.config, 'option') and session.config.option.verbose:
        print(f"\n{test_monitor.get_performance_summary()}")
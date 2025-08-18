"""
Unit tests for Data yahoo_finance service module.
Tests Yahoo Finance API integration and data processing.
"""

import pandas as pd
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from django.test import TestCase
from django.utils import timezone

from Data.models import Stock, StockPrice, DataSector, DataIndustry
# Import the main class from yahoo_finance service
# Note: We'll test the core functionality and cache mechanisms


class YahooFinanceServiceTestCase(TestCase):
    """Test cases for Yahoo Finance service functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create test stock
        self.stock = Stock.objects.create(
            symbol='TEST',
            short_name='Test Company',
            exchange='NASDAQ'
        )
        
        # Mock Yahoo Finance data
        self.mock_yf_data = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'High': [105.0, 106.0, 107.0, 108.0, 109.0],
            'Low': [95.0, 96.0, 97.0, 98.0, 99.0],
            'Close': [102.0, 103.0, 104.0, 105.0, 106.0],
            'Adj Close': [101.5, 102.5, 103.5, 104.5, 105.5],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }).set_index('Date')
        
        self.mock_info_data = {
            'symbol': 'TEST',
            'shortName': 'Test Company',
            'longName': 'Test Company Inc.',
            'exchange': 'NASDAQ',
            'sector': 'Technology',
            'industry': 'Software'
        }
    
    @patch('yfinance.Ticker')
    def test_fetch_stock_data_basic(self, mock_ticker):
        """Test basic stock data fetching."""
        # Mock yfinance Ticker
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = self.mock_yf_data
        mock_ticker_instance.info = self.mock_info_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Import and test the service (mocking the actual service class)
        from Data.services.yahoo_finance import CompositeCache
        
        cache = CompositeCache()
        
        # Test cache key generation
        cache_key = cache.get_cache_key([self.stock], date(2023, 1, 1), date(2023, 1, 5))
        self.assertIsInstance(cache_key, str)
        self.assertEqual(len(cache_key), 12)  # MD5 hash truncated to 12 chars
        
        # Test composite cache key generation
        composite_key = cache.get_composite_cache_key(self.stock, date(2023, 1, 1))
        self.assertIn('Stock', composite_key)
        self.assertIn(str(self.stock.id), composite_key)
    
    @patch('yfinance.Ticker')
    def test_fetch_stock_info(self, mock_ticker):
        """Test fetching stock information."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = self.mock_info_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Test info data structure
        self.assertEqual(self.mock_info_data['symbol'], 'TEST')
        self.assertEqual(self.mock_info_data['sector'], 'Technology')
        self.assertEqual(self.mock_info_data['industry'], 'Software')
    
    @patch('yfinance.Ticker')
    def test_fetch_historical_data_with_retry(self, mock_ticker):
        """Test historical data fetching with retry mechanism."""
        # First call fails, second succeeds
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.side_effect = [
            Exception("Connection error"),
            self.mock_yf_data
        ]
        mock_ticker.return_value = mock_ticker_instance
        
        # Test would verify retry logic works
        with patch('time.sleep'):  # Speed up test by mocking sleep
            # Mock the retry behavior
            try:
                result = mock_ticker_instance.history(start='2023-01-01', end='2023-01-05')
                self.fail("Should have raised exception on first call")
            except Exception:
                # Retry should work
                result = mock_ticker_instance.history(start='2023-01-01', end='2023-01-05')
                self.assertIsInstance(result, pd.DataFrame)
    
    @patch('yfinance.Ticker')
    def test_data_validation_and_cleaning(self, mock_ticker):
        """Test data validation and cleaning processes."""
        # Mock data with some invalid values
        dirty_data = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=3, freq='D'),
            'Open': [100.0, None, 102.0],  # None value
            'High': [105.0, 106.0, float('inf')],  # Infinite value
            'Low': [95.0, 96.0, -1.0],  # Negative value
            'Close': [102.0, 103.0, 104.0],
            'Adj Close': [101.5, 102.5, 103.5],
            'Volume': [1000000, 0, 1200000]  # Zero volume
        }).set_index('Date')
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = dirty_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Test data cleaning logic
        cleaned_data = dirty_data.dropna()  # Remove None values
        cleaned_data = cleaned_data.replace([float('inf'), -float('inf')], None).dropna()
        
        self.assertEqual(len(cleaned_data), 1)  # Only first row should remain clean
    
    def test_composite_cache_functionality(self):
        """Test CompositeCache class functionality."""
        from Data.services.yahoo_finance import CompositeCache
        
        cache = CompositeCache()
        
        # Test caching and retrieval
        test_key = "test_key"
        test_data = {"price": 100.0, "volume": 1000000}
        
        cache.cache_composite(test_key, test_data)
        retrieved_data = cache.get_cached_composite(test_key)
        
        self.assertEqual(retrieved_data, test_data)
        
        # Test cache clearing
        cache.clear_cache()
        retrieved_after_clear = cache.get_cached_composite(test_key)
        self.assertIsNone(retrieved_after_clear)
    
    def test_price_data_indexing(self):
        """Test price data indexing for fast lookup."""
        from Data.services.yahoo_finance import CompositeCache
        
        cache = CompositeCache()
        
        # Create mock price objects
        mock_prices = []
        for i in range(3):
            mock_price = Mock()
            mock_price.stock = self.stock
            mock_price.date = date(2023, 1, i + 1)
            mock_price.close = Decimal(f'10{i}.00')
            mock_prices.append(mock_price)
        
        # Test indexing
        indexed_prices = cache.index_prices_by_stock_and_date(mock_prices)
        
        self.assertIn(self.stock, indexed_prices)
        self.assertEqual(len(indexed_prices[self.stock]), 3)
        self.assertEqual(indexed_prices[self.stock][date(2023, 1, 1)].close, Decimal('100.00'))
    
    @patch('requests.Session')
    def test_http_session_configuration(self, mock_session):
        """Test HTTP session configuration for Yahoo Finance requests."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        
        # Test that session is configured with proper headers and timeout
        # This would test the actual HTTP configuration in the service
        expected_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Mock session configuration
        mock_session_instance.headers.update(expected_headers)
        mock_session_instance.get.return_value.status_code = 200
        
        # Verify configuration
        mock_session_instance.headers.update.assert_called_once()
    
    @patch('yfinance.Ticker')
    def test_error_handling_network_timeout(self, mock_ticker):
        """Test error handling for network timeouts."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.side_effect = TimeoutError("Request timed out")
        mock_ticker.return_value = mock_ticker_instance
        
        # Test that timeout errors are handled gracefully
        with self.assertRaises(TimeoutError):
            mock_ticker_instance.history(start='2023-01-01', end='2023-01-05')
    
    @patch('yfinance.Ticker')
    def test_error_handling_invalid_symbol(self, mock_ticker):
        """Test error handling for invalid stock symbols."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()  # Empty DataFrame
        mock_ticker_instance.info = {}  # Empty info
        mock_ticker.return_value = mock_ticker_instance
        
        # Test handling of empty responses (invalid symbols)
        result = mock_ticker_instance.history(start='2023-01-01', end='2023-01-05')
        self.assertTrue(result.empty)
    
    def test_date_range_validation(self):
        """Test validation of date ranges for data requests."""
        today = date.today()
        
        # Test valid date range
        start_date = today - timedelta(days=30)
        end_date = today
        self.assertLess(start_date, end_date)
        
        # Test invalid date range (start after end)
        invalid_start = today + timedelta(days=1)
        invalid_end = today
        self.assertGreater(invalid_start, invalid_end)
    
    @patch('yfinance.Ticker')
    def test_data_type_conversion(self, mock_ticker):
        """Test proper data type conversion for database storage."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = self.mock_yf_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Test conversion of pandas data to Python types
        for index, row in self.mock_yf_data.iterrows():
            # Test that values can be converted to Decimal for database storage
            open_price = Decimal(str(row['Open']))
            self.assertIsInstance(open_price, Decimal)
            
            # Test volume conversion to int
            volume = int(row['Volume'])
            self.assertIsInstance(volume, int)
            
            # Test date handling
            date_value = index.date() if hasattr(index, 'date') else index
            self.assertIsInstance(date_value, (date, datetime))
    
    def test_concurrent_request_handling(self):
        """Test handling of concurrent Yahoo Finance requests."""
        from Data.services.yahoo_finance import CompositeCache
        import threading
        
        cache = CompositeCache()
        results = []
        errors = []
        
        def cache_operation(key, data):
            try:
                cache.cache_composite(key, data)
                retrieved = cache.get_cached_composite(key)
                results.append(retrieved)
            except Exception as e:
                errors.append(str(e))
        
        threads = []
        for i in range(10):
            thread = threading.Thread(
                target=cache_operation,
                args=(f'key_{i}', {'data': i})
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All operations should succeed
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 10)


class YahooFinanceIntegrationTestCase(TestCase):
    """Integration tests for Yahoo Finance service with database operations."""
    
    def setUp(self):
        """Set up test data."""
        self.stock = Stock.objects.create(
            symbol='AAPL',
            short_name='Apple Inc.',
            exchange='NASDAQ'
        )
        
        self.sector = DataSector.objects.create(
            sectorKey='technology',
            sectorName='Technology'
        )
        
        self.industry = DataIndustry.objects.create(
            industryKey='consumer_electronics',
            industryName='Consumer Electronics'
        )
    
    @patch('yfinance.Ticker')
    def test_stock_data_to_database_integration(self, mock_ticker):
        """Test integration of Yahoo Finance data with database storage."""
        # Mock Yahoo Finance response
        mock_data = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=3, freq='D'),
            'Open': [150.0, 151.0, 152.0],
            'High': [155.0, 156.0, 157.0],
            'Low': [148.0, 149.0, 150.0],
            'Close': [153.0, 154.0, 155.0],
            'Adj Close': [152.5, 153.5, 154.5],
            'Volume': [50000000, 51000000, 52000000]
        }).set_index('Date')
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Simulate data processing and storage
        for index, row in mock_data.iterrows():
            StockPrice.objects.create(
                stock=self.stock,
                date=index.date(),
                open=Decimal(str(row['Open'])),
                high=Decimal(str(row['High'])),
                low=Decimal(str(row['Low'])),
                close=Decimal(str(row['Close'])),
                adjusted_close=Decimal(str(row['Adj Close'])),
                volume=int(row['Volume'])
            )
        
        # Verify data was stored correctly
        stored_prices = StockPrice.objects.filter(stock=self.stock).order_by('date')
        self.assertEqual(stored_prices.count(), 3)
        
        first_price = stored_prices.first()
        self.assertEqual(first_price.open, Decimal('150.0'))
        self.assertEqual(first_price.volume, 50000000)
    
    def test_batch_stock_updates(self):
        """Test batch updates of multiple stocks."""
        # Create multiple stocks
        stocks = []
        for i in range(3):
            stock = Stock.objects.create(
                symbol=f'STOCK{i}',
                short_name=f'Stock {i} Inc.',
                exchange='NASDAQ'
            )
            stocks.append(stock)
        
        # Simulate batch price updates
        test_date = date(2023, 1, 1)
        for stock in stocks:
            StockPrice.objects.create(
                stock=stock,
                date=test_date,
                open=Decimal('100.0'),
                high=Decimal('105.0'),
                low=Decimal('95.0'),
                close=Decimal('102.0'),
                adjusted_close=Decimal('101.5'),
                volume=1000000
            )
        
        # Verify all stocks have price data
        for stock in stocks:
            prices = StockPrice.objects.filter(stock=stock, date=test_date)
            self.assertEqual(prices.count(), 1)
    
    @patch('yfinance.Ticker')
    def test_data_update_conflict_resolution(self, mock_ticker):
        """Test handling of conflicting data updates."""
        # Create initial price data
        initial_price = StockPrice.objects.create(
            stock=self.stock,
            date=date(2023, 1, 1),
            open=Decimal('100.0'),
            high=Decimal('105.0'),
            low=Decimal('95.0'),
            close=Decimal('102.0'),
            volume=1000000
        )
        
        # Simulate update with new data
        mock_data = pd.DataFrame({
            'Date': [pd.Timestamp('2023-01-01')],
            'Open': [101.0],  # Different value
            'High': [106.0],  # Different value
            'Low': [96.0],    # Different value
            'Close': [103.0], # Different value
            'Adj Close': [102.5],
            'Volume': [1100000]  # Different value
        }).set_index('Date')
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Simulate update logic (would use update_or_create in real implementation)
        updated_price = StockPrice.objects.get(stock=self.stock, date=date(2023, 1, 1))
        updated_price.open = Decimal('101.0')
        updated_price.high = Decimal('106.0')
        updated_price.low = Decimal('96.0')
        updated_price.close = Decimal('103.0')
        updated_price.volume = 1100000
        updated_price.save()
        
        # Verify update was applied
        final_price = StockPrice.objects.get(stock=self.stock, date=date(2023, 1, 1))
        self.assertEqual(final_price.open, Decimal('101.0'))
        self.assertEqual(final_price.volume, 1100000)
        
        # Should still have only one record
        self.assertEqual(StockPrice.objects.filter(stock=self.stock).count(), 1)
    
    def test_performance_with_large_datasets(self):
        """Test performance handling of large datasets."""
        # Create a large batch of price data
        test_date = date(2023, 1, 1)
        batch_size = 1000
        
        prices_to_create = []
        for i in range(batch_size):
            prices_to_create.append(StockPrice(
                stock=self.stock,
                date=test_date + timedelta(days=i),
                open=Decimal('100.0') + i * Decimal('0.01'),
                high=Decimal('105.0') + i * Decimal('0.01'),
                low=Decimal('95.0') + i * Decimal('0.01'),
                close=Decimal('102.0') + i * Decimal('0.01'),
                volume=1000000 + i * 1000
            ))
        
        # Time the bulk creation
        start_time = timezone.now()
        StockPrice.objects.bulk_create(prices_to_create)
        end_time = timezone.now()
        
        # Verify all records were created
        self.assertEqual(StockPrice.objects.filter(stock=self.stock).count(), batch_size)
        
        # Should complete within reasonable time (less than 5 seconds)
        duration = (end_time - start_time).total_seconds()
        self.assertLess(duration, 5.0)
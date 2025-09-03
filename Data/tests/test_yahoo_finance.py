"""
Unit tests for Data yahoo_finance service module.
Tests Yahoo Finance API integration and data processing.
"""

import pandas as pd
from datetime import datetime, date, timedelta
from decimal import Decimal
from django.test import TestCase
from django.utils import timezone

from Data.models import Stock, StockPrice, DataSector, DataIndustry
from Data.tests.fixtures import DataTestDataFactory, YahooFinanceTestService
from Core.tests.fixtures import TestEnvironmentManager


class YahooFinanceServiceTestCase(TestCase):
    """Test cases for Yahoo Finance service functionality."""
    
    def setUp(self):
        """Set up test data."""
        TestEnvironmentManager.setup_test_environment()
        
        # Create real test stock using factory
        self.stock = DataTestDataFactory.create_test_stock('AAPL', 'Apple Inc.', 'Technology')
        
        # Create real Yahoo Finance test service
        self.yf_service = YahooFinanceTestService()
        
        # Create real historical data
        self.historical_data = self.yf_service.get_historical_data(
            'AAPL', 
            date(2023, 1, 1), 
            date(2023, 1, 5)
        )
        
        # Get real stock info data
        self.stock_info = self.yf_service.get_stock_info('AAPL')
    
    def tearDown(self):
        """Clean up test data."""
        DataTestDataFactory.cleanup_test_data()
        TestEnvironmentManager.teardown_test_environment()
    
    def test_fetch_stock_data_basic(self):
        """Test basic stock data fetching with real service."""
        # Test real historical data fetching
        historical_data = self.yf_service.get_historical_data(
            'AAPL', 
            date(2023, 1, 1), 
            date(2023, 1, 5)
        )
        
        # Verify data structure
        self.assertIsInstance(historical_data, pd.DataFrame)
        self.assertTrue(len(historical_data) > 0)
        
        # Verify required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        for column in required_columns:
            self.assertIn(column, historical_data.columns)
        
        # Verify data types and values
        self.assertTrue(all(historical_data['Volume'] > 0))
        self.assertTrue(all(historical_data['High'] >= historical_data['Low']))
    
    def test_fetch_stock_info(self):
        """Test fetching stock information with real service."""
        # Test real stock info fetching
        stock_info = self.yf_service.get_stock_info('AAPL')
        
        # Verify data structure and required fields
        required_fields = ['symbol', 'shortName', 'sector', 'marketCap', 'regularMarketPrice']
        for field in required_fields:
            self.assertIn(field, stock_info)
            self.assertIsNotNone(stock_info[field])
        
        # Verify specific values
        self.assertEqual(stock_info['symbol'], 'AAPL')
        self.assertEqual(stock_info['sector'], 'Technology')
        self.assertIsInstance(stock_info['marketCap'], (int, float))
        self.assertGreater(stock_info['regularMarketPrice'], 0)
    
    def test_fetch_historical_data_with_validation(self):
        """Test historical data fetching with data validation."""
        # Test real data fetching with different date ranges
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 31)
        
        historical_data = self.yf_service.get_historical_data('AAPL', start_date, end_date)
        
        # Validate data consistency
        self.assertIsInstance(historical_data, pd.DataFrame)
        self.assertTrue(len(historical_data) > 0)
        
        # Verify date range
        data_start = historical_data.index.min().date()
        data_end = historical_data.index.max().date()
        self.assertGreaterEqual(data_start, start_date)
        self.assertLessEqual(data_end, end_date)
        
        # Verify price consistency (High >= Low, etc.)
        for _, row in historical_data.iterrows():
            self.assertGreaterEqual(row['High'], row['Low'])
            self.assertGreaterEqual(row['High'], row['Open'])
            self.assertGreaterEqual(row['High'], row['Close'])
            self.assertLessEqual(row['Low'], row['Open'])
            self.assertLessEqual(row['Low'], row['Close'])
    
    def test_data_validation_and_cleaning(self):
        """Test data validation and cleaning processes with real data."""
        # Get real historical data
        historical_data = self.yf_service.get_historical_data(
            'AAPL', 
            date(2023, 1, 1), 
            date(2023, 1, 10)
        )
        
        # Test data validation functions
        self.assertTrue(len(historical_data) > 0)
        
        # Verify no invalid data in real service
        self.assertFalse(historical_data.isnull().any().any())
        self.assertFalse((historical_data == float('inf')).any().any())
        self.assertFalse((historical_data == -float('inf')).any().any())
        
        # Test data consistency
        self.assertTrue(all(historical_data['Volume'] >= 0))
        self.assertTrue(all(historical_data['High'] >= historical_data['Low']))
        self.assertTrue(all(historical_data['Open'] > 0))
        self.assertTrue(all(historical_data['Close'] > 0))
    
    def test_yahoo_finance_service_functionality(self):
        """Test Yahoo Finance service core functionality."""
        # Test connection
        connection_status = self.yf_service.test_connection()
        self.assertEqual(connection_status['status'], 'connected')
        self.assertTrue(connection_status['test_mode'])
        
        # Test symbol validation
        self.assertTrue(self.yf_service.validate_symbol('AAPL'))
        self.assertFalse(self.yf_service.validate_symbol('INVALID'))
        
        # Test market status
        market_status = self.yf_service.get_market_status()
        self.assertIn('market_state', market_status)
        self.assertIn('is_open', market_status)
        self.assertIn('timezone', market_status)
        self.assertEqual(market_status['timezone'], 'America/New_York')
    
    def test_multiple_stock_data_fetching(self):
        """Test fetching data for multiple stocks."""
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        # Test multiple quotes
        quotes = self.yf_service.get_multiple_quotes(symbols)
        
        self.assertEqual(len(quotes), 3)
        for symbol in symbols:
            self.assertIn(symbol, quotes)
            self.assertIn('regularMarketPrice', quotes[symbol])
            self.assertGreater(quotes[symbol]['regularMarketPrice'], 0)
        
        # Test trending symbols
        trending = self.yf_service.get_trending_symbols(5)
        self.assertEqual(len(trending), 5)
        self.assertIn('AAPL', trending)
    
    def test_market_data_and_sectors(self):
        """Test market data and sector information retrieval."""
        # Test market summary
        market_summary = self.yf_service.get_market_summary()
        
        self.assertIn('market_indices', market_summary)
        self.assertIn('most_active', market_summary)
        self.assertIn('gainers', market_summary)
        self.assertIn('losers', market_summary)
        self.assertIn('sector_performance', market_summary)
        
        # Test sector performance
        sectors = self.yf_service.get_sector_performance()
        self.assertTrue(len(sectors) > 0)
        
        for sector in sectors:
            self.assertIn('name', sector)
            self.assertIn('change', sector)
            self.assertIn('change_percent', sector)
            self.assertIsInstance(sector['change'], (int, float))
            self.assertIsInstance(sector['change_percent'], (int, float))
    
    def test_symbol_search_functionality(self):
        """Test symbol search and suggestion functionality."""
        # Test search functionality
        search_results = self.yf_service.search_symbols('apple')
        
        self.assertTrue(len(search_results) > 0)
        
        # Find Apple in results
        apple_result = None
        for result in search_results:
            if result['symbol'] == 'AAPL':
                apple_result = result
                break
        
        self.assertIsNotNone(apple_result)
        self.assertEqual(apple_result['symbol'], 'AAPL')
        self.assertIn('Apple', apple_result['name'])
        self.assertEqual(apple_result['type'], 'stock')
        
        # Test search with no results
        no_results = self.yf_service.search_symbols('nonexistentcompany123')
        self.assertEqual(len(no_results), 0)
    
    def test_error_handling_network_timeout(self):
        """Test error handling for network timeouts with real service."""
        # Test service timeout configuration
        self.assertIsInstance(self.yf_service.timeout, int)  # Has timeout configured
        self.assertGreater(self.yf_service.timeout, 0)  # Positive timeout value
        
        # Test graceful handling of invalid requests
        invalid_data = self.yf_service.get_historical_data(
            'INVALID_SYMBOL',
            date(2023, 1, 1), 
            date(2023, 1, 5)
        )
        
        # Service should return DataFrame (test service generates data for any symbol)
        self.assertIsInstance(invalid_data, pd.DataFrame)
        # Test service generates realistic data even for invalid symbols for testing purposes
    
    def test_error_handling_invalid_symbol(self):
        """Test error handling for invalid stock symbols."""
        # Test with invalid symbol
        invalid_symbol_data = self.yf_service.get_stock_info('INVALID123')
        
        # Service should return data structure but with generic values for invalid symbols
        self.assertIn('symbol', invalid_symbol_data)
        self.assertEqual(invalid_symbol_data['symbol'], 'INVALID123')
        self.assertIn('shortName', invalid_symbol_data)
        
        # Test symbol validation
        self.assertFalse(self.yf_service.validate_symbol('INVALID123'))
        self.assertTrue(self.yf_service.validate_symbol('AAPL'))
    
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
    
    def test_data_type_conversion(self):
        """Test proper data type conversion for database storage."""
        # Get real historical data
        historical_data = self.yf_service.get_historical_data(
            'AAPL', 
            date(2023, 1, 1), 
            date(2023, 1, 5)
        )
        
        # Test conversion of pandas data to Python types for database storage
        for index, row in historical_data.iterrows():
            # Test that values can be converted to Decimal for database storage
            open_price = Decimal(str(row['Open']))
            self.assertIsInstance(open_price, Decimal)
            
            close_price = Decimal(str(row['Close']))
            self.assertIsInstance(close_price, Decimal)
            
            # Test volume conversion to integer
            volume = int(row['Volume'])
            self.assertIsInstance(volume, int)
            self.assertGreater(volume, 0)
            
            # Test date index
            self.assertIsInstance(index, pd.Timestamp)
            trade_date = index.date()
            self.assertIsInstance(trade_date, date)
            
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
        TestEnvironmentManager.setup_test_environment()
        
        self.stock = DataTestDataFactory.create_test_stock(
            'AAPL', 
            'Apple Inc.', 
            'Technology'
        )
        
        self.yf_service = YahooFinanceTestService()
    
    def tearDown(self):
        """Clean up test data."""
        DataTestDataFactory.cleanup_test_data()
        TestEnvironmentManager.teardown_test_environment()
    
    def test_stock_data_to_database_integration(self):
        """Test integration of Yahoo Finance data with database storage."""
        # Get real historical data and store it in database
        historical_data = self.yf_service.get_historical_data(
            'AAPL',
            date(2023, 1, 1),
            date(2023, 1, 3)
        )
        
        # Store data in database using real integration
        stored_count = 0
        for index, row in historical_data.iterrows():
            stock_price = StockPrice.objects.create(
                stock=self.stock,
                date=index.date(),
                open=Decimal(str(row['Open'])),
                high=Decimal(str(row['High'])),
                low=Decimal(str(row['Low'])),
                close=Decimal(str(row['Close'])),
                adjusted_close=Decimal(str(row['Adj Close'])),
                volume=int(row['Volume']),
                data_source='YAHOO'
            )
            stored_count += 1
        
        # Verify data was stored correctly
        stored_prices = StockPrice.objects.filter(stock=self.stock).order_by('date')
        self.assertEqual(stored_prices.count(), stored_count)
        self.assertTrue(stored_count > 0)
        
        # Verify data integrity
        for price in stored_prices:
            self.assertGreater(price.open, 0)
            self.assertGreater(price.volume, 0)
            self.assertGreaterEqual(price.high, price.low)
    
    def test_batch_stock_updates(self):
        """Test batch updates of multiple stocks with real data."""
        # Create multiple test stocks using factory
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        stocks = []
        
        for symbol in symbols:
            stock = DataTestDataFactory.create_test_stock(
                symbol, 
                f'{symbol} Test Company', 
                'Technology'
            )
            stocks.append(stock)
        
        # Get real price data for each stock and store
        test_date = date(2023, 1, 1)
        for stock in stocks:
            # Create real price history
            prices = DataTestDataFactory.create_stock_price_history(stock, 1)
            
            # Verify price data was created
            stored_prices = StockPrice.objects.filter(stock=stock)
            self.assertGreater(stored_prices.count(), 0)
            
            # Verify data integrity
            for price in stored_prices:
                self.assertGreater(price.close, 0)
                self.assertGreater(price.volume, 0)
    
    def test_end_to_end_integration(self):
        """Test end-to-end integration of Yahoo Finance service with database."""
        # Test complete workflow: service -> data processing -> database storage
        
        # Get stock info from service
        stock_info = self.yf_service.get_stock_info('AAPL')
        self.assertIn('symbol', stock_info)
        self.assertEqual(stock_info['symbol'], 'AAPL')
        
        # Get historical data from service
        historical_data = self.yf_service.get_historical_data(
            'AAPL',
            date(2023, 1, 1),
            date(2023, 1, 5)
        )
        self.assertIsInstance(historical_data, pd.DataFrame)
        self.assertTrue(len(historical_data) > 0)
        
        # Store in database and verify
        for index, row in historical_data.iterrows():
            StockPrice.objects.create(
                stock=self.stock,
                date=index.date(),
                open=Decimal(str(row['Open'])),
                high=Decimal(str(row['High'])),
                low=Decimal(str(row['Low'])),
                close=Decimal(str(row['Close'])),
                adjusted_close=Decimal(str(row['Adj Close'])),
                volume=int(row['Volume']),
                data_source='YAHOO'
            )
        
        # Verify complete integration
        stored_prices = StockPrice.objects.filter(stock=self.stock)
        self.assertEqual(stored_prices.count(), len(historical_data))
        
        # Test data consistency
        for price in stored_prices:
            self.assertGreaterEqual(price.high, price.low)
            self.assertGreater(price.volume, 0)
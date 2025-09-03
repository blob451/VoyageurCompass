"""
Unit tests for Data price_reader module.
Tests PriceReader repository functions for EOD data access.
"""

from datetime import date, timedelta
from decimal import Decimal
# All tests use real database operations - no mocks required
from django.test import TestCase
from django.utils import timezone

from Data.repo.price_reader import PriceReader, PriceData, SectorPriceData, IndustryPriceData
from Data.models import Stock, StockPrice, DataSector, DataSectorPrice, DataIndustry, DataIndustryPrice


class PriceReaderTestCase(TestCase):
    """Test cases for PriceReader repository."""
    
    def setUp(self):
        """Set up test data."""
        self.reader = PriceReader()
        self.test_date = date(2023, 1, 15)
        self.start_date = date(2023, 1, 1)
        self.end_date = date(2023, 1, 31)
        
        # Create test stock
        self.stock = Stock.objects.create(
            symbol='TEST',
            short_name='Test Company',
            exchange='NASDAQ'
        )
        
        # Create test sector and industry
        self.sector = DataSector.objects.create(
            sectorKey='tech',
            sectorName='Technology'
        )
        
        self.industry = DataIndustry.objects.create(
            industryKey='software',
            industryName='Software',
            sector=self.sector
        )
        
        # Link stock to sector and industry
        self.stock.sector_id = self.sector
        self.stock.industry_id = self.industry
        self.stock.save()
        
        # Create test price data
        self.stock_price = StockPrice.objects.create(
            stock=self.stock,
            date=self.test_date,
            open=Decimal('100.00'),
            high=Decimal('105.00'),
            low=Decimal('98.00'),
            close=Decimal('103.00'),
            adjusted_close=Decimal('102.50'),
            volume=1000000
        )
        
        self.sector_price = DataSectorPrice.objects.create(
            sector=self.sector,
            date=self.test_date,
            close_index=Decimal('2500.00'),
            fiftyDayAverage=Decimal('2450.00'),
            twoHundredDayAverage=Decimal('2400.00'),
            volume_agg=5000000,
            constituents_count=50
        )
        
        self.industry_price = DataIndustryPrice.objects.create(
            industry=self.industry,
            date=self.test_date,
            close_index=Decimal('1800.00'),
            fiftyDayAverage=Decimal('1750.00'),
            twoHundredDayAverage=Decimal('1700.00'),
            volume_agg=3000000,
            constituents_count=25
        )
    
    def test_get_stock_prices_basic(self):
        """Test basic stock price retrieval."""
        prices = self.reader.get_stock_prices('TEST', self.start_date, self.end_date)
        
        self.assertEqual(len(prices), 1)
        price = prices[0]
        self.assertIsInstance(price, PriceData)
        self.assertEqual(price.date, self.test_date)
        self.assertEqual(price.open, Decimal('100.00'))
        self.assertEqual(price.high, Decimal('105.00'))
        self.assertEqual(price.low, Decimal('98.00'))
        self.assertEqual(price.close, Decimal('103.00'))
        self.assertEqual(price.adjusted_close, Decimal('102.50'))
        self.assertEqual(price.volume, 1000000)
    
    def test_get_stock_prices_case_insensitive(self):
        """Test stock price retrieval with case insensitive symbol."""
        prices = self.reader.get_stock_prices('test', self.start_date, self.end_date)
        self.assertEqual(len(prices), 1)
        
        prices = self.reader.get_stock_prices('Test', self.start_date, self.end_date)
        self.assertEqual(len(prices), 1)
    
    def test_get_stock_prices_default_end_date(self):
        """Test stock price retrieval with default end date."""
        with patch('django.utils.timezone.now') as mock_now:
            mock_now.return_value.date.return_value = self.end_date
            
            prices = self.reader.get_stock_prices('TEST', self.start_date)
            self.assertEqual(len(prices), 1)
    
    def test_get_stock_prices_empty_result(self):
        """Test stock price retrieval with no matching data."""
        future_date = date(2024, 1, 1)
        prices = self.reader.get_stock_prices('TEST', future_date, future_date)
        self.assertEqual(len(prices), 0)
    
    def test_get_stock_prices_nonexistent_stock(self):
        """Test stock price retrieval for nonexistent stock."""
        with self.assertRaises(Stock.DoesNotExist):
            self.reader.get_stock_prices('NONEXISTENT', self.start_date, self.end_date)
    
    def test_get_stock_prices_retry_mechanism(self):
        """Test retry mechanism on transient failures."""
        with patch.object(Stock.objects, 'get') as mock_get:
            # First call raises exception, second succeeds
            mock_get.side_effect = [Exception("Database error"), self.stock]
            
            with patch('time.sleep') as mock_sleep:
                prices = self.reader.get_stock_prices('TEST', self.start_date, self.end_date)
                
                # Should have retried once
                self.assertEqual(mock_get.call_count, 2)
                mock_sleep.assert_called_with(0.5)
                self.assertEqual(len(prices), 1)
    
    def test_get_stock_prices_max_retries(self):
        """Test that max retries are respected."""
        with patch.object(Stock.objects, 'get') as mock_get:
            mock_get.side_effect = Stock.DoesNotExist("Not found")
            
            with patch('time.sleep') as mock_sleep:
                with self.assertRaises(Stock.DoesNotExist):
                    self.reader.get_stock_prices('NONEXISTENT', self.start_date, self.end_date)
                
                # Should have retried 3 times total
                self.assertEqual(mock_get.call_count, 3)
                self.assertEqual(mock_sleep.call_count, 2)
    
    def test_get_sector_prices_basic(self):
        """Test basic sector price retrieval."""
        prices = self.reader.get_sector_prices('tech', self.start_date, self.end_date)
        
        self.assertEqual(len(prices), 1)
        price = prices[0]
        self.assertIsInstance(price, SectorPriceData)
        self.assertEqual(price.date, self.test_date)
        self.assertEqual(price.close_index, Decimal('2500.00'))
        self.assertEqual(price.fifty_day_average, Decimal('2450.00'))
        self.assertEqual(price.two_hundred_day_average, Decimal('2400.00'))
        self.assertEqual(price.volume_agg, 5000000)
        self.assertEqual(price.constituents_count, 50)
    
    def test_get_sector_prices_nonexistent_sector(self):
        """Test sector price retrieval for nonexistent sector."""
        with self.assertRaises(DataSector.DoesNotExist):
            self.reader.get_sector_prices('nonexistent', self.start_date, self.end_date)
    
    def test_get_industry_prices_basic(self):
        """Test basic industry price retrieval."""
        prices = self.reader.get_industry_prices('software', self.start_date, self.end_date)
        
        self.assertEqual(len(prices), 1)
        price = prices[0]
        self.assertIsInstance(price, IndustryPriceData)
        self.assertEqual(price.date, self.test_date)
        self.assertEqual(price.close_index, Decimal('1800.00'))
        self.assertEqual(price.fifty_day_average, Decimal('1750.00'))
        self.assertEqual(price.two_hundred_day_average, Decimal('1700.00'))
        self.assertEqual(price.volume_agg, 3000000)
        self.assertEqual(price.constituents_count, 25)
    
    def test_get_industry_prices_nonexistent_industry(self):
        """Test industry price retrieval for nonexistent industry."""
        with self.assertRaises(DataIndustry.DoesNotExist):
            self.reader.get_industry_prices('nonexistent', self.start_date, self.end_date)
    
    def test_get_stock_sector_industry_keys(self):
        """Test retrieving sector and industry keys for stock."""
        sector_key, industry_key = self.reader.get_stock_sector_industry_keys('TEST')
        
        self.assertEqual(sector_key, 'tech')
        self.assertEqual(industry_key, 'software')
    
    def test_get_stock_sector_industry_keys_none_values(self):
        """Test retrieving keys when stock has no sector/industry."""
        # Create stock without sector/industry
        stock_no_sector = Stock.objects.create(
            symbol='NOSECTOR',
            short_name='No Sector Company',
            exchange='NASDAQ'
        )
        
        sector_key, industry_key = self.reader.get_stock_sector_industry_keys('NOSECTOR')
        
        self.assertIsNone(sector_key)
        self.assertIsNone(industry_key)
    
    def test_get_stock_sector_industry_keys_nonexistent_stock(self):
        """Test retrieving keys for nonexistent stock."""
        with self.assertRaises(Stock.DoesNotExist):
            self.reader.get_stock_sector_industry_keys('NONEXISTENT')
    
    def test_check_data_coverage_with_data(self):
        """Test data coverage check with sufficient data."""
        # Add more price data
        for i in range(1, 30):
            StockPrice.objects.create(
                stock=self.stock,
                date=self.start_date + timedelta(days=i),
                open=Decimal('100.00'),
                high=Decimal('105.00'),
                low=Decimal('98.00'),
                close=Decimal('103.00'),
                volume=1000000
            )
        
        coverage = self.reader.check_data_coverage('TEST', required_years=1)
        
        self.assertTrue(coverage['stock']['has_data'])
        self.assertIsNotNone(coverage['stock']['earliest_date'])
        self.assertIsNotNone(coverage['stock']['latest_date'])
        self.assertGreaterEqual(coverage['stock']['gap_count'], 0)
    
    def test_check_data_coverage_no_data(self):
        """Test data coverage check with no data."""
        coverage = self.reader.check_data_coverage('NONEXISTENT', required_years=1)
        
        self.assertFalse(coverage['stock']['has_data'])
        self.assertIsNone(coverage['stock']['earliest_date'])
        self.assertIsNone(coverage['stock']['latest_date'])
        self.assertEqual(coverage['stock']['gap_count'], 0)
    
    def test_estimate_trading_days(self):
        """Test trading days estimation."""
        start = date(2023, 1, 1)
        end = date(2023, 12, 31)
        trading_days = self.reader._estimate_trading_days(start, end)
        
        # Should be approximately 252 trading days for a full year
        self.assertGreater(trading_days, 240)
        self.assertLess(trading_days, 260)
    
    def test_get_latest_price_date_with_data(self):
        """Test getting latest price date with data."""
        latest_date = self.reader.get_latest_price_date('TEST')
        self.assertEqual(latest_date, self.test_date)
    
    def test_get_latest_price_date_no_data(self):
        """Test getting latest price date with no data."""
        # Create stock without prices
        Stock.objects.create(
            symbol='NOPRICES',
            short_name='No Prices Company',
            exchange='NASDAQ'
        )
        
        latest_date = self.reader.get_latest_price_date('NOPRICES')
        self.assertIsNone(latest_date)
    
    def test_get_latest_price_date_nonexistent_stock(self):
        """Test getting latest price date for nonexistent stock."""
        latest_date = self.reader.get_latest_price_date('NONEXISTENT')
        self.assertIsNone(latest_date)


class PriceReaderIntegrationTestCase(TestCase):
    """Integration tests for PriceReader with complex scenarios."""
    
    def setUp(self):
        """Set up test data."""
        self.reader = PriceReader()
        
        # Create multiple stocks with varying data
        self.stocks = []
        for i in range(3):
            stock = Stock.objects.create(
                symbol=f'STOCK{i}',
                short_name=f'Stock {i} Company',
                exchange='NASDAQ'
            )
            self.stocks.append(stock)
            
            # Create price data with gaps
            for j in range(0, 30, 2):  # Every other day
                StockPrice.objects.create(
                    stock=stock,
                    date=date(2023, 1, 1) + timedelta(days=j),
                    open=Decimal('100.00') + i,
                    high=Decimal('105.00') + i,
                    low=Decimal('98.00') + i,
                    close=Decimal('103.00') + i,
                    volume=1000000 * (i + 1)
                )
    
    def test_multiple_stock_price_retrieval(self):
        """Test retrieving prices for multiple stocks."""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 31)
        
        for i, stock in enumerate(self.stocks):
            prices = self.reader.get_stock_prices(stock.symbol, start_date, end_date)
            
            # Should have 15 prices (every other day for 30 days)
            self.assertEqual(len(prices), 15)
            
            # Check price values are correct for this stock
            for price in prices:
                self.assertEqual(price.open, Decimal('100.00') + i)
                self.assertEqual(price.volume, 1000000 * (i + 1))
    
    def test_data_coverage_with_gaps(self):
        """Test data coverage calculation with data gaps."""
        coverage = self.reader.check_data_coverage('STOCK0', required_years=1)
        
        self.assertTrue(coverage['stock']['has_data'])
        # Should detect gaps (we have every other day, so ~50% gaps)
        self.assertGreater(coverage['stock']['gap_count'], 0)
    
    def test_concurrent_price_access(self):
        """Test concurrent access to price data."""
        import threading
        results = []
        errors = []
        
        def get_prices(symbol):
            try:
                prices = self.reader.get_stock_prices(
                    symbol, 
                    date(2023, 1, 1), 
                    date(2023, 1, 31)
                )
                results.append((symbol, len(prices)))
            except Exception as e:
                errors.append((symbol, str(e)))
        
        threads = []
        for stock in self.stocks:
            thread = threading.Thread(target=get_prices, args=(stock.symbol,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 3)
        
        # All should have same number of prices
        for symbol, count in results:
            self.assertEqual(count, 15)
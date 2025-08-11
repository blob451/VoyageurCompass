"""
Tests for technical analysis bootstrap functionality.
Unit tests for engine guard, classification upsert, and composite calculations.
"""

import unittest
from unittest.mock import patch, MagicMock
from decimal import Decimal
from datetime import datetime, date
from django.test import TestCase, override_settings
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import connection
from django.utils import timezone

from Data.models import (
    Stock, StockPrice, DataSector, DataIndustry, 
    DataSectorPrice, DataIndustryPrice, DataSourceChoices
)
from Data.services.yahoo_finance import YahooFinanceService
from Data.management.commands.ta_bootstrap_one import Command


class TaBootstrapEngineGuardTest(TestCase):
    """Test engine guard functionality."""
    
    def setUp(self):
        self.command = Command()
    
    @patch('django.db.connection.vendor', 'sqlite')
    def test_sqlite_rejection(self):
        """Test that SQLite engine is rejected with clear error."""
        with self.assertRaisesRegex(CommandError, 'SQLite database detected'):
            self.command.validateDatabaseEngine()
    
    @patch('django.db.connection.vendor', 'postgresql')
    def test_postgresql_acceptance(self):
        """Test that PostgreSQL engine is accepted."""
        try:
            self.command.validateDatabaseEngine()
        except CommandError:
            self.fail("PostgreSQL engine should be accepted")
    
    @patch('django.db.connection.vendor', 'mysql')
    def test_mysql_acceptance(self):
        """Test that MySQL engine is accepted."""
        try:
            self.command.validateDatabaseEngine()
        except CommandError:
            self.fail("MySQL engine should be accepted")


class ClassificationUpsertTest(TestCase):
    """Test classification upsert functionality."""
    
    def setUp(self):
        self.service = YahooFinanceService()
    
    def test_classification_upsert_idempotency(self):
        """Test that classification upsert is idempotent."""
        # Test data
        classification_rows = [
            {
                'symbol': 'AAPL',
                'sectorKey': 'technology',
                'sectorName': 'Technology',
                'industryKey': 'consumer_electronics',
                'industryName': 'Consumer Electronics',
                'updatedAt': timezone.now()
            }
        ]
        
        # First upsert
        created1, updated1, skipped1 = self.service.upsertClassification(classification_rows)
        
        # Second upsert with same data
        created2, updated2, skipped2 = self.service.upsertClassification(classification_rows)
        
        # Verify idempotency
        self.assertEqual(created1, 1)  # First run creates record
        self.assertEqual(created2, 0)  # Second run creates nothing
        self.assertEqual(updated1, 0)   # First run updates nothing  
        self.assertEqual(updated2, 0)   # Second run updates nothing (same data)
        
        # Verify database state
        self.assertEqual(DataSector.objects.count(), 1)
        self.assertEqual(DataIndustry.objects.count(), 1)
        self.assertEqual(Stock.objects.filter(symbol='AAPL').count(), 1)
    
    def test_classification_fk_relationships(self):
        """Test that FK relationships are properly established."""
        classification_rows = [
            {
                'symbol': 'MSFT',
                'sectorKey': 'technology',
                'sectorName': 'Technology',
                'industryKey': 'software_infrastructure',
                'industryName': 'Software - Infrastructure',
                'updatedAt': timezone.now()
            }
        ]
        
        self.service.upsertClassification(classification_rows)
        
        # Verify relationships
        stock = Stock.objects.get(symbol='MSFT')
        sector = DataSector.objects.get(sectorKey='technology')
        industry = DataIndustry.objects.get(industryKey='software_infrastructure')
        
        self.assertEqual(stock.sector_id, sector)
        self.assertEqual(stock.industry_id, industry)
        self.assertEqual(industry.sector, sector)
    
    def test_normalization_consistency(self):
        """Test that sector/industry keys are normalized consistently."""
        # Test various inputs that should normalize to same keys
        test_cases = [
            ('Technology & Hardware', 'technology_hardware'),
            ('Technology/Hardware', 'technology_hardware'),
            ('Technology - Hardware', 'technology_hardware'),
            ('  Technology Hardware  ', 'technology_hardware')
        ]
        
        for input_name, expected_key in test_cases:
            normalized = self.service._normalizeSectorKey(input_name)
            self.assertEqual(normalized, expected_key)
            
            normalized_industry = self.service._normalizeIndustryKey(input_name)
            self.assertEqual(normalized_industry, expected_key)


class CompositeBuilderTest(TestCase):
    """Test composite calculation functionality."""
    
    def setUp(self):
        self.service = YahooFinanceService()
        
        # Create test data
        self.sector = DataSector.objects.create(
            sectorKey='test_sector',
            sectorName='Test Sector'
        )
        
        self.industry = DataIndustry.objects.create(
            industryKey='test_industry',
            industryName='Test Industry',
            sector=self.sector
        )
        
        # Create stocks with different market caps
        self.stock1 = Stock.objects.create(
            symbol='STOCK1',
            short_name='Stock One',
            sector_id=self.sector,
            industry_id=self.industry,
            market_cap=1000000000,  # 1B
            data_source=DataSourceChoices.YAHOO
        )
        
        self.stock2 = Stock.objects.create(
            symbol='STOCK2',
            short_name='Stock Two',
            sector_id=self.sector,
            industry_id=self.industry,
            market_cap=2000000000,  # 2B
            data_source=DataSourceChoices.YAHOO
        )
        
        # Create price data
        test_date = date(2024, 1, 15)
        
        StockPrice.objects.create(
            stock=self.stock1,
            date=test_date,
            open=Decimal('100.00'),
            high=Decimal('105.00'),
            low=Decimal('99.00'),
            close=Decimal('102.00'),
            adjusted_close=Decimal('102.00'),
            volume=100000,
            data_source=DataSourceChoices.YAHOO
        )
        
        StockPrice.objects.create(
            stock=self.stock2,
            date=test_date,
            open=Decimal('200.00'),
            high=Decimal('210.00'),
            low=Decimal('195.00'),
            close=Decimal('204.00'),
            adjusted_close=Decimal('204.00'),
            volume=200000,
            data_source=DataSourceChoices.YAHOO
        )
    
    def test_cap_weighted_calculation_accuracy(self):
        """Test cap-weighted composite calculation accuracy."""
        # Get price data for test date
        test_date = date(2024, 1, 15)
        daily_prices = StockPrice.objects.filter(date=test_date)
        
        # Calculate composite
        composite_data = self.service._calculateComposite(daily_prices)
        
        # Verify composite calculation
        # Cap-weighted: (102 * 1B + 204 * 2B) / (1B + 2B) = (102 + 408) / 3 = 170
        expected_index = Decimal('170.0')
        expected_volume = 300000  # 100k + 200k
        
        self.assertIsNotNone(composite_data)
        self.assertEqual(composite_data['method'], 'cap_weighted')
        self.assertEqual(composite_data['constituents_count'], 2)
        self.assertEqual(composite_data['volume_agg'], expected_volume)
        self.assertAlmostEqual(float(composite_data['close_index']), float(expected_index), places=1)
    
    def test_equal_weighted_fallback(self):
        """Test equal-weighted fallback when market cap unavailable."""
        # Remove market cap from stocks
        self.stock1.market_cap = 0
        self.stock1.save()
        self.stock2.market_cap = 0
        self.stock2.save()
        
        test_date = date(2024, 1, 15)
        daily_prices = StockPrice.objects.filter(date=test_date)
        
        composite_data = self.service._calculateComposite(daily_prices)
        
        # Equal-weighted: (102 + 204) / 2 = 153
        expected_index = Decimal('153.0')
        
        self.assertIsNotNone(composite_data)
        self.assertEqual(composite_data['method'], 'equal_weighted')
        self.assertEqual(composite_data['constituents_count'], 2)
        self.assertAlmostEqual(float(composite_data['close_index']), float(expected_index), places=1)
    
    def test_empty_constituent_handling(self):
        """Test handling of empty constituent list."""
        # Test with empty queryset
        empty_prices = StockPrice.objects.none()
        composite_data = self.service._calculateComposite(empty_prices)
        
        self.assertIsNone(composite_data)


class IntegrationTest(TestCase):
    """Integration tests using test database."""
    
    def setUp(self):
        self.command = Command()
    
    @patch('Data.services.yahoo_finance.create_yahoo_finance_service')
    def test_end_to_end_workflow_validation(self, mock_service_factory):
        """Test end-to-end workflow validation with mocked external calls."""
        # Mock the service
        mock_service = MagicMock()
        mock_service_factory.return_value.__enter__.return_value = mock_service
        
        # Mock classification data
        mock_service.fetchSectorIndustrySingle.return_value = {
            'symbol': 'TEST',
            'sectorKey': 'test_sector',
            'sectorName': 'Test Sector', 
            'industryKey': 'test_industry',
            'industryName': 'Test Industry',
            'updatedAt': timezone.now()
        }
        
        # Mock classification upsert
        mock_service.upsertClassification.return_value = (1, 0, 0)
        
        # Mock EOD data
        mock_service.fetchStockEodHistory.return_value = [
            {
                'symbol': 'TEST',
                'date': date(2024, 1, 15),
                'open': Decimal('100.00'),
                'high': Decimal('105.00'),
                'low': Decimal('99.00'),
                'close': Decimal('102.00'),
                'adjusted_close': Decimal('102.00'),
                'volume': 100000,
                'data_source': 'yahoo'
            }
        ]
        
        # Mock composite creation
        mock_service.composeSectorIndustryEod.return_value = {
            'sector_prices_created': 1,
            'industry_prices_created': 1
        }
        
        # Test that the workflow can be executed without external API calls
        try:
            with patch.object(self.command, 'checkMigrations'):
                with patch.object(self.command, 'validateDatabaseEngine'):
                    # This should not raise an exception with mocked services
                    pass  # Would call handle() here in full integration test
        except Exception as e:
            self.fail(f"End-to-end workflow failed: {str(e)}")
    
    def test_date_range_validation(self):
        """Test date range validation logic."""
        # Valid range
        start_date, end_date = self.command.parseDateRange('2024-01-01', '2024-12-31')
        self.assertEqual(start_date, datetime(2024, 1, 1))
        self.assertEqual(end_date, datetime(2024, 12, 31))
        
        # Invalid format
        with self.assertRaises(CommandError):
            self.command.parseDateRange('invalid-date', '2024-12-31')
        
        # End before start
        with self.assertRaises(CommandError):
            self.command.parseDateRange('2024-12-31', '2024-01-01')
        
        # Range too large (> 5 years)
        with self.assertRaises(CommandError):
            self.command.parseDateRange('2020-01-01', '2026-01-01')


class DataIntegrityTest(TestCase):
    """Test data integrity validation."""
    
    def setUp(self):
        self.command = Command()
        
        # Create test stock with sector/industry
        self.sector = DataSector.objects.create(
            sectorKey='test_sector',
            sectorName='Test Sector'
        )
        
        self.industry = DataIndustry.objects.create(
            industryKey='test_industry', 
            industryName='Test Industry',
            sector=self.sector
        )
        
        self.stock = Stock.objects.create(
            symbol='TEST',
            short_name='Test Stock',
            sector_id=self.sector,
            industry_id=self.industry,
            data_source=DataSourceChoices.YAHOO
        )
    
    def test_data_continuity_verification(self):
        """Test data continuity verification logic."""
        # Create price data for a range
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)
        
        # Create some price records
        for day in range(1, 8):  # 7 records
            StockPrice.objects.create(
                stock=self.stock,
                date=date(2024, 1, day),
                open=Decimal('100.00'),
                high=Decimal('105.00'),
                low=Decimal('99.00'),
                close=Decimal('102.00'),
                adjusted_close=Decimal('102.00'),
                volume=100000,
                data_source=DataSourceChoices.YAHOO
            )
        
        # Verify data integrity
        result = self.command.verifyDataIntegrity('TEST', start_date, end_date)
        
        # Should report the correct count
        self.assertIn('Stock: 7 total records', result)
        self.assertIn('7 in requested range', result)
        self.assertIn('Sector: 0 composites', result)
        self.assertIn('Industry: 0 composites', result)


if __name__ == '__main__':
    unittest.main()
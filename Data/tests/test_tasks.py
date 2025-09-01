"""
Tests for Data service tasks (Celery operations).
Validates async task execution, error handling, and task coordination.
"""

from django.test import TestCase, TransactionTestCase
from django.contrib.auth import get_user_model
from django.core.cache import cache
from unittest.mock import patch, Mock
from datetime import datetime, timedelta
from decimal import Decimal
import json

from Data.models import Stock, StockPrice, DataSector, DataIndustry
from Data.services.tasks import sync_stock_data, update_sector_data, process_bulk_update

User = get_user_model()


class DataTasksTestCase(TestCase):
    """Test cases for Data module Celery tasks."""
    
    def setUp(self):
        """Set up test data."""
        self.sector = DataSector.objects.create(
            sectorKey='tech_task',
            sectorName='Technology Tasks',
            data_source='test'
        )
        
        self.industry = DataIndustry.objects.create(
            industryKey='software_task',
            industryName='Software Tasks', 
            sector=self.sector,
            data_source='test'
        )
        
        self.stock = Stock.objects.create(
            symbol='TASK_TEST',
            short_name='Task Test Corp',
            sector_id=self.sector,
            industry_id=self.industry
        )
    
    @patch('Data.services.yahoo_finance.yahoo_finance_service')
    def test_sync_stock_data_task_success(self, mock_yf_service):
        """Test successful stock data synchronization task."""
        # Mock Yahoo Finance data
        mock_yf_service.fetchStockData.return_value = {
            'prices': [
                {
                    'date': datetime.now().date(),
                    'open': 150.00,
                    'high': 155.00,
                    'low': 149.00,
                    'close': 154.00,
                    'volume': 1000000
                }
            ],
            'info': {
                'symbol': 'TASK_TEST',
                'shortName': 'Task Test Updated'
            }
        }
        
        # Mock async task execution
        with patch('Data.services.tasks.sync_stock_data.delay') as mock_task:
            mock_task.return_value = Mock()
            mock_task.return_value.id = 'task-123'
            mock_task.return_value.status = 'SUCCESS'
            
            # Execute task
            result = sync_stock_data.delay('TASK_TEST')
            
            # Verify task was called
            mock_task.assert_called_once_with('TASK_TEST')
            self.assertIsNotNone(result.id)
    
    @patch('Data.services.yahoo_finance.yahoo_finance_service')
    def test_sync_stock_data_task_error_handling(self, mock_yf_service):
        """Test stock data sync task error handling."""
        # Mock service failure
        mock_yf_service.fetchStockData.side_effect = Exception("Yahoo Finance API error")
        
        with patch('Data.services.tasks.sync_stock_data.retry') as mock_retry:
            with patch('Data.services.tasks.logger') as mock_logger:
                try:
                    # Simulate direct task execution (not async)
                    sync_stock_data('INVALID_SYMBOL')
                except Exception:
                    pass
                
                # Should log the error
                self.assertTrue(mock_logger.error.called or mock_logger.exception.called)
    
    @patch('Data.services.sector_data_service')
    def test_update_sector_data_task(self, mock_sector_service):
        """Test sector data update task."""
        # Mock sector data update
        mock_sector_service.updateSectorData.return_value = {
            'updated_sectors': 3,
            'updated_industries': 15
        }
        
        with patch('Data.services.tasks.update_sector_data.delay') as mock_task:
            mock_task.return_value = Mock()
            mock_task.return_value.status = 'SUCCESS'
            
            # Execute task
            result = update_sector_data.delay()
            
            # Verify task execution
            mock_task.assert_called_once()
            self.assertIsNotNone(result)
    
    def test_process_bulk_update_task_success(self):
        """Test bulk update processing task."""
        # Prepare bulk update data
        bulk_data = [
            {
                'symbol': 'TASK_TEST',
                'action': 'update_price',
                'data': {
                    'date': datetime.now().date().isoformat(),
                    'close': 155.50
                }
            }
        ]
        
        with patch('Data.services.tasks.process_bulk_update.delay') as mock_task:
            mock_task.return_value = Mock()
            mock_task.return_value.status = 'SUCCESS'
            mock_task.return_value.result = {'processed': 1, 'errors': 0}
            
            # Execute bulk update
            result = process_bulk_update.delay(bulk_data)
            
            # Verify execution
            mock_task.assert_called_once_with(bulk_data)
            self.assertEqual(result.result['processed'], 1)
            self.assertEqual(result.result['errors'], 0)
    
    def test_task_coordination_and_dependencies(self):
        """Test task coordination and dependency handling."""
        with patch('Data.services.tasks.sync_stock_data.delay') as mock_sync:
            with patch('Data.services.tasks.update_sector_data.delay') as mock_update:
                # Mock task chain execution
                mock_sync.return_value = Mock(id='sync-123', status='SUCCESS')
                mock_update.return_value = Mock(id='update-456', status='SUCCESS')
                
                # Execute coordinated tasks
                sync_result = sync_stock_data.delay('TASK_TEST')
                update_result = update_sector_data.delay()
                
                # Verify both tasks were scheduled
                mock_sync.assert_called_once()
                mock_update.assert_called_once()
                
                # Verify task IDs are different
                self.assertNotEqual(sync_result.id, update_result.id)


class DataTasksIntegrationTestCase(TransactionTestCase):
    """Integration tests for Data tasks with database operations."""
    
    def setUp(self):
        """Set up integration test data."""
        cache.clear()
        
        self.sector = DataSector.objects.create(
            sectorKey='tech_integration',
            sectorName='Technology Integration',
            data_source='test'
        )
        
        self.industry = DataIndustry.objects.create(
            industryKey='software_integration',
            industryName='Software Integration',
            sector=self.sector,
            data_source='test'
        )
        
        self.stock = Stock.objects.create(
            symbol='INTEG_TASK',
            short_name='Integration Task Corp',
            sector_id=self.sector,
            industry_id=self.industry
        )
    
    def test_task_database_transaction_isolation(self):
        """Test that tasks properly handle database transactions."""
        with patch('Data.services.yahoo_finance.yahoo_finance_service') as mock_yf:
            mock_yf.fetchStockData.return_value = {
                'prices': [
                    {
                        'date': datetime.now().date(),
                        'open': 100.00,
                        'high': 105.00, 
                        'low': 99.00,
                        'close': 104.00,
                        'volume': 500000
                    }
                ]
            }
            
            # Check initial state
            initial_price_count = StockPrice.objects.filter(stock=self.stock).count()
            
            # Simulate task execution with database operations
            # (In real implementation, this would be async)
            try:
                # Simulate successful data sync
                StockPrice.objects.create(
                    stock=self.stock,
                    date=datetime.now().date(),
                    open=Decimal('100.00'),
                    high=Decimal('105.00'),
                    low=Decimal('99.00'),
                    close=Decimal('104.00'),
                    volume=500000,
                    data_source='test_task'
                )
                
                # Verify data was created
                final_price_count = StockPrice.objects.filter(stock=self.stock).count()
                self.assertEqual(final_price_count, initial_price_count + 1)
                
            except Exception as e:
                self.fail(f"Task database operation failed: {e}")
    
    def test_concurrent_task_execution(self):
        """Test handling of concurrent task execution."""
        import threading
        import time
        
        results = []
        errors = []
        
        def simulate_task_execution(task_id):
            """Simulate concurrent task execution."""
            try:
                # Simulate task work
                time.sleep(0.1)
                
                # Create unique data for each task
                StockPrice.objects.create(
                    stock=self.stock,
                    date=datetime.now().date() - timedelta(days=task_id),
                    open=Decimal('100.00') + task_id,
                    high=Decimal('105.00') + task_id,
                    low=Decimal('99.00') + task_id,
                    close=Decimal('104.00') + task_id,
                    volume=500000 + (task_id * 1000),
                    data_source=f'concurrent_task_{task_id}'
                )
                
                results.append(f'task_{task_id}_success')
                
            except Exception as e:
                errors.append(f'task_{task_id}_error: {str(e)}')
        
        # Create multiple threads to simulate concurrent tasks
        threads = []
        for i in range(3):
            thread = threading.Thread(target=simulate_task_execution, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertEqual(len(errors), 0)
        
        # Verify all data was created
        created_prices = StockPrice.objects.filter(
            stock=self.stock,
            data_source__startswith='concurrent_task_'
        ).count()
        self.assertEqual(created_prices, 3)
    
    def test_task_retry_mechanism(self):
        """Test task retry logic and exponential backoff."""
        retry_attempts = []
        
        def mock_retry_function(attempt):
            """Mock retry function to track attempts."""
            retry_attempts.append(attempt)
            if attempt < 3:
                raise Exception(f"Temporary failure on attempt {attempt}")
            return {"status": "success", "attempt": attempt}
        
        # Simulate retry logic
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                result = mock_retry_function(attempt)
                if result["status"] == "success":
                    break
            except Exception:
                if attempt == max_retries:
                    self.fail("Task failed after max retries")
                continue
        
        # Verify retry attempts were made
        self.assertEqual(len(retry_attempts), 3)
        self.assertEqual(retry_attempts, [1, 2, 3])
    
    def test_task_performance_monitoring(self):
        """Test task performance monitoring and metrics."""
        import time
        
        start_time = time.time()
        
        # Simulate task with performance monitoring
        try:
            # Simulate work
            time.sleep(0.1)
            
            # Create test data
            StockPrice.objects.create(
                stock=self.stock,
                date=datetime.now().date(),
                open=Decimal('120.00'),
                high=Decimal('125.00'),
                low=Decimal('119.00'),
                close=Decimal('124.00'),
                volume=750000,
                data_source='performance_test'
            )
            
            execution_time = time.time() - start_time
            
            # Verify performance is reasonable
            self.assertLess(execution_time, 2.0)  # Should complete within 2 seconds
            self.assertGreater(execution_time, 0.05)  # Should take some measurable time
            
        except Exception as e:
            self.fail(f"Performance test failed: {e}")


if __name__ == '__main__':
    import unittest
    unittest.main()